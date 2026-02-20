"""
Simple Scale Tuner for NNCF Quantizers.

Tunes quantization scales by minimizing MSE between quantized and original weights directly.
No calibration data needed - just optimizes scales to best preserve weight values.

Features:
- Separate learning rates for 2-bit and 4-bit quantizers
- Layer pattern filtering to tune only specific layers
- Early stopping when no improvement
- Automatic restoration of best/initial scales if tuning makes loss worse

Usage:
    from nncf_layerwise_ptq_tuner import tune_all_scales, TunerConfig, ScaleTuner

    # After nncf.compress_weights(...)

    # Option 1: Tune all layers
    config = TunerConfig(num_steps=100, learning_rate_2bit=1e-4, learning_rate_4bit=1e-4)
    results = tune_all_scales(tb, model, config)

    # Option 2: Tune specific layers by pattern
    config = TunerConfig(
        num_steps=200,
        learning_rate_2bit=1e-5,
        learning_rate_4bit=1e-4,
        layer_patterns=["layers:0:", "layers:4:", "layers:15:"],  # Match specific layers
        early_stop_patience=50,
        restore_best=True,
    )
    results = tune_all_scales(tb, model, config)

    # Option 3: Using ScaleTuner wrapper
    tuner = ScaleTuner(model)
    tuner.tune(
        tb,
        num_steps=100,
        layer_patterns=["down_proj"],  # Only tune down_proj layers
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nncf.torch.function_hook.hook_storage import decode_hook_name
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.model_graph_manager import get_module_by_name
from nncf.torch.model_graph_manager import split_const_name
from nncf.torch.quantization.layers import AsymmetricLoraQuantizer
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import SymmetricLoraQuantizer
from nncf.torch.quantization.layers import SymmetricQuantizer


@dataclass
class TunerConfig:
    """Configuration for scale tuning."""

    num_steps: int = 100
    learning_rate_2bit: float = 1e-4
    learning_rate_4bit: float = 1e-4
    verbose: bool = True
    log_interval: int = 20
    num_bits_to_tune: tuple[int, ...] = (2, 4)  # Only tune quantizers with these bit widths
    layer_patterns: Optional[list[str]] = None  # If set, only tune layers matching these patterns
    early_stop_patience: int = 50  # Stop if no improvement for this many steps
    restore_best: bool = True  # Restore best scale if final is worse
    warmup_steps: int = 10  # Number of warmup steps (linear warmup)
    min_lr_ratio: float = 0.01  # Minimum LR as ratio of max LR (for cosine annealing)

    def get_learning_rate(self, num_bits: int) -> float:
        """Get learning rate for a specific bit width."""
        if num_bits == 2:
            return self.learning_rate_2bit
        if num_bits == 4:
            return self.learning_rate_4bit
        return self.learning_rate_4bit  # Default to 4-bit lr

    def get_lr_at_step(self, step: int, base_lr: float) -> float:
        """
        Compute learning rate at a given step with warmup and cosine annealing.

        - Warmup: Linear increase from min_lr to base_lr over warmup_steps
        - Annealing: Cosine decay from base_lr to min_lr after warmup
        """
        min_lr = base_lr * self.min_lr_ratio

        if step < self.warmup_steps:
            # Linear warmup
            warmup_factor = step / max(1, self.warmup_steps)
            return min_lr + (base_lr - min_lr) * warmup_factor
        # Cosine annealing after warmup
        progress = (step - self.warmup_steps) / max(1, self.num_steps - self.warmup_steps)
        progress = min(1.0, progress)  # Clamp to [0, 1]
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine_factor

    def matches_layer_pattern(self, hook_name: str) -> bool:
        """Check if hook_name matches any of the layer patterns."""
        if self.layer_patterns is None:
            return True
        return any(pattern in hook_name for pattern in self.layer_patterns)


def get_scale_params(quantizer: BaseQuantizer) -> list[nn.Parameter]:
    """Get scale parameters from a quantizer."""
    if isinstance(quantizer, (SymmetricQuantizer, SymmetricLoraQuantizer)):
        return [quantizer._scale_param_storage]
    if isinstance(quantizer, (AsymmetricQuantizer, AsymmetricLoraQuantizer)):
        return [quantizer.input_low, quantizer.input_range]
    return []


def tune_quantizer_scale(
    tb, hook_name, quantizer: BaseQuantizer, original_weight: torch.Tensor, config: TunerConfig, model
) -> dict[str, Any]:
    """
    Tune a single quantizer's scale to minimize weight quantization error.

    Args:
        quantizer: NNCF quantizer module
        original_weight: Original (unquantized) weight tensor
        config: Tuning configuration

    Returns:
        Dictionary with tuning statistics
    """
    fq_params = list(quantizer.get_trainable_params().values())
    if not fq_params:
        return {"initial_loss": None, "final_loss": None, "num_steps": 0, "restored_best": False}

    # Store original weight (detached, no grad)
    orig_weight = original_weight.detach().clone()

    # Print info about weight and quantizer
    if config.verbose:
        print(
            f"  Weight: shape={orig_weight.shape}, range=[{orig_weight.min().item():.4f}, {orig_weight.max().item():.4f}]"
        )
        if isinstance(quantizer, (SymmetricQuantizer, SymmetricLoraQuantizer)):
            scale = quantizer.scale
            print(
                f"  Scale: shape={scale.shape}, range=[{scale.min().item():.4f}, {scale.max().item():.4f}], mean={scale.mean().item():.6f}"
            )
        if isinstance(quantizer, (SymmetricLoraQuantizer, AsymmetricLoraQuantizer)):
            lora_out = quantizer.lora_B @ quantizer.lora_A
            print(
                f"  LoRA: A_norm={quantizer.lora_A.norm().item():.4f}, B_norm={quantizer.lora_B.norm().item():.4f}, output_norm={lora_out.norm().item():.6f}"
            )

    # Save initial scale values for potential restoration
    initial_scale_values = [p.data.clone() for p in fq_params]
    best_scale_values = [p.data.clone() for p in fq_params]
    best_loss = float("inf")
    steps_without_improvement = 0

    # Enable gradients for scale params
    for p in fq_params:
        p.requires_grad = True

    params = list(model.parameters())
    num_train_params = sum(p.numel() for p in params if p.requires_grad)
    num_fq_params = sum(p.numel() for p in fq_params)
    num_all_params = sum(p.numel() for p in params)
    print(
        f"trainable params: {num_train_params:,d} || "
        f"all params: {num_all_params:,d} || "
        f"FQ params: {num_fq_params:,d} || "
        f"trainable%: {100 * num_train_params / num_all_params:.4f}"
    )
    assert num_fq_params == num_train_params

    base_lr = config.get_learning_rate(quantizer.num_bits)
    # Use SGD instead of Adam - Adam's momentum causes overshooting when scales are well-calibrated
    optimizer = torch.optim.SGD(fq_params, lr=base_lr)

    initial_loss = None
    final_loss = None
    actual_steps = 0

    for step in range(config.num_steps):
        # Update learning rate with warmup + cosine annealing
        current_lr = config.get_lr_at_step(step, base_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # Compute quantized weight
        quantizer.enable_quantization()
        q_weight = quantizer(orig_weight)

        # MSE loss between quantized and original
        loss = F.mse_loss(q_weight, orig_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Clamp params - only clamp absolute value away from zero, preserving sign
        with torch.no_grad():
            if isinstance(quantizer, (SymmetricQuantizer, SymmetricLoraQuantizer)):
                if not quantizer._is_using_log_scale_storage:
                    # Preserve sign but prevent absolute value from being too small
                    scale_data = quantizer._scale_param_storage.data
                    min_abs_scale = 1e-8
                    # For very small values, set to min_abs_scale with original sign (or positive if zero)
                    small_mask = scale_data.abs() < min_abs_scale
                    scale_data[small_mask] = torch.where(
                        scale_data[small_mask] >= 0,
                        torch.tensor(min_abs_scale, device=scale_data.device, dtype=scale_data.dtype),
                        torch.tensor(-min_abs_scale, device=scale_data.device, dtype=scale_data.dtype),
                    )
            elif isinstance(quantizer, (AsymmetricQuantizer, AsymmetricLoraQuantizer)):
                quantizer.input_range.data.clamp_(min=1e-8)

        loss_val = loss.item()

        tb.add_scalar(hook_name + "__loss", loss_val, step)
        tb.add_scalar(hook_name + "__lr", current_lr, step)

        # Log gradient norm
        grad_norms = [p.grad.norm().item() if p.grad is not None else 0.0 for p in fq_params]
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        tb.add_scalar(hook_name + "__grad_norm", avg_grad_norm, step)

        # Log scale norm
        if isinstance(quantizer, (SymmetricQuantizer, SymmetricLoraQuantizer)):
            scale_norm = quantizer.scale.norm().item()
            tb.add_scalar(hook_name + "__scale_norm", scale_norm, step)
        elif isinstance(quantizer, (AsymmetricQuantizer, AsymmetricLoraQuantizer)):
            input_range_norm = quantizer.input_range.norm().item()
            tb.add_scalar(hook_name + "__input_range_norm", input_range_norm, step)

        # Log LoRA adapter norms
        if isinstance(quantizer, (SymmetricLoraQuantizer, AsymmetricLoraQuantizer)):
            lora_A_norm = quantizer.lora_A.norm().item()
            lora_B_norm = quantizer.lora_B.norm().item()
            tb.add_scalar(hook_name + "__lora_A_norm", lora_A_norm, step)
            tb.add_scalar(hook_name + "__lora_B_norm", lora_B_norm, step)

        if initial_loss is None:
            initial_loss = loss_val
            best_loss = loss_val
        final_loss = loss_val
        actual_steps = step + 1

        # Track best scales
        if loss_val < best_loss:
            best_loss = loss_val
            best_scale_values = [p.data.clone() for p in fq_params]
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        if config.verbose and step % config.log_interval == 0:
            print(f"    Step {step}: loss = {loss_val:.8f}, lr = {current_lr:.2e}, grad = {avg_grad_norm:.2e}")

        # Early stopping
        if config.early_stop_patience > 0 and steps_without_improvement >= config.early_stop_patience:
            if config.verbose:
                print(f"    Early stopping at step {step} (no improvement for {config.early_stop_patience} steps)")
            break

    # Disable gradients
    for p in fq_params:
        p.requires_grad = False

    # Restore best scales if configured and beneficial
    restored_best = False
    if config.restore_best and best_loss < final_loss:
        with torch.no_grad():
            for p, best_val in zip(fq_params, best_scale_values):
                p.data.copy_(best_val)
        final_loss = best_loss
        restored_best = True
        if config.verbose:
            print(f"    Restored best scales (loss: {best_loss:.8f})")

    # Check if we made things worse than initial
    if config.restore_best and initial_loss is not None and final_loss > initial_loss:
        with torch.no_grad():
            for p, init_val in zip(fq_params, initial_scale_values):
                p.data.copy_(init_val)
        if config.verbose:
            print("    WARNING: Tuning made loss worse! Restored initial scales.")
        final_loss = initial_loss
        restored_best = True

    reduction = (initial_loss - final_loss) / initial_loss if initial_loss else 0
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "loss_reduction": reduction,
        "num_steps": actual_steps,
        "restored_best": restored_best,
    }


def get_weight_for_quantizer(model: nn.Module, hook_name: str) -> Optional[torch.Tensor]:
    """
    Extract the original weight tensor that a quantizer operates on.

    Uses NNCF's existing utilities: decode_hook_name, split_const_name, get_module_by_name.
    """
    try:
        hook_type, op_name, port_id = decode_hook_name(hook_name)
        module_name, weight_attr_name = split_const_name(op_name)
        module = get_module_by_name(module_name, model)
        weight_param = getattr(module, weight_attr_name)
        return weight_param.data if hasattr(weight_param, "data") else weight_param
    except Exception:
        return None

    return None


def tune_all_scales(
    tb,
    model: nn.Module,
    config: Optional[TunerConfig] = None,
) -> dict[str, dict[str, Any]]:
    """
    Tune scales for all quantizers in the model.

    Args:
        model: NNCF-compressed model
        config: Tuning configuration

    Returns:
        Dictionary mapping quantizer names to tuning results
    """
    config = config or TunerConfig()
    results = {}

    # Freeze model
    model.requires_grad_(False)

    hook_storage = get_hook_storage(model)
    quantizers = []

    # Collect quantizers
    for hook_name, module in hook_storage.named_hooks():
        if isinstance(
            module, (SymmetricQuantizer, AsymmetricQuantizer, SymmetricLoraQuantizer, AsymmetricLoraQuantizer)
        ):
            if config.num_bits_to_tune and module.num_bits not in config.num_bits_to_tune:
                continue
            if not config.matches_layer_pattern(hook_name):
                continue
            quantizers.append((hook_name, module))

    print(f"Tuning {len(quantizers)} quantizers...")

    for i, (hook_name, quantizer) in enumerate(quantizers):
        if config.verbose:
            qtype = "sym" if isinstance(quantizer, (SymmetricQuantizer, SymmetricLoraQuantizer)) else "asym"
            lora = "_lora" if isinstance(quantizer, (SymmetricLoraQuantizer, AsymmetricLoraQuantizer)) else ""
            lr = config.get_learning_rate(quantizer.num_bits)
            print(f"\n[{i + 1}/{len(quantizers)}] {hook_name}")
            print(f"  Type: {qtype}{lora}, Bits: {quantizer.num_bits}, LR: {lr}")

        # Get original weight
        orig_weight = get_weight_for_quantizer(model, hook_name)

        if orig_weight is None:
            raise RuntimeError("Can not find weight for hook: ", hook_name)
            # # Fallback: create a representative weight tensor from scale shape
            # if isinstance(quantizer, (SymmetricQuantizer, SymmetricLoraQuantizer)):
            #     scale_shape = quantizer.scale.shape
            # else:
            #     scale_shape = quantizer.input_low.shape

            # # Create test tensor matching quantizer's expected input shape
            # orig_weight = torch.randn(
            #     scale_shape,
            #     device=quantizer.scale.device if hasattr(quantizer, "scale") else quantizer.input_low.device,
            #     dtype=torch.float32,
            # )
            # if config.verbose:
            #     print(f"  Warning: Using synthetic weight of shape {scale_shape}")

        result = tune_quantizer_scale(tb, hook_name, quantizer, orig_weight, config, model)
        results[hook_name] = result

        if config.verbose and result["initial_loss"]:
            print(
                f"  Loss: {result['initial_loss']:.8f} -> {result['final_loss']:.8f} "
                f"({result['loss_reduction'] * 100:.1f}% reduction)"
            )

    return results


class ScaleTuner:
    """
    Simple class wrapper for scale tuning.

    Example:
        tuner = ScaleTuner(model)
        tuner.tune(num_steps=100, learning_rate_2bit=1e-2, learning_rate_4bit=1e-3)
        tuner.print_summary()
    """

    def __init__(self, model: nn.Module, num_bits_to_tune: tuple[int, ...] = (2, 4)):
        self.model = model
        self.num_bits_to_tune = num_bits_to_tune
        self.results: dict[str, dict[str, Any]] = {}

    def tune(
        self,
        tb,
        num_steps: int = 100,
        learning_rate_2bit: float = 1e-4,
        learning_rate_4bit: float = 1e-4,
        verbose: bool = True,
        layer_patterns: Optional[list[str]] = None,
        early_stop_patience: int = 50,
        restore_best: bool = True,
        warmup_steps: int = 10,
        min_lr_ratio: float = 0.01,
    ) -> dict[str, dict[str, Any]]:
        """
        Tune all quantizer scales.

        Args:
            tb: TensorBoard writer
            num_steps: Number of optimization steps
            learning_rate_2bit: Learning rate for 2-bit quantizers
            learning_rate_4bit: Learning rate for 4-bit quantizers
            verbose: Print progress
            layer_patterns: If set, only tune layers containing these substrings
            early_stop_patience: Stop if no improvement for this many steps (0 to disable)
            restore_best: Restore best scale if tuning makes loss worse
            warmup_steps: Number of linear warmup steps
            min_lr_ratio: Minimum LR as ratio of max LR (for cosine annealing)
        """
        config = TunerConfig(
            num_steps=num_steps,
            learning_rate_2bit=learning_rate_2bit,
            learning_rate_4bit=learning_rate_4bit,
            verbose=verbose,
            num_bits_to_tune=self.num_bits_to_tune,
            layer_patterns=layer_patterns,
            early_stop_patience=early_stop_patience,
            restore_best=restore_best,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )
        self.results = tune_all_scales(tb, self.model, config)
        return self.results

    def print_summary(self) -> None:
        """Print summary of tuning results."""
        if not self.results:
            print("No tuning results. Run tune() first.")
            return

        print("\n" + "=" * 60)
        print("Scale Tuning Summary")
        print("=" * 60)

        total_reduction = 0
        count = 0

        for name, result in self.results.items():
            if result["initial_loss"]:
                reduction = result["loss_reduction"] * 100
                total_reduction += reduction
                count += 1
                print(f"{name[:50]:50s} {reduction:6.1f}%")

        if count > 0:
            print("-" * 60)
            print(f"{'Average reduction:':50s} {total_reduction / count:6.1f}%")


# =============================================================================
# Direct Weight-based Tuning (Alternative simpler approach)
# =============================================================================


def tune_scale_for_weight(
    weight: torch.Tensor,
    num_bits: int = 4,
    symmetric: bool = True,
    group_size: int = -1,
    num_steps: int = 100,
    lr: float = 1e-3,
) -> torch.Tensor:
    """
    Directly compute optimal scale for a weight tensor by gradient descent.

    Args:
        weight: Original weight tensor
        num_bits: Number of quantization bits
        symmetric: Use symmetric quantization
        group_size: Group size for quantization (-1 for per-tensor)
        num_steps: Optimization steps
        lr: Learning rate

    Returns:
        Optimized scale tensor
    """
    weight = weight.detach().float()

    # Determine scale shape
    if group_size > 0 and weight.dim() >= 2:
        # Per-group quantization
        num_groups = (weight.shape[-1] + group_size - 1) // group_size
        scale_shape = list(weight.shape[:-1]) + [num_groups]
    else:
        # Per-tensor or per-channel
        scale_shape = [1]

    # Initialize scale from weight statistics
    if symmetric:
        level_high = 2 ** (num_bits - 1) - 1
        init_scale = weight.abs().max() / level_high
    else:
        level_high = 2**num_bits - 1
        init_scale = (weight.max() - weight.min()) / level_high

    scale = torch.full(scale_shape, init_scale.item(), device=weight.device, requires_grad=True)

    optimizer = torch.optim.AdamW([scale], lr=lr, weight_decay=0)

    if symmetric:
        level_low = -(2 ** (num_bits - 1))
        level_high = 2 ** (num_bits - 1) - 1
    else:
        level_low = 0
        level_high = 2**num_bits - 1

    for _ in range(num_steps):
        # Quantize
        scale_clamped = scale.clamp(min=1e-8)
        scaled = weight / scale_clamped
        quantized = torch.clamp(torch.round(scaled), level_low, level_high)
        dequantized = quantized * scale_clamped

        # Loss
        loss = F.mse_loss(dequantized, weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return scale.detach().clamp(min=1e-8)


# =============================================================================
# Tests
# =============================================================================


def test_direct_scale_tuning():
    """Test direct scale optimization."""
    print("\n=== Test: Direct Scale Tuning ===")

    # Create random weight
    weight = torch.randn(64, 128)

    # Tune scale
    scale = tune_scale_for_weight(weight, num_bits=4, num_steps=50, lr=1e-2)

    # Compute final quantization error
    level_low, level_high = -8, 7
    scaled = weight / scale
    quantized = torch.clamp(torch.round(scaled), level_low, level_high)
    dequantized = quantized * scale

    mse = F.mse_loss(dequantized, weight).item()
    print(f"  Final MSE: {mse:.8f}")
    print(f"  Scale: {scale.item():.6f}")
    print("✓ Direct scale tuning works")


def test_quantizer_scale_tuning():
    """Test tuning NNCF quantizer scales."""
    print("\n=== Test: Quantizer Scale Tuning ===")

    from nncf.torch.quantization.layers import PTQuantizerSpec

    # Create quantizer
    qspec = PTQuantizerSpec(
        num_bits=4,
        mode=None,
        signedness_to_force=True,
        narrow_range=False,
        half_range=False,
        logarithm_scale=False,
        compression_lr_multiplier=1.0,
        is_quantized_on_export=False,
        scale_shape=(1,),
    )
    quantizer = SymmetricQuantizer(qspec)

    # Create weight
    weight = torch.randn(64, 128)

    # Set bad initial scale
    quantizer._scale_param_storage.data.fill_(0.5)

    # Measure initial error
    quantizer.enable_quantization()
    with torch.no_grad():
        q_initial = quantizer(weight)
    initial_mse = F.mse_loss(q_initial, weight).item()

    # Tune
    config = TunerConfig(num_steps=50, learning_rate_2bit=1e-2, learning_rate_4bit=1e-2, verbose=False)
    result = tune_quantizer_scale(quantizer, weight, config)

    # Measure final error
    with torch.no_grad():
        q_final = quantizer(weight)
    final_mse = F.mse_loss(q_final, weight).item()

    print(f"  Initial MSE: {initial_mse:.8f}")
    print(f"  Final MSE: {final_mse:.8f}")
    print(f"  Reduction: {(1 - final_mse / initial_mse) * 100:.1f}%")

    assert final_mse < initial_mse, "Tuning should reduce MSE"
    print("✓ Quantizer scale tuning works")


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("Scale Tuner Tests")
    print("=" * 60)

    test_direct_scale_tuning()
    test_quantizer_scale_tuning()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NNCF Scale Tuner")
    parser.add_argument("--test", action="store_true", help="Run tests")
    args = parser.parse_args()

    if args.test:
        run_tests()
    else:
        print("NNCF Scale Tuner - Tune quantization scales to minimize weight error")
        print("\nUsage:")
        print("  python nncf_layerwise_ptq_tuner.py --test")
        print("\nAs module:")
        print("  from nncf_layerwise_ptq_tuner import tune_all_scales, ScaleTuner")
        print("  results = tune_all_scales(model)")
        print("  # or")
        print("  tuner = ScaleTuner(model)")
        print("  tuner.tune(num_steps=100)")
