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
from torch.amp import GradScaler

from nncf.torch.function_hook.hook_storage import decode_hook_name
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.model_graph_manager import get_module_by_name
from nncf.torch.model_graph_manager import split_const_name
from nncf.torch.quantization.layers import AsymmetricLoraQuantizer
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import SymmetricLoraQuantizer
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.quantization.quantize_functions import set_use_autograd_quantize


@dataclass
class TunerConfig:
    """Configuration for scale tuning."""

    num_steps: int = 100
    learning_rate_2bit: float = 1e-4
    learning_rate_4bit: float = 1e-4
    learning_rate_lora: float = 1e-4  # Separate LR for LoRA adapters (common for all bit widths)
    scheduler_type_scale: str = "cosine"  # "cosine" or "constant" for scale params
    scheduler_type_lora: str = "cosine"  # "cosine" or "constant" for LoRA params
    outlier_ratio: float = 0.001  # Exclude top-k% largest losses (0.001 = 0.1%)
    loss_type: str = "mse"  # Loss function: "mse", "nmse", "cosine", "sqnr", "log_mse"
    verbose: bool = True
    log_interval: int = 20
    num_bits_to_tune: tuple[int, ...] = (2, 4)  # Only tune quantizers with these bit widths
    layer_patterns: Optional[list[str]] = None  # If set, only tune layers matching these patterns
    early_stop_patience: int = 50  # Stop if no improvement for this many steps
    restore_best: bool = True  # Restore best scale if final is worse
    warmup_steps: int = 10  # Number of warmup steps (linear warmup)
    min_lr_ratio: float = 0.01  # Minimum LR as ratio of max LR (for cosine annealing)
    use_autograd_quantize: bool = False  # Use STE-based autograd for gradient computation

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


def compute_loss(
    output: torch.Tensor, target: torch.Tensor, loss_type: str = "mse", outlier_ratio: float = 0.001
) -> torch.Tensor:
    """
    Compute loss between quantized and original weights.

    Supported loss types:
    - "mse": Mean squared error (optionally with outlier removal)
    - "nmse": Normalized MSE - scales error by weight magnitude (better gradient signal)
    - "cosine": Cosine distance (1 - cosine_similarity) - preserves direction
    - "sqnr": Negative SQNR (Signal-to-Quantization-Noise Ratio) - standard metric
    - "log_mse": Log of scaled MSE - compresses dynamic range, larger gradients

    Args:
        output: Quantized tensor
        target: Original (reference) tensor
        loss_type: Type of loss function to use
        outlier_ratio: Fraction of elements to exclude for MSE variants (0.001 = 0.1%)

    Returns:
        Scalar loss value
    """
    output_f = output.float().flatten()
    target_f = target.float().flatten()

    if loss_type == "mse":
        # Standard MSE with optional outlier removal
        se = (output_f - target_f).pow(2)
        n = se.numel()
        if outlier_ratio > 0 and n > 1:
            k = max(1, int(n * outlier_ratio))
            threshold, _ = se.kthvalue(n - k + 1)
            mask = se < threshold
            return se[mask].mean()
        return se.mean()

    if loss_type == "nmse":
        # Normalized MSE: error scaled by weight magnitude
        # This gives ~constant relative gradient regardless of weight magnitude
        eps = 1e-8
        se = ((output_f - target_f) / (target_f.abs() + eps)).pow(2)
        n = se.numel()
        if outlier_ratio > 0 and n > 1:
            k = max(1, int(n * outlier_ratio))
            threshold, _ = se.kthvalue(n - k + 1)
            mask = se < threshold
            return se[mask].mean()
        return se.mean()

    if loss_type == "cosine":
        # Cosine distance: 1 - cos_similarity
        # Range [0, 2], 0 = identical direction
        cos_sim = F.cosine_similarity(output_f.unsqueeze(0), target_f.unsqueeze(0))
        return 1.0 - cos_sim.squeeze()

    if loss_type == "sqnr":
        # Negative SQNR: -10 * log10(signal_power / noise_power)
        # We minimize this, so higher SQNR = lower loss
        # Typical range: -20 to -60 dB for good quantization
        signal_power = (target_f**2).mean()
        noise_power = ((output_f - target_f) ** 2).mean().clamp(min=1e-10)
        sqnr = 10.0 * torch.log10(signal_power / noise_power)
        return -sqnr  # Negative so minimizing increases SQNR

    if loss_type == "log_mse":
        # Log-scaled MSE: log(1 + mse * scale)
        # Compresses dynamic range, provides larger gradients for small MSE
        se = (output_f - target_f).pow(2)
        n = se.numel()
        if outlier_ratio > 0 and n > 1:
            k = max(1, int(n * outlier_ratio))
            threshold, _ = se.kthvalue(n - k + 1)
            mask = se < threshold
            mse = se[mask].mean()
        else:
            mse = se.mean()
        # Scale factor to make log more sensitive (1e6 makes MSE=1e-6 -> log(2))
        scale = 1e6
        return torch.log1p(mse * scale)

    raise ValueError(f"Unknown loss_type: {loss_type}. Supported: mse, nmse, cosine, sqnr, log_mse")


def robust_mse_loss(output: torch.Tensor, target: torch.Tensor, outlier_ratio: float = 0.001) -> torch.Tensor:
    """
    Robust MSE loss that excludes top-k largest element-wise losses.
    DEPRECATED: Use compute_loss(output, target, loss_type="mse", outlier_ratio=...) instead.
    """
    return compute_loss(output, target, loss_type="mse", outlier_ratio=outlier_ratio)


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
    # Enable/disable autograd-based quantization with STE
    set_use_autograd_quantize(config.use_autograd_quantize)

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

    # Separate scale params from LoRA adapter params for different learning rates
    scale_params = []
    lora_params = []
    if isinstance(quantizer, (SymmetricLoraQuantizer, AsymmetricLoraQuantizer)):
        adapters = quantizer.get_adapters()
        adapter_set = set(adapters.values())
        print(f"  DEBUG: LoRA quantizer detected, adapters: {list(adapters.keys())}")
        print(f"  DEBUG: adapter_set has {len(adapter_set)} unique params")
        for p in fq_params:
            if p in adapter_set:
                lora_params.append(p)
                print(f"    -> Found LoRA param: shape={p.shape}")
            else:
                scale_params.append(p)
                print(f"    -> Found scale param: shape={p.shape}")
    else:
        scale_params = fq_params
        print(f"  DEBUG: Non-LoRA quantizer, all {len(fq_params)} params are scale params")

    print(f"  DEBUG: scale_params={len(scale_params)}, lora_params={len(lora_params)}")

    base_lr = config.get_learning_rate(quantizer.num_bits)
    lora_lr = config.learning_rate_lora

    # Use SGD instead of Adam - Adam's momentum causes overshooting when scales are well-calibrated
    # Create separate optimizers for scales and LoRA, each with its own scheduler
    scale_optimizer = torch.optim.SGD(scale_params, lr=base_lr) if scale_params else None
    lora_optimizer = torch.optim.SGD(lora_params, lr=lora_lr) if lora_params else None

    print(f"  DEBUG: scale_optimizer created: {scale_optimizer is not None}, scale_lr={base_lr}")
    print(f"  DEBUG: lora_optimizer created: {lora_optimizer is not None}, lora_lr={lora_lr}")

    # Create schedulers based on config
    def create_scheduler(optimizer, scheduler_type: str, base_lr: float):
        """Create scheduler based on type."""
        if optimizer is None:
            return None
        if scheduler_type == "constant":
            # ConstantLR with factor=1.0 keeps LR unchanged
            return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=config.num_steps)

        # Default: cosine annealing (with optional warmup)
        if config.warmup_steps <= 0:
            # No warmup - just use CosineAnnealingLR directly
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.num_steps,
                eta_min=base_lr * config.min_lr_ratio,
            )

        # With warmup: LinearLR for warmup, then CosineAnnealingLR
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=config.min_lr_ratio,
            end_factor=1.0,
            total_iters=config.warmup_steps,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_steps - config.warmup_steps,
            eta_min=base_lr * config.min_lr_ratio,
        )
        # Chain them: warmup first, then cosine
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.warmup_steps],
        )

    scale_scheduler = create_scheduler(scale_optimizer, config.scheduler_type_scale, base_lr)
    lora_scheduler = create_scheduler(lora_optimizer, config.scheduler_type_lora, lora_lr)

    # Debug: print scheduler type info
    print(f"  DEBUG: scale_scheduler type: {config.scheduler_type_scale}, created: {scale_scheduler is not None}")
    print(f"  DEBUG: lora_scheduler type: {config.scheduler_type_lora}, created: {lora_scheduler is not None}")
    if lora_scheduler:
        print(f"  DEBUG: lora_scheduler class: {type(lora_scheduler).__name__}")

    # Only use GradScaler for fp16 manual backward (not autograd quantize)
    # With autograd quantize (STE), computation is in fp32 so scaler is not needed
    use_grad_scaler = not config.use_autograd_quantize and orig_weight.dtype == torch.float16
    scaler = GradScaler(init_scale=2**16, enabled=use_grad_scaler)

    initial_loss = None
    final_loss = None
    actual_steps = 0

    for step in range(config.num_steps):
        # Compute quantized weight
        quantizer.enable_quantization()
        q_weight = quantizer(orig_weight)

        # Compute loss using selected loss type
        loss = compute_loss(q_weight, orig_weight, config.loss_type, config.outlier_ratio)

        # Zero gradients for both optimizers
        if scale_optimizer:
            scale_optimizer.zero_grad()
        if lora_optimizer:
            lora_optimizer.zero_grad()

        # Backward pass
        scaler.scale(loss).backward()

        # Step both optimizers
        if scale_optimizer:
            scaler.step(scale_optimizer)
        if lora_optimizer:
            scaler.step(lora_optimizer)
        scaler.update()

        # Step schedulers (after optimizer step)
        if scale_scheduler:
            scale_scheduler.step()
        if lora_scheduler:
            lora_scheduler.step()

        # Debug: print LR after scheduler step for first few iterations
        if config.verbose and step < 3:
            scale_lr_after = scale_optimizer.param_groups[0]["lr"] if scale_optimizer else 0.0
            lora_lr_after = lora_optimizer.param_groups[0]["lr"] if lora_optimizer else 0.0
            print(
                f"    DEBUG step {step}: after scheduler.step() -> scale_lr={scale_lr_after:.2e}, lora_lr={lora_lr_after:.2e}"
            )

        # Clamp params - only clamp absolute value away from zero, preserving sign
        # with torch.no_grad():
        #     if isinstance(quantizer, (SymmetricQuantizer, SymmetricLoraQuantizer)):
        #         if not quantizer._is_using_log_scale_storage:
        #             # Preserve sign but prevent absolute value from being too small
        #             scale_data = quantizer._scale_param_storage.data
        #             min_abs_scale = 1e-8
        #             # For very small values, set to min_abs_scale with original sign (or positive if zero)
        #             small_mask = scale_data.abs() < min_abs_scale
        #             scale_data[small_mask] = torch.where(
        #                 scale_data[small_mask] >= 0,
        #                 torch.tensor(min_abs_scale, device=scale_data.device, dtype=scale_data.dtype),
        #                 torch.tensor(-min_abs_scale, device=scale_data.device, dtype=scale_data.dtype),
        #             )
        #     elif isinstance(quantizer, (AsymmetricQuantizer, AsymmetricLoraQuantizer)):
        #         quantizer.input_range.data.clamp_(min=1e-8)

        loss_val = loss.item()

        # Log learning rates (scale LR and LoRA LR from separate optimizers)
        tb.add_scalar(hook_name + "__loss", loss_val, step)
        if scale_optimizer:
            scale_lr_current = scale_optimizer.param_groups[0]["lr"]
            tb.add_scalar(hook_name + "__lr_scale", scale_lr_current, step)
        if lora_optimizer:
            lora_lr_current = lora_optimizer.param_groups[0]["lr"]
            tb.add_scalar(hook_name + "__lr_lora", lora_lr_current, step)

        # Log gradient norm (unscaled if using GradScaler)
        if use_grad_scaler:
            # Get unscaled gradient norm
            inv_scale = 1.0 / scaler.get_scale()
            grad_norms = [(p.grad.norm().item() * inv_scale) if p.grad is not None else 0.0 for p in fq_params]
        else:
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
            scale_lr_log = scale_optimizer.param_groups[0]["lr"] if scale_optimizer else 0.0
            lora_lr_log = lora_optimizer.param_groups[0]["lr"] if lora_optimizer else 0.0
            lr_str = f"lr_scale={scale_lr_log:.2e}"
            if lora_optimizer:
                lr_str += f", lr_lora={lora_lr_log:.2e}"
            print(f"    Step {step}: loss = {loss_val:.10f}, {lr_str}, grad = {avg_grad_norm:.4e}")

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
            print(f"  Type: {qtype}{lora}, Bits: {quantizer.num_bits}, LR: {lr}, Loss: {config.loss_type}")

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
        learning_rate_lora: float = 1e-4,
        scheduler_type_scale: str = "cosine",
        scheduler_type_lora: str = "cosine",
        outlier_ratio: float = 0.001,
        loss_type: str = "mse",
        verbose: bool = True,
        layer_patterns: Optional[list[str]] = None,
        early_stop_patience: int = 50,
        restore_best: bool = True,
        warmup_steps: int = 10,
        min_lr_ratio: float = 0.01,
        use_autograd_quantize=True,
    ) -> dict[str, dict[str, Any]]:
        """
        Tune all quantizer scales.

        Args:
            tb: TensorBoard writer
            num_steps: Number of optimization steps
            learning_rate_2bit: Learning rate for 2-bit quantizers (scales)
            learning_rate_4bit: Learning rate for 4-bit quantizers (scales)
            learning_rate_lora: Learning rate for LoRA adapters (common for all bit widths)
            scheduler_type_scale: LR scheduler for scales - "cosine" or "constant"
            scheduler_type_lora: LR scheduler for LoRA - "cosine" or "constant"
            outlier_ratio: Exclude top-k% largest losses (0.001 = 0.1%) for stability
            loss_type: Loss function - "mse", "nmse", "cosine", "sqnr", "log_mse"
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
            learning_rate_lora=learning_rate_lora,
            scheduler_type_scale=scheduler_type_scale,
            scheduler_type_lora=scheduler_type_lora,
            outlier_ratio=outlier_ratio,
            loss_type=loss_type,
            verbose=verbose,
            num_bits_to_tune=self.num_bits_to_tune,
            layer_patterns=layer_patterns,
            early_stop_patience=early_stop_patience,
            restore_best=restore_best,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
            use_autograd_quantize=use_autograd_quantize,
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

    def lr_find(
        self,
        layer_pattern: Optional[str] = None,
        min_lr: float = 1e-7,
        max_lr: float = 10.0,
        num_steps: int = 100,
        loss_type: str = "nmse",
        smooth_factor: float = 0.05,
        divergence_threshold: float = 4.0,
        use_autograd_quantize: bool = True,
        param_type: str = "all",
    ) -> dict[str, Any]:
        """
        Find optimal learning rate using Leslie Smith's LR Range Test.

        Exponentially increases LR from min_lr to max_lr, recording loss at each step.
        Returns the LR where loss decreases fastest (steepest negative gradient).

        Args:
            layer_pattern: Pattern to match layer name (uses first match). None = first quantizer.
            min_lr: Starting learning rate
            max_lr: Maximum learning rate
            num_steps: Number of steps for the sweep
            loss_type: Loss function to use ("mse", "nmse", "cosine", "sqnr", "log_mse")
            smooth_factor: Exponential smoothing factor for loss (0 = no smoothing)
            divergence_threshold: Stop if loss exceeds this multiple of best loss
            use_autograd_quantize: Use STE-based autograd for gradients
            param_type: Which params to sweep - "scale", "lora", or "all"

        Returns:
            Dictionary with:
                - suggested_lr: Recommended learning rate
                - lr_at_min_loss: LR where loss was minimum
                - lrs: List of learning rates tested
                - losses: List of losses at each LR
                - smoothed_losses: Smoothed losses
                - gradients: Loss gradient at each point
        """
        set_use_autograd_quantize(use_autograd_quantize)

        # Find quantizers (use named_hooks() same as tune_all_scales)
        hook_storage = get_hook_storage(self.model)
        quantizers = []
        for hook_name, hook in hook_storage.named_hooks():
            if isinstance(hook, BaseQuantizer):
                if hook.num_bits not in self.num_bits_to_tune:
                    continue
                if layer_pattern is None or layer_pattern in hook_name:
                    quantizers.append((hook_name, hook))
                    break  # Use first match

        if not quantizers:
            print("No matching quantizer found!")
            return {}

        hook_name, quantizer = quantizers[0]
        print(f"LR Range Test on: {hook_name}")
        print(f"  Param type: {param_type}, Loss: {loss_type}")
        print(f"  LR range: [{min_lr:.2e}, {max_lr:.2e}], Steps: {num_steps}")

        # Get original weight
        orig_weight = get_weight_for_quantizer(self.model, hook_name)
        if orig_weight is None:
            msg = f"Cannot find weight for hook: {hook_name}"
            raise RuntimeError(msg)

        orig_weight = orig_weight.detach().clone()

        # Get trainable params
        fq_params = list(quantizer.get_trainable_params().values())
        if not fq_params:
            print("No trainable params found!")
            return {}

        # Separate scale vs LoRA params
        scale_params = []
        lora_params = []
        if isinstance(quantizer, (SymmetricLoraQuantizer, AsymmetricLoraQuantizer)):
            adapters = quantizer.get_adapters()
            adapter_set = set(adapters.values())
            for p in fq_params:
                if p in adapter_set:
                    lora_params.append(p)
                else:
                    scale_params.append(p)
        else:
            scale_params = fq_params

        # Select params based on param_type
        if param_type == "scale":
            params = scale_params
        elif param_type == "lora":
            params = lora_params
        else:  # "all"
            params = fq_params

        if not params:
            print(f"No {param_type} params found!")
            return {}

        print(f"  Testing {len(params)} params")

        # Save original parameter states
        original_states = [p.data.clone() for p in params]

        # Compute LR multiplier per step: lr = min_lr * mult^step
        # At step num_steps-1: max_lr = min_lr * mult^(num_steps-1)
        mult = (max_lr / min_lr) ** (1.0 / (num_steps - 1))

        # Create optimizer with min_lr
        optimizer = torch.optim.SGD(params, lr=min_lr)

        # Storage for results
        lrs = []
        losses = []
        smoothed_losses = []
        smoothed_loss = None
        best_loss = float("inf")

        for step in range(num_steps):
            current_lr = min_lr * (mult**step)

            # Update LR
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            # Forward pass
            quantizer.enable_quantization()
            q_weight = quantizer(orig_weight)

            # Compute loss
            loss = compute_loss(q_weight, orig_weight, loss_type, outlier_ratio=0.0)
            loss_val = loss.item()

            # Smooth loss
            if smoothed_loss is None:
                smoothed_loss = loss_val
            else:
                smoothed_loss = smooth_factor * loss_val + (1 - smooth_factor) * smoothed_loss

            lrs.append(current_lr)
            losses.append(loss_val)
            smoothed_losses.append(smoothed_loss)

            # Track best
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            # Check for divergence
            if smoothed_loss > best_loss * divergence_threshold:
                print(f"  Stopped at step {step}: loss diverged")
                break

            # Backward and step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Restore original parameters
        for p, orig_state in zip(params, original_states):
            p.data.copy_(orig_state)

        # Compute gradient of smoothed loss w.r.t. log(lr)
        # Gradient = d(loss) / d(log_lr) ≈ (loss[i+1] - loss[i-1]) / (log_lr[i+1] - log_lr[i-1])
        log_lrs = [math.log10(lr) for lr in lrs]
        gradients = [0.0]  # First point has no gradient
        for i in range(1, len(smoothed_losses) - 1):
            grad = (smoothed_losses[i + 1] - smoothed_losses[i - 1]) / (log_lrs[i + 1] - log_lrs[i - 1])
            gradients.append(grad)
        if len(smoothed_losses) > 1:
            gradients.append(0.0)  # Last point

        # Find LR with steepest negative gradient (fastest loss decrease)
        min_grad_idx = 0
        min_grad = 0.0
        for i, grad in enumerate(gradients):
            if grad < min_grad:
                min_grad = grad
                min_grad_idx = i

        # Find LR at minimum loss
        min_loss_idx = smoothed_losses.index(min(smoothed_losses))

        # Suggested LR: where gradient is most negative
        # Common practice: use LR slightly before the steepest point or at steepest
        suggested_lr = lrs[min_grad_idx] if min_grad_idx > 0 else lrs[min_loss_idx]

        # Alternative: LR one order of magnitude before min_loss point
        safe_lr = lrs[min_loss_idx] / 10 if min_loss_idx > 0 else lrs[0]

        print("\n  Results:")
        print(f"    LR at steepest descent: {lrs[min_grad_idx]:.2e} (gradient={min_grad:.4f})")
        print(f"    LR at minimum loss:     {lrs[min_loss_idx]:.2e} (loss={smoothed_losses[min_loss_idx]:.6f})")
        print(f"    Safe LR (1/10 of min):  {safe_lr:.2e}")
        print(f"    Suggested LR:           {suggested_lr:.2e}")

        return {
            "suggested_lr": suggested_lr,
            "lr_at_min_loss": lrs[min_loss_idx],
            "lr_at_steepest": lrs[min_grad_idx],
            "safe_lr": safe_lr,
            "lrs": lrs,
            "losses": losses,
            "smoothed_losses": smoothed_losses,
            "gradients": gradients,
            "min_loss": min(smoothed_losses),
            "best_loss_idx": min_loss_idx,
            "steepest_idx": min_grad_idx,
        }

    def plot_lr_find(self, result: dict[str, Any], save_path: Optional[str] = None) -> None:
        """
        Plot LR finder results.

        Args:
            result: Output from lr_find()
            save_path: If provided, save plot to this path
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return

        if not result:
            print("No results to plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        lrs = result["lrs"]
        losses = result["losses"]
        smoothed = result["smoothed_losses"]
        gradients = result["gradients"]

        # Left plot: Loss vs LR
        ax1.plot(lrs, losses, "b-", alpha=0.3, label="Raw loss")
        ax1.plot(lrs, smoothed, "b-", linewidth=2, label="Smoothed loss")
        ax1.axvline(result["suggested_lr"], color="r", linestyle="--", label=f"Suggested: {result['suggested_lr']:.2e}")
        ax1.axvline(
            result["lr_at_min_loss"], color="g", linestyle=":", label=f"Min loss: {result['lr_at_min_loss']:.2e}"
        )
        ax1.set_xscale("log")
        ax1.set_xlabel("Learning Rate")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss vs Learning Rate")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right plot: Gradient vs LR
        ax2.plot(lrs, gradients, "r-", linewidth=2)
        ax2.axhline(0, color="k", linestyle="-", alpha=0.3)
        ax2.axvline(
            result["lr_at_steepest"], color="r", linestyle="--", label=f"Steepest: {result['lr_at_steepest']:.2e}"
        )
        ax2.set_xscale("log")
        ax2.set_xlabel("Learning Rate")
        ax2.set_ylabel("d(Loss)/d(log LR)")
        ax2.set_title("Loss Gradient (negative = loss decreasing)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


# =============================================================================
# Direct Weight-based Tuning (Alternative simpler approach)
# =============================================================================


# def tune_scale_for_weight(
#     weight: torch.Tensor,
#     num_bits: int = 4,
#     symmetric: bool = True,
#     group_size: int = -1,
#     num_steps: int = 100,
#     lr: float = 1e-3,
# ) -> torch.Tensor:
#     """
#     Directly compute optimal scale for a weight tensor by gradient descent.

#     Args:
#         weight: Original weight tensor
#         num_bits: Number of quantization bits
#         symmetric: Use symmetric quantization
#         group_size: Group size for quantization (-1 for per-tensor)
#         num_steps: Optimization steps
#         lr: Learning rate

#     Returns:
#         Optimized scale tensor
#     """
#     weight = weight.detach().float()

#     # Determine scale shape
#     if group_size > 0 and weight.dim() >= 2:
#         # Per-group quantization
#         num_groups = (weight.shape[-1] + group_size - 1) // group_size
#         scale_shape = list(weight.shape[:-1]) + [num_groups]
#     else:
#         # Per-tensor or per-channel
#         scale_shape = [1]

#     # Initialize scale from weight statistics
#     if symmetric:
#         level_high = 2 ** (num_bits - 1) - 1
#         init_scale = weight.abs().max() / level_high
#     else:
#         level_high = 2**num_bits - 1
#         init_scale = (weight.max() - weight.min()) / level_high

#     scale = torch.full(scale_shape, init_scale.item(), device=weight.device, requires_grad=True)

#     optimizer = torch.optim.AdamW([scale], lr=lr, weight_decay=0)

#     if symmetric:
#         level_low = -(2 ** (num_bits - 1))
#         level_high = 2 ** (num_bits - 1) - 1
#     else:
#         level_low = 0
#         level_high = 2**num_bits - 1

#     for _ in range(num_steps):
#         # Quantize
#         scale_clamped = scale.clamp(min=1e-8)
#         scaled = weight / scale_clamped
#         quantized = torch.clamp(torch.round(scaled), level_low, level_high)
#         dequantized = quantized * scale_clamped

#         # Loss
#         loss = F.mse_loss(dequantized, weight)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     return scale.detach().clamp(min=1e-8)


# =============================================================================
# Tests
# =============================================================================


# def test_direct_scale_tuning():
#     """Test direct scale optimization."""
#     print("\n=== Test: Direct Scale Tuning ===")

#     # Create random weight
#     weight = torch.randn(64, 128)

#     # Tune scale
#     scale = tune_scale_for_weight(weight, num_bits=4, num_steps=50, lr=1e-2)

#     # Compute final quantization error
#     level_low, level_high = -8, 7
#     scaled = weight / scale
#     quantized = torch.clamp(torch.round(scaled), level_low, level_high)
#     dequantized = quantized * scale

#     mse = F.mse_loss(dequantized, weight).item()
#     print(f"  Final MSE: {mse:.8f}")
#     print(f"  Scale: {scale.item():.6f}")
#     print("✓ Direct scale tuning works")


# def test_quantizer_scale_tuning():
#     """Test tuning NNCF quantizer scales."""
#     print("\n=== Test: Quantizer Scale Tuning ===")

#     from nncf.torch.quantization.layers import PTQuantizerSpec

#     # Create quantizer
#     qspec = PTQuantizerSpec(
#         num_bits=4,
#         mode=None,
#         signedness_to_force=True,
#         narrow_range=False,
#         half_range=False,
#         logarithm_scale=False,
#         compression_lr_multiplier=1.0,
#         is_quantized_on_export=False,
#         scale_shape=(1,),
#     )
#     quantizer = SymmetricQuantizer(qspec)

#     # Create weight
#     weight = torch.randn(64, 128)

#     # Set bad initial scale
#     quantizer._scale_param_storage.data.fill_(0.5)

#     # Measure initial error
#     quantizer.enable_quantization()
#     with torch.no_grad():
#         q_initial = quantizer(weight)
#     initial_mse = F.mse_loss(q_initial, weight).item()

#     # Tune
#     config = TunerConfig(num_steps=50, learning_rate_2bit=1e-2, learning_rate_4bit=1e-2, verbose=False)
#     result = tune_quantizer_scale(quantizer, weight, config)

#     # Measure final error
#     with torch.no_grad():
#         q_final = quantizer(weight)
#     final_mse = F.mse_loss(q_final, weight).item()

#     print(f"  Initial MSE: {initial_mse:.8f}")
#     print(f"  Final MSE: {final_mse:.8f}")
#     print(f"  Reduction: {(1 - final_mse / initial_mse) * 100:.1f}%")

#     assert final_mse < initial_mse, "Tuning should reduce MSE"
#     print("✓ Quantizer scale tuning works")


# def run_tests():
#     """Run all tests."""
#     print("=" * 60)
#     print("Scale Tuner Tests")
#     print("=" * 60)

#     test_direct_scale_tuning()
#     test_quantizer_scale_tuning()

#     print("\n" + "=" * 60)
#     print("All tests passed!")
#     print("=" * 60)


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="NNCF Scale Tuner")
#     parser.add_argument("--test", action="store_true", help="Run tests")
#     args = parser.parse_args()

#     if args.test:
#         run_tests()
#     else:
#         print("NNCF Scale Tuner - Tune quantization scales to minimize weight error")
#         print("\nUsage:")
#         print("  python nncf_layerwise_ptq_tuner.py --test")
#         print("\nAs module:")
#         print("  from nncf_layerwise_ptq_tuner import tune_all_scales, ScaleTuner")
#         print("  results = tune_all_scales(model)")
#         print("  # or")
#         print("  tuner = ScaleTuner(model)")
#         print("  tuner.tune(num_steps=100)")
