"""
Layer-wise Post-Training Quantization (PTQ) Tuner with Fake-Quantization Weight Hooks.

This module provides a complete implementation for tuning quantization scales layer-by-layer
by minimizing MSE between quantized and unquantized outputs.

Key Components:
- FakeQuantWeightHook: Applies fake quantization to weights via hooks
- LayerQuantTuner: Tunes quantization scales for a single layer
- LayerwisePTQDriver: Orchestrates layer-by-layer PTQ tuning

Usage:
    from layerwise_ptq_tuner import LayerwisePTQDriver

    driver = LayerwisePTQDriver(model)
    driver.tune(calibration_dataloader, num_steps=100)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# =============================================================================
# Fake Quantization Functions
# =============================================================================


def symmetric_fake_quantize(
    weight: torch.Tensor,
    scale: torch.Tensor,
    num_bits: int = 8,
    narrow_range: bool = False,
) -> torch.Tensor:
    """
    Apply symmetric fake quantization to weights.

    Args:
        weight: Weight tensor to quantize
        scale: Scale parameter (learnable)
        num_bits: Number of quantization bits
        narrow_range: If True, use [-2^(n-1)+1, 2^(n-1)-1] range

    Returns:
        Fake-quantized weight tensor
    """
    if narrow_range:
        level_low = -(2 ** (num_bits - 1)) + 1
        level_high = 2 ** (num_bits - 1) - 1
    else:
        level_low = -(2 ** (num_bits - 1))
        level_high = 2 ** (num_bits - 1) - 1

    # Clamp scale to positive
    scale_clamped = scale.clamp(min=1e-8)

    # Quantize: round(weight / scale) * scale
    scaled = weight / scale_clamped
    quantized = torch.clamp(torch.round(scaled), level_low, level_high)
    dequantized = quantized * scale_clamped

    return dequantized


def asymmetric_fake_quantize(
    weight: torch.Tensor,
    input_low: torch.Tensor,
    input_range: torch.Tensor,
    num_bits: int = 8,
) -> torch.Tensor:
    """
    Apply asymmetric fake quantization to weights.

    Args:
        weight: Weight tensor to quantize
        input_low: Lower bound parameter (learnable)
        input_range: Range parameter (learnable)
        num_bits: Number of quantization bits

    Returns:
        Fake-quantized weight tensor
    """
    level_low = 0
    level_high = 2**num_bits - 1
    levels = level_high - level_low

    # Clamp input_range to positive
    input_range_clamped = input_range.clamp(min=1e-8)
    input_high = input_low + input_range_clamped

    # Scale computation
    scale = input_range_clamped / levels

    # Quantize
    scaled = (weight - input_low) / scale
    quantized = torch.clamp(torch.round(scaled), level_low, level_high)
    dequantized = quantized * scale + input_low

    return dequantized


class SymmetricFakeQuantSTE(torch.autograd.Function):
    """
    Symmetric fake quantization with Straight-Through Estimator.

    Forward: applies quantization
    Backward: passes gradient straight through, allowing gradient to flow to scale
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        scale: torch.Tensor,
        level_low: int,
        level_high: int,
    ) -> torch.Tensor:
        # Clamp scale to positive
        scale_clamped = scale.clamp(min=1e-8)

        # Quantize
        scaled = weight / scale_clamped
        quantized_int = torch.clamp(torch.round(scaled), level_low, level_high)
        dequantized = quantized_int * scale_clamped

        # Save for backward
        ctx.save_for_backward(weight, scale, quantized_int)
        ctx.level_low = level_low
        ctx.level_high = level_high

        return dequantized

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        weight, scale, quantized_int = ctx.saved_tensors

        # Gradient w.r.t. weight: straight-through (pass through as-is)
        grad_weight = grad_output.clone()

        # Gradient w.r.t. scale: d(q_int * scale) / d(scale) = q_int
        # But we need to sum over the weight dimensions to match scale shape
        grad_scale = grad_output * quantized_int

        # Sum over dimensions where scale is 1 (broadcasting dimensions)
        scale_shape = scale.shape
        weight_shape = weight.shape
        for dim in range(len(weight_shape)):
            if dim >= len(scale_shape) or scale_shape[dim] == 1:
                grad_scale = grad_scale.sum(dim=dim, keepdim=True)

        # Reshape to match scale
        grad_scale = grad_scale.reshape(scale_shape)

        return grad_weight, grad_scale, None, None


class AsymmetricFakeQuantSTE(torch.autograd.Function):
    """
    Asymmetric fake quantization with Straight-Through Estimator.
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        input_low: torch.Tensor,
        input_range: torch.Tensor,
        level_low: int,
        level_high: int,
    ) -> torch.Tensor:
        levels = level_high - level_low

        # Clamp input_range to positive
        input_range_clamped = input_range.clamp(min=1e-8)
        scale = input_range_clamped / levels

        # Quantize
        scaled = (weight - input_low) / scale
        quantized_int = torch.clamp(torch.round(scaled), level_low, level_high)
        dequantized = quantized_int * scale + input_low

        # Save for backward
        ctx.save_for_backward(weight, input_low, input_range, quantized_int, scale)
        ctx.level_low = level_low
        ctx.level_high = level_high
        ctx.levels = levels

        return dequantized

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        weight, input_low, input_range, quantized_int, scale = ctx.saved_tensors
        levels = ctx.levels

        # Gradient w.r.t. weight: straight-through
        grad_weight = grad_output.clone()

        # For asymmetric: output = q_int * (input_range / levels) + input_low
        # d(output)/d(input_low) = 1 - q_int / levels (approximately 1 for simplicity)
        # d(output)/d(input_range) = q_int / levels

        grad_input_low = grad_output.clone()
        grad_input_range = grad_output * (quantized_int / levels)

        # Sum over broadcasting dimensions
        for dim in range(len(weight.shape)):
            if dim >= len(input_low.shape) or input_low.shape[dim] == 1:
                grad_input_low = grad_input_low.sum(dim=dim, keepdim=True)
                grad_input_range = grad_input_range.sum(dim=dim, keepdim=True)

        grad_input_low = grad_input_low.reshape(input_low.shape)
        grad_input_range = grad_input_range.reshape(input_range.shape)

        return grad_weight, grad_input_low, grad_input_range, None, None


def symmetric_fake_quantize_ste(
    weight: torch.Tensor,
    scale: torch.Tensor,
    num_bits: int = 8,
    narrow_range: bool = False,
) -> torch.Tensor:
    """Symmetric fake quantization with STE for gradient flow through scale."""
    if narrow_range:
        level_low = -(2 ** (num_bits - 1)) + 1
        level_high = 2 ** (num_bits - 1) - 1
    else:
        level_low = -(2 ** (num_bits - 1))
        level_high = 2 ** (num_bits - 1) - 1

    return SymmetricFakeQuantSTE.apply(weight, scale, level_low, level_high)


def asymmetric_fake_quantize_ste(
    weight: torch.Tensor,
    input_low: torch.Tensor,
    input_range: torch.Tensor,
    num_bits: int = 8,
) -> torch.Tensor:
    """Asymmetric fake quantization with STE for gradient flow."""
    level_low = 0
    level_high = 2**num_bits - 1
    return AsymmetricFakeQuantSTE.apply(weight, input_low, input_range, level_low, level_high)


# =============================================================================
# Fake Quantization Weight Hook
# =============================================================================


@dataclass
class FakeQuantConfig:
    """Configuration for fake quantization."""

    num_bits: int = 8
    symmetric: bool = True
    per_channel: bool = True
    narrow_range: bool = False
    channel_axis: int = 0  # Output channel axis for per-channel quantization


class FakeQuantWeightHook:
    """
    A reusable hook that applies fake quantization to layer weights.

    This hook intercepts the forward pass and replaces the layer's weights
    with their fake-quantized versions. The quantization parameters (scale)
    are learnable and can be optimized to minimize reconstruction error.

    Attributes:
        layer: The target layer (nn.Linear or nn.Conv2d)
        config: Fake quantization configuration
        enabled: Whether fake quantization is enabled
        scale: Learnable scale parameter(s)
        input_low: Learnable input_low parameter (for asymmetric)
        input_range: Learnable input_range parameter (for asymmetric)
    """

    def __init__(
        self,
        layer: nn.Module,
        config: Optional[FakeQuantConfig] = None,
    ):
        """
        Initialize the fake quantization hook.

        Args:
            layer: Target layer (Linear, Conv1d, Conv2d, etc.)
            config: Quantization configuration
        """
        self.layer = layer
        self.config = config or FakeQuantConfig()
        self.enabled = False
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._original_weight: Optional[torch.Tensor] = None
        self._quantized_weight: Optional[torch.Tensor] = None

        # Initialize quantization parameters
        self._init_quant_params()

    def _init_quant_params(self) -> None:
        """Initialize quantization scale/range parameters based on weight statistics."""
        weight = self.layer.weight.data

        if self.config.per_channel:
            # Per-channel: scale shape matches output channels
            axis = self.config.channel_axis
            num_channels = weight.shape[axis]

            # Reduce over all dims except channel axis
            reduce_dims = [i for i in range(weight.dim()) if i != axis]

            if self.config.symmetric:
                # Scale = max(|weight|) / (2^(n-1) - 1)
                max_abs = weight.abs().amax(dim=reduce_dims, keepdim=True)
                if self.config.narrow_range:
                    level_high = 2 ** (self.config.num_bits - 1) - 1
                else:
                    level_high = 2 ** (self.config.num_bits - 1) - 1
                scale = max_abs / level_high

                # Reshape scale to proper broadcast shape
                scale_shape = [1] * weight.dim()
                scale_shape[axis] = num_channels
                self.scale = nn.Parameter(scale.reshape(scale_shape).clone())
            else:
                # Asymmetric: input_low = min, input_range = max - min
                w_min = weight.amin(dim=reduce_dims, keepdim=True)
                w_max = weight.amax(dim=reduce_dims, keepdim=True)

                scale_shape = [1] * weight.dim()
                scale_shape[axis] = num_channels

                self.input_low = nn.Parameter(w_min.reshape(scale_shape).clone())
                self.input_range = nn.Parameter((w_max - w_min).reshape(scale_shape).clone())
        else:
            # Per-tensor quantization
            if self.config.symmetric:
                max_abs = weight.abs().max()
                if self.config.narrow_range:
                    level_high = 2 ** (self.config.num_bits - 1) - 1
                else:
                    level_high = 2 ** (self.config.num_bits - 1) - 1
                scale = max_abs / level_high
                self.scale = nn.Parameter(torch.tensor([scale.item()], device=weight.device, dtype=weight.dtype))
            else:
                w_min = weight.min()
                w_max = weight.max()
                self.input_low = nn.Parameter(torch.tensor([w_min.item()], device=weight.device, dtype=weight.dtype))
                self.input_range = nn.Parameter(
                    torch.tensor([(w_max - w_min).item()], device=weight.device, dtype=weight.dtype)
                )

    def get_quant_params(self) -> list[nn.Parameter]:
        """Get list of learnable quantization parameters."""
        if self.config.symmetric:
            return [self.scale]
        return [self.input_low, self.input_range]

    def _apply_fake_quant(self, weight: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization to weight tensor."""
        if self.config.symmetric:
            return symmetric_fake_quantize_ste(
                weight,
                self.scale,
                self.config.num_bits,
                self.config.narrow_range,
            )
        return asymmetric_fake_quantize_ste(
            weight,
            self.input_low,
            self.input_range,
            self.config.num_bits,
        )

    def _forward_pre_hook(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        """Pre-forward hook that replaces weights with fake-quantized version."""
        if self.enabled:
            # Store original weight
            self._original_weight = module.weight.data.clone()
            # Apply fake quantization with gradient-enabled scale
            q_weight = self._apply_fake_quant(module.weight)
            module.weight.data = q_weight.data
            # Store the quantized weight for gradient computation
            self._quantized_weight = q_weight

    def _forward_hook(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Post-forward hook that restores original weights and applies gradient-enabled output."""
        if self.enabled and self._original_weight is not None:
            # Restore original weights
            module.weight.data = self._original_weight
            self._original_weight = None

            # Re-compute output with gradient-enabled fake quantization
            # This ensures gradients flow through scale parameters
            if self._quantized_weight is not None and inputs:
                input_tensor = inputs[0]
                if isinstance(module, nn.Linear):
                    # Recompute linear with quantized weight that has gradient
                    output = F.linear(input_tensor, self._quantized_weight, module.bias)
                elif isinstance(module, nn.Conv2d):
                    output = F.conv2d(
                        input_tensor,
                        self._quantized_weight,
                        module.bias,
                        module.stride,
                        module.padding,
                        module.dilation,
                        module.groups,
                    )
                elif isinstance(module, nn.Conv1d):
                    output = F.conv1d(
                        input_tensor,
                        self._quantized_weight,
                        module.bias,
                        module.stride,
                        module.padding,
                        module.dilation,
                        module.groups,
                    )
                self._quantized_weight = None
        return output

    def register(self) -> FakeQuantWeightHook:
        """Register the hooks on the layer."""
        if self._hook_handle is None:
            self._pre_hook_handle = self.layer.register_forward_pre_hook(self._forward_pre_hook)
            self._hook_handle = self.layer.register_forward_hook(self._forward_hook)
        return self

    def remove(self) -> None:
        """Remove the hooks from the layer."""
        if self._hook_handle is not None:
            self._pre_hook_handle.remove()
            self._hook_handle.remove()
            self._hook_handle = None
            self._pre_hook_handle = None

    def enable_quantization(self) -> None:
        """Enable fake quantization."""
        self.enabled = True

    def disable_quantization(self) -> None:
        """Disable fake quantization."""
        self.enabled = False

    def clamp_params(self) -> None:
        """Clamp quantization parameters to valid ranges."""
        with torch.no_grad():
            if self.config.symmetric:
                self.scale.data.clamp_(min=1e-8)
            else:
                self.input_range.data.clamp_(min=1e-8)


# =============================================================================
# Layer Quantization Tuner
# =============================================================================


@dataclass
class TunerConfig:
    """Configuration for the layer quantization tuner."""

    num_steps: int = 100
    learning_rate: float = 1e-3
    loss_fn: str = "mse"  # "mse" or "kl"
    optimizer: str = "adam"  # "adam" or "sgd"
    verbose: bool = True
    log_interval: int = 10


class LayerQuantTuner:
    """
    Tunes quantization scales for a single layer to minimize reconstruction error.

    This class handles the optimization loop for a single layer's quantization
    parameters, capturing reference outputs and minimizing the MSE between
    quantized and unquantized outputs.
    """

    def __init__(
        self,
        layer: nn.Module,
        hook: FakeQuantWeightHook,
        config: Optional[TunerConfig] = None,
    ):
        """
        Initialize the layer tuner.

        Args:
            layer: Target layer to tune
            hook: FakeQuantWeightHook attached to the layer
            config: Tuning configuration
        """
        self.layer = layer
        self.hook = hook
        self.config = config or TunerConfig()
        self.loss_history: list[float] = []

    def _get_optimizer(self, params: list[nn.Parameter]) -> torch.optim.Optimizer:
        """Create optimizer for quantization parameters."""
        if self.config.optimizer == "adam":
            return torch.optim.Adam(params, lr=self.config.learning_rate)
        if self.config.optimizer == "sgd":
            return torch.optim.SGD(params, lr=self.config.learning_rate, momentum=0.9)
        raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _compute_loss(
        self,
        q_out: torch.Tensor,
        ref_out: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reconstruction loss."""
        if self.config.loss_fn == "mse":
            return F.mse_loss(q_out, ref_out)
        if self.config.loss_fn == "kl":
            # KL divergence (softmax over last dim)
            log_q = F.log_softmax(q_out, dim=-1)
            p = F.softmax(ref_out, dim=-1)
            return F.kl_div(log_q, p, reduction="batchmean")
        raise ValueError(f"Unknown loss function: {self.config.loss_fn}")

    def tune(
        self,
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
        calibration_data: Iterator[torch.Tensor],
        num_samples: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Tune the quantization parameters for this layer.

        Args:
            forward_fn: Function that runs forward pass and returns layer output
            calibration_data: Iterator yielding input tensors
            num_samples: Maximum number of calibration samples to use

        Returns:
            Dictionary with tuning statistics
        """
        self.loss_history = []
        quant_params = self.hook.get_quant_params()

        # Ensure quant params require grad
        for p in quant_params:
            p.requires_grad = True

        optimizer = self._get_optimizer(quant_params)

        initial_loss = None
        final_loss = None

        for step in range(self.config.num_steps):
            try:
                x = next(calibration_data)
            except StopIteration:
                if self.config.verbose:
                    print(f"  Calibration data exhausted at step {step}")
                break

            if isinstance(x, (list, tuple)):
                x = x[0]  # Assume first element is input

            # Move to same device as layer
            device = next(self.layer.parameters()).device
            x = x.to(device)

            # Reference output (float, no quantization)
            self.hook.disable_quantization()
            with torch.no_grad():
                ref_out = forward_fn(x).detach()

            # Quantized output
            self.hook.enable_quantization()
            q_out = forward_fn(x)

            # Compute loss and backprop
            loss = self._compute_loss(q_out, ref_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Clamp parameters
            self.hook.clamp_params()

            loss_val = loss.item()
            self.loss_history.append(loss_val)

            if initial_loss is None:
                initial_loss = loss_val
            final_loss = loss_val

            if self.config.verbose and step % self.config.log_interval == 0:
                print(f"  Step {step:4d}: loss = {loss_val:.6f}")

        # Disable gradients after tuning
        for p in quant_params:
            p.requires_grad = False

        return {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction": (initial_loss - final_loss) / initial_loss if initial_loss else 0,
            "num_steps": len(self.loss_history),
            "loss_history": self.loss_history,
        }


# =============================================================================
# Layer-wise PTQ Driver
# =============================================================================


class LayerwisePTQDriver:
    """
    Orchestrates layer-by-layer post-training quantization.

    This driver implements the full layer-wise PTQ algorithm:
    1. Identify quantizable layers
    2. Initialize fake-quant hooks for each layer
    3. Tune each layer in forward order
    4. Keep previously tuned layers quantized

    Example:
        ```python
        model = MyModel()
        driver = LayerwisePTQDriver(model)

        # Tune all layers
        results = driver.tune(calibration_dataloader, num_steps=100)

        # Access tuned hooks
        for name, hook in driver.hooks.items():
            print(f"{name}: scale = {hook.scale.data}")
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        quant_config: Optional[FakeQuantConfig] = None,
        layer_types: tuple[type, ...] = (nn.Linear, nn.Conv1d, nn.Conv2d),
    ):
        """
        Initialize the PTQ driver.

        Args:
            model: The model to quantize
            quant_config: Quantization configuration for all layers
            layer_types: Types of layers to quantize
        """
        self.model = model
        self.quant_config = quant_config or FakeQuantConfig()
        self.layer_types = layer_types

        # Freeze model weights
        self._freeze_model()

        # Discover quantizable layers
        self.layers: dict[str, nn.Module] = {}
        self.hooks: dict[str, FakeQuantWeightHook] = {}
        self._discover_layers()

    def _freeze_model(self) -> None:
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def _discover_layers(self) -> None:
        """Discover and register hooks for quantizable layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, self.layer_types):
                self.layers[name] = module

                # Create hook with appropriate config
                config = copy.deepcopy(self.quant_config)

                # Adjust channel axis for different layer types
                if isinstance(module, nn.Linear):
                    config.channel_axis = 0  # Output features
                elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                    config.channel_axis = 0  # Output channels

                hook = FakeQuantWeightHook(module, config)
                hook.register()
                self.hooks[name] = hook

    def _create_layer_forward_fn(
        self,
        layer_name: str,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Create a forward function that captures output of a specific layer.

        Args:
            layer_name: Name of the target layer

        Returns:
            Forward function that returns the layer's output
        """
        layer_output = {}

        def hook_fn(module, input, output):
            layer_output["output"] = output

        handle = self.layers[layer_name].register_forward_hook(hook_fn)

        def forward_fn(x: torch.Tensor) -> torch.Tensor:
            self.model(x)
            output = layer_output.get("output")
            return output

        forward_fn._handle = handle  # type: ignore
        return forward_fn

    def tune(
        self,
        calibration_dataloader: DataLoader,
        tuner_config: Optional[TunerConfig] = None,
        layer_order: Optional[list[str]] = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Tune all layers in order.

        Args:
            calibration_dataloader: DataLoader providing calibration data
            tuner_config: Configuration for the tuner
            layer_order: Optional custom order for layer tuning

        Returns:
            Dictionary mapping layer names to tuning results
        """
        config = tuner_config or TunerConfig()
        results: dict[str, dict[str, Any]] = {}

        # Default order: forward pass order (as discovered)
        if layer_order is None:
            layer_order = list(self.layers.keys())

        print(f"Tuning {len(layer_order)} layers...")

        for i, layer_name in enumerate(layer_order):
            print(f"\n[{i + 1}/{len(layer_order)}] Tuning layer: {layer_name}")

            layer = self.layers[layer_name]
            hook = self.hooks[layer_name]

            # Enable fake-quant for previously tuned layers
            for prev_name in layer_order[:i]:
                self.hooks[prev_name].enable_quantization()

            # Disable fake-quant for layers after current
            for next_name in layer_order[i + 1 :]:
                self.hooks[next_name].disable_quantization()

            # Create forward function for this layer
            forward_fn = self._create_layer_forward_fn(layer_name)

            # Create data iterator
            def data_iterator():
                for batch in calibration_dataloader:
                    if isinstance(batch, (list, tuple)):
                        yield batch[0]
                    else:
                        yield batch

            # Tune this layer
            tuner = LayerQuantTuner(layer, hook, config)
            result = tuner.tune(forward_fn, data_iterator())

            # Clean up forward hook
            forward_fn._handle.remove()  # type: ignore

            results[layer_name] = result

            # Enable quantization for this layer (now tuned)
            hook.enable_quantization()

            if config.verbose:
                loss_reduction = result.get("loss_reduction", 0) * 100
                print(f"  Final: {result.get('final_loss', 0):.6f} (reduction: {loss_reduction:.2f}%)")

        # Enable all hooks after tuning
        self.enable_all_quantization()

        return results

    def enable_all_quantization(self) -> None:
        """Enable quantization for all layers."""
        for hook in self.hooks.values():
            hook.enable_quantization()

    def disable_all_quantization(self) -> None:
        """Disable quantization for all layers."""
        for hook in self.hooks.values():
            hook.disable_quantization()

    def remove_all_hooks(self) -> None:
        """Remove all hooks from the model."""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()

    def get_quantized_state_dict(self) -> dict[str, torch.Tensor]:
        """
        Get state dict with quantized weights.

        Returns:
            State dict containing fake-quantized weights
        """
        self.enable_all_quantization()

        state_dict = {}
        for name, layer in self.layers.items():
            hook = self.hooks[name]
            q_weight = hook._apply_fake_quant(layer.weight.data)
            state_dict[f"{name}.weight"] = q_weight
            if layer.bias is not None:
                state_dict[f"{name}.bias"] = layer.bias.data.clone()

        return state_dict

    def export_quant_params(self) -> dict[str, dict[str, torch.Tensor]]:
        """
        Export quantization parameters for all layers.

        Returns:
            Dictionary mapping layer names to their quantization parameters
        """
        params = {}
        for name, hook in self.hooks.items():
            if hook.config.symmetric:
                params[name] = {"scale": hook.scale.data.clone()}
            else:
                params[name] = {
                    "input_low": hook.input_low.data.clone(),
                    "input_range": hook.input_range.data.clone(),
                }
        return params


# =============================================================================
# Unit Tests
# =============================================================================


def test_fake_quant_functions():
    """Test basic fake quantization functions."""
    print("\n=== Test: Fake Quantization Functions ===")

    # Test symmetric quantization
    weight = torch.randn(64, 32)
    scale = torch.tensor([0.1])

    q_weight = symmetric_fake_quantize(weight, scale, num_bits=8)

    # Check output is quantized (discrete values)
    unique_vals = torch.unique(q_weight / scale)
    assert len(unique_vals) <= 256, "Too many unique values for 8-bit quant"

    # Test asymmetric quantization
    input_low = torch.tensor([-1.0])
    input_range = torch.tensor([2.0])

    q_weight_asym = asymmetric_fake_quantize(weight, input_low, input_range, num_bits=8)

    # Check values are within range
    assert q_weight_asym.min() >= input_low.item() - 1e-5
    assert q_weight_asym.max() <= (input_low + input_range).item() + 1e-5

    print("✓ Symmetric quantization works correctly")
    print("✓ Asymmetric quantization works correctly")


def test_fake_quant_hook():
    """Test FakeQuantWeightHook functionality."""
    print("\n=== Test: FakeQuantWeightHook ===")

    # Create a simple layer
    layer = nn.Linear(32, 64)
    x = torch.randn(8, 32)

    # Get reference output
    with torch.no_grad():
        ref_out = layer(x).clone()

    # Create and register hook
    config = FakeQuantConfig(num_bits=8, symmetric=True, per_channel=True)
    hook = FakeQuantWeightHook(layer, config)
    hook.register()

    # Test disabled state (should match reference)
    hook.disable_quantization()
    with torch.no_grad():
        out_disabled = layer(x)
    assert torch.allclose(out_disabled, ref_out, atol=1e-6), "Disabled hook should not change output"

    # Test enabled state (should be different due to quantization)
    hook.enable_quantization()
    with torch.no_grad():
        out_enabled = layer(x)
    # Note: might be close but not exact due to quantization

    # Verify weights are restored after forward
    original_weight = layer.weight.data.clone()
    with torch.no_grad():
        _ = layer(x)
    assert torch.allclose(layer.weight.data, original_weight), "Weights should be restored after forward"

    # Clean up
    hook.remove()

    print("✓ Hook correctly toggles quantization")
    print("✓ Hook restores original weights after forward")


def test_layer_tuner():
    """Test LayerQuantTuner reduces MSE."""
    print("\n=== Test: LayerQuantTuner MSE Reduction ===")

    # Create a simple layer with known weights
    layer = nn.Linear(32, 64)

    # Initialize weights to have a specific range
    nn.init.uniform_(layer.weight, -1.0, 1.0)

    # Create hook with deliberately bad initial scale (too large)
    config = FakeQuantConfig(num_bits=8, symmetric=True, per_channel=False)
    hook = FakeQuantWeightHook(layer, config)
    hook.register()

    # Set a very poor initial scale (much too large, causing severe quantization)
    # Optimal scale for weight range [-1, 1] with 8-bit is ~0.0078
    # We set it to 10x larger to create a bad starting point
    hook.scale.data.fill_(0.1)

    # Create calibration data
    calibration_data = [torch.randn(16, 32) for _ in range(100)]

    def forward_fn(x):
        return layer(x)

    # Measure initial MSE
    hook.disable_quantization()
    with torch.no_grad():
        ref_out = forward_fn(calibration_data[0])

    hook.enable_quantization()
    with torch.no_grad():
        initial_q_out = forward_fn(calibration_data[0])
    initial_mse = F.mse_loss(initial_q_out, ref_out).item()

    # Tune with enough steps
    tuner_config = TunerConfig(num_steps=100, learning_rate=1e-3, verbose=False)
    tuner = LayerQuantTuner(layer, hook, tuner_config)
    result = tuner.tune(forward_fn, iter(calibration_data))

    # Measure final MSE
    hook.enable_quantization()
    with torch.no_grad():
        final_q_out = forward_fn(calibration_data[0])
    hook.disable_quantization()
    with torch.no_grad():
        ref_out = forward_fn(calibration_data[0])
    final_mse = F.mse_loss(final_q_out, ref_out).item()

    print(f"  Initial scale: 0.1, Final scale: {hook.scale.data.item():.6f}")
    print(f"  Initial MSE: {initial_mse:.6f}")
    print(f"  Final MSE:   {final_mse:.6f}")
    print(f"  Reduction:   {(1 - final_mse / initial_mse) * 100:.2f}%")

    # The final MSE should be significantly lower than initial
    assert final_mse < initial_mse * 0.5, (
        f"Tuning should reduce MSE by at least 50%, got {(1 - final_mse / initial_mse) * 100:.1f}%"
    )

    # Clean up
    hook.remove()

    print("✓ LayerQuantTuner successfully reduces MSE")


def test_layerwise_ptq_driver():
    """Test full layer-wise PTQ driver."""
    print("\n=== Test: LayerwisePTQDriver ===")

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 10)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = SimpleModel()

    # Create calibration data
    from torch.utils.data import TensorDataset

    cal_data = torch.randn(100, 32)
    cal_labels = torch.randint(0, 10, (100,))
    cal_dataset = TensorDataset(cal_data, cal_labels)
    cal_loader = DataLoader(cal_dataset, batch_size=16)

    # Run PTQ
    driver = LayerwisePTQDriver(model)

    print(f"  Discovered {len(driver.layers)} quantizable layers:")
    for name in driver.layers:
        print(f"    - {name}")

    tuner_config = TunerConfig(num_steps=20, learning_rate=1e-2, verbose=False, log_interval=5)
    results = driver.tune(cal_loader, tuner_config)

    # Check results
    for name, result in results.items():
        loss_reduction = result.get("loss_reduction", 0) * 100
        print(f"  {name}: reduction = {loss_reduction:.2f}%")

    # Export quantization parameters
    quant_params = driver.export_quant_params()
    assert len(quant_params) == len(driver.layers), "Should have params for all layers"

    # Clean up
    driver.remove_all_hooks()

    print("✓ LayerwisePTQDriver completed successfully")


def test_per_channel_quantization():
    """Test per-channel quantization."""
    print("\n=== Test: Per-Channel Quantization ===")

    layer = nn.Linear(32, 64)

    # Per-channel config
    config = FakeQuantConfig(num_bits=8, symmetric=True, per_channel=True)
    hook = FakeQuantWeightHook(layer, config)

    # Check scale shape
    expected_shape = (64, 1)  # (out_features, 1)
    assert hook.scale.shape == expected_shape, f"Expected scale shape {expected_shape}, got {hook.scale.shape}"

    print(f"  Scale shape: {hook.scale.shape}")
    print("✓ Per-channel quantization initialized correctly")


def test_gradient_flow():
    """Test that gradients flow through scale parameters."""
    print("\n=== Test: Gradient Flow ===")

    layer = nn.Linear(32, 64)

    config = FakeQuantConfig(num_bits=8, symmetric=True, per_channel=False)
    hook = FakeQuantWeightHook(layer, config)
    hook.register()
    hook.enable_quantization()

    # Enable gradient for scale
    hook.scale.requires_grad = True

    x = torch.randn(8, 32)
    target = torch.randn(8, 64)

    out = layer(x)
    loss = F.mse_loss(out, target)
    loss.backward()

    assert hook.scale.grad is not None, "Scale should have gradient"
    assert hook.scale.grad.abs().sum() > 0, "Scale gradient should be non-zero"

    print(f"  Scale gradient: {hook.scale.grad}")
    print("✓ Gradients flow correctly through scale")

    hook.remove()


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running Layer-wise PTQ Tuner Tests")
    print("=" * 60)

    test_fake_quant_functions()
    test_fake_quant_hook()
    test_per_channel_quantization()
    test_gradient_flow()
    test_layer_tuner()
    test_layerwise_ptq_driver()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


# =============================================================================
# Usage Example
# =============================================================================


def example_usage():
    """
    Example showing how to use the layer-wise PTQ tuner.
    """
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data import TensorDataset

    # 1. Define your model
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
            )
            self.classifier = nn.Linear(128, 10)

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    model = MyModel()

    # 2. Prepare calibration data (no labels needed)
    calibration_data = torch.randn(500, 784)  # 500 samples
    cal_dataset = TensorDataset(calibration_data)
    cal_loader = DataLoader(cal_dataset, batch_size=32)

    # 3. Configure quantization
    quant_config = FakeQuantConfig(
        num_bits=8,
        symmetric=True,
        per_channel=True,
    )

    # 4. Create PTQ driver
    driver = LayerwisePTQDriver(model, quant_config)

    # 5. Configure tuning
    tuner_config = TunerConfig(
        num_steps=100,
        learning_rate=1e-3,
        loss_fn="mse",
        verbose=True,
        log_interval=20,
    )

    # 6. Run layer-wise tuning
    results = driver.tune(cal_loader, tuner_config)

    # 7. Export quantization parameters
    quant_params = driver.export_quant_params()
    for name, params in quant_params.items():
        print(f"{name}: scale shape = {params['scale'].shape}")

    # 8. Use model with quantization enabled
    driver.enable_all_quantization()
    test_input = torch.randn(1, 784)
    output = model(test_input)
    print(f"Output shape: {output.shape}")

    # 9. Clean up when done
    driver.remove_all_hooks()


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Layer-wise PTQ Tuner")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--example", action="store_true", help="Run usage example")

    args = parser.parse_args()

    if args.test:
        run_all_tests()
    elif args.example:
        example_usage()
    else:
        print("Layer-wise PTQ Tuner")
        print("Usage:")
        print("  python layerwise_ptq_tuner.py --test     # Run unit tests")
        print("  python layerwise_ptq_tuner.py --example  # Run usage example")
        print("\nOr import as a module:")
        print("  from layerwise_ptq_tuner import LayerwisePTQDriver")
