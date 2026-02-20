#!/usr/bin/env python
"""
Debug script to verify quantization scale convergence.

Tests:
1. Pure PyTorch fake quantization (known to work)
2. NNCF's QuantizeSymmetricTorch
3. Full SymmetricQuantizer module

Verifies gradients are correct and scales converge to optimal values.
"""

import torch
import torch.nn.functional as F


def pytorch_fake_quantize(x, scale, num_bits=4):
    """
    Pure PyTorch symmetric fake quantization.
    Reference implementation to compare against.
    """
    # 4-bit signed: levels = 16, range = [-8, 7]
    levels = 2**num_bits
    level_low = -(levels // 2)
    level_high = levels // 2 - 1

    # Quantize: clamp, scale, round, unscale
    # Range: [level_low * scale, level_high * scale]
    x_clamped = torch.clamp(x, level_low * scale, level_high * scale)
    x_int = torch.round(x_clamped / scale)
    x_quant = x_int * scale
    return x_quant


def pytorch_fake_quantize_ste(x, scale, num_bits=4):
    """
    Fake quantization with Straight-Through Estimator.
    Uses detach trick for gradient flow.
    """
    levels = 2**num_bits
    level_low = -(levels // 2)
    level_high = levels // 2 - 1

    x_clamped = torch.clamp(x, level_low * scale, level_high * scale)
    x_int = torch.round(x_clamped / scale)
    x_quant = x_int * scale

    # STE: forward uses quantized, backward uses identity for x
    return x_quant


def test_pytorch_reference():
    """Test that pure PyTorch quantization can optimize scales."""
    print("=" * 60)
    print("Test 1: Pure PyTorch Fake Quantization")
    print("=" * 60)

    torch.manual_seed(42)
    # Create a random weight tensor
    weight = torch.randn(128, 128) * 0.1

    # Initial scale (intentionally wrong)
    scale = torch.tensor([0.5], requires_grad=True)

    optimizer = torch.optim.Adam([scale], lr=1e-2)

    print(
        f"Weight stats: mean={weight.mean():.4f}, std={weight.std():.4f}, "
        f"min={weight.min():.4f}, max={weight.max():.4f}"
    )
    print(f"Optimal scale (approx): {weight.abs().max() / 7:.4f}")
    print(f"Initial scale: {scale.item():.4f}")

    for step in range(100):
        q_weight = pytorch_fake_quantize_ste(weight, scale)
        loss = F.mse_loss(q_weight, weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scale.data.clamp_(min=1e-8)

        if step % 20 == 0:
            print(f"  Step {step}: loss={loss.item():.8f}, scale={scale.item():.6f}, grad={scale.grad.item():.6f}")

    print(f"\nFinal scale: {scale.item():.6f}")
    print(f"Final loss: {loss.item():.8f}")
    return loss.item()


def test_nncf_quantizer():
    """Test NNCF's QuantizeSymmetricTorch function."""
    print("\n" + "=" * 60)
    print("Test 2: NNCF QuantizeSymmetricTorch")
    print("=" * 60)

    from nncf.torch.quantization.quantize_functions import QuantizeSymmetricTorch

    torch.manual_seed(42)
    weight = torch.randn(128, 128) * 0.1

    # NNCF uses logarithm_scale=True by default for symmetric
    # But let's test with raw scale first
    scale = torch.tensor([0.5], requires_grad=True)

    # 4-bit signed symmetric: levels=16, level_low=-8, level_high=7
    levels = 16
    level_low = -8
    level_high = 7

    optimizer = torch.optim.Adam([scale], lr=1e-2)

    print(f"Weight stats: mean={weight.mean():.4f}, std={weight.std():.4f}")
    print(f"Initial scale: {scale.item():.4f}")

    for step in range(100):
        q_weight = QuantizeSymmetricTorch.apply(weight, weight.shape, scale, level_low, level_high, levels)
        loss = F.mse_loss(q_weight, weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scale.data.clamp_(min=1e-8)

        if step % 20 == 0:
            grad_val = scale.grad.item() if scale.grad is not None else 0.0
            print(f"  Step {step}: loss={loss.item():.8f}, scale={scale.item():.6f}, grad={grad_val:.6f}")

    print(f"\nFinal scale: {scale.item():.6f}")
    print(f"Final loss: {loss.item():.8f}")
    return loss.item()


def test_nncf_full_quantizer():
    """Test full NNCF SymmetricQuantizer module."""
    print("\n" + "=" * 60)
    print("Test 3: Full NNCF SymmetricQuantizer")
    print("=" * 60)

    from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
    from nncf.torch.quantization.layers import PTQuantizerSpec
    from nncf.torch.quantization.layers import SymmetricQuantizer

    torch.manual_seed(42)
    weight = torch.randn(128, 128) * 0.1

    # Create quantizer spec
    qspec = PTQuantizerSpec(
        num_bits=4,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=True,
        narrow_range=False,
        half_range=False,
        scale_shape=(1,),
        logarithm_scale=False,  # Use raw scale for testing
        is_quantized_on_export=False,
        compression_lr_multiplier=None,
    )

    quantizer = SymmetricQuantizer(qspec)
    # Set initial scale
    quantizer._scale_param_storage.data.fill_(0.5)
    quantizer._scale_param_storage.requires_grad = True

    optimizer = torch.optim.Adam([quantizer._scale_param_storage], lr=1e-2)

    print(f"Weight stats: mean={weight.mean():.4f}, std={weight.std():.4f}")
    print(f"Initial scale: {quantizer.scale.item():.4f}")
    print(f"Using log scale: {quantizer._is_using_log_scale_storage}")

    for step in range(100):
        quantizer.enable_quantization()
        q_weight = quantizer(weight)
        loss = F.mse_loss(q_weight, weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        quantizer._scale_param_storage.data.clamp_(min=1e-8)

        if step % 20 == 0:
            grad = quantizer._scale_param_storage.grad
            grad_val = grad.item() if grad is not None else 0.0
            print(f"  Step {step}: loss={loss.item():.8f}, scale={quantizer.scale.item():.6f}, grad={grad_val:.6f}")

    print(f"\nFinal scale: {quantizer.scale.item():.6f}")
    print(f"Final loss: {loss.item():.8f}")
    return loss.item()


def test_nncf_lora_quantizer():
    """Test NNCF SymmetricLoraQuantizer to see if LoRA affects convergence."""
    print("\n" + "=" * 60)
    print("Test 4: NNCF SymmetricLoraQuantizer")
    print("=" * 60)

    from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
    from nncf.torch.quantization.layers import PTLoraSpec
    from nncf.torch.quantization.layers import PTQuantizerSpec
    from nncf.torch.quantization.layers import SymmetricLoraQuantizer

    torch.manual_seed(42)
    weight = torch.randn(128, 128) * 0.1

    # Create quantizer spec
    qspec = PTQuantizerSpec(
        num_bits=4,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=True,
        narrow_range=False,
        half_range=False,
        scale_shape=(1,),
        logarithm_scale=False,
        is_quantized_on_export=False,
        compression_lr_multiplier=None,
    )

    lspec = PTLoraSpec(
        lora_rank=8,
        orig_weight_shape=[128, 128],
        weight_shape=[128, 128],
    )

    quantizer = SymmetricLoraQuantizer(qspec, lspec)
    quantizer._scale_param_storage.data.fill_(0.5)
    quantizer._scale_param_storage.requires_grad = True

    optimizer = torch.optim.Adam([quantizer._scale_param_storage], lr=1e-2)

    print(f"Weight stats: mean={weight.mean():.4f}, std={weight.std():.4f}")
    print(f"Initial scale: {quantizer.scale.item():.4f}")
    print(f"LoRA A shape: {quantizer.lora_A.shape}, B shape: {quantizer.lora_B.shape}")
    print(f"LoRA A norm: {quantizer.lora_A.norm().item():.6f}")
    print(f"LoRA B norm: {quantizer.lora_B.norm().item():.6f}")

    # Check LoRA contribution
    lora_out = quantizer.lora_B @ quantizer.lora_A
    print(f"LoRA output norm: {lora_out.norm().item():.6f}")

    for step in range(100):
        quantizer.enable_quantization()
        q_weight = quantizer(weight)
        loss = F.mse_loss(q_weight, weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        quantizer._scale_param_storage.data.clamp_(min=1e-8)

        if step % 20 == 0:
            grad = quantizer._scale_param_storage.grad
            grad_val = grad.item() if grad is not None else 0.0
            print(f"  Step {step}: loss={loss.item():.8f}, scale={quantizer.scale.item():.6f}, grad={grad_val:.6f}")

    print(f"\nFinal scale: {quantizer.scale.item():.6f}")
    print(f"Final loss: {loss.item():.8f}")
    return loss.item()


def verify_gradient_correctness():
    """Verify NNCF gradient against numerical gradient."""
    print("\n" + "=" * 60)
    print("Test 5: Gradient Verification (Numerical vs Analytical)")
    print("=" * 60)

    from nncf.torch.quantization.quantize_functions import QuantizeSymmetricTorch

    torch.manual_seed(42)
    weight = torch.randn(8, 8) * 0.1  # Small for numerical grad check

    scale = torch.tensor([0.05], requires_grad=True)
    levels = 16
    level_low = -8
    level_high = 7

    # Analytical gradient
    q_weight = QuantizeSymmetricTorch.apply(weight, weight.shape, scale.clone(), level_low, level_high, levels)
    loss = F.mse_loss(q_weight, weight)
    loss.backward()
    analytical_grad = scale.grad.item()

    # Numerical gradient
    eps = 1e-5
    with torch.no_grad():
        scale_plus = scale + eps
        q_plus = QuantizeSymmetricTorch.apply(weight, weight.shape, scale_plus.clone(), level_low, level_high, levels)
        loss_plus = F.mse_loss(q_plus, weight).item()

        scale_minus = scale - eps
        q_minus = QuantizeSymmetricTorch.apply(weight, weight.shape, scale_minus.clone(), level_low, level_high, levels)
        loss_minus = F.mse_loss(q_minus, weight).item()

    numerical_grad = (loss_plus - loss_minus) / (2 * eps)

    print(f"Analytical gradient: {analytical_grad:.8f}")
    print(f"Numerical gradient:  {numerical_grad:.8f}")
    print(f"Difference: {abs(analytical_grad - numerical_grad):.8f}")
    print(f"Relative error: {abs(analytical_grad - numerical_grad) / (abs(numerical_grad) + 1e-10):.4%}")

    if abs(analytical_grad - numerical_grad) / (abs(numerical_grad) + 1e-10) > 0.1:
        print("\n*** WARNING: Gradients don't match! ***")
    else:
        print("\n*** Gradients match! ***")


def analyze_gradient_components():
    """Analyze the gradient computation step by step."""
    print("\n" + "=" * 60)
    print("Test 6: Detailed Gradient Analysis")
    print("=" * 60)

    from nncf.torch.quantization.reference import ReferenceBackendType
    from nncf.torch.quantization.reference import ReferenceQuantize

    torch.manual_seed(42)
    weight = torch.randn(8, 8) * 0.1
    scale = torch.tensor([0.05])

    levels = 16
    level_low = -8
    level_high = 7

    # NNCF formula
    # input_low = -scale (when scale > 0)
    # input_range = abs((2 + 1/level_low) * scale) = abs((2 - 0.125) * scale) = 1.875 * scale
    input_low = torch.where(scale > 0, -scale, -scale / level_low * level_high)
    input_range = torch.abs((2 + 1 / level_low) * scale)

    print(f"Scale: {scale.item():.6f}")
    print(f"input_low: {input_low.item():.6f}")
    print(f"input_range: {input_range.item():.6f}")
    print(f"Quantization range: [{input_low.item():.6f}, {(input_low + input_range).item():.6f}]")

    # Chain rule derivatives
    # d(input_low)/d(scale) = -1 (for scale > 0)
    # d(input_range)/d(scale) = 1.875 (for scale > 0)
    d_input_low_d_scale = -1.0
    d_input_range_d_scale = 1.875

    # Forward pass
    rq = ReferenceQuantize(ReferenceBackendType.TORCH)
    q_weight = rq.forward(weight, input_low, input_range, levels)
    loss = F.mse_loss(q_weight, weight)

    # Backward pass (using reference impl)
    grad_output = 2 * (q_weight - weight) / weight.numel()  # MSE gradient
    grad_input, grad_low, grad_range = rq.backward(
        grad_output, weight, input_low, input_range, levels, level_low, level_high
    )

    print("\nGradient components:")
    print(f"  grad_low (summed): {grad_low.sum().item():.8f}")
    print(f"  grad_range (summed): {grad_range.sum().item():.8f}")

    # Correct chain rule
    grad_low_sum = grad_low.sum().item()
    grad_range_sum = grad_range.sum().item()
    correct_grad_scale = grad_low_sum * d_input_low_d_scale + grad_range_sum * d_input_range_d_scale
    print(f"\nCorrect grad_scale (chain rule): {correct_grad_scale:.8f}")
    print("  = grad_low * (-1) + grad_range * 1.875")
    print(f"  = {grad_low_sum:.8f} * (-1) + {grad_range_sum:.8f} * 1.875")

    # What NNCF returns (just grad_range)
    nncf_grad_scale = grad_range_sum
    print(f"\nNNCF grad_scale (just grad_range): {nncf_grad_scale:.8f}")

    # Numerical verification
    eps = 1e-5
    with torch.no_grad():
        scale_plus = scale + eps
        input_low_plus = torch.where(scale_plus > 0, -scale_plus, -scale_plus / level_low * level_high)
        input_range_plus = torch.abs((2 + 1 / level_low) * scale_plus)
        q_plus = rq.forward(weight, input_low_plus, input_range_plus, levels)
        loss_plus = F.mse_loss(q_plus, weight).item()

        scale_minus = scale - eps
        input_low_minus = torch.where(scale_minus > 0, -scale_minus, -scale_minus / level_low * level_high)
        input_range_minus = torch.abs((2 + 1 / level_low) * scale_minus)
        q_minus = rq.forward(weight, input_low_minus, input_range_minus, levels)
        loss_minus = F.mse_loss(q_minus, weight).item()

    numerical_grad = (loss_plus - loss_minus) / (2 * eps)
    print(f"\nNumerical gradient: {numerical_grad:.8f}")
    print("\nError analysis:")
    print(
        f"  Correct vs Numerical: {abs(correct_grad_scale - numerical_grad):.8f} ({abs(correct_grad_scale - numerical_grad) / abs(numerical_grad) * 100:.2f}%)"
    )
    print(
        f"  NNCF vs Numerical: {abs(nncf_grad_scale - numerical_grad):.8f} ({abs(nncf_grad_scale - numerical_grad) / abs(numerical_grad) * 100:.2f}%)"
    )


def main():
    print("Debugging Scale Convergence\n")

    loss1 = test_pytorch_reference()
    loss2 = test_nncf_quantizer()
    loss3 = test_nncf_full_quantizer()
    loss4 = test_nncf_lora_quantizer()
    verify_gradient_correctness()
    analyze_gradient_components()

    # Additional test using actual tuner function
    test_actual_tuner()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"PyTorch reference final loss: {loss1:.8f}")
    print(f"NNCF QuantizeSymmetricTorch:  {loss2:.8f}")
    print(f"NNCF SymmetricQuantizer:      {loss3:.8f}")
    print(f"NNCF SymmetricLoraQuantizer:  {loss4:.8f}")


def test_actual_tuner():
    """Test using our actual tune_quantizer_scale function."""
    print("\n" + "=" * 60)
    print("Test 7: Using actual tune_quantizer_scale")
    print("=" * 60)

    from unittest.mock import MagicMock

    from nncf_layerwise_ptq_tuner import TunerConfig
    from nncf_layerwise_ptq_tuner import tune_quantizer_scale

    from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
    from nncf.torch.quantization.layers import PTQuantizerSpec
    from nncf.torch.quantization.layers import SymmetricQuantizer

    torch.manual_seed(42)
    weight = torch.randn(128, 128)

    # Create quantizer
    qspec = PTQuantizerSpec(
        num_bits=4,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=True,
        scale_shape=(1,),
        narrow_range=False,
        half_range=False,
        logarithm_scale=False,
    )
    quantizer = SymmetricQuantizer(qspec)

    # Initialize scale
    with torch.no_grad():
        quantizer._scale_param_storage.data.fill_(0.5)

    print(f"Weight stats: mean={weight.mean():.4f}, std={weight.std():.4f}")
    print(f"Initial scale: {quantizer.scale.item():.6f}")

    # Create mock tensorboard
    class FakeTB:
        def add_scalar(self, *args, **kwargs):
            pass

    # Create mock model
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter(quantizer.get_trainable_params().values())

    # Test config with small LR
    config = TunerConfig(
        num_steps=100,
        learning_rate_4bit=1e-3,
        verbose=True,
        log_interval=20,
    )

    result = tune_quantizer_scale(
        tb=FakeTB(),
        hook_name="test",
        quantizer=quantizer,
        original_weight=weight,
        config=config,
        model=mock_model,
    )

    print(f"\nFinal scale: {quantizer.scale.item():.6f}")
    print(f"Initial loss: {result['initial_loss']:.8f}")
    print(f"Final loss: {result['final_loss']:.8f}")
    print(f"Loss reduction: {result['loss_reduction'] * 100:.1f}%")


if __name__ == "__main__":
    main()
