#!/usr/bin/env python3
import numpy as np
import torch

from nncf.torch.quantization.reference import torch_backward
from nncf.torch.quantization.triton.reference import backward as triton_backward


# Reproduce the exact failing test case
def test_asymmetric_single_scale_case():
    device = torch.device("cuda")

    # Test parameters from the failing test
    input_size = [1, 16, 64, 64]
    bits = 8
    is_weights = True
    is_fp16 = True
    scale_mode = "single_scale"

    # Generate the same data as the test
    level_low, level_high, levels = 0, 2**bits - 1, 2**bits
    fixed = {"input_low": -(2 ** (bits - 1)), "input_range": 2**bits - 1}

    # For single_scale, create [1] shaped tensors
    input_low = torch.tensor([fixed["input_low"]], device=device, dtype=torch.float16)
    input_range = torch.tensor([fixed["input_range"]], device=device, dtype=torch.float16)

    print(f"input_low: {input_low} (shape: {input_low.shape})")
    print(f"input_range: {input_range} (shape: {input_range.shape})")
    print(f"levels: {levels}, level_low: {level_low}, level_high: {level_high}")

    # Generate input tensor
    input_tensor = torch.randn(input_size, device=device, dtype=torch.float16)
    grad_output = torch.randn(input_size, device=device, dtype=torch.float16)

    # Test reference implementation
    print("\nTesting REFERENCE implementation:")
    ref_grads = torch_backward(
        grad_output, input_tensor, input_low, input_range, levels, level_low, level_high, is_asymmetric=True
    )
    print(f"ref_grads[1] (grad_low): {ref_grads[1]} (shape: {ref_grads[1].shape})")
    print(f"ref_grads[2] (grad_range): {ref_grads[2]} (shape: {ref_grads[2].shape})")

    # Test triton implementation
    print("\nTesting TRITON implementation:")
    triton_grads = triton_backward(
        grad_output, input_tensor, input_low, input_range, levels, level_low, level_high, is_asymmetric=True
    )
    print(f"triton_grads[1] (grad_low): {triton_grads[1]} (shape: {triton_grads[1].shape})")
    print(f"triton_grads[2] (grad_range): {triton_grads[2]} (shape: {triton_grads[2].shape})")

    # Compare
    print("\nComparison:")
    if ref_grads[1].numel() > 0 and triton_grads[1].numel() > 0:
        print(f"grad_low ratio: {(ref_grads[1] / triton_grads[1]).item():.4f}")
    if ref_grads[2].numel() > 0 and triton_grads[2].numel() > 0:
        print(f"grad_range ratio: {(ref_grads[2] / triton_grads[2]).item():.4f}")

    # Test with fixed seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create the same test data multiple times
    for i in range(3):
        input_tensor = torch.randn(input_size, device=device, dtype=torch.float16)
        grad_output = torch.randn(input_size, device=device, dtype=torch.float16)

        ref_grads = torch_backward(
            grad_output, input_tensor, input_low, input_range, levels, level_low, level_high, is_asymmetric=True
        )
        triton_grads = triton_backward(
            grad_output, input_tensor, input_low, input_range, levels, level_low, level_high, is_asymmetric=True
        )

        print(f"\nTest {i + 1}:")
        print(f"  Reference grad_range: {ref_grads[2].item():.4f}")
        print(f"  Triton grad_range: {triton_grads[2].item():.4f}")
        print(f"  Ratio: {(ref_grads[2] / triton_grads[2]).item():.4f}")


test_asymmetric_single_scale_case()
