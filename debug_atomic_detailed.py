#!/usr/bin/env python3

import numpy as np
import torch

from nncf.torch.quantization.triton.reference import backward_kernel_separate_with_reduction


def debug_atomic_accumulation():
    """Debug the atomic accumulation issue in detail"""

    # Create a simple test case
    input_shape = (256, 256)  # 65536 elements
    input_tensor = torch.randn(input_shape, dtype=torch.float16, device="cuda", requires_grad=True)

    # Single scale case
    input_low = torch.tensor([-1.0], dtype=torch.float16, device="cuda")
    input_range = torch.tensor([2.0], dtype=torch.float16, device="cuda")
    levels = 255

    # Gradient from upstream
    grad_output = torch.ones_like(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Total elements: {input_tensor.numel()}")
    print(f"Input low shape: {input_low.shape}")
    print(f"Input range shape: {input_range.shape}")
    print(f"Grad output shape: {grad_output.shape}")

    # Allocate output gradients
    grad_input = torch.zeros_like(input_tensor)
    grad_low = torch.zeros_like(input_low)
    grad_range = torch.zeros_like(input_range)

    print(f"\nBefore kernel:")
    print(f"grad_low: {grad_low}")
    print(f"grad_range: {grad_range}")

    # Call the kernel
    backward_kernel_separate_with_reduction(
        grad_output, input_tensor, input_low, input_range, levels, 0, 255, grad_input, grad_low, grad_range
    )

    print(f"\nAfter kernel:")
    print(f"grad_low: {grad_low}")
    print(f"grad_range: {grad_range}")
    print(f"grad_low sum: {grad_low.sum()}")
    print(f"grad_range sum: {grad_range.sum()}")

    # Let's also check what the expected values should be
    # by manually computing gradients for a few elements
    print(f"\nManual computation for first few elements:")

    # Sample a few values
    sample_indices = [(0, 0), (0, 1), (1, 0), (127, 127)]

    for i, j in sample_indices:
        input_val = input_tensor[i, j].item()
        grad_out_val = grad_output[i, j].item()

        # Compute clipped value
        clipped = max(input_low.item(), min(input_val, input_low.item() + input_range.item()))

        # Compute gradients manually
        if input_val < input_low.item():
            grad_input_manual = 0.0
            grad_low_manual = grad_out_val
            grad_range_manual = 0.0
        elif input_val > input_low.item() + input_range.item():
            grad_input_manual = 0.0
            grad_low_manual = grad_out_val
            grad_range_manual = grad_out_val
        else:
            grad_input_manual = grad_out_val
            grad_low_manual = 0.0
            grad_range_manual = 0.0

        print(f"  [{i},{j}]: input={input_val:.3f}, clipped={clipped:.3f}")
        print(f"    Expected grad_low={grad_low_manual:.3f}, grad_range={grad_range_manual:.3f}")

    # Count elements in different regions
    input_vals = input_tensor.cpu().numpy()
    low_val = input_low.cpu().item()
    range_val = input_range.cpu().item()

    below_low = np.sum(input_vals < low_val)
    above_high = np.sum(input_vals > low_val + range_val)
    in_range = np.sum((input_vals >= low_val) & (input_vals <= low_val + range_val))

    print(f"\nElement distribution:")
    print(f"  Below low ({low_val}): {below_low}")
    print(f"  Above high ({low_val + range_val}): {above_high}")
    print(f"  In range: {in_range}")
    print(f"  Total: {below_low + above_high + in_range}")

    # Expected gradients
    expected_grad_low = below_low + above_high
    expected_grad_range = above_high

    print(f"\nExpected gradients:")
    print(f"  grad_low: {expected_grad_low}")
    print(f"  grad_range: {expected_grad_range}")

    print(f"\nActual vs Expected:")
    print(
        f"  grad_low: {grad_low.item():.1f} vs {expected_grad_low} (ratio: {grad_low.item() / expected_grad_low if expected_grad_low > 0 else 'N/A'})"
    )
    print(
        f"  grad_range: {grad_range.item():.1f} vs {expected_grad_range} (ratio: {grad_range.item() / expected_grad_range if expected_grad_range > 0 else 'N/A'})"
    )


if __name__ == "__main__":
    debug_atomic_accumulation()
