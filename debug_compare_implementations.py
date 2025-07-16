#!/usr/bin/env python3
import numpy as np
import torch

from nncf.torch.quantization.reference import torch_backward
from nncf.torch.quantization.triton.reference import backward as triton_backward

# Test with single_scale mode
device = torch.device("cuda")

# Create test data similar to what fails
input_size = [1, 16, 64, 64]
input_tensor = torch.randn(input_size, device=device, dtype=torch.float16)
grad_output = torch.randn(input_size, device=device, dtype=torch.float16)

# For single_scale mode, input_low and input_range should have shape [1]
input_low = torch.tensor([-128.0], device=device, dtype=torch.float16)
input_range = torch.tensor([255.0], device=device, dtype=torch.float16)

print("input_tensor shape:", input_tensor.shape)
print("grad_output shape:", grad_output.shape)
print("input_low shape:", input_low.shape)
print("input_range shape:", input_range.shape)

# Test parameters
levels = 256
level_low = 0
level_high = 255

print("\nTesting REFERENCE implementation:")
try:
    ref_grads = torch_backward(
        grad_output, input_tensor, input_low, input_range, levels, level_low, level_high, is_asymmetric=True
    )
    print("ref_grads[0] (grad_input) shape:", ref_grads[0].shape)
    print("ref_grads[1] (grad_low) shape:", ref_grads[1].shape)
    print("ref_grads[2] (grad_range) shape:", ref_grads[2].shape)
    print("ref_grads[1] (grad_low) value:", ref_grads[1])
    print("ref_grads[2] (grad_range) value:", ref_grads[2])
except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()

print("\nTesting TRITON implementation:")
try:
    triton_grads = triton_backward(
        grad_output, input_tensor, input_low, input_range, levels, level_low, level_high, is_asymmetric=True
    )
    print("triton_grads[0] (grad_input) shape:", triton_grads[0].shape)
    print("triton_grads[1] (grad_low) shape:", triton_grads[1].shape)
    print("triton_grads[2] (grad_range) shape:", triton_grads[2].shape)
    print("triton_grads[1] (grad_low) value:", triton_grads[1])
    print("triton_grads[2] (grad_range) value:", triton_grads[2])
except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()

# Compare the results
print("\nComparison:")
if "ref_grads" in locals() and "triton_grads" in locals():
    print("grad_low difference:", torch.abs(ref_grads[1] - triton_grads[1]).max().item())
    print("grad_range difference:", torch.abs(ref_grads[2] - triton_grads[2]).max().item())
    print("grad_low ratio:", (triton_grads[1] / ref_grads[1]).item() if ref_grads[1] != 0 else "ref is zero")
    print("grad_range ratio:", (triton_grads[2] / ref_grads[2]).item() if ref_grads[2] != 0 else "ref is zero")
