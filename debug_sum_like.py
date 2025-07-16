#!/usr/bin/env python3
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
print("input_low shape:", input_low.shape)
print("input_range shape:", input_range.shape)

# Test parameters
levels = 256
level_low = 0
level_high = 255

print("\nTesting REFERENCE implementation:")
ref_grads = torch_backward(
    grad_output, input_tensor, input_low, input_range, levels, level_low, level_high, is_asymmetric=True
)
print("ref_grads[0] (grad_input) shape:", ref_grads[0].shape)
print("ref_grads[1] (grad_low) shape (before sum_like):", ref_grads[1].shape)
print("ref_grads[2] (grad_range) shape (before sum_like):", ref_grads[2].shape)

# Now let's manually apply sum_like to see what the reference should actually return
from nncf.torch.quantization.reference import sum_like

ref_grad_low_summed = sum_like(ref_grads[1], input_low)
ref_grad_range_summed = sum_like(ref_grads[2], input_range)

print("ref_grad_low_summed shape:", ref_grad_low_summed.shape)
print("ref_grad_range_summed shape:", ref_grad_range_summed.shape)
print("ref_grad_low_summed value:", ref_grad_low_summed)
print("ref_grad_range_summed value:", ref_grad_range_summed)

print("\nTesting TRITON implementation:")
triton_grads = triton_backward(
    grad_output, input_tensor, input_low, input_range, levels, level_low, level_high, is_asymmetric=True
)
print("triton_grads[1] (grad_low) value:", triton_grads[1])
print("triton_grads[2] (grad_range) value:", triton_grads[2])

# Compare the summed results
print("\nComparison (after sum_like):")
print("grad_low difference:", torch.abs(ref_grad_low_summed - triton_grads[1]).max().item())
print("grad_range difference:", torch.abs(ref_grad_range_summed - triton_grads[2]).max().item())
print("grad_low ratio:", (triton_grads[1] / ref_grad_low_summed).item() if ref_grad_low_summed != 0 else "ref is zero")
print(
    "grad_range ratio:",
    (triton_grads[2] / ref_grad_range_summed).item() if ref_grad_range_summed != 0 else "ref is zero",
)
