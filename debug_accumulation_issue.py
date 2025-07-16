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
print("ref_grads[1] (grad_low) shape:", ref_grads[1].shape)
print("ref_grads[2] (grad_range) shape:", ref_grads[2].shape)
print("ref_grads[1] (grad_low) value:", ref_grads[1])
print("ref_grads[2] (grad_range) value:", ref_grads[2])

print("\nTesting TRITON implementation:")
triton_grads = triton_backward(
    grad_output, input_tensor, input_low, input_range, levels, level_low, level_high, is_asymmetric=True
)
print("triton_grads[0] (grad_input) shape:", triton_grads[0].shape)
print("triton_grads[1] (grad_low) shape:", triton_grads[1].shape)
print("triton_grads[2] (grad_range) shape:", triton_grads[2].shape)
print("triton_grads[1] (grad_low) value:", triton_grads[1])
print("triton_grads[2] (grad_range) value:", triton_grads[2])

# Compare the results
print("\nComparison:")
print("grad_low difference:", torch.abs(ref_grads[1] - triton_grads[1]).max().item())
print("grad_range difference:", torch.abs(ref_grads[2] - triton_grads[2]).max().item())
print("grad_low ratio:", (ref_grads[1] / triton_grads[1]).item() if triton_grads[1] != 0 else "triton is zero")
print("grad_range ratio:", (ref_grads[2] / triton_grads[2]).item() if triton_grads[2] != 0 else "triton is zero")

# Check tensor stats
print("\nTensor statistics:")
print("input_tensor elements:", input_tensor.numel())
print("input_low elements:", input_low.numel())
print("input_range elements:", input_range.numel())

# Check the gradients before reduction
print("\nGradient analysis:")
print("Expected total elements to accumulate:", input_tensor.numel())  # Should be 1*16*64*64 = 65536
print("Actual accumulation factor:", input_tensor.numel() / input_low.numel())  # Should be 65536/1 = 65536
