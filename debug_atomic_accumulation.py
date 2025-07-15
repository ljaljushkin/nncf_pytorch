#!/usr/bin/env python3
import torch

from nncf.torch.quantization.triton.reference import backward

# Test the integrated kernel vs separate kernel
device = torch.device("cuda")

# Create simple test data
input_size = [4, 16, 16, 16]
input_tensor = torch.randn(input_size, device=device, dtype=torch.float16, requires_grad=True)

# For per_channel_scale activations, input_low and input_range have shape [1, 16, 1, 1]
input_low = torch.randn(1, 16, 1, 1, device=device, dtype=torch.float16, requires_grad=True)
input_range = torch.abs(torch.randn(1, 16, 1, 1, device=device, dtype=torch.float16)) + 0.1
input_range.requires_grad = True

grad_output = torch.randn_like(input_tensor)
levels = 16
level_low = 0
level_high = 15

print("Input shapes:")
print(f"input_tensor: {input_tensor.shape}")
print(f"input_low: {input_low.shape}")
print(f"input_range: {input_range.shape}")

# Test our new integrated kernel
print("\nTesting integrated kernel...")
grad_input, grad_low, grad_range = backward(
    grad_output, input_tensor, input_low, input_range, levels, level_low, level_high
)

print(f"grad_input shape: {grad_input.shape}")
print(f"grad_low shape: {grad_low.shape}")
print(f"grad_range shape: {grad_range.shape}")

# Check some values
print(f"\nSample grad_low values: {grad_low.flatten()[:5]}")
print(f"Sample grad_range values: {grad_range.flatten()[:5]}")

# Test if values are reasonable by checking if they're all the same (which would indicate over-accumulation)
print(f"\ngrad_low std: {grad_low.std().item()}")
print(f"grad_range std: {grad_range.std().item()}")
print(f"grad_low mean: {grad_low.mean().item()}")
print(f"grad_range mean: {grad_range.mean().item()}")

# Also check the expected accumulation
expected_accumulation = 4 * 16 * 16  # batch_size * height * width
print(f"\nExpected accumulation factor: {expected_accumulation}")
print(f"Actual grad_low mean / expected: {grad_low.mean().item() / expected_accumulation}")
