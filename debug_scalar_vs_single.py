#!/usr/bin/env python3
import numpy as np
import torch

# Recreate the exact test case that's failing
torch.manual_seed(42)
device = torch.device("cuda")

input_size = [1, 16, 64, 64]
bits = 8
is_weights = True
is_fp16 = True
scale_mode = "single_scale"

# Generate the same data as the test
level_low, level_high, levels = 0, 2**bits - 1, 2**bits
fixed = {"input_low": -(2 ** (bits - 1)), "input_range": 2**bits - 1}

# For single_scale with weights, input_low and input_range are scalars
if scale_mode == "single_scale":
    ref_input_low = np.array(fixed["input_low"], dtype=np.float16)
    ref_input_range = np.array(fixed["input_range"], dtype=np.float16)

print("numpy ref_input_low shape:", ref_input_low.shape)
print("numpy ref_input_range shape:", ref_input_range.shape)

# Convert to tensors with requires_grad=True
input_low = torch.from_numpy(ref_input_low).to(device)
input_range = torch.from_numpy(ref_input_range).to(device)
input_low.requires_grad = True
input_range.requires_grad = True

print("torch input_low shape:", input_low.shape)
print("torch input_range shape:", input_range.shape)
print("torch input_low numel:", input_low.numel())
print("torch input_range numel:", input_range.numel())

# Check if they're scalar or single-element tensors
print("input_low is scalar:", input_low.dim() == 0)
print("input_range is scalar:", input_range.dim() == 0)

# Now test what shape PyTorch expects for gradients
print("\nTesting gradient shape expectations...")


# Create a simple function that just returns the input multiplied by the parameters
def test_function(input_tensor, low_param, range_param):
    return input_tensor * low_param * range_param


input_tensor = torch.randn(input_size, device=device, requires_grad=True)
output = test_function(input_tensor, input_low, input_range)
loss = output.sum()

try:
    loss.backward()
    print("✓ Success - gradients computed correctly")
    print(f"input_low.grad shape: {input_low.grad.shape}")
    print(f"input_range.grad shape: {input_range.grad.shape}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test with [1] shaped tensors instead
print("\nTesting with [1] shaped tensors...")
input_low_1 = torch.tensor([ref_input_low], device=device, requires_grad=True)
input_range_1 = torch.tensor([ref_input_range], device=device, requires_grad=True)

print("input_low_1 shape:", input_low_1.shape)
print("input_range_1 shape:", input_range_1.shape)

input_tensor2 = torch.randn(input_size, device=device, requires_grad=True)
output2 = test_function(input_tensor2, input_low_1, input_range_1)
loss2 = output2.sum()

try:
    loss2.backward()
    print("✓ Success - gradients computed correctly")
    print(f"input_low_1.grad shape: {input_low_1.grad.shape}")
    print(f"input_range_1.grad shape: {input_range_1.grad.shape}")
except Exception as e:
    print(f"✗ Error: {e}")
