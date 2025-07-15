#!/usr/bin/env python3
import numpy as np
import torch

# Test what the actual generate_range function returns
# Based on the code I saw, for single_scale it should return arrays with shape [1]
input_size = [1, 16, 64, 64]
bits = 8
is_weights = True
is_fp16 = True
scale_mode = "single_scale"

# Generate the same data as the test
level_low, level_high, levels = 0, 2**bits - 1, 2**bits
fixed = {"input_low": -(2 ** (bits - 1)), "input_range": 2**bits - 1}

# Based on the code I saw, for single_scale it does this:
if scale_mode == "single_scale":
    input_low, input_range = fixed["input_low"], fixed["input_range"]
    ref_input_low = np.array([input_low])  # This creates shape [1]
    ref_input_range = np.array([input_range])  # This creates shape [1]

print("CORRECT ref_input_low shape:", ref_input_low.shape)
print("CORRECT ref_input_range shape:", ref_input_range.shape)
print("CORRECT ref_input_low value:", ref_input_low)
print("CORRECT ref_input_range value:", ref_input_range)

# Convert to tensors
device = torch.device("cuda")
input_low = torch.from_numpy(ref_input_low.astype(np.float16)).to(device)
input_range = torch.from_numpy(ref_input_range.astype(np.float16)).to(device)

print("torch input_low shape:", input_low.shape)
print("torch input_range shape:", input_range.shape)
print("torch input_low:", input_low)
print("torch input_range:", input_range)

# So the issue is that for single_scale, we should have shape [1] not []
# This means our triton_sum_like function needs to preserve the [1] shape
# when the target tensor has shape [1]

# Let's test our triton_sum_like function with the correct shapes
print("\nTesting triton_sum_like with correct shapes...")
from nncf.torch.quantization.triton.reference import triton_sum_like

# Create a test full-size grad
full_grad = torch.randn(1, 16, 64, 64, device=device, dtype=torch.float16)
print("full_grad shape:", full_grad.shape)

# Test with shape [1] target
reduced_grad = triton_sum_like(full_grad, input_low)
print("reduced_grad shape:", reduced_grad.shape)
print("target shape:", input_low.shape)
print("Shape match:", reduced_grad.shape == input_low.shape)

# The issue is that triton_sum_like is returning [] when it should return [1]
