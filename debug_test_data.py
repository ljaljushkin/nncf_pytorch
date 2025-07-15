#!/usr/bin/env python3
import numpy as np
import torch

# Let's look at the exact test generation code to understand the issue
# Check what get_test_data does
from tests.torch.quantization.test_functions import get_test_data

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

# This is what the test does
test_input_low, test_input_range = get_test_data(
    [ref_input_low, ref_input_range], True, is_backward=True, is_fp16=is_fp16
)

print("test_input_low shape:", test_input_low.shape)
print("test_input_range shape:", test_input_range.shape)
print("test_input_low requires_grad:", test_input_low.requires_grad)
print("test_input_range requires_grad:", test_input_range.requires_grad)

# Check if there's a difference in how the tensors are created
print("\nComparing tensor creation methods...")
direct_tensor_low = torch.from_numpy(ref_input_low).to(device)
direct_tensor_range = torch.from_numpy(ref_input_range).to(device)
print("direct_tensor_low shape:", direct_tensor_low.shape)
print("direct_tensor_range shape:", direct_tensor_range.shape)
