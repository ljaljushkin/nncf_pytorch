#!/usr/bin/env python3
import torch

# Let's look at the exact test generation code to understand the issue
# Check what generate_range returns for single_scale
from tests.torch.quantization.test_functions import BaseParametrized

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

# Use the exact same function as the test
ref_input_low, ref_input_range = BaseParametrized.generate_range(
    input_size, scale_mode, is_weights, is_fp16, levels, fixed
)

print("CORRECT ref_input_low shape:", ref_input_low.shape)
print("CORRECT ref_input_range shape:", ref_input_range.shape)
print("CORRECT ref_input_low value:", ref_input_low)
print("CORRECT ref_input_range value:", ref_input_range)

# Convert to tensors
input_low = torch.from_numpy(ref_input_low).to(device)
input_range = torch.from_numpy(ref_input_range).to(device)

print("torch input_low shape:", input_low.shape)
print("torch input_range shape:", input_range.shape)

# So the issue is that for single_scale, we should have shape [1] not []
# This means our triton_sum_like function needs to preserve the [1] shape
# when the target tensor has shape [1]
