#!/usr/bin/env python3
import numpy as np
import torch

from nncf.torch.utils import sum_like

# Test what sum_like actually does with [1] shaped tensor
device = torch.device("cuda")

# Create test data
full_tensor = torch.ones(1, 16, 64, 64, device=device, dtype=torch.float16)
target_tensor = torch.tensor([1.0], device=device, dtype=torch.float16)

print("full_tensor shape:", full_tensor.shape)
print("full_tensor.sum():", full_tensor.sum().item())
print("target_tensor shape:", target_tensor.shape)
print("target_tensor.numel():", target_tensor.numel())

# Test sum_like
result = sum_like(full_tensor, target_tensor)
print("sum_like result shape:", result.shape)
print("sum_like result:", result)

# The real issue seems to be that sum_like is broken and always returns the original shape
# Let me check the actual implementation bug

# Based on the code, sum_like checks `ref_tensor.size == 1` which is wrong
# It should check `ref_tensor.numel() == 1`

# Let me check what happens with the actual bug
print("\nChecking the actual bug:")
print("target_tensor.size:", target_tensor.size)
print("target_tensor.size == 1:", target_tensor.size == 1)
print("target_tensor.numel():", target_tensor.numel())
print("target_tensor.numel() == 1:", target_tensor.numel() == 1)

# The issue is that sum_like is not reducing when it should!
# For [1] shaped tensors, it should return the sum as a scalar or with shape [1]
# But it's returning the original shape because the condition is wrong


# Let me write a corrected version
def corrected_sum_like(tensor_to_sum, ref_tensor):
    """Corrected version of sum_like that checks numel() instead of size"""
    if ref_tensor.numel() == 1:
        return tensor_to_sum.sum()

    for dim, size in enumerate(ref_tensor.shape):
        if size == 1:
            tensor_to_sum = tensor_to_sum.sum(dim, keepdim=True)
    return tensor_to_sum


corrected_result = corrected_sum_like(full_tensor, target_tensor)
print("\nCorrected sum_like result shape:", corrected_result.shape)
print("Corrected sum_like result:", corrected_result)
print("Corrected sum_like result value:", corrected_result.item())

# Now the question is: should our implementation return a scalar or [1]?
# Looking at the test failure, the reference returns scalar -156.8
# But our implementation returns [-10.695]

# The issue is that the test expects scalar but we return [1] shaped tensor
# However, the actual values are also different, suggesting a computation error
