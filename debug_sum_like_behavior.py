#!/usr/bin/env python3
import torch

from nncf.torch.utils import sum_like

# Test what sum_like actually returns
device = torch.device("cuda")

# Create a full-size tensor
full_tensor = torch.randn(1, 16, 64, 64, device=device, dtype=torch.float16)
print("full_tensor shape:", full_tensor.shape)
print("full_tensor.size:", full_tensor.size())

# Create a [1] shaped tensor
ref_tensor = torch.tensor([1.0], device=device, dtype=torch.float16)
print("ref_tensor shape:", ref_tensor.shape)
print("ref_tensor.size:", ref_tensor.size())

# Test sum_like
result = sum_like(full_tensor, ref_tensor)
print("sum_like result shape:", result.shape)
print("sum_like result:", result)

# Test with scalar reference
ref_scalar = torch.tensor(1.0, device=device, dtype=torch.float16)
print("\nref_scalar shape:", ref_scalar.shape)
print("ref_scalar.size:", ref_scalar.size())

result_scalar = sum_like(full_tensor, ref_scalar)
print("sum_like result_scalar shape:", result_scalar.shape)
print("sum_like result_scalar:", result_scalar)
