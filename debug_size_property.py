#!/usr/bin/env python3
import torch

from nncf.torch.utils import sum_like

# Test what .size vs .size() returns
device = torch.device("cuda")

# Create a [1] shaped tensor
ref_tensor = torch.tensor([1.0], device=device, dtype=torch.float16)
print("ref_tensor shape:", ref_tensor.shape)
print("ref_tensor.size:", ref_tensor.size)
print("ref_tensor.size():", ref_tensor.size())
print("ref_tensor.size == 1:", ref_tensor.size == 1)

# Test with scalar
ref_scalar = torch.tensor(1.0, device=device, dtype=torch.float16)
print("\nref_scalar shape:", ref_scalar.shape)
print("ref_scalar.size:", ref_scalar.size)
print("ref_scalar.size():", ref_scalar.size())
print("ref_scalar.size == 1:", ref_scalar.size == 1)

# Test with different sizes
ref_2d = torch.tensor([[1.0]], device=device, dtype=torch.float16)
print("\nref_2d shape:", ref_2d.shape)
print("ref_2d.size:", ref_2d.size)
print("ref_2d.size == 1:", ref_2d.size == 1)

# Test sum_like with [1] shaped tensor
full_tensor = torch.ones(1, 16, 64, 64, device=device, dtype=torch.float16)
result = sum_like(full_tensor, ref_tensor)
print("\nsum_like result shape:", result.shape)
print("sum_like result:", result)
print("sum_like result value:", result.item() if result.numel() == 1 else "not scalar")
