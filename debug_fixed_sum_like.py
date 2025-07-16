#!/usr/bin/env python3
import torch

from nncf.torch.utils import sum_like

# Test the corrected sum_like function
device = torch.device("cuda")

# Create a full-size tensor
full_tensor = torch.ones(1, 16, 64, 64, device=device, dtype=torch.float16)
print("full_tensor shape:", full_tensor.shape)
print("full_tensor sum:", full_tensor.sum().item())

# Test with [1] shaped tensor
ref_tensor = torch.tensor([1.0], device=device, dtype=torch.float16)
print("ref_tensor shape:", ref_tensor.shape)
print("ref_tensor.numel():", ref_tensor.numel())

result = sum_like(full_tensor, ref_tensor)
print("sum_like result shape:", result.shape)
print("sum_like result:", result)
print("sum_like result value:", result.item())

# Test with scalar reference
ref_scalar = torch.tensor(1.0, device=device, dtype=torch.float16)
print("\nref_scalar shape:", ref_scalar.shape)
print("ref_scalar.numel():", ref_scalar.numel())

result_scalar = sum_like(full_tensor, ref_scalar)
print("sum_like result_scalar shape:", result_scalar.shape)
print("sum_like result_scalar:", result_scalar)
print("sum_like result_scalar value:", result_scalar.item())

# Test with 2D [1,1] tensor
ref_2d = torch.tensor([[1.0]], device=device, dtype=torch.float16)
print("\nref_2d shape:", ref_2d.shape)
print("ref_2d.numel():", ref_2d.numel())

result_2d = sum_like(full_tensor, ref_2d)
print("sum_like result_2d shape:", result_2d.shape)
print("sum_like result_2d:", result_2d)
print("sum_like result_2d value:", result_2d.item())
