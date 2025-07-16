#!/usr/bin/env python3
import torch

# Test what .numel() returns
device = torch.device("cuda")

# Create a [1] shaped tensor
ref_tensor = torch.tensor([1.0], device=device, dtype=torch.float16)
print("ref_tensor shape:", ref_tensor.shape)
print("ref_tensor.numel():", ref_tensor.numel())
print("ref_tensor.numel() == 1:", ref_tensor.numel() == 1)

# Test with scalar
ref_scalar = torch.tensor(1.0, device=device, dtype=torch.float16)
print("\nref_scalar shape:", ref_scalar.shape)
print("ref_scalar.numel():", ref_scalar.numel())
print("ref_scalar.numel() == 1:", ref_scalar.numel() == 1)

# Test with different sizes
ref_2d = torch.tensor([[1.0]], device=device, dtype=torch.float16)
print("\nref_2d shape:", ref_2d.shape)
print("ref_2d.numel():", ref_2d.numel())
print("ref_2d.numel() == 1:", ref_2d.numel() == 1)

# Test with larger tensor
ref_large = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float16)
print("\nref_large shape:", ref_large.shape)
print("ref_large.numel():", ref_large.numel())
print("ref_large.numel() == 1:", ref_large.numel() == 1)


# Test manual sum_like with corrected logic
def corrected_sum_like(tensor_to_sum, ref_tensor):
    """Corrected version of sum_like"""
    if ref_tensor.numel() == 1:
        return tensor_to_sum.sum()

    for dim, size in enumerate(ref_tensor.shape):
        if size == 1:
            tensor_to_sum = tensor_to_sum.sum(dim, keepdim=True)
    return tensor_to_sum


# Test corrected sum_like
full_tensor = torch.ones(1, 16, 64, 64, device=device, dtype=torch.float16)
result = corrected_sum_like(full_tensor, ref_tensor)
print("\ncorrected_sum_like result shape:", result.shape)
print("corrected_sum_like result:", result)
print("corrected_sum_like result value:", result.item())
