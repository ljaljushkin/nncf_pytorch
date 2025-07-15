#!/usr/bin/env python3
import torch

from nncf.torch.quantization.triton.reference import triton_sum_like

# Test triton_sum_like with the failing case
device = torch.device("cuda")

# Create test tensor
input_tensor = torch.randn(1, 16, 64, 64, device=device, dtype=torch.float16)
target_tensor = torch.zeros(1, device=device, dtype=torch.float16)

print(f"Input tensor shape: {input_tensor.shape}")
print(f"Target tensor shape: {target_tensor.shape}")

# Test triton_sum_like
triton_result = triton_sum_like(input_tensor, target_tensor)
print(f"Triton result shape: {triton_result.shape}")
print(f"Triton result value: {triton_result.item()}")

# Test torch sum
torch_result = input_tensor.sum()
print(f"Torch result shape: {torch_result.shape}")
print(f"Torch result value: {torch_result.item()}")

print(f"Results match: {torch.allclose(triton_result, torch_result, rtol=1e-2)}")
print(f"Difference: {abs(triton_result.item() - torch_result.item())}")
