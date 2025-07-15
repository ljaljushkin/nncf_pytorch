#!/usr/bin/env python3
import torch

from nncf.torch.quantization.reference import ReferenceBackendType
from nncf.torch.quantization.reference import ReferenceQuantize
from nncf.torch.quantization.triton.reference import backward

# Test the sum_like behavior
device = torch.device("cuda")

# Create test data matching the failing test case
input_size = [4, 16, 16, 16]
levels = 16
level_low = 0
level_high = 15

# Create reference implementation
RQ = ReferenceQuantize(ReferenceBackendType.TORCH)

# Generate test data
ref_input = torch.randn(input_size, device=device, dtype=torch.float16)
ref_input_low = torch.tensor([-8.0], device=device, dtype=torch.float16)
ref_input_range = torch.tensor([15.0], device=device, dtype=torch.float16)

print(f"Input shape: {ref_input.shape}")
print(f"Input_low shape: {ref_input_low.shape}")
print(f"Input_range shape: {ref_input_range.shape}")

# Test reference backward
mock_grad_output = torch.ones(input_size, device=device, dtype=torch.float16)
ref_grads = RQ.backward(mock_grad_output, ref_input, ref_input_low, ref_input_range, levels, level_low, level_high)

print("\nReference implementation results:")
print(f"grad_input sum: {ref_grads[0].sum().item()}")
print(f"grad_low sum: {ref_grads[1].sum().item()}")
print(f"grad_range sum: {ref_grads[2].sum().item()}")
print(f"grad_low shape: {ref_grads[1].shape}")
print(f"grad_range shape: {ref_grads[2].shape}")

# Test our triton implementation
triton_grads = backward(mock_grad_output, ref_input, ref_input_low, ref_input_range, levels, level_low, level_high)

print("\nTriton implementation results:")
print(f"grad_input sum: {triton_grads[0].sum().item()}")
print(f"grad_low sum: {triton_grads[1].sum().item()}")
print(f"grad_range sum: {triton_grads[2].sum().item()}")
print(f"grad_low shape: {triton_grads[1].shape}")
print(f"grad_range shape: {triton_grads[2].shape}")

# Check the differences
print("\nDifferences:")
print(f"grad_input close: {torch.allclose(ref_grads[0], triton_grads[0], rtol=1e-2)}")
print(f"grad_low close: {torch.allclose(ref_grads[1], triton_grads[1], rtol=1e-2)}")
print(f"grad_range close: {torch.allclose(ref_grads[2], triton_grads[2], rtol=1e-2)}")

# Check ratio
if ref_grads[1].sum().item() != 0:
    ratio_low = triton_grads[1].sum().item() / ref_grads[1].sum().item()
    print(f"grad_low ratio: {ratio_low}")

if ref_grads[2].sum().item() != 0:
    ratio_range = triton_grads[2].sum().item() / ref_grads[2].sum().item()
    print(f"grad_range ratio: {ratio_range}")
