#!/usr/bin/env python3
import torch

from nncf.torch.quantization.reference import ReferenceBackendType
from nncf.torch.quantization.reference import ReferenceQuantize
from nncf.torch.quantization.triton.reference import triton_sum_like

# Test direct sum_like behavior
device = torch.device("cuda")

# Test case - failing case configuration
input_size = [1, 16, 64, 64]
levels = 256
level_low = 0
level_high = 255

# Create reference implementation
RQ = ReferenceQuantize(ReferenceBackendType.TORCH)

# Generate test data
ref_input = torch.randn(input_size, device=device, dtype=torch.float16)
ref_input_low = torch.tensor([-128.0], device=device, dtype=torch.float16)
ref_input_range = torch.tensor([255.0], device=device, dtype=torch.float16)

print(f"Input shape: {ref_input.shape}")
print(f"Input_low shape: {ref_input_low.shape}")
print(f"Input_range shape: {ref_input_range.shape}")

# Test reference backward to get the full-size gradients
mock_grad_output = torch.ones(input_size, device=device, dtype=torch.float16)
ref_grads = RQ.backward(mock_grad_output, ref_input, ref_input_low, ref_input_range, levels, level_low, level_high)

print("\nReference gradients full:")
print(f"grad_low shape: {ref_grads[1].shape}")
print(f"grad_range shape: {ref_grads[2].shape}")
print(f"grad_low sum: {ref_grads[1].sum().item()}")
print(f"grad_range sum: {ref_grads[2].sum().item()}")

# Test triton_sum_like on the reference gradients
triton_grad_low = triton_sum_like(ref_grads[1], ref_input_low)
triton_grad_range = triton_sum_like(ref_grads[2], ref_input_range)

print("\nTriton sum_like results:")
print(f"grad_low reduced: {triton_grad_low.item()}")
print(f"grad_range reduced: {triton_grad_range.item()}")

# Test PyTorch sum for comparison
torch_grad_low = ref_grads[1].sum()
torch_grad_range = ref_grads[2].sum()

print("\nPyTorch sum results:")
print(f"grad_low sum: {torch_grad_low.item()}")
print(f"grad_range sum: {torch_grad_range.item()}")

# Check if they match
print("\nMatches:")
print(f"grad_low: {torch.allclose(triton_grad_low, torch_grad_low, rtol=1e-2)}")
print(f"grad_range: {torch.allclose(triton_grad_range, torch_grad_range, rtol=1e-2)}")

# Check shape compatibility
print("\nShape compatibility:")
print(f"ref_grads[1] can be reduced to {ref_input_low.shape}: {ref_grads[1].shape}")
print(f"ref_grads[2] can be reduced to {ref_input_range.shape}: {ref_grads[2].shape}")

# Check reduction manually
manual_sum_low = ref_grads[1].sum()
manual_sum_range = ref_grads[2].sum()
print("\nManual sum check:")
print(f"grad_low manual sum: {manual_sum_low.item()}")
print(f"grad_range manual sum: {manual_sum_range.item()}")
