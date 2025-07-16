#!/usr/bin/env python3
import numpy as np
import torch

from nncf.torch.quantization.quantize_functions import asymmetric_quantize
from nncf.torch.quantization.reference import ReferenceBackendType
from nncf.torch.quantization.reference import ReferenceQuantize

# Test the specific case that's failing
device = torch.device("cuda")
torch.manual_seed(42)

# Test parameters from the failing test
input_size = [1, 16, 64, 64]  # 65536 elements
bits = 8
level_low = 0
level_high = 255
levels = 256

# Fixed values for FP16 test
input_low = -(2 ** (bits - 1))  # -128
input_range = 2**bits - 1  # 255

print("=== Test Parameters ===")
print(f"input_size: {input_size}")
print(f"bits: {bits}")
print(f"levels: {levels}")
print(f"level_low: {level_low}")
print(f"level_high: {level_high}")
print(f"input_low: {input_low}")
print(f"input_range: {input_range}")
print(f"Total elements: {np.prod(input_size)}")

# Create test data - use single scale
ref_input_low = np.array([input_low], dtype=np.float16)
ref_input_range = np.array([input_range], dtype=np.float16)

# Generate input data (similar to the test)
np.random.seed(42)
# Use safe values that won't cause quantization issues
ref_input = np.random.uniform(-120, 120, input_size).astype(np.float16)

print("\n=== Reference Implementation ===")
# Create reference quantizer
RQ = ReferenceQuantize(backend_type=ReferenceBackendType.NUMPY)

# Run reference implementation
ref_output = RQ.forward(ref_input, ref_input_low, ref_input_range, levels)
mock_prev_output_grads = np.ones(input_size, dtype=np.float16)
ref_grads = RQ.backward(
    mock_prev_output_grads, ref_input, ref_input_low, ref_input_range, levels, level_low, level_high
)

print(f"Reference grad_low: {ref_grads[1]} (shape: {ref_grads[1].shape})")
print(f"Reference grad_range: {ref_grads[2]} (shape: {ref_grads[2].shape})")

print("\n=== Triton Implementation ===")
# Convert to torch tensors
test_input = torch.from_numpy(ref_input).to(device).requires_grad_(True)
test_input_low = torch.from_numpy(ref_input_low).to(device).requires_grad_(True)
test_input_range = torch.from_numpy(ref_input_range).to(device).requires_grad_(True)

# Run Triton implementation
test_value = asymmetric_quantize(test_input, levels, level_low, level_high, test_input_low, test_input_range, eps=1e-8)
test_value.sum().backward()

print(f"Triton grad_low: {test_input_low.grad} (shape: {test_input_low.grad.shape})")
print(f"Triton grad_range: {test_input_range.grad} (shape: {test_input_range.grad.shape})")

print("\n=== Comparison ===")
print(f"grad_low - Reference: {ref_grads[1][0]:.1f}, Triton: {test_input_low.grad.item():.1f}")
print(f"grad_range - Reference: {ref_grads[2][0]:.1f}, Triton: {test_input_range.grad.item():.1f}")
print(f"Ratio grad_low: {ref_grads[1][0] / test_input_low.grad.item():.2f}")
print(f"Ratio grad_range: {ref_grads[2][0] / test_input_range.grad.item():.2f}")

# Check if the ratio is a power of 2
ratio_low = ref_grads[1][0] / test_input_low.grad.item()
ratio_range = ref_grads[2][0] / test_input_range.grad.item()

print("\nIs ratio a power of 2?")
print(f"grad_low ratio: {ratio_low:.2f} = 2^{np.log2(ratio_low):.2f}")
print(f"grad_range ratio: {ratio_range:.2f} = 2^{np.log2(ratio_range):.2f}")
