#!/usr/bin/env python3
import numpy as np
import torch

from nncf.torch.quantization.quantize_functions import asymmetric_quantize

# Test the current implementation
device = torch.device("cuda")
torch.manual_seed(42)

# Test parameters from the failing test
input_size = [4, 16, 16, 16]
bits = 4
level_low = 0
level_high = 15
levels = 16

# Fixed values for FP16 test
input_low = -(2 ** (bits - 1))  # -8
input_range = 2**bits - 1  # 15

print("=== Test Parameters ===")
print(f"input_size: {input_size}")
print(f"bits: {bits}")
print(f"levels: {levels}")
print(f"level_low: {level_low}")
print(f"level_high: {level_high}")
print(f"input_low: {input_low}")
print(f"input_range: {input_range}")

# Create test data
ref_input_low = np.full([1, 16, 1, 1], input_low, dtype=np.float16)
ref_input_range = np.full([1, 16, 1, 1], input_range, dtype=np.float16)

# Generate input data (similar to the test)
np.random.seed(42)
ref_input = np.random.uniform(-7, 7, input_size).astype(np.float16)

# Convert to torch tensors
test_input = torch.from_numpy(ref_input).to(device).requires_grad_(True)
test_input_low = torch.from_numpy(ref_input_low).to(device).requires_grad_(True)
test_input_range = torch.from_numpy(ref_input_range).to(device).requires_grad_(True)

print("\n=== Input Tensors ===")
print(f"test_input shape: {test_input.shape}")
print(f"test_input_low shape: {test_input_low.shape}")
print(f"test_input_range shape: {test_input_range.shape}")
print(f"test_input_low values: {test_input_low.flatten()[:5]}")
print(f"test_input_range values: {test_input_range.flatten()[:5]}")

# Run forward pass
test_value = asymmetric_quantize(test_input, levels, level_low, level_high, test_input_low, test_input_range, eps=1e-8)

print("\n=== Forward Pass ===")
print(f"test_value shape: {test_value.shape}")
print(f"test_value values: {test_value.flatten()[:5]}")

# Run backward pass
test_value.sum().backward()

print("\n=== Backward Pass ===")
print(f"test_input.grad shape: {test_input.grad.shape}")
print(f"test_input_low.grad shape: {test_input_low.grad.shape}")
print(f"test_input_range.grad shape: {test_input_range.grad.shape}")

print(f"test_input.grad values: {test_input.grad.flatten()[:5]}")
print(f"test_input_low.grad values: {test_input_low.grad.flatten()[:5]}")
print(f"test_input_range.grad values: {test_input_range.grad.flatten()[:5]}")

# Check gradient values
print("\n=== Gradient Analysis ===")
print(f"test_input_low.grad min: {test_input_low.grad.min().item()}")
print(f"test_input_low.grad max: {test_input_low.grad.max().item()}")
print(f"test_input_low.grad mean: {test_input_low.grad.mean().item()}")

print(f"test_input_range.grad min: {test_input_range.grad.min().item()}")
print(f"test_input_range.grad max: {test_input_range.grad.max().item()}")
print(f"test_input_range.grad mean: {test_input_range.grad.mean().item()}")

print(f"test_input.grad min: {test_input.grad.min().item()}")
print(f"test_input.grad max: {test_input.grad.max().item()}")
print(f"test_input.grad mean: {test_input.grad.mean().item()}")

# Check for any NaN or inf values
print("\n=== Numerical Stability Check ===")
print(f"test_input_low.grad has NaN: {torch.isnan(test_input_low.grad).any().item()}")
print(f"test_input_range.grad has NaN: {torch.isnan(test_input_range.grad).any().item()}")
print(f"test_input.grad has NaN: {torch.isnan(test_input.grad).any().item()}")

print(f"test_input_low.grad has inf: {torch.isinf(test_input_low.grad).any().item()}")
print(f"test_input_range.grad has inf: {torch.isinf(test_input_range.grad).any().item()}")
print(f"test_input.grad has inf: {torch.isinf(test_input.grad).any().item()}")

# Check actual values from test failure
print("\n=== Expected vs Actual Values ===")
print(f"Expected values around 247-248, got: {test_input_low.grad.flatten()[:5]}")
print(f"Expected values around 247-248, got: {test_input_range.grad.flatten()[:5]}")

# Check if values are in expected range
low_grad_mean = test_input_low.grad.mean().item()
range_grad_mean = test_input_range.grad.mean().item()
print(f"Low grad mean: {low_grad_mean}")
print(f"Range grad mean: {range_grad_mean}")
print(f"Are values in expected range (247-254)? Low: {200 < low_grad_mean < 300}, Range: {200 < range_grad_mean < 300}")
