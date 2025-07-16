#!/usr/bin/env python3
import numpy as np
import torch

# Test the actual function calls used in the test
from nncf.torch.quantization.quantize_functions import symmetric_quantize
from nncf.torch.quantization.reference import ReferenceQuantizedFunctions as RQ


def get_grads(tensors):
    """Get gradients from tensors - from the test helpers"""
    return [tensor.grad for tensor in tensors]


# Test the EXACT same setup as the failing test
device = torch.device("cuda")
is_fp16 = True
input_size = [1, 16, 64, 64]
bits = 8
scale_mode = "single_scale"

np.random.seed(0)
torch.manual_seed(0)


# Generate test data exactly like the test
def generate_scale(input_size, scale_mode, is_weights, is_fp16, fixed=None):
    if scale_mode == "single_scale":
        return np.array([fixed if fixed is not None else np.random.rand()])
    return np.array([1.0])


# Create scale
fixed = 2 ** (bits - 1) - 1  # 127 for 8-bit
ref_scale = generate_scale(input_size, scale_mode, True, is_fp16, fixed=fixed).astype(np.float16)
print("ref_scale:", ref_scale)

# Create test tensor
test_scale = torch.from_numpy(ref_scale).to(device).requires_grad_(True)
print("test_scale shape:", test_scale.shape)

# Create input tensor
ref_input = np.random.randn(*input_size).astype(np.float16)
test_input = torch.from_numpy(ref_input).to(device).requires_grad_(True)

# Test parameters
level_low = -(2 ** (bits - 1))  # -128
level_high = 2 ** (bits - 1) - 1  # 127
levels = 2**bits  # 256

print("level_low:", level_low)
print("level_high:", level_high)
print("levels:", levels)

# Get reference values
ref_scale = abs(ref_scale) + 1e-4  # EPS
ref_input_low = ref_scale * (level_low / level_high)
ref_input_range = ref_scale - ref_input_low

print("ref_input_low:", ref_input_low)
print("ref_input_range:", ref_input_range)

# Test reference implementation
mock_prev_output_grads = np.ones(input_size, dtype=np.float16)
ref_grads = RQ.Quantize_backward(
    mock_prev_output_grads, ref_input, ref_input_low, ref_input_range, levels, level_low, level_high
)

print("\nReference gradients:")
print("ref_grads[0] shape:", ref_grads[0].shape)
print("ref_grads[1] shape:", ref_grads[1].shape)
print("ref_grads[2] shape:", ref_grads[2].shape)

# Remove grad_low (symmetric quantization doesn't have input_low)
del ref_grads[1]

print("\nAfter deleting ref_grads[1]:")
print("ref_grads[0] shape:", ref_grads[0].shape)
print("ref_grads[1] shape:", ref_grads[1].shape)
print("ref_grads[1] (scale gradient):", ref_grads[1])

# Test our implementation
test_value = symmetric_quantize(test_input, levels, level_low, level_high, test_scale, eps=1e-4)
test_value.sum().backward()
test_grads = get_grads([test_input, test_scale])

print("\nTest gradients:")
print("test_grads[0] shape:", test_grads[0].shape)
print("test_grads[1] shape:", test_grads[1].shape)
print("test_grads[1] (scale gradient):", test_grads[1])

print("\nComparison:")
print("Expected:", ref_grads[1])
print("Actual:", test_grads[1])
print("Difference:", abs(float(ref_grads[1]) - float(test_grads[1][0])))
print("Ratio:", float(test_grads[1][0]) / float(ref_grads[1]) if ref_grads[1] != 0 else "ref is zero")
