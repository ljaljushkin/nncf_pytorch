#!/usr/bin/env python3
import numpy as np
import torch

from nncf.torch.quantization.triton.reference import backward

# Reproduce the failing test case
torch.manual_seed(42)
device = torch.device("cuda")

input_size = [1, 16, 64, 64]
bits = 8
is_weights = True
is_fp16 = True
scale_mode = "single_scale"

# Generate test data
level_low, level_high, levels = 0, 2**bits - 1, 2**bits
fixed = {"input_low": -(2 ** (bits - 1)), "input_range": 2**bits - 1}

# For single_scale with weights, input_low and input_range are scalars
ref_input_low = np.array(fixed["input_low"], dtype=np.float16)
ref_input_range = np.array(fixed["input_range"], dtype=np.float16)

print("ref_input_low shape:", ref_input_low.shape)
print("ref_input_range shape:", ref_input_range.shape)

# Convert to tensors
input_low = torch.from_numpy(ref_input_low).to(device)
input_range = torch.from_numpy(ref_input_range).to(device)

print("input_low shape:", input_low.shape)
print("input_range shape:", input_range.shape)

# Create input tensor
input_tensor = torch.randn(input_size, device=device, dtype=torch.float16, requires_grad=True)
print("input_tensor shape:", input_tensor.shape)

# Create grad_output
grad_output = torch.randn_like(input_tensor)
print("grad_output shape:", grad_output.shape)

print("\nTesting backward function directly...")
try:
    grad_input, grad_low, grad_range = backward(
        grad_output, input_tensor, input_low, input_range, levels, level_low, level_high, is_asymmetric=False
    )

    print("grad_input shape:", grad_input.shape)
    print("grad_low shape:", grad_low.shape)
    print("grad_range shape:", grad_range.shape)

    print("\nExpected shapes:")
    print("grad_input expected:", input_tensor.shape)
    print("grad_low expected:", input_low.shape)
    print("grad_range expected:", input_range.shape)

    print("\nShape matches:")
    print("grad_input match:", grad_input.shape == input_tensor.shape)
    print("grad_low match:", grad_low.shape == input_low.shape)
    print("grad_range match:", grad_range.shape == input_range.shape)

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
