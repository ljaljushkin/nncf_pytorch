import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from nncf.torch.quantization.triton.reference import two_stage_sum_reduction

# Test shape preservation
input_tensor = torch.randn(1, 16, 64, 64, device="cuda")
ref_tensor = torch.ones(1, 1, 1, 1, device="cuda")

result = two_stage_sum_reduction(input_tensor, ref_tensor)
print(f"Input shape: {input_tensor.shape}")
print(f"Reference shape: {ref_tensor.shape}")
print(f"Result shape: {result.shape}")
print(f"Result value: {result.item()}")
print(f"Expected value: {input_tensor.sum().item()}")
