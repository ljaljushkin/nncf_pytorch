# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Simple test for 2-bit symmetric quantization to verify the formula.
"""

import torch


def reference_symmetric_quantize_2bit(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation of 2-bit symmetric quantization WITH zero level.

    For 2-bit symmetric:
    - level_low = -2
    - level_high = 1
    - levels = 4
    - Output levels: {-2, -1, 0, 1} * scale

    Formula:
    - quantize: q = round(x / scale)
    - clip: q = clip(q, level_low, level_high)
    - dequantize: x' = q * scale
    """
    level_low = -2
    level_high = 1

    # Quantize
    q = torch.round(x / scale)
    # Clip
    q = torch.clamp(q, level_low, level_high)
    # Dequantize
    return q * scale


def reference_symmetric_quantize_2bit_no_zero(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation of 2-bit symmetric quantization WITHOUT zero level.

    For 2-bit symmetric (no zero):
    - levels = 4
    - Output levels: {-1.5, -0.5, 0.5, 1.5} * scale / (levels/2)

    Formula:
    k = levels
    q = round(clamp(x / scale, -1, 1) * k/2 - 1/2)
    q = clamp(q, -k/2, k/2 - 1)  # clip q to valid range
    output = (q + 0.5) * scale / k * 2
    """
    levels = 4
    k = levels
    q = torch.round(torch.clamp(x / scale, -1, 1) * k / 2 - 1 / 2)
    q = torch.clamp(q, -k / 2, k / 2 - 1)  # clip q to [-2, 1]
    return (q + 0.5) * scale / k * 2


def test_formula_trace():
    """Debug trace through the formula step by step."""
    print("\n" + "=" * 80)
    print("FORMULA TRACE: q = round(clamp(x/scale, -1, 1) * k/2 - 0.5)")
    print("               out = (q + 0.5) * scale / k * 2")
    print("=" * 80)

    for scale_val in [1.0, 1.5, 2.0]:
        scale = torch.tensor(scale_val)
        levels = 4
        k = levels

        print(f"\n--- scale={scale.item()}, k={k} ---")

        test_inputs = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        for x in test_inputs:
            x_t = torch.tensor(x)
            normalized = x_t / scale
            clamped = torch.clamp(normalized, -1, 1)
            pre_round = clamped * k / 2 - 0.5
            q = torch.round(pre_round)
            out = (q + 0.5) * scale / k * 2
            print(
                f"x={x:5.2f} → norm={normalized.item():5.2f} → clamp={clamped.item():5.2f} "
                f"→ pre_round={pre_round.item():5.2f} → q={q.item():5.2f} → out={out.item():6.3f}"
            )


def test_symmetric_2bit_simple_values():
    """
    Test 2-bit symmetric quantization with 10 simple values (WITH zero).
    """
    scale = torch.tensor(1.0)

    # Test 10 input values spanning the expected quantization range
    # For 2-bit symmetric with scale=1: valid outputs are -2, -1, 0, 1
    test_cases = [
        # (input, expected_output)
        (-3.0, -2.0),  # clips to level_low
        (-2.5, -2.0),  # clips to level_low
        (-2.0, -2.0),  # exact level_low
        (-1.5, -2.0),  # rounds to -2
        (-1.0, -1.0),  # exact
        (-0.5, 0.0),  # rounds to 0
        (0.0, 0.0),  # exact zero
        (0.5, 0.0),  # rounds to 0
        (1.0, 1.0),  # exact level_high
        (1.5, 1.0),  # clips to level_high
    ]

    print("\n" + "=" * 80)
    print("Testing 2-bit symmetric quantization WITH ZERO with scale=1.0")
    print("For 2-bit signed: level_low=-2, level_high=1, levels=4")
    print("Valid quantized values: -2, -1, 0, 1")
    print("=" * 80)

    for input_val, expected in test_cases:
        input_tensor = torch.tensor(input_val)
        ref_output = reference_symmetric_quantize_2bit(input_tensor, scale)
        print(f"Input: {input_val:6.2f} -> Reference: {ref_output.item():6.2f} (expected: {expected:6.2f})")
        assert ref_output.item() == expected, f"Reference mismatch for input {input_val}"

    print("\nReference implementation (WITH ZERO) passes all tests.\n")


def test_symmetric_2bit_no_zero_simple_values():
    """
    Test 2-bit symmetric quantization with 10 simple values (NO zero).
    Output levels: {-1.5, -0.5, 0.5, 1.5}

    Formula works with scale=2.0 (input range is [-scale, scale] = [-2, 2])
    """
    scale = torch.tensor(2.0)  # scale=2 means input range [-2, 2]

    # Test 10 input values spanning the expected quantization range
    # For 2-bit symmetric NO ZERO with scale=2: valid outputs are -1.5, -0.5, 0.5, 1.5
    test_cases = [
        # (input, expected_output)
        (-2.0, -1.5),  # clips to min level
        (-1.5, -1.5),  # exact min level
        (-1.0, -1.5),  # rounds to -1.5 (q=-2)
        (-0.5, -0.5),  # exact -0.5 (q=-1)
        (-0.25, -0.5),  # rounds to 0.5
        (0.0, 0.5),  # rounds to 0.5 (q=0) - NO ZERO OUTPUT!
        (0.25, 0.5),  # rounds to 0.5
        (0.5, 0.5),  # rounds to 0.5 (q=0)
        (1.0, 0.5),  # rounds to 0.5 (q=0)
        (1.5, 1.5),  # exact max level
        (2.0, 1.5),  # clips to max level
    ]

    print("\n" + "=" * 80)
    print("Testing 2-bit symmetric quantization NO ZERO with scale=2.0")
    print("Output levels: {-1.5, -0.5, 0.5, 1.5}")
    print("Formula: q = round(clamp(x/scale, -1, 1) * k/2 - 0.5), output = (q + 0.5) * scale / k * 2")
    print("=" * 80)

    all_passed = True
    for input_val, expected in test_cases:
        input_tensor = torch.tensor(input_val)
        ref_output = reference_symmetric_quantize_2bit_no_zero(input_tensor, scale)
        status = "✓" if abs(ref_output.item() - expected) < 1e-5 else "✗"
        if status == "✗":
            all_passed = False
        print(f"Input: {input_val:6.2f} -> Reference: {ref_output.item():6.2f} (expected: {expected:6.2f}) {status}")

    print(f"\nReference implementation (NO ZERO): {'All tests passed!' if all_passed else 'SOME TESTS FAILED!'}\n")


def test_quantize_symmetric_original():
    """
    Test the original QuantizeSymmetric implementation.
    """
    from nncf.torch.quantization.quantize_functions import QuantizeSymmetric

    scale = torch.tensor(1.0)
    level_low = -2  # 2-bit signed
    level_high = 1
    levels = 4

    test_cases = [
        (-3.0, -2.0),
        (-2.5, -2.0),
        (-2.0, -2.0),
        (-1.5, -2.0),
        (-1.0, -1.0),
        (-0.5, 0.0),
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 1.0),
        (1.5, 1.0),
    ]

    print("\n" + "=" * 80)
    print("Testing QuantizeSymmetric (original) with scale=1.0")
    print("=" * 80)

    all_passed = True
    for input_val, expected in test_cases:
        input_tensor = torch.tensor(input_val)
        output = QuantizeSymmetric.apply(input_tensor, scale, level_low, level_high, levels)
        status = "✓" if abs(output.item() - expected) < 1e-5 else "✗"
        if status == "✗":
            all_passed = False
        print(f"Input: {input_val:6.2f} -> Output: {output.item():6.2f} (expected: {expected:6.2f}) {status}")

    print(f"\nQuantizeSymmetric: {'All tests passed!' if all_passed else 'SOME TESTS FAILED!'}\n")


def test_quantize_symmetric_torch():
    """
    Test the QuantizeSymmetricTorch implementation (used for LoRA).
    """
    from nncf.torch.quantization.quantize_functions import QuantizeSymmetricTorch

    scale = torch.tensor(1.0)
    level_low = -2  # 2-bit signed
    level_high = 1
    levels = 4
    input_shape = (1,)

    test_cases = [
        (-3.0, -2.0),
        (-2.5, -2.0),
        (-2.0, -2.0),
        (-1.5, -2.0),
        (-1.0, -1.0),
        (-0.5, 0.0),
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 1.0),
        (1.5, 1.0),
    ]

    print("\n" + "=" * 80)
    print("Testing QuantizeSymmetricTorch (for LoRA) with scale=1.0")
    print("=" * 80)

    all_passed = True
    for input_val, expected in test_cases:
        input_tensor = torch.tensor([input_val])
        output = QuantizeSymmetricTorch.apply(input_tensor, input_shape, scale, level_low, level_high, levels)
        status = "✓" if abs(output.item() - expected) < 1e-5 else "✗"
        if status == "✗":
            all_passed = False
        print(f"Input: {input_val:6.2f} -> Output: {output.item():6.2f} (expected: {expected:6.2f}) {status}")

    print(f"\nQuantizeSymmetricTorch: {'All tests passed!' if all_passed else 'SOME TESTS FAILED!'}\n")


def analyze_formula_difference():
    """
    Analyze the difference between QuantizeSymmetric and QuantizeSymmetricTorch formulas
    and show the correct formula for 2-bit no-zero quantization.
    """
    scale = 1.0
    level_low = -2
    level_high = 1

    print("\n" + "=" * 80)
    print("Formula Analysis for 2-bit Symmetric Quantization")
    print("scale=1.0, level_low=-2, level_high=1, levels=4")
    print("=" * 80)

    # QuantizeSymmetric formula (CORRECT for levels {-2, -1, 0, 1})
    input_low_orig = scale * (level_low / level_high)  # scale * (-2/1) = -2
    input_range_orig = scale - input_low_orig  # 1 - (-2) = 3

    print("\nQuantizeSymmetric formula (WITH ZERO levels: {-2, -1, 0, 1}):")
    print(f"  input_low = scale * (level_low / level_high) = {scale} * ({level_low}/{level_high}) = {input_low_orig}")
    print(f"  input_range = scale - input_low = {scale} - ({input_low_orig}) = {input_range_orig}")
    print(f"  Range: [{input_low_orig}, {input_low_orig + input_range_orig}]")

    # QuantizeSymmetricTorch formula (BROKEN for 2-bit)
    input_low_torch = -scale  # -1
    input_range_torch = abs((2 + 1 / level_low) * scale)  # |(2 - 0.5) * 1| = 1.5

    print("\nQuantizeSymmetricTorch formula (BROKEN for 2-bit):")
    print(f"  input_low = -scale = {input_low_torch}")
    print(f"  input_range = |(2 + 1/level_low) * scale| = |(2 + 1/{level_low}) * {scale}| = {input_range_torch}")
    print(f"  Range: [{input_low_torch}, {input_low_torch + input_range_torch}]")

    # NO ZERO formula for levels {-1.5, -0.5, 0.5, 1.5}
    input_low_nozero = -1.5 * scale  # -1.5
    input_range_nozero = 3.0 * scale  # 3.0

    print("\nNO ZERO formula (levels: {-1.5, -0.5, 0.5, 1.5}):")
    print(f"  input_low = -1.5 * scale = {input_low_nozero}")
    print(f"  input_range = 3.0 * scale = {input_range_nozero}")
    print(f"  Range: [{input_low_nozero}, {input_low_nozero + input_range_nozero}]")
    print("  zero_point = -0.5 (offset to avoid zero in output)")
    print("  ")
    print("  Generic formula for no-zero symmetric:")
    print("    input_low = (level_low + 0.5) * scale / level_high")
    print(f"             = ({level_low} + 0.5) * {scale} / {level_high} = {(level_low + 0.5) * scale / level_high}")
    print("    input_range = scale * (level_high - level_low) / level_high")
    print(
        f"               = {scale} * ({level_high} - {level_low}) / {level_high} = {scale * (level_high - level_low) / level_high}"
    )

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"  WITH ZERO (current):   input_low={input_low_orig}, input_range={input_range_orig}")
    print(f"  NO ZERO (desired):     input_low={input_low_nozero}, input_range={input_range_nozero}")
    print(f"  QuantizeSymmetricTorch: input_low={input_low_torch}, input_range={input_range_torch} <- WRONG!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_formula_trace()
    analyze_formula_difference()
    test_symmetric_2bit_simple_values()
    test_symmetric_2bit_no_zero_simple_values()
    test_quantize_symmetric_original()
    test_quantize_symmetric_torch()
