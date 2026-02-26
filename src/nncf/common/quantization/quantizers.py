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


def calculate_symmetric_level_ranges(num_bits: int, signed: bool, narrow_range: bool = False) -> tuple[int, int]:
    """
    Calculates the numbers of the low and high quant and the number of
    quantization levels for the symmetric quantization scheme.

    :param num_bits: The bitwidth of the quantization.
    :param signed: The flag specifying type of the symmetric quantization scheme
        if it is True then the symmetric quantization scheme is the signed and
        the un-signed otherwise.
    :param narrow_range: True if the range of quantized values is reduced by 1 compared to the
        naive case, False otherwise.
    :return: A Tuple
        level_low - the low quant number
        level_high - the high quant number
    """
    levels = 2**num_bits

    if signed:
        level_high = (levels // 2) - 1
        level_low = -(levels // 2)
    else:
        level_high = levels - 1
        level_low = 0

    if narrow_range:
        if level_low < 0:
            level_low += 1
        else:
            level_high -= 1

    return level_low, level_high


def calculate_asymmetric_level_ranges(num_bits: int, narrow_range: bool = False) -> tuple[int, int]:
    """
    Calculates the numbers of the low and high quant and the number of
    quantization levels for the asymmetric quantization scheme.

    :param num_bits: The bitwidth of the quantization
    :param narrow_range: The flag specifying quantization range if it is True
        then [1; 2^num_bits - 1] and [0; 2^num_bits - 1] otherwise
    :return: A Tuple
        level_low - the low quant number
        level_high - the high quant number
    """
    levels = 2**num_bits
    level_high = levels - 1
    level_low = 0

    if narrow_range:
        level_low = level_low + 1

    return level_low, level_high


def get_num_levels(level_low: int, level_high: int) -> int:
    return level_high - level_low + 1


def calculate_stretched_symmetric_params(num_bits: int) -> dict:
    """
    Calculates parameters for stretched symmetric quantization (ParetoQ-style).

    In stretched quantization, the quantization grid is shifted by 0.5 to avoid
    wasting a level on zero. For example, with 2 bits the representable values
    are {-0.75, -0.25, 0.25, 0.75} * alpha instead of {-2, -1, 0, 1} * scale.

    :param num_bits: The bitwidth of the quantization.
    :return: A dictionary with the following keys:
        n_levels - number of half-levels (2^(num_bits - 1))
        shift - the half-level shift value (0.5)
        Qp - positive clipping bound ((n_levels - shift) / n_levels)
        Qn - negative clipping bound (-Qp)
        clip_val - clipping threshold (1 - 1e-2)
        levels - total number of quantization levels (2^num_bits)
    """
    n_levels = 2 ** (num_bits - 1)
    shift = 0.5
    Qp = (n_levels - shift) / n_levels
    Qn = -Qp
    clip_val = 1 - 1e-2
    levels = 2**num_bits
    return {
        "n_levels": n_levels,
        "shift": shift,
        "Qp": Qp,
        "Qn": Qn,
        "clip_val": clip_val,
        "levels": levels,
    }
