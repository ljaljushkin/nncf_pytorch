# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Any, Optional

import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.quantization.layers import get_per_channel_scale_shape
from nncf.torch.quantization.reference import ReferenceBackendType
from nncf.torch.quantization.reference import ReferenceQuantize
from tools.benchmark import run_profile
from tools.benchmark import run_wall
from tools.benchmark import run_worker

TIME_SCALES = {"ms": 1000}
NBITS = 8
GPU_RUNS_LOW_BATCH = 3000
GPU_RUNS_HIGH_BATCH = 300
CPU_RUNS = 100
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", 128))
LOW_BATCH_INPUT_SIZE_2D = [128, 2048]
TYPICAL_INPUT_SIZE_2D = [2048, 4096]
HIGH_BATCH_INPUT_SIZE_2D = [2048, 128256]
LOW_BATCH_INPUT_SIZE = [2, 96, 64, 64]
HIGH_BATCH_INPUT_SIZE = [128, 96, 64, 64]


class BatchMode(Enum):
    LOW = "low"
    HIGH = "high"


class ExecutionType(Enum):
    REGULAR = "regular"
    DATA_PARALLEL = "data_parallel"
    DISTRIBUTED_DATA_PARALLEL = "distributed_data_parallel"


class TimingMode(Enum):
    KERNEL = "kernel"
    WALL = "wall"


@dataclass
class BatchDescriptor:
    mode: BatchMode
    input_size: list[int]
    num_runs: dict[torch.device, int]


class TensorType(Enum):
    WEIGHTS = "weights"
    ACTIVATIONS = "activations"


class GranularityType(Enum):
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_GROUP = "per_group"


TEST_TENSOR_TYPES: list[TensorType] = [
    TensorType.WEIGHTS,
    # TensorType.ACTIVATIONS
]
TEST_GRANULARITY: list[GranularityType] = [
    # GranularityType.PER_TENSOR,
    # GranularityType.PER_CHANNEL,
    GranularityType.PER_GROUP
]
TEST_SYMMETRIC: list[bool] = [
    # True,
    False
]

TEST_DEVICES: list[torch.device] = [
    torch.device("cuda"),
    # torch.device("cpu")
]


def get_repeat_count(shape):
    """Determine kernel repeat count based on matrix dimensions."""
    total_elements = shape[0] * shape[1]
    if total_elements > 50_000_000:  # Very large matrices
        return 50
    elif total_elements > 10_000_000:  # Large matrices
        return 100
    elif total_elements > 1_000_000:  # Medium matrices
        return 500
    else:  # Small matrices
        return 1000


matmul_shapes = {
    "HuggingFaceTB/SmolLM-1.7B-Instruct": {(2048, 2048), (2048, 8192), (8192, 2048), (49152, 2048)},
    "Qwen/Qwen2.5-1.5B-Instruct": {(256, 1536), (1536, 1536), (1536, 8960), (8960, 1536), (151936, 1536)},
    "Qwen/Qwen2.5-3B-Instruct": {(256, 2048), (2048, 2048), (2048, 11008), (11008, 2048), (151936, 2048)},
    "google/gemma-2-2b": {(1024, 2304), (2048, 2304), (2304, 2048), (2304, 9216), (9216, 2304), (256000, 2304)},
    "meta-llama/Meta-Llama-3-8B-Instruct": {(1024, 4096), (4096, 4096), (4096, 14336), (14336, 4096), (128256, 4096)},
    "microsoft/Phi-3-mini-4k-instruct": {(3072, 3072), (3072, 8192), (9216, 3072), (16384, 3072), (32064, 3072)},
    "bigcode/starcoder2-3b": {(256, 3072), (3072, 3072), (3072, 12288), (12288, 3072), (49152, 3072)},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        (256, 1536),
        (1536, 1536),
        (1536, 8960),
        (8960, 1536),
        (151936, 1536),
    },
    "meta-llama/Llama-2-7b-chat-hf": {(4096, 4096), (4096, 11008), (11008, 4096), (32000, 4096)},
    "meta-llama/Llama-3.2-1B-Instruct": {(512, 2048), (2048, 2048), (2048, 8192), (8192, 2048), (128256, 2048)},
    "meta-llama/Llama-3.2-3B-Instruct": {(1024, 3072), (3072, 3072), (3072, 8192), (8192, 3072), (128256, 3072)},
    "microsoft/Phi-3.5-mini-instruct": {(3072, 3072), (3072, 8192), (9216, 3072), (16384, 3072), (32064, 3072)},
    "mistralai/Mistral-7B-v0.1": {(1024, 4096), (4096, 4096), (4096, 14336), (14336, 4096), (32000, 4096)},
    "stabilityai/stablelm-3b-4e1t": {(2560, 2560), (2560, 6912), (6912, 2560), (50304, 2560)},
    "tinyllama/tinyllama-1.1b-step-50k-105b": {(256, 2048), (2048, 2048), (2048, 5632), (5632, 2048), (32000, 2048)},
}

unique_shapes = {shape for shapes_set in matmul_shapes.values() for shape in shapes_set} | {
    tuple(LOW_BATCH_INPUT_SIZE_2D),
    tuple(HIGH_BATCH_INPUT_SIZE_2D),
    tuple(TYPICAL_INPUT_SIZE_2D),
}

TEST_BATCHES = [
    BatchDescriptor(
        mode=BatchMode.HIGH,
        input_size=list(shape),
        num_runs={torch.device("cuda"): get_repeat_count(shape), torch.device("cpu"): CPU_RUNS},
    )
    for shape in unique_shapes
]

# TEST_BATCHES: list[BatchDescriptor] = [
# BatchDescriptor(
#     mode=BatchMode.LOW,
#     input_size=LOW_BATCH_INPUT_SIZE,
#     num_runs={torch.device("cuda"): GPU_RUNS_LOW_BATCH, torch.device("cpu"): CPU_RUNS},
# ),
# BatchDescriptor(
#     mode=BatchMode.HIGH,
#     input_size=HIGH_BATCH_INPUT_SIZE,
#     num_runs={torch.device("cuda"): GPU_RUNS_HIGH_BATCH, torch.device("cpu"): CPU_RUNS},
# ),
# ]

TEST_DTYPES: list[torch.dtype] = [
    # torch.float,
    # torch.half,
    torch.bfloat16
]
TEST_EXEC_TYPES: list[ExecutionType] = [
    ExecutionType.REGULAR,
    # ExecutionType.DISTRIBUTED_DATA_PARALLEL,
    # ExecutionType.DATA_PARALLEL,
]
TEST_NARROW_RANGE: list[bool] = [
    False,
    # True
]
TEST_TIMING_MODE: list[TimingMode] = [
    # TimingMode.WALL,
    TimingMode.KERNEL
]
TEST_REFERENCE: list[bool] = [
    False,
    # True
]


@dataclass
class ParamStruct:
    dtype: torch.dtype
    device: torch.device
    exec_type: ExecutionType
    batch: BatchDescriptor
    tensor_type: TensorType
    granularity: GranularityType
    symmetric: bool
    narrow_range: bool
    timing_mode: TimingMode
    ref: bool

    def to_dict(self) -> dict:
        dct = asdict(self)
        dct.pop("batch")
        dct["num_runs"] = self.batch.num_runs[self.device]
        dct["input_size"] = self.batch.input_size
        return dct


TEST_PARAM_STRUCTS: list[ParamStruct] = [
    ParamStruct(
        dtype=dtype,
        device=device,
        exec_type=exec_type,
        batch=batch,
        tensor_type=tensor_type,
        granularity=granularity,
        symmetric=symmetric,
        narrow_range=narrow_range,
        timing_mode=timing,
        ref=ref,
    )
    for ref, timing, narrow_range, dtype, exec_type, batch, device, tensor_type, granularity, symmetric in product(
        TEST_REFERENCE,
        TEST_TIMING_MODE,
        TEST_NARROW_RANGE,
        TEST_DTYPES,
        TEST_EXEC_TYPES,
        TEST_BATCHES,
        TEST_DEVICES,
        TEST_TENSOR_TYPES,
        TEST_GRANULARITY,
        TEST_SYMMETRIC,
    )
    if not (device == torch.device("cpu") and dtype == torch.half)
    and not (device == torch.device("cpu") and exec_type == ExecutionType.DISTRIBUTED_DATA_PARALLEL)
]


class DefaultedPTQuantizerSpec(PTQuantizerSpec):
    def __init__(
        self,
        scale_shape: tuple[int, ...],
        weight_shape: tuple[int, ...],
        num_bits: int = 8,
        mode: QuantizationMode = QuantizationMode.SYMMETRIC,
        signedness_to_force: Optional[bool] = None,
        narrow_range: bool = False,
        half_range: bool = False,
        logarithm_scale: bool = None,
    ):
        super().__init__(
            num_bits, mode, signedness_to_force, narrow_range, half_range, scale_shape, weight_shape, logarithm_scale
        )


RQ = ReferenceQuantize(backend_type=ReferenceBackendType.TORCH)


def get_module(params_struct: ParamStruct) -> BaseQuantizer:
    input_shape = params_struct.batch.input_size
    is_weights = params_struct.tensor_type == TensorType.WEIGHTS
    weight_shape = list(input_shape)

    if params_struct.granularity == GranularityType.PER_GROUP:
        channel_axis = 1
        assert is_weights, "Per-group quantization is only supported for weights"
        assert len(weight_shape) == 2, "Weight shape must have exactly two dimensions"
        assert weight_shape[channel_axis] % GROUP_SIZE == 0, "Number of channels must be divisible by GROUP_SIZE"
        num_groups = weight_shape[channel_axis] // GROUP_SIZE
        # weight: [2048, 4096] -> [2048, 4096//128, 128]
        weight_shape[channel_axis : channel_axis + 1] = (num_groups, GROUP_SIZE)
        # scale: [2048, 4096//128, 1]
        scale_shape = list(weight_shape)
        scale_shape[channel_axis + 1] = 1
    elif params_struct.granularity == GranularityType.PER_CHANNEL:
        scale_shape = get_per_channel_scale_shape(input_shape, is_weights=is_weights)
    else:
        scale_shape = [
            1,
        ]
    specs = DefaultedPTQuantizerSpec(
        scale_shape=scale_shape, weight_shape=weight_shape, narrow_range=params_struct.narrow_range, num_bits=NBITS
    )

    module_cls = SymmetricQuantizer if params_struct.symmetric else AsymmetricQuantizer
    m = module_cls(specs)
    m = m.to(params_struct.device)
    if params_struct.dtype == torch.half:
        m.half()

    return m


if __name__ == "__main__":
    file_name = "benchmark.csv" if len(sys.argv) == 1 else sys.argv[1]
    print(f"Benchmark results will be saved to file {file_name}")

    benchmark_data: list[dict[str, Any]] = []
    device_ids = range(torch.cuda.device_count())
    ngpus_per_node = len(device_ids)
    world_size = ngpus_per_node
    for param_struct in tqdm(TEST_PARAM_STRUCTS):
        param_struct: ParamStruct
        print(param_struct)
        module = get_module(param_struct)
        call_fn = run_wall if param_struct.timing_mode == TimingMode.WALL else run_profile
        num_runs = param_struct.batch.num_runs[param_struct.device]

        input_size = param_struct.batch.input_size
        if param_struct.exec_type == ExecutionType.DISTRIBUTED_DATA_PARALLEL:
            output: list[dict[str, float]] = []
            try:
                mp.spawn(
                    run_worker,
                    nprocs=ngpus_per_node,
                    args=(world_size, module, input_size, num_runs, param_struct.dtype, output),
                )
                run_data = output[0]
            except:  # noqa: E722
                run_data = {"time": -1}
        else:
            run_data = call_fn(
                module, input_size, param_struct.device, num_runs, dtype=param_struct.dtype, forward_only=False
            )

        runtime = next(iter(run_data.values()))
        # benchmark_data.append({**param_struct.to_dict(), "time_ms": runtime})
        d = param_struct.to_dict()
        d.update(run_data)
        benchmark_data.append({**d})

        df = pd.DataFrame(benchmark_data)

        df.to_csv(file_name, index=False)
    print("Done!")
