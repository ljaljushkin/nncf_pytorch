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

import math
import time

import torch
import torch.distributed as dist
from torch import nn

TIME_SCALES = {"ms": 1000}


def warmup(layer, input_, runs, forward_only=False):
    for _ in range(runs):
        new_i = layer(input_)
        if not forward_only:
            new_i[0].sum().backward()


def run_wall(
    layer, input_size_, device, runs, is_print=True, dtype=torch.float, forward_only=False
) -> dict[str, float]:
    input_ = torch.randn(input_size_, device=torch.device(device), dtype=dtype)
    input_.requires_grad_(True)

    # Force CUDA initialization & warm up
    warmup(layer, input_, 100)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        layer.zero_grad()
        new_i = layer(input_)
        new_i[0].sum().backward()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    ctime, scale = list(TIME_SCALES.items())[0]
    fbtime = elapsed / runs * scale

    if is_print:
        print(f"Forward&Backward: {fbtime:.3f} {ctime}")
    return {"forward_backward": fbtime}


def run_profile(layer, input_size_, device, runs, forward_only=False, dtype=torch.float) -> dict[str, float]:
    input_ = torch.randn(input_size_, device=torch.device(device), dtype=dtype)
    # input_.requires_grad_(True)

    # Force CUDA initialization & warm up
    warmup(layer, input_, 100, forward_only)

    forward_min = math.inf
    forward_time = 0
    backward_min = math.inf
    backward_time = 0

    # Memory measurement: collect peak memory for each iteration
    forward_memory_peaks = []
    backward_memory_peaks = []

    for _ in range(runs):
        # Clear all cached memory and gradients before each measurement
        layer.zero_grad()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

        # Measure forward pass
        torch.cuda.synchronize()
        start = time.time()
        new_i = layer(input_)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        forward_min = min(forward_min, elapsed)
        forward_time += elapsed

        # Record peak memory for this forward pass
        forward_peak_mb = torch.cuda.max_memory_allocated() / (1024**3)
        forward_memory_peaks.append(forward_peak_mb)
        torch.cuda.reset_max_memory_allocated()

        if not forward_only:
            # Measure backward pass
            torch.cuda.synchronize()
            start = time.time()
            new_i[0].sum().backward()
            torch.cuda.synchronize()
            elapsed = time.time() - start
            backward_min = min(backward_min, elapsed)
            backward_time += elapsed

            # Record peak memory for this backward pass
            backward_peak_mb = torch.cuda.max_memory_allocated() / (1024**3)
            backward_memory_peaks.append(backward_peak_mb)

    ctime, scale = list(TIME_SCALES.items())[0]
    forward_min *= scale
    backward_min *= scale
    forward_average = forward_time / runs * scale
    backward_average = backward_time / runs * scale

    # Calculate meaningful memory statistics
    forward_gb_avg = sum(forward_memory_peaks) / len(forward_memory_peaks)
    forward_gb_max = max(forward_memory_peaks)
    forward_gb_min = min(forward_memory_peaks)

    if backward_memory_peaks:
        backward_gb_avg = sum(backward_memory_peaks) / len(backward_memory_peaks)
        backward_gb_max = max(backward_memory_peaks)
        backward_gb_min = min(backward_memory_peaks)
    else:
        backward_gb_avg = backward_gb_max = backward_gb_min = 0

    print(
        f"Forward: mem avg {forward_gb_avg:.3f}GB (max {forward_gb_max:.3f}GB) / "
        f"time avg {forward_average:.3f}{ctime} | "
        f"Backward: mem avg {backward_gb_avg:.3f}GB (max {backward_gb_max:.3f}GB) / "
        f"time avg {backward_average:.3f}{ctime}"
    )

    return {
        "forward_avg": forward_average,
        "backward_avg": backward_average,
        "forward_gb_avg": forward_gb_avg,
        # "forward_gb_max": forward_gb_max,
        # "forward_gb_min": forward_gb_min,
        "backward_gb_avg": backward_gb_avg,
        # "backward_gb_max": backward_gb_max,
        # "backward_gb_min": backward_gb_min,
    }


def run_worker(gpu, world_size, layer, input_size_, runs, dtype=torch.float, output: list[dict[str, int]] = None):
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:8899", world_size=world_size, rank=gpu)

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(gpu)

    batch = (int)(input_size_[0] / world_size)
    if gpu == 0:
        run_size = input_size_.copy()
        run_size[0] = input_size_[0] - batch * (world_size - 1)
    else:
        run_size = input_size_.copy()
        run_size[0] = batch

    run_model = layer.to(device)
    run_model = nn.parallel.DistributedDataParallel(run_model, device_ids=[gpu])

    retval = run_wall(run_model, run_size, device, runs, (gpu == 0), dtype)
    if output is not None:
        output.append(retval)
