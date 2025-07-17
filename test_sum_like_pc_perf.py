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

import torch
import triton

from sum_like_kernels import sum_like_baseline
from sum_like_kernels import sum_like_two_stage


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N_ELEMENTS"],
        x_vals=[64 * 128 * s * s for s in [16, 32, 64, 128, 256]],
        line_arg="provider",
        line_vals=[
            "pytorch",
            "kernel_baseline",
            "kernel_two_stage",
        ],
        line_names=[
            "PyTorch",
            "Kernel Baseline (tl.atomic_add)",
            "Kernel Two Stage (tl.sum per-block/per-channel)",
        ],
        styles=[
            ("blue", "-"),
            ("red", "--"),
            ("green", "-."),
        ],
        ylabel="ms",
        plot_name="sum-like-4d-performance-comparison",
        args={"D1_size": 128},
    )
)
def benchmark(D1_size, N_ELEMENTS, provider):
    # Infer shapes from total elements
    D0_size = 64
    D2_size = int((N_ELEMENTS / (D0_size * D1_size)) ** 0.5)
    D3_size = int(N_ELEMENTS / (D0_size * D1_size * D2_size))

    shape = (D0_size, D1_size, D2_size, D3_size)
    ref_shape = (1, D1_size, 1, 1)

    x = torch.randn(shape, device="cuda", dtype=torch.float16)
    ref = torch.empty(ref_shape, device="cuda", dtype=torch.float16)

    quantiles = [0.2, 0.5, 0.8]

    if provider == "pytorch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.sum(x, axis=(0, 2, 3), keepdim=True), quantiles=quantiles
        )
    elif provider == "kernel_baseline":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sum_like_baseline(x, ref), quantiles=quantiles)
    elif provider == "kernel_two_stage":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sum_like_two_stage(x, ref), quantiles=quantiles)

    return ms, min_ms, max_ms


if __name__ == "__main__":
    print("Running benchmark...")
    benchmark.run(show_plots=True, print_data=True, save_path="out_bench")
