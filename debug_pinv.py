# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from time import time

import numpy as np
import scipy
import torch

"""
70.87x faster lstsq for [8x8]   0.018604 vs 0.000262
17.47x faster lstsq for [1028x8]   0.007878 vs 0.000451
1.02x faster lstsq for [2024x256]   0.028259 vs 0.027669
4.53x faster lstsq for [2024x2024]   0.619389 vs 0.136586
1.26x faster lstsq for [20240x2560]   2.749271 vs 2.178973

3.70x faster lstsq for [8x8]   0.005147 vs 0.001390
2.45x faster lstsq for [1028x8]   0.002087 vs 0.000851
2.18x faster lstsq for [2024x256]   0.047880 vs 0.021922
5.15x faster lstsq for [2024x2024]   0.620695 vs 0.120440
1.46x faster lstsq for [20240x2560]   2.677321 vs 1.836982

92.45x faster lstsq for [8x8]   0.022130 vs 0.000239
27.04x faster lstsq for [1028x8]   0.009998 vs 0.000370
1.10x faster lstsq for [2024x256]   0.027362 vs 0.024975
5.21x faster lstsq for [2024x2024]   0.589248 vs 0.113002
1.15x faster lstsq for [20240x2560]   2.566608 vs 2.230172
"""
for x, y in [(8, 8), (1028, 8), (2024, 256), (2024, 2024), (20240, 2560)]:
    A = torch.randn(x, y)
    B = torch.randn(x, y)
    C = torch.randn(x, y)
    start = time()
    t1 = torch.linalg.pinv(A)
    t2 = t1 @ B
    t3 = t1 @ C
    time1 = time() - start
    start = time()
    t1 = torch.linalg.lstsq(A, B).solution
    t2 = torch.linalg.lstsq(A, C).solution
    time2 = time() - start
    print(f"{time1 / time2:.2f}x faster lstsq for [{x}x{y}]   {time1:4f} vs {time2:4f}\n")

"""
numpy results:
0.61x faster lstsq for [8x8]   0.000861 vs 0.001421
0.49x faster lstsq for [1028x8]   0.000526 vs 0.001076
0.41x faster lstsq for [2024x256]   0.121046 vs 0.292080
0.21x faster lstsq for [2024x2024]   1.730708 vs 8.315658
0.35x faster lstsq for [20240x2560]   10.547004 vs 30.238744
"""

for x, y in [(8, 8), (1028, 8), (2024, 256), (2024, 2024), (20240, 2560)]:
    A = np.random.rand(x, y)
    B = np.random.rand(x, y)
    C = np.random.rand(x, y)
    start = time()
    t1 = np.linalg.pinv(A)
    t2 = t1 @ B
    t3 = t1 @ C
    time1 = time() - start
    start = time()
    t1 = np.linalg.lstsq(A, B)[0]
    t2 = np.linalg.lstsq(A, C)[0]
    time2 = time() - start
    print(f"{time1 / time2:.2f}x faster lstsq for [{x}x{y}]   {time1:4f} vs {time2:4f}\n")


"""
scipy results:
133.92x faster lstsq for [8x8]   0.046999 vs 0.000351
0.55x faster lstsq for [1028x8]   0.000697 vs 0.001264
0.40x faster lstsq for [2024x256]   0.115728 vs 0.292409
0.19x faster lstsq for [2024x2024]   1.244587 vs 6.648910
0.28x faster lstsq for [20240x2560]   10.331703 vs 37.240068

"""
for x, y in [(8, 8), (1028, 8), (2024, 256), (2024, 2024), (20240, 2560)]:
    A = np.random.rand(x, y)
    B = np.random.rand(x, y)
    C = np.random.rand(x, y)
    start = time()
    t1 = scipy.linalg.pinv(A)
    t2 = t1 @ B
    t3 = t1 @ C
    time1 = time() - start
    start = time()
    t1 = scipy.linalg.lstsq(A, B)[0]
    t2 = scipy.linalg.lstsq(A, C)[0]
    time2 = time() - start
    print(f"{time1 / time2:.2f}x faster lstsq for [{x}x{y}]   {time1:4f} vs {time2:4f}\n")
