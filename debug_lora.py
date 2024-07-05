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

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

p1 = Path("/home/nlyaly/projects/nncf/tests/post_training/nncf_debug/lora/fns_losses.csv")
# p1 = Path("/home/nlyaly/projects/nncf/tests/post_training/nncf_debug/lora/curr_losses.csv")
# p1 = Path("/home/nlyaly/projects/nncf/tests/post_training/nncf_debug/lora/gold_losses_0.83682.csv")
df = pd.read_csv(p1)

df = df.drop(df.columns[[0]], axis=1)

# if layer data in row
delta = df.iloc[0] - df.iloc[-1]
print("is all layers improved: ", all(delta > 0))


fig, ax = plt.subplots(1)
ax.plot(delta)
ax.set_xticklabels([])
# plt.show()
plt.savefig(p1.parent / "fns.jpg")


# print(df)
# df = df.T
# df.to_csv(p.parent / "losses.csv")

# for column in df.columns:

# print(df[column])
