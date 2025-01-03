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

import os

import yaml

mlruns_dir = "/home/nlyaly/projects/nncf2/examples/llm_compression/torch/mlruns"

for root, dirs, files in os.walk(mlruns_dir):
    for file in files:
        if file == "meta.yaml":
            file_path = os.path.join(root, file)
            print("found file:", file_path)
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
            data["experiment_id"] = "847852149191820453"
            with open(file_path, "w") as f:
                yaml.safe_dump(data, f)
