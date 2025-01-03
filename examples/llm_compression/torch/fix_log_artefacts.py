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


def update_artifact_location(root_dir, old_path, new_path):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file == "meta.yaml":
                # print(subdir)
                file_path = os.path.join(subdir, file)
                with open(file_path, "r") as f:
                    data = yaml.safe_load(f)

                if "artifact_uri" in data and old_path in data["artifact_uri"]:
                    # print('old_path={}\n new_path={}\n'.
                    # format(data['artifact_uri'], data['artifact_uri'].replace(old_path, new_path)))
                    data["artifact_uri"] = data["artifact_uri"].replace(old_path, new_path)
                    # print('new_path={}\n'.format(data['artifact_uri']))
                with open(file_path, "w") as f:
                    yaml.safe_dump(data, f)


root_directory = "/home/nlyaly/projects/nncf/examples/llm_compression/torch"
old_artifact_location = (
    "file:///local_ssd2/nlyalyus/projects/nncf/examples/llm_compression/torch/mlruns/641700664634679570"
)
new_artifact_location = "file:///home/nlyaly/projects/nncf/examples/llm_compression/torch/mlruns/947665597319518370"

update_artifact_location(root_directory, old_artifact_location, new_artifact_location)
