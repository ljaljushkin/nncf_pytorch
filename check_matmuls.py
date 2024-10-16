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

import openvino.runtime as ov

# p = Path('/home/nlyaly/sandbox/tmp/whisper')
p = Path("/mnt/cifs/ov-share-05/cv_bench_cache/WW40_llm-optimum_2024.5.0-16901-32aaa2fbd96")
num_dims = set()
for model_dir in p.iterdir():
    for ov_xml in model_dir.glob("**/*.xml"):
        # print(ov_xml)
        try:
            ov_model = ov.Core().read_model(ov_xml)
        except RuntimeError:
            # print('Runtime Error')
            continue
        operations = ov_model.get_ops()
        matmul_nodes = [op for op in operations if op.get_type_name() == "MVN"]
        if matmul_nodes:
            print(len(matmul_nodes), model_dir.name, ov_xml)
        num_dims.add(len(matmul_nodes))
        # for node in matmul_nodes:
        #     num_dims.add(tuple(len(i.get_partial_shape()) for i in node.inputs()))
print(num_dims)


# p = Path('/mnt/cifs/ov-share-05/cv_bench_cache/WW40_llm-optimum_2024.5.0-16901-32aaa2fbd96')

# print(f"{ndims} MatMul Node: {node.get_friendly_name()}")
# all_ndims.add(*ndims)

#     global_first.union(first_dims)
#     global_second.union(second_dims)
#     print(model_dir.name, first_dims, second_dims)
#     map_model_vs_ndims_for_matmuls[model_dir.name] = (first_dims, second_dims)

# print(global_first, global_second)

# ov_xml = p / 'bloomz-560m/pytorch/ov/FP16/openvino_model.xml'
# ov_model = ov.Core().read_model(ov_xml)

# nncf_graph = NNCFGraphFactory.create(ov_model)

# for node in nncf_graph.get_all_nodes():
#     if node.metatype == OVMatMulMetatype:
#         print(node.node_name)
#         input_edges = nncf_graph.get_input_edges(node)
#         for input_edge in input_edges:
#             print(input_edge.tensor_shape)
#             break
