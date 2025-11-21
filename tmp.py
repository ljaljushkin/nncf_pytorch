import numpy as np
import openvino as ov
from openvino import opset13 as opset

import nncf


def get_one_layer_model():
    input_node = opset.parameter([2, 3, 10], name="Input_1")
    weights_data = np.random.randn(64, 10)
    weights_data[-1, -1] = 10000
    current_weights = opset.constant(weights_data, dtype=np.float32, name="weights")
    matmul_node = opset.matmul(input_node, current_weights, transpose_a=False, transpose_b=True, name="MatMul")

    weights_data_2 = np.random.randn(4, 64)
    current_weights_2 = opset.constant(weights_data_2, dtype=np.float32, name="weights")
    matmul_node_2 = opset.matmul(matmul_node, current_weights_2, transpose_a=False, transpose_b=True, name="MatMul")

    result = opset.result(matmul_node_2, name="Result")
    result.get_output_tensor(0).set_names(set(["Result"]))
    model = ov.Model([result], [input_node])
    return model


model = get_one_layer_model()
ov.save_model(model, "original.xml")

model = ov.Core().read_model("e2m1_compressed.xml")
model = nncf.compress_weights(
    model,
    # mode=nncf.CompressWeightsMode.MXFP4,
    mode=nncf.CompressWeightsMode.INT4_SYM,
    group_size=8,
    all_layers=True,
)
ov.save_model(model, "int4_e2m1_compressed.xml")
# ov.save_model(model, "e2m1_compressed.xml")


# if not type_list:
#     return all_nodes_of_type
# for nncf_node in self.nodes.values():
#     if nncf_node.node_type in type_list:
#         all_nodes_of_type.append(nncf_node)
# return all_nodes_of_type

# tests??
# Main case:
# TODO: data-free - test for matching MXFP4 pattern (compress model with 2 linear layers (SequentialMatmulModel) to mxfp4 then to int4)
# TODO: reproducer for issue with double-compression when first layer compressed to int4 with second in ignored scope,
#  and then all model compressed to int8
# TODO: data-aware algorithm with mxfp4->int4
# TODO: test for not matching MXFP4 decompression pattern (e.g. when e2m1 is directly converted to fp16)
# -> expected to skip weight from compression and modification

# Corner cases:
# TODO: test for matching MXFP8 pattern (has different types for weight and scales)
# TODO: handle "Convert" op in the decompression subgraph correctly (different cases with activations and weights in fp16, bf16, f32)
# TODO: reproduce issue with layout (reshaped const and scale for group quantization + 3D MOE case)
