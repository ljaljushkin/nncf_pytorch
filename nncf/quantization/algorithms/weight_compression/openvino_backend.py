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
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import openvino as ov
from openvino.runtime import opset13 as opset

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.utils import get_reduction_axes
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.openvino.graph.metatypes import openvino_metatypes as om
from nncf.openvino.graph.metatypes.groups import ATOMIC_ACTIVATIONS_OPERATIONS
from nncf.openvino.graph.model_transformer import OVModelTransformer
from nncf.openvino.graph.node_utils import get_const_value
from nncf.openvino.graph.node_utils import get_weight_channel_axes
from nncf.openvino.graph.transformations.command_creation import OVCommandCreator
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.rt_info import dump_parameters
from nncf.openvino.statistics.collectors import get_raw_stat_collector
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.awq_patterns import get_awq_patterns
from nncf.quantization.algorithms.weight_compression.backend import AWQAlgoBackend
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.weight_lowering import compress_weight
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_dequantization
from nncf.tensor import Tensor
from nncf.tensor.definitions import TensorDataType


@dataclass
class LoraParams:
    rank: int = 8
    n_iters: int = 3
    w_regulation: bool = False


class OVWeightCompressionAlgoBackend(WeightCompressionAlgoBackend):
    def __init__(self, model: ov.Model, name_to_node_mapping: Dict = None):
        if name_to_node_mapping is None:
            self.name_to_node_mapping = OVModelTransformer._get_name_to_node_mapping(model)
        else:
            self.name_to_node_mapping = name_to_node_mapping

    @property
    def matmul_metatypes(self) -> List[OperatorMetatype]:
        return [om.OVMatMulMetatype]

    @property
    def convolution_metatypes(self) -> List[OperatorMetatype]:
        return [
            om.OVConvolutionMetatype,
            om.OVDepthwiseConvolutionMetatype,
            om.OVConvolutionBackpropDataMetatype,
            om.OVGroupConvolutionMetatype,
            om.OVGroupConvolutionBackpropDataMetatype,
        ]

    @property
    def embedding_metatypes(self) -> List[OperatorMetatype]:
        return [om.OVEmbeddingMetatype]

    @staticmethod
    def is_node_with_weights(node: NNCFNode, graph: NNCFGraph) -> bool:
        return node.layer_attributes and node.layer_attributes.constant_attributes

    @staticmethod
    def get_reduction_axes(node_with_weight: NNCFNode, weight_port_id: int, graph: NNCFGraph) -> Optional[Tuple[int]]:
        channel_axes = get_weight_channel_axes(node_with_weight)
        const_shape = node_with_weight.layer_attributes.constant_attributes[weight_port_id]["shape"]
        return get_reduction_axes(channel_axes, const_shape)

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def raw_statistic_collector(num_samples: Optional[int] = None) -> TensorCollector:
        return get_raw_stat_collector(num_samples)

    @staticmethod
    def get_activation_port_id(node: NNCFNode, nncf_graph: NNCFGraph) -> int:
        constant_ports = node.layer_attributes.get_const_port_ids()
        activation_ports = [
            e.input_port_id for e in nncf_graph.get_input_edges(node) if e.input_port_id not in constant_ports
        ]
        assert len(activation_ports) == 1
        return activation_ports[0]

    @staticmethod
    def get_weight_names_and_port_ids(node: NNCFNode, graph: NNCFGraph) -> List[Tuple[str, int]]:
        result = []
        for weight_port_id in node.layer_attributes.get_const_port_ids():
            weight_name = node.layer_attributes.constant_attributes[weight_port_id]["name"]
            result.append((weight_name, weight_port_id))
        return result

    def get_weight(self, node_with_weight: NNCFNode, weight_port_id: int, model: ov.Model, graph: NNCFGraph) -> Tensor:
        weight_name = node_with_weight.layer_attributes.constant_attributes[weight_port_id]["name"]
        weight_node = self.name_to_node_mapping[weight_name]
        weight_tensor = get_const_value(weight_node)
        return Tensor(weight_tensor)

    def set_weight(
        self, node_with_weight: NNCFNode, weight_port_id: int, model: ov.Model, graph: NNCFGraph, weight: Tensor
    ):
        node_with_const = self.name_to_node_mapping[node_with_weight.node_name]

        const_port = node_with_const.input(weight_port_id)
        const_node = node_with_const.input_value(weight_port_id).get_node()

        new_const_node = ov.runtime.op.Constant(weight.data, shared_memory=True)
        new_const_node.set_friendly_name(const_node.get_friendly_name())
        const_port.replace_source_output(new_const_node.output(0))

        const_name = node_with_weight.layer_attributes.constant_attributes[weight_port_id]["name"]
        self.name_to_node_mapping[const_name] = new_const_node

        new_output = new_const_node.output(0)
        for target_input in const_node.output(0).get_target_inputs():
            target_input.replace_source_output(new_output)

        del const_node

    @staticmethod
    def _get_int_mul(compressed_weight, compression_dtype, const_node_name, wc_params, const_dtype):
        compressed_const = opset.constant(compressed_weight.tensor.data, dtype=compression_dtype, name=const_node_name)
        converted_const = opset.convert(compressed_const, ov.Type.f16)
        if compressed_weight.zero_point is not None:
            zero_point_const = opset.constant(
                compressed_weight.zero_point.data,
                dtype=compression_dtype,
                name=f"{const_node_name}/zero_point",
            )
            converted_zero_point = opset.convert(zero_point_const, ov.Type.f16)
            converted_const = opset.subtract(
                converted_const, converted_zero_point, name=f"{const_node_name}/zero_point/subtract"
            )

        scale_const = opset.constant(compressed_weight.scale.data, dtype=ov.Type.f16, name=f"{const_node_name}/scale")
        mul = opset.multiply(
            converted_const,
            scale_const,
            name=f"{const_node_name}/fq_weights_{wc_params.weight_port_id}",
        )

        mul = opset.convert(mul, const_dtype, name=f"{const_node_name}/fq_weights_{wc_params.weight_port_id}/convert")

        return mul

    # NOTE: not backend specific. can be common. Lora adapters algorithm itself!
    @staticmethod
    def calculate_adapters(weight, compressed_weight, wc_params, lora_params):
        import numpy as np
        import numpy.linalg as linalg
        import scipy.linalg as slinalg

        rank, n_iters, w_regulation = lora_params.rank, lora_params.n_iters, lora_params.w_regulation

        # TODO: support NF4
        fq_weights = do_dequantization(
            compressed_weight.tensor,
            compressed_weight.scale,
            compressed_weight.zero_point,
            wc_params.reduction_axes[0],
        )
        fq_weights_data = fq_weights.data

        X = wc_params.X.data
        diff_before = np.mean(np.abs(weight.data @ X - fq_weights_data @ X))

        # q_w + USV = w => USV = w - q_w
        residual = (weight.data - fq_weights_data).astype(np.float32)
        w_residual = residual.copy()
        if wc_params.reduction_axes == 0:
            residual = np.transpose(residual)
        if wc_params.stat is not None:  # and False:
            s = wc_params.stat.data
            if wc_params.compression_config.group_size > 0:
                gs = wc_params.compression_config.group_size
                n_gs = s.shape[0] // gs
                for i in range(n_gs):
                    offset = i * gs
                    denum = np.sum(s[offset : offset + gs])
                    s[offset : offset + gs] = s[offset : offset + gs] / denum
                    denum = np.max(s[offset : offset + gs])
                    s[offset : offset + gs] = s[offset : offset + gs] / denum
                s = np.expand_dims(s, 0)
                residual = residual * s

        svd = linalg.svd(residual, compute_uv=True, full_matrices=False)
        U = svd[0]
        S = svd[1]
        V = svd[2]

        Ur = U[:, :rank]
        Sr = np.diag(S[:rank])
        Vr = V[:rank, :]

        Vr = Sr @ Vr
        US = Ur

        if wc_params.X is not None:  # rectification by data
            X = wc_params.X.data
            # NOTE: big matmul
            dY = w_residual @ X  # [O, C] * [C, 1] = [O, 1]

            # US @ Vr = res
            # US @ Vr @ X = dY
            # US @ |VR VR @ X| = |res dY|

            for i in range(n_iters):
                VX = Vr @ X
                if not w_regulation:
                    sol = slinalg.lstsq(np.transpose(VX), np.transpose(dY))
                else:
                    VrVX = np.concatenate((Vr, VX), axis=1)
                    dYR = np.concatenate((w_residual, dY), axis=1)
                    sol = slinalg.lstsq(np.transpose(VrVX), np.transpose(dYR), lapack_driver="gelsy")

                diff_after_svd = np.mean(np.abs(weight.data @ X - fq_weights_data @ X - (US @ Vr) @ X))
                # if i == 0:
                # loss.extend([diff_before, diff_after_svd])
                # wandb.log({layer_name: diff_before})
                # wandb.log({layer_name: diff_after_svd})

                US = np.transpose(sol[0])

                diff_after_svd_rectification = np.mean(np.abs(weight.data @ X - fq_weights_data @ X - (US @ Vr) @ X))
                # loss.append(diff_after_svd_rectification)
                # wandb.log({layer_name: diff_after_svd_rectification})
                print(f"{i} Rectification 1: ", diff_before, diff_after_svd, diff_after_svd_rectification)

                USI = linalg.pinv(US)
                if not w_regulation:
                    dYU = USI @ dY
                    sol = slinalg.lstsq(np.transpose(X), np.transpose(dYU), lapack_driver="gelsy")
                else:
                    Ind = np.eye(Vr.shape[1])
                    IX = np.concatenate((Ind, X), axis=1)
                    dYR = np.concatenate((USI @ w_residual, USI @ dY), axis=1)
                    sol = slinalg.lstsq(np.transpose(IX), np.transpose(dYR), lapack_driver="gelsy")

                Vr = np.transpose(sol[0])

                # diff_after_svd_rectification = np.mean(np.abs(weight.data @ X - q_weights_data @ X - (US @ Vr) @ X))
                # loss.append(diff_after_svd_rectification)
                # wandb.log({layer_name: diff_after_svd_rectification})
                # if n_iters - i < 3:
                print(f"{i} Rectification 2: ", diff_before, diff_after_svd, diff_after_svd_rectification)
        # new_residual = US @ Vr
        # print("Before: ", np.mean(np.abs(residual)), " After: ", np.mean(np.abs(residual - new_residual)), rank)
        # weight_delta = np.mean(np.abs(residual)) - np.mean(np.abs(residual - new_residual))
        # original_l1_noise = diff_before
        # l1_noise_delta = diff_before - diff_after_svd_rectification
        # wandb.log(
        #     {
        #         "weight_delta": weight_delta,
        #         "original_l1_noise": original_l1_noise,
        #         "l1_noise_delta": l1_noise_delta,
        #     }
        # )

        return Vr, US

    # NOTE: backend specific code
    def insert_adapters(self, wc_params, V, US, int8_lora):
        input_node = self.name_to_node_mapping[wc_params.node_with_weight.node_name].input_value(0)
        mm_node = self.name_to_node_mapping[wc_params.node_with_weight.node_name]

        const_attributes = wc_params.node_with_weight.layer_attributes.constant_attributes[wc_params.weight_port_id]
        const_node_name = const_attributes["name"]
        const_node = self.name_to_node_mapping[const_node_name]
        const_dtype = const_node.output(0).get_element_type()

        if int8_lora:
            compression_config = WeightCompressionConfig()
            V = Tensor(V)
            compressed_V = compress_weight(
                V,
                wc_params.reduction_axes,
                compression_config,
            )

            US = Tensor(US)
            compressed_US = compress_weight(
                US,
                wc_params.reduction_axes,
                compression_config,
            )
            V_W = self._get_int_mul(
                compressed_V, ov.Type.u8, wc_params.node_with_weight.node_name + "_V", wc_params, const_dtype
            )

            V_MM = opset.matmul(input_node, V_W, transpose_a=False, transpose_b=True)

            US_W = self._get_int_mul(
                compressed_US, ov.Type.u8, wc_params.node_with_weight.node_name + "_U", wc_params, const_dtype
            )

            US_MM = opset.matmul(V_MM, US_W, transpose_a=False, transpose_b=True)
        else:
            V_W = opset.constant(V)
            V_MM = opset.matmul(input_node, V_W, transpose_a=False, transpose_b=True)

            US_W = opset.constant(US)
            US_MM = opset.matmul(V_MM, US_W, transpose_a=False, transpose_b=True)

        node_output_port = mm_node.output(0)
        node_output_source_ports = node_output_port.get_target_inputs()

        add = opset.add(mm_node, US_MM)

        for node_output_source_port in node_output_source_ports:
            node_output_source_port.replace_source_output(add.output(0))

    def insert_lora_residual(
        self,
        model: ov.Model,
        graph: NNCFGraph,
        wc_params: WeightCompressionParameters,
        weight,
        compressed_weight,
        rank=8,
        int8_lora=True,
        w_regulation=False,
        n_iters=3,
        # TODO: pre-computed adapter weights!
        cached_adapter_weights: Dict[str, np.ndarray] = None,
    ):
        # print(wc_params.node_with_weight.node_name)
        # layer_name = wc_params.node_with_weight.node_name.split("/")[0].split("__module.model.layers.")[1]

        # print(compressed_weight.tensor.shape)
        # loss = []

        # print(
        #     f"Tune {layer_name} with rank={rank} w_regulation={w_regulation} n_iters={n_iters} "
        #     f"int8_lora={int8_lora} num_cached={len(cached_adapter_weights)}"
        # )

        # TODO: do only once per weight!
        # get FQ weights compress-decompress via OV model??
        #   is compression should happen after??

        lora_params = LoraParams(rank, n_iters, w_regulation)
        # NOTE: wc_params hides stat and X
        V, US = self.calculate_adapters(weight, compressed_weight, wc_params, lora_params)
        self.insert_adapters(wc_params, V, US, int8_lora=True)

        # return {layer_name: loss}

    def transform_model(
        self,
        model: ov.Model,
        graph: NNCFGraph,
        weight_compression_parameters: Iterable[WeightCompressionParameters],
        precomputed_scales: Dict[str, Tensor] = None,
        precomputed_zero_points: Dict[str, Tensor] = None,
        lora=True,
        num_params=None,
    ) -> ov.Model:
        w_regulation = False
        n_iters = 3
        rank = 8
        for wc_params in weight_compression_parameters:
            compression_config = wc_params.compression_config
            if compression_config.mode == CompressWeightsMode.NF4:
                compression_dtype = ov.Type.nf4
            elif compression_config.mode == CompressWeightsMode.E2M1:
                compression_dtype = ov.Type.f4e2m1
            elif compression_config.mode == CompressWeightsMode.INT4_SYM:
                compression_dtype = ov.Type.i4
            elif compression_config.mode == CompressWeightsMode.INT4_ASYM:
                compression_dtype = ov.Type.u4
            elif compression_config.mode == CompressWeightsMode.INT8_SYM:
                compression_dtype = ov.Type.i8
            elif compression_config.mode == CompressWeightsMode.INT8_ASYM:
                compression_dtype = ov.Type.u8
            else:
                raise ValueError(f"{compression_config.mode.value} is not supported.")

            const_attributes = wc_params.node_with_weight.layer_attributes.constant_attributes[wc_params.weight_port_id]
            const_node_name = const_attributes["name"]
            const_node = self.name_to_node_mapping[const_node_name]
            const_node_output = const_node.output(0)
            const_dtype = const_node_output.get_element_type()

            should_add_convert_node = False
            if const_dtype != ov.Type.f16:
                for inp in const_node_output.get_target_inputs():
                    if inp.get_node().get_type_name() != "Convert":
                        should_add_convert_node = True
                        break

            weight = Tensor(get_const_value(const_node, np.float32 if const_dtype == ov.Type.bf16 else None))
            original_shape = weight.shape
            compressed_weight = compress_weight(
                weight,
                wc_params.reduction_axes,
                compression_config,
                None if precomputed_scales is None else precomputed_scales.get(wc_params.weight_name),
                None if precomputed_zero_points is None else precomputed_zero_points.get(wc_params.weight_name),
            )

            compressed_const = opset.constant(
                compressed_weight.tensor.data, dtype=compression_dtype, name=const_node_name
            )
            converted_const = opset.convert(compressed_const, ov.Type.f16)
            if compressed_weight.zero_point is not None and compressed_weight.tensor.dtype == TensorDataType.uint8:
                zero_point_const = opset.constant(
                    compressed_weight.zero_point.data,
                    dtype=compression_dtype,
                    name=f"{const_node_name}/zero_point",
                )
                converted_zero_point = opset.convert(zero_point_const, ov.Type.f16)
                converted_const = opset.subtract(
                    converted_const, converted_zero_point, name=f"{const_node_name}/zero_point/subtract"
                )

            scale_const = opset.constant(
                compressed_weight.scale.data, dtype=ov.Type.f16, name=f"{const_node_name}/scale"
            )
            mul = opset.multiply(
                converted_const,
                scale_const,
                name=f"{const_node_name}/fq_weights_{wc_params.weight_port_id}",
            )

            if compression_config.group_size != -1:
                mul = opset.reshape(mul, output_shape=original_shape, special_zero=False)

            if should_add_convert_node:
                mul = opset.convert(
                    mul, const_dtype, name=f"{const_node_name}/fq_weights_{wc_params.weight_port_id}/convert"
                )

                mul_output = mul.output(0)
                for target_input in const_node.output(0).get_target_inputs():
                    target_input.replace_source_output(mul_output)

            if wc_params.compression_config.num_bits == 4 and lora:
                self.insert_lora_residual(
                    model,
                    graph,
                    wc_params,
                    weight,
                    compressed_weight,
                    rank=rank,
                    w_regulation=w_regulation,
                    n_iters=n_iters,
                )

        # reset name_to_node_mapping
        self.name_to_node_mapping = None
        return model

    @staticmethod
    def dump_parameters(
        model: ov.Model, parameters: Dict, algo_name: Optional[str] = "quantization", path: Optional[List] = None
    ) -> None:
        dump_parameters(model, parameters, algo_name, path)

    @staticmethod
    def get_compress_decompress_pipeline(
        weight_compression_parameter: WeightCompressionParameters, w_shape, s_shape, z_p_shape=None
    ):
        parameters, clamp = OVWeightCompressionAlgoBackend.get_compress_pipeline(
            weight_compression_parameter, w_shape, s_shape, z_p_shape, True
        )

        if len(parameters) == 3:
            _, s, zp = parameters
            result = (clamp - zp) * s
        else:
            s = parameters[1]
            result = clamp * s

        model = ov.Model([result], parameters)

        compiled_model = ov.compile_model(model)

        return lambda parameters: compiled_model(parameters)[0]

    @staticmethod
    def get_compress_pipeline(
        weight_compression_parameter: WeightCompressionParameters, w_shape, s_shape, z_p_shape=None, return_nodes=False
    ):
        config = weight_compression_parameter.compression_config
        mode = config.mode
        assert mode in [CompressWeightsMode.INT4_SYM, CompressWeightsMode.INT4_ASYM]
        num_bits = config.num_bits

        level_low = 0
        level_high = 2**num_bits - 1

        w = opset.parameter(w_shape, name="w")
        s = opset.parameter(s_shape, name="s")
        parameters = [w, s]
        compressed_w = w / s
        if z_p_shape is not None:
            zp = opset.parameter(z_p_shape, name="zp")
            parameters.append(zp)
            compressed_w += zp

        result = opset.clamp(opset.round(compressed_w), level_low, level_high, name="compressed_weights")

        if return_nodes:
            return parameters, result

        model = ov.Model([result], parameters)

        compiled_model = ov.compile_model(model)

        return lambda parameters: compiled_model(parameters)[0]


class OVAWQAlgoAlgoBackend(AWQAlgoBackend, OVWeightCompressionAlgoBackend):
    @staticmethod
    def get_awq_patterns():
        return get_awq_patterns(om.OVMatMulMetatype, om.OVMultiplyMetatype, ATOMIC_ACTIVATIONS_OPERATIONS)

    @staticmethod
    def scale_insertion_command(source_node, next_nodes, source_node_output_port, scale):
        return OVCommandCreator.multiply_insertion_command(
            source_node, next_nodes, source_node_output_port, scale, f"{source_node.node_name}/awq_mul"
        )
