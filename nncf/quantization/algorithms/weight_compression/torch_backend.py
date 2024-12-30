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

from typing import Dict, Iterable, List, Optional, Tuple

import torch

import nncf
import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import CONST_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.quantization.structs import QuantizationScheme
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.tensor_statistics.statistics import WCTensorStatistic
from nncf.experimental.common.tensor_statistics.collectors import MeanReducer
from nncf.experimental.common.tensor_statistics.collectors import NoopAggregator
from nncf.experimental.common.tensor_statistics.collectors import ShapeReducer
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.lora_correction import LoraCorrectionAlgorithm
from nncf.tensor import Tensor
from nncf.tensor.definitions import TensorDataType
from nncf.torch.graph.graph import PTTargetPoint
from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.model_graph_manager import find_const_node_in_constant_subgraph
from nncf.torch.model_graph_manager import get_const_node
from nncf.torch.model_graph_manager import get_module_by_name
from nncf.torch.model_graph_manager import split_const_name
from nncf.torch.model_transformer import PTModelTransformer
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec


class PTWeightCompressionAlgoBackend(WeightCompressionAlgoBackend):
    TARGET_TYPE_TO_PT_INS_TYPE_MAP = {
        TargetType.PRE_LAYER_OPERATION: TargetType.OPERATOR_PRE_HOOK,
        TargetType.POST_LAYER_OPERATION: TargetType.OPERATOR_POST_HOOK,
    }
    MATMUL_METATYPES = [om.PTLinearMetatype, om.PTMatMulMetatype, om.PTAddmmMetatype]
    EMBEDDING_METATYPES = [om.PTEmbeddingMetatype, om.PTAtenEmbeddingMetatype]
    CONVOLUTION_METATYPES = [
        om.PTConv1dMetatype,
        om.PTConv2dMetatype,
        om.PTConv3dMetatype,
        om.PTDepthwiseConv1dSubtype,
        om.PTDepthwiseConv2dSubtype,
        om.PTDepthwiseConv3dSubtype,
        om.PTConvTranspose1dMetatype,
        om.PTConvTranspose2dMetatype,
        om.PTConvTranspose3dMetatype,
    ]

    @property
    def matmul_metatypes(self) -> List[OperatorMetatype]:
        return PTWeightCompressionAlgoBackend.MATMUL_METATYPES

    @property
    def embedding_metatypes(self) -> List[OperatorMetatype]:
        return PTWeightCompressionAlgoBackend.EMBEDDING_METATYPES

    @property
    def convolution_metatypes(self) -> List[OperatorMetatype]:
        return PTWeightCompressionAlgoBackend.CONVOLUTION_METATYPES

    @staticmethod
    def is_node_with_weights(node: NNCFNode, graph: NNCFGraph) -> bool:
        if (
            node.metatype not in PTWeightCompressionAlgoBackend.MATMUL_METATYPES
            and node.metatype not in PTWeightCompressionAlgoBackend.EMBEDDING_METATYPES
            and node.metatype not in PTWeightCompressionAlgoBackend.CONVOLUTION_METATYPES
        ):
            return False
        for prev_node in graph.get_previous_nodes(node):
            edge = graph.get_edge(prev_node, node)
            if edge.input_port_id not in node.metatype.weight_port_ids:
                continue
            weight_node = find_const_node_in_constant_subgraph(prev_node, graph)
            if weight_node is not None:
                return True
        return False

    @staticmethod
    def get_weight_names_and_port_ids(node: NNCFNode, graph: NNCFGraph) -> List[Tuple[str, int]]:
        weight_port_ids = []
        for prev_node in graph.get_previous_nodes(node):
            weight_node = find_const_node_in_constant_subgraph(prev_node, graph)
            if weight_node is None:
                continue
            edge = graph.get_edge(prev_node, node)
            if edge.input_port_id in node.metatype.weight_port_ids:
                weight_port_ids.append((weight_node.layer_attributes.name, edge.input_port_id))
        return weight_port_ids

    @staticmethod
    def get_reduction_axes(node_with_weight: NNCFNode, weight_port_id: int, graph: NNCFGraph) -> Optional[Tuple[int]]:
        weight_node = get_const_node(node_with_weight, weight_port_id, graph)

        ndims = len(weight_node.layer_attributes.shape)
        reduction_axes = None
        if node_with_weight.metatype == om.PTEmbeddingMetatype:
            reduction_axes = [1]
        elif node_with_weight.metatype == om.PTLinearMetatype:
            reduction_axes = [ndims - 1]
        elif node_with_weight.metatype == om.PTMatMulMetatype:
            if weight_port_id == 0:
                reduction_axes = [ndims - 1]
            elif weight_port_id == 1:
                reduction_axes = [max(0, ndims - 2)]
        elif node_with_weight.metatype == om.PTAddmmMetatype:
            if weight_port_id == 1:
                reduction_axes = [ndims - 1]
            elif weight_port_id == 2:
                reduction_axes = [max(0, ndims - 2)]
        elif node_with_weight.metatype in PTWeightCompressionAlgoBackend.CONVOLUTION_METATYPES:
            channel_idx = (
                1
                if node_with_weight.metatype
                in [om.PTConvTranspose1dMetatype, om.PTConvTranspose2dMetatype, om.PTConvTranspose3dMetatype]
                else 0
            )
            reduction_axes = [i for i in range(ndims) if i != channel_idx]
        return tuple(reduction_axes)

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> PTTargetPoint:
        if NNCFGraphNodeType.INPUT_NODE in target_node_name or target_type == TargetType.POST_LAYER_OPERATION:
            port_id = None
        if target_type in PTWeightCompressionAlgoBackend.TARGET_TYPE_TO_PT_INS_TYPE_MAP:
            target_type = PTWeightCompressionAlgoBackend.TARGET_TYPE_TO_PT_INS_TYPE_MAP[target_type]
        return PTTargetPoint(target_type, target_node_name, input_port_id=port_id)

    def mean_statistic_collector(
        self, reduction_axes: Tuple[int], subset_size: Optional[int] = None
    ) -> TensorCollector:
        mean_reducer = MeanReducer(reduction_axes)
        shape_reducer = ShapeReducer()
        collector = TensorCollector(WCTensorStatistic)
        collector.register_statistic_branch(WCTensorStatistic.MEAN_STAT, mean_reducer, NoopAggregator(subset_size))
        collector.register_statistic_branch(WCTensorStatistic.SHAPE_STAT, shape_reducer, NoopAggregator(subset_size))
        return collector

    @staticmethod
    def get_activation_port_id(node: NNCFNode, graph: NNCFGraph) -> int:
        activation_ports = []
        for prev_node in graph.get_previous_nodes(node):
            if prev_node.metatype in CONST_NOOP_METATYPES:
                continue
            edge = graph.get_edge(prev_node, node)
            activation_ports.append(edge.input_port_id)
        assert len(activation_ports) == 1
        return activation_ports[0]

    def get_weight(
        self, node_with_weight: NNCFNode, weight_port_id: int, model: torch.nn.Module, graph: NNCFGraph
    ) -> Tensor:
        weight_node = get_const_node(node_with_weight, weight_port_id, graph)
        weight_name = weight_node.layer_attributes.name
        module_name, weight_attr_name = split_const_name(weight_name)
        module = get_module_by_name(module_name, model)
        weight = getattr(module, weight_attr_name)
        if weight is None or not isinstance(weight, torch.nn.Parameter):
            raise nncf.InternalError(f"Could not find a torch.nn.Parameter in the model by name {weight_name}.")

        return Tensor(weight)

    def get_weight_dtype(
        self, node_with_weight: NNCFNode, weight_port_id: int, model: torch.nn.Module, graph: NNCFGraph
    ) -> TensorDataType:
        return self.get_weight(node_with_weight, weight_port_id, model, graph).dtype

    @staticmethod
    def get_weight_shape(node_with_weight: NNCFNode, weight_port_id: int, graph: NNCFGraph) -> Tuple:
        weight_node = get_const_node(node_with_weight, weight_port_id, graph)
        return tuple(weight_node.layer_attributes.shape)

    def set_weight(
        self, node_with_weight: NNCFNode, weight_port_id: int, model: torch.nn.Module, graph: NNCFGraph, weight: Tensor
    ):
        pass

    def insert_adapters(
        self, wc_params: WeightCompressionParameters, lora_A: Tensor, lora_B: Tensor, int8_lora: bool
    ) -> None:
        pass

    @staticmethod
    def init_lora_adapters(svd_residual, reduction_axes=None, rank=None):
        # O stands for output dimension, H - input dimension or hidden size, R - rank.
        U_full, S_full, V_full = torch.linalg.svd(svd_residual, full_matrices=False)
        U = U_full[:, :rank]  # [H, R]
        S_sqrt = torch.sqrt(S_full)
        S = torch.diag(S_sqrt[:rank])  # [R, R]
        V = V_full[:rank, :]  # [R, O]
        V = S @ V  # [R, O]
        U = U @ S  # [H, R]
        return U, V

    def transform_model(
        self,
        model: NNCFNetwork,
        graph: NNCFGraph,
        weight_compression_parameters: Iterable[WeightCompressionParameters],
        precomputed_scales: Dict[str, Tensor] = None,
        precomputed_zero_points: Dict[str, Tensor] = None,
        lora_correction_algo: LoraCorrectionAlgorithm = None,
    ) -> NNCFNetwork:
        transformation_layout = TransformationLayout()
        for wc_params in weight_compression_parameters:
            compression_config = wc_params.compression_config
            mode = compression_config.mode
            print(f"Quantize {wc_params.weight_name} to {compression_config.mode.value}")

            weight_node = get_const_node(wc_params.node_with_weight, wc_params.weight_port_id, graph)
            weight_name = weight_node.layer_attributes.name
            module_name, weight_attr_name = split_const_name(weight_name)
            module = get_module_by_name(module_name, model)
            weight = getattr(module, weight_attr_name)
            if weight is None or not isinstance(weight, torch.nn.Parameter):
                raise nncf.InternalError(f"Could not find a torch.nn.Parameter in the model by name {weight_name}.")

            orig_weight_shape = weight_shape = list(weight.shape)
            scale_shape = weight_shape.copy()
            scale_shape[wc_params.reduction_axes[0]] = 1

            schema = QuantizationScheme.SYMMETRIC
            if mode in [CompressWeightsMode.INT4_ASYM, CompressWeightsMode.INT8_ASYM]:
                schema = QuantizationScheme.ASYMMETRIC

            quantizer_config = QuantizerConfig(
                num_bits=compression_config.num_bits,
                mode=schema,
                # TODO: extend for group-wise case
            )

            lora_rank = 256
            group_size = compression_config.group_size
            out_features, in_features = weight_shape
            group_reduction_axes = wc_params.reduction_axes[0]

            if compression_config.num_bits == 4 and group_size > 0:
                group_reduction_axes = 2
                weight_shape = [out_features, in_features // group_size, group_size]
                scale_shape = [out_features, in_features // group_size, 1]

            reshaped_weight = weight.reshape(weight_shape)

            # Group-wise:  Weight [a1, r, a2] -> Scale [a1, 1, a2]
            # Per-channel: Weight [a1, a2]    -> Scale [a1, 1]
            input_low = torch.amin(reshaped_weight, dim=group_reduction_axes, keepdim=True).float()
            input_high = torch.amax(reshaped_weight, dim=group_reduction_axes, keepdim=True).float()

            quantizer_spec = PTQuantizerSpec.from_config(
                quantizer_config,
                narrow_range=False,
                scale_shape=scale_shape,
                weight_shape=weight_shape,
                half_range=False,
                logarithm_scale=False,
                is_quantized_on_export=False,
                compression_lr_multiplier=None,
                lora_rank=lora_rank,
                group_size=group_size,
                module_name=module_name,
            )
            quantizer_cls = QUANTIZATION_MODULES.get(quantizer_config.mode)
            quantizer = quantizer_cls(quantizer_spec)

            if isinstance(quantizer, AsymmetricQuantizer):
                quantizer.input_low = torch.nn.Parameter(input_low)
                input_range = input_high - input_low
                # Subtract eps from the input_range to make quantizer parameters equal to
                # original parameters on the forward call.
                quantizer.input_range = torch.nn.Parameter(input_range - quantizer.eps)
            else:
                signed_scale = True
                quantizer.signed = bool(torch.any(input_low.data < 0))
                quantizer.set_levels()
                ll_lh = quantizer.level_low / quantizer.level_high
                if signed_scale:
                    w_abs_min = torch.abs(input_low)
                    w_max = input_high
                    scale = torch.where(w_abs_min >= w_max, w_abs_min, -w_max)
                    eps = quantizer.eps
                    scale = quantizer.scale = torch.nn.Parameter(torch.where(torch.abs(scale) < eps, eps, scale))
                    # range: [-s, 7/8s] if s>0 else [7/8s,-s]
                    input_low = torch.where(scale > 0, -scale, -scale / ll_lh)
                    input_range = torch.abs((2 + 1 / quantizer.level_low) * scale)  # 15/8s or (2-1/8)s
                    quantizer.scale = torch.nn.Parameter(torch.where(torch.abs(scale) < eps, eps, scale))
                else:
                    quantizer.scale = torch.nn.Parameter(input_high.data - quantizer.eps)
                    input_low = quantizer.scale * ll_lh
                    input_range = quantizer.scale - input_low
            quantizer.to(weight.device)

            if compression_config.num_bits == 4:
                quantizer._lora_A = torch.nn.Parameter(quantizer._lora_A.type(dtype=weight.dtype))
                quantizer._lora_B = torch.nn.Parameter(quantizer._lora_B.type(dtype=weight.dtype))

                weight = reshaped_weight.reshape(orig_weight_shape)
                fq_weight = quantizer.quantize(weight)
                print("quant noise before SVD={:.2f}".format(torch.linalg.norm(fq_weight - weight, ord="fro").item()))
                svd_residual = (torch.rand(weight_shape, dtype=weight.dtype).to(weight.device) / 100) * input_range / 15
                svd_residual = svd_residual.reshape(orig_weight_shape)
                svd_residual = svd_residual.type(
                    torch.float32
                )  # otherwise "svd_cuda_gesvdj" not implemented for 'BFloat16'
                B, A = self.init_lora_adapters(svd_residual, rank=quantizer.lora_rank)
                quantizer._lora_A = torch.nn.Parameter(A.type(dtype=weight.dtype))
                quantizer._lora_B = torch.nn.Parameter(B.type(dtype=weight.dtype))
                fq_weight = quantizer.quantize(weight)
                print(
                    "quant noise right after SVD={:.2f}".format(torch.linalg.norm(fq_weight - weight, ord="fro").item())
                )

            node_name = weight_node.node_name
            fq_node_name = wc_params.node_with_weight.node_name
            # TODO: why not wc_params.weight_port_id)???
            #  because post hook for constant??
            target_point = PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS, fq_node_name, input_port_id=1)
            storage_key = "FQ_LORA_for_node_{}".format(node_name.replace(".", "_"))

            transformation_layout.register(
                PTSharedFnInsertionCommand(
                    target_points=[target_point],
                    fn=quantizer,
                    op_unique_name=storage_key,
                    compression_module_type=ExtraCompressionModuleType.EXTERNAL_QUANTIZER,
                    priority=TransformationPriority.QUANTIZATION_PRIORITY,
                )
            )

        # apply transformations
        transformed_model = PTModelTransformer(model).transform(transformation_layout)
        return transformed_model
