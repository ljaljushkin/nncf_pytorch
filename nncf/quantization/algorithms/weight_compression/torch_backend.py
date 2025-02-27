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

from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch

import nncf
import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.graph import NNCFGraph, NNCFNode
from nncf.common.graph.operator_metatypes import CONST_NOOP_METATYPES, OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType, TransformationPriority
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.quantization.structs import QuantizationScheme
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.experimental.common.tensor_statistics.collectors import (
    MaxVarianceReducer,
    MeanAbsMaxReducer,
    MeanAggregator,
    MeanReducer,
    MeanVarianceReducer,
    NoopAggregator,
    ShapeReducer,
    TensorCollector,
)
from nncf.experimental.common.tensor_statistics.statistics import (
    MaxVarianceTensorStatistic,
    MeanMagnitudeTensorStatistic,
    MeanVarianceTensorStatistic,
    WCTensorStatistic,
)
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.backend import (
    MixedPrecisionAlgoBackend,
    WeightCompressionAlgoBackend,
)
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.lora_correction import LoraCorrectionAlgorithm
from nncf.quantization.algorithms.weight_compression.weight_lowering import compress_weight
from nncf.tensor import Tensor
from nncf.tensor.definitions import TensorDataType
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.graph.graph import PTTargetPoint
from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType, PTSharedFnInsertionCommand
from nncf.torch.model_graph_manager import (
    find_const_node_in_constant_subgraph,
    get_const_data,
    get_const_node,
    get_module_by_name,
    split_const_name,
)
from nncf.torch.model_transformer import PTModelTransformer
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.layers import (
    QUANTIZATION_MODULES,
    INT4AsymmetricWeightsDecompressor,
    INT4SymmetricWeightsDecompressor,
    INT8AsymmetricWeightsDecompressor,
    INT8SymmetricWeightsDecompressor,
    PTLoraSpec,
    PTQuantizerSpec,
)


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
        weight = get_const_data(weight_node, model)
        if weight is None:
            msg = f"Could not find a torch.nn.Parameter in the model by name {weight_name}."
            raise nncf.InternalError(msg)
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
    def init_lora_adapters(svd_residual, rank=None):
        # O stands for output dimension, H - input dimension or hidden size, R - rank.
        U_full, S_full, V_full = torch.linalg.svd(svd_residual, full_matrices=False)
        U = U_full[:, :rank]  # [H, R]
        S_sqrt = torch.sqrt(S_full)
        S = torch.diag(S_sqrt[:rank])  # [R, R]
        V = V_full[:rank, :]  # [R, O]
        V = S @ V  # [R, O]
        U = U @ S  # [H, R]
        return U, V

    @staticmethod
    def get_filter_fn_for_statistics(activation_port_id: int, algorithm_key: str) -> Callable[[StatisticPoint], bool]:
        def filter_func(point: StatisticPoint) -> bool:
            return (
                algorithm_key in point.algorithm_to_tensor_collectors
                and point.target_point.type
                == PTWeightCompressionAlgoBackend.TARGET_TYPE_TO_PT_INS_TYPE_MAP[TargetType.POST_LAYER_OPERATION]
            )

        return filter_func

    # TODO: reduce number of params
    @staticmethod
    def get_fq_insertion_command(
        compressed_weight,
        wc_params,
        orig_weight_shape,
    ):
        compression_config = wc_params.compression_config
        mode_vs_schema_map = {
            CompressWeightsMode.INT4_ASYM: QuantizationScheme.ASYMMETRIC_LORA,
            CompressWeightsMode.INT4_SYM: QuantizationScheme.SYMMETRIC_LORA,
            CompressWeightsMode.INT8_ASYM: QuantizationScheme.ASYMMETRIC,
            CompressWeightsMode.INT8_SYM: QuantizationScheme.SYMMETRIC,
        }
        schema = mode_vs_schema_map[compression_config.mode]

        lora_rank = 256
        device = compressed_weight.tensor.data.device
        scale = compressed_weight.scale.data
        zero_point = compressed_weight.zero_point.data
        weight_shape = compressed_weight.tensor.shape

        quantizer_spec = PTQuantizerSpec(
            num_bits=compression_config.num_bits,
            mode=schema,
            signedness_to_force=True,
            narrow_range=False,
            half_range=False,
            scale_shape=scale.shape,
            weight_shape=weight_shape,
            logarithm_scale=False,
        )

        quantizer_cls = QUANTIZATION_MODULES.get(schema)
        if schema in [QuantizationScheme.ASYMMETRIC_LORA, QuantizationScheme.SYMMETRIC_LORA]:
            lora_spec = PTLoraSpec(lora_rank=lora_rank, orig_weight_shape=orig_weight_shape)
            quantizer = quantizer_cls(quantizer_spec, lora_spec)
            # TODO: re-evaluate for all types: float32, float16, bfloat16
            lora_dtype = quantizer._lora_A.dtype
            svd_residual = torch.rand(weight_shape).to(device) * scale / 100  # value on [0,1] * (1/100 of quant size)
            svd_residual = svd_residual.reshape(orig_weight_shape)
            B, A = PTWeightCompressionAlgoBackend.init_lora_adapters(svd_residual, rank=lora_rank)
            quantizer._lora_A = torch.nn.Parameter(A.type(dtype=lora_dtype))
            quantizer._lora_B = torch.nn.Parameter(B.type(dtype=lora_dtype))
        else:
            quantizer = quantizer_cls(quantizer_spec)

        levels = quantizer.levels
        if schema in [QuantizationScheme.ASYMMETRIC_LORA, QuantizationScheme.ASYMMETRIC]:
            dtype = quantizer.input_low.dtype
            # NOTE: loose some accuracy, because of invertion of round
            input_low = -zero_point * scale
            input_range = scale * (levels - 1)
            quantizer.input_low = torch.nn.Parameter(input_low.type(dtype))
            quantizer.input_range = torch.nn.Parameter(input_range.type(dtype))
        else:
            scale = scale.type(quantizer.scale.dtype)
            quantizer.scale = torch.nn.Parameter(scale * levels / 2)

        target_node_name = wc_params.node_with_weight.node_name
        target_point = PTTargetPoint(
            TargetType.OPERATION_WITH_WEIGHTS, target_node_name=target_node_name, input_port_id=wc_params.weight_port_id
        )
        storage_key = "FQ_LORA_{}".format(wc_params.weight_name.replace(".", "_"))

        return PTSharedFnInsertionCommand(
                target_points=[target_point],
                fn=quantizer,
                op_unique_name=storage_key,
                compression_module_type=ExtraCompressionModuleType.EXTERNAL_QUANTIZER,
                priority=TransformationPriority.QUANTIZATION_PRIORITY,
            )

    @staticmethod
    def get_dq_insertion_command(
        compressed_weight,
        wc_params,
        model,
        graph,
        weight_node,
    ):
        weight_name = weight_node.layer_attributes.name
        module_name, weight_attr_name = split_const_name(weight_name)
        module = get_module_by_name(module_name, model)
        weight = getattr(module, weight_attr_name)
        if not isinstance(weight, torch.nn.Parameter):
            msg = f"Weight is not a torch.nn.Parameter in the model by name {weight_name}."
            raise nncf.InternalError(msg)
        weight_dtype = weight.dtype
        weight_shape = weight.shape

        compression_config = wc_params.compression_config
        # creates weight decompressor
        if compression_config.mode == CompressWeightsMode.INT8_SYM:
            decompressor = INT8SymmetricWeightsDecompressor(compressed_weight.scale.data, result_dtype=weight_dtype)
        elif compression_config.mode == CompressWeightsMode.INT8_ASYM:
            decompressor = INT8AsymmetricWeightsDecompressor(
                compressed_weight.scale.data, compressed_weight.zero_point.data, result_dtype=weight_dtype
            )
        elif compression_config.mode == CompressWeightsMode.INT4_SYM:
            decompressor = INT4SymmetricWeightsDecompressor(
                scale=compressed_weight.scale.data,
                compressed_weight_shape=compressed_weight.tensor.shape,
                result_shape=weight_shape,
                result_dtype=weight_dtype,
            )
        elif compression_config.mode == CompressWeightsMode.INT4_ASYM:
            decompressor = INT4AsymmetricWeightsDecompressor(
                scale=compressed_weight.scale.data,
                zero_point=compressed_weight.zero_point.data,
                compressed_weight_shape=compressed_weight.tensor.shape,
                result_shape=weight_shape,
                result_dtype=weight_dtype,
            )

        # pack tensor
        packed_tensor = decompressor.pack_weight(compressed_weight.tensor.data)

        # sets compressed tensor
        # TODO:(AlexanderDokuchaev): update set_const_data
        compressed_parameter = torch.nn.Parameter(packed_tensor, requires_grad=False)
        setattr(module, weight_attr_name, compressed_parameter)

        consumer_nodes = graph.get_next_nodes(weight_node)
        if len(consumer_nodes) > 1:
            for c_node in consumer_nodes:
                c_module = model.nncf.get_module_by_scope(Scope.from_str(c_node.layer_name))
                for name, param in c_module.named_parameters(recurse=False, remove_duplicate=False):
                    if id(param) == id(weight):
                        setattr(c_module, name, compressed_parameter)

        # registry weight decompression module in the model
        decompressor_name = f"weights_decompressor_{weight_node.node_name.replace('.', '_')}"

        # inserts the weight decompressor into the model as the post hook on the model weight
        return PTSharedFnInsertionCommand(
                [PTTargetPoint(TargetType.OPERATOR_POST_HOOK, target_node_name=weight_node.node_name)],
                decompressor,
                decompressor_name,
            )

    def transform_model(
        self,
        model: NNCFNetwork,
        graph: NNCFGraph,
        weight_compression_parameters: Iterable[WeightCompressionParameters],
        precomputed_scales: Dict[str, Tensor] = None,
        precomputed_zero_points: Dict[str, Tensor] = None,
        lora_correction_algo: LoraCorrectionAlgorithm = None,
    ) -> NNCFNetwork:
        # TODO: pass somehow?
        fq_lora = True
        # TODO: support compression format
        # TODO: compression_format.FQ
        #   doesn't work for group_size != -1
        #   only per-channel, need to remove specifics, affects PTQ
        transformation_layout = TransformationLayout()
        for wc_params in weight_compression_parameters:
            compression_config = wc_params.compression_config
            if compression_config.mode in [
                CompressWeightsMode.NF4,
                CompressWeightsMode.E2M1,
            ]:
                msg = f"{compression_config.mode.value} is not supported."
                raise nncf.ParameterNotSupportedError(msg)

            print(f"Quantize {wc_params.weight_name} to {compression_config.mode.value}")

            weight_node = get_const_node(wc_params.node_with_weight, wc_params.weight_port_id, graph)
            weight_name = weight_node.layer_attributes.name
            weight = get_const_data(weight_node, model)
            if weight is None:
                msg = f"Could not find a torch.nn.Parameter in the model by name {weight_name}."
                raise nncf.InternalError(msg)

            # calculates compressed weights and decompression parameters
            compressed_weight = compress_weight(
                Tensor(weight),
                wc_params.reduction_axes,
                compression_config,
                None if precomputed_scales is None else precomputed_scales.get(wc_params.weight_name),
                None if precomputed_zero_points is None else precomputed_zero_points.get(wc_params.weight_name),
            )

            if fq_lora:
                command = self.get_fq_insertion_command(compressed_weight, wc_params, weight.shape)
            else:
                command = self.get_dq_insertion_command(compressed_weight, wc_params, model, graph, weight_node)
            transformation_layout.register(command)

        # apply transformations
        transformed_model = PTModelTransformer(model).transform(transformation_layout)
        return transformed_model


class PTMixedPrecisionAlgoBackend(MixedPrecisionAlgoBackend, PTWeightCompressionAlgoBackend):
    @staticmethod
    def mean_variance_statistic_collector(
        reduction_axes: Tuple[int], subset_size: Optional[int] = None
    ) -> TensorCollector:
        reducer = MeanVarianceReducer(reduction_axes)
        aggregator = MeanAggregator(num_samples=subset_size)
        collector = TensorCollector(MeanVarianceTensorStatistic)
        collector.register_statistic_branch(MeanVarianceTensorStatistic.MEAN_VARIANCE_STAT, reducer, aggregator)
        return collector

    @staticmethod
    def max_variance_statistic_collector(
        reduction_axes: Tuple[int], subset_size: Optional[int] = None
    ) -> TensorCollector:
        reducer = MaxVarianceReducer(reduction_axes)
        aggregator = MeanAggregator(num_samples=subset_size)
        collector = TensorCollector(MaxVarianceTensorStatistic)
        collector.register_statistic_branch(MaxVarianceTensorStatistic.MAX_VARIANCE_STAT, reducer, aggregator)
        return collector

    @staticmethod
    def mean_abs_max_statistic_collector(
        reduction_axes: Tuple[int], subset_size: Optional[int] = None
    ) -> TensorCollector:
        reducer = MeanAbsMaxReducer(reduction_axes)
        aggregator = MeanAggregator(num_samples=subset_size)
        collector = TensorCollector(MeanMagnitudeTensorStatistic)
        collector.register_statistic_branch(MeanMagnitudeTensorStatistic.MEAN_MAGNITUDE_STAT, reducer, aggregator)
        return collector
