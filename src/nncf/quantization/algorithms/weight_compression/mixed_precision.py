# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC
from abc import abstractmethod
from typing import Iterable, Optional, TypeVar

import nncf
from nncf import Dataset
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.logging import nncf_logger
from nncf.common.logging.track_progress import track
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.common.utils.registry import Registry
from nncf.parameters import CompressWeightsMode
from nncf.parameters import SensitivityMetric
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.weight_lowering import get_integer_quantization_error
from nncf.tensor import Tensor
from nncf.tensor import functions as fns

TModel = TypeVar("TModel")
MIXED_PRECISION_CRITERIA = Registry("mixed_precision_criteria")
THE_LOWEST_SENSITIVITY = 0

# Supported bit-widths for mixed precision selection (in descending order)
SUPPORTED_BITS = [8, 4, 2]
DEFAULT_BACKUP_BITS = 8


def choose_bits_per_layer_with_path(
    layers: dict[str, list[tuple[int, int, float]]], target_bits: int
) -> tuple[Optional[float], Optional[list[tuple[str, int]]]]:
    """
    Dynamic programming algorithm to select optimal bit-width per layer.

    Args:
        layers: A dict mapping each layer name to a list of candidate options.
                Each option is a tuple of (bits, bits_cost, loss_cost).
        target_bits: Upper bound on the total bit budget.

    Returns:
        (min_loss, best_path), where best_path is a list of
        (layer_name, bits) for each layer, or (None, None) if no feasible
        solution exists.
    """
    # dp: total_bits -> (accumulated_loss, chosen_path)
    # The path explicitly stores the selected options.
    dp: dict[int, tuple[float, list[tuple[str, int]]]] = {0: (0.0, [])}

    for layer_name, opts in layers.items():
        new_dp: dict[int, tuple[float, list[tuple[str, int]]]] = {}
        for cur_bits, (cur_loss, cur_path) in dp.items():
            for opt in opts:
                bits, bits_cost, loss_cost = opt
                new_total = cur_bits + bits_cost
                if new_total > target_bits:
                    continue

                new_loss = cur_loss + loss_cost
                new_path = cur_path + [(layer_name, bits)]

                # Keep the path with smaller loss for the same bit budget
                if new_total not in new_dp or new_loss < new_dp[new_total][0]:
                    new_dp[new_total] = (new_loss, new_path)

        if not new_dp:
            return None, None

        # Pareto pruning: remove dominated (bits, loss) states
        items = sorted(new_dp.items(), key=lambda x: x[0])  # (bits, (loss, path))
        pruned: dict[int, tuple[float, list[tuple[str, int]]]] = {}
        best_loss_so_far = float("inf")
        for bits_val, (loss_val, path_val) in items:
            if loss_val < best_loss_so_far:
                pruned[bits_val] = (loss_val, path_val)
                best_loss_so_far = loss_val

        dp = pruned

    # Select the solution with the minimum loss
    if not dp:
        return None, None
    best_bits = min(dp.keys(), key=lambda k: dp[k][0])
    best_loss, best_path = dp[best_bits]
    return best_loss, best_path


def compute_layer_bits(weight_param: WeightCompressionParameters, num_bits: int) -> int:
    """
    Compute total bits used for a layer given the number of bits per weight.

    :param weight_param: Weight parameters for the layer.
    :param num_bits: Number of bits per weight element.
    :return: Total bits for the layer.
    """
    return int(weight_param.num_weights) * num_bits


class MixedPrecisionCriterion(Algorithm):
    """
    Computes mixed quantization scheme (e.g. uniform int8, int4, or int2)
    for weights based on sensitivity criteria using dynamic programming optimization.

    The algorithm selects bit-widths for each layer to minimize total loss (sensitivity-weighted
    quantization error) while staying within a bit budget derived from the ratio parameter.
    """

    def __init__(
        self,
        avg_bits: float,
        subset_size: Optional[int] = None,
        primary_bits: int = 4,
        backup_bits: int = DEFAULT_BACKUP_BITS,
        available_bits: Optional[list[int]] = None,
        ratio: Optional[float] = None,
        use_dp: bool = False,
    ):
        """
        :param avg_bits: Target average bits per weight value. The algorithm selects bit-widths
            for each layer to achieve this target while minimizing loss. E.g., avg_bits=4.0
            targets 4 bits per weight on average.
        :param subset_size: Size of dataset subset for statistics.
        :param primary_bits: Primary (lowest) precision bits to use (default: 4).
        :param backup_bits: Backup (highest) precision bits (default: 8).
        :param available_bits: List of available bit-widths to use. Defaults to [backup_bits, primary_bits].
        :param ratio: Original ratio parameter for backward compatibility with FP8 modes.
            When primary_bits == backup_bits, this ratio is used directly since avg_bits
            cannot encode the ratio information.
        :param use_dp: If True, force DP selection regardless of number of bit options.
            Set to True when available_bits is explicitly provided by user.
        """
        self._avg_bits = avg_bits
        self._subset_size = subset_size
        self._algorithm_key = f"MPC_{hash(self)}"
        self._backend_entity = None
        self._primary_bits = primary_bits
        self._backup_bits = backup_bits
        self._ratio = ratio
        self._use_dp = use_dp
        self._available_bits = (
            sorted(set(available_bits), reverse=True)
            if available_bits is not None
            else sorted(set([backup_bits, primary_bits]), reverse=True)
        )

    @abstractmethod
    def _calc_sensitivity(
        self,
        model: TModel,
        graph: NNCFGraph,
        weight_params: list[WeightCompressionParameters],
        statistic_points: Optional[StatisticPointsContainer] = None,
    ) -> list[float]:
        """
        Calculates sensitivity of each layer according to a criterion.

        :return: List of values per node to be quantized.
        """

    def _get_compression_mode_for_bits(self, num_bits: int, is_asym: bool = False) -> CompressWeightsMode:
        """
        Get the compression mode for a given bit-width.

        :param num_bits: Number of bits.
        :param is_asym: Whether to use asymmetric quantization.
        :return: CompressWeightsMode enum value.
        """
        if num_bits == 8:
            return CompressWeightsMode.INT8_ASYM if is_asym else CompressWeightsMode.INT8_SYM
        if num_bits == 4:
            return CompressWeightsMode.INT4_ASYM if is_asym else CompressWeightsMode.INT4_SYM
        if num_bits == 2:
            return CompressWeightsMode.INT2_ASYM if is_asym else CompressWeightsMode.INT2_SYM
        msg = f"Unsupported bit-width: {num_bits}. Supported values are 2, 4, 8."
        raise ValueError(msg)

    def _build_layer_options(
        self,
        weight_params: list[WeightCompressionParameters],
        scores: list[float],
    ) -> dict[str, list[tuple[int, int, float]]]:
        """
        Build options for each layer with different bit-widths.

        :param weight_params: List of weight parameters.
        :param scores: Sensitivity scores for each layer.
        :return: Dict mapping layer name to list of (bits, bits_cost, loss_cost) tuples.
        """
        layer_options = {}
        for wp, score in zip(weight_params, scores):
            layer_name = wp.node_with_weight.node_name
            options = []
            for bits in self._available_bits:
                bits_cost = compute_layer_bits(wp, bits)
                # Loss cost: higher sensitivity layers should prefer higher bits
                # Lower bits = higher loss contribution for sensitive layers
                loss_cost = score * (self._backup_bits - bits) / self._backup_bits
                options.append((bits, bits_cost, loss_cost))
            layer_options[layer_name] = options
        return layer_options

    def _apply_greedy_selection(
        self,
        weight_params: list[WeightCompressionParameters],
        scores: list[float],
    ) -> list[WeightCompressionParameters]:
        """
        Original greedy selection for backward compatibility with 2-bit-option cases.

        Selects layers with lowest sensitivity until adding more would exceed the target.
        Uses avg_bits to compute the target ratio of weights in primary precision.

        :param weight_params: List of weight parameters.
        :param scores: Sensitivity scores for each layer.
        :return: List of weight parameters selected for primary precision.
        """
        num_all_weights = sum(wp.num_weights for wp in weight_params)

        # Handle edge case: when primary and backup bits are the same (e.g., FP8 vs INT8),
        # use the original ratio directly since avg_bits cannot encode ratio info
        if self._backup_bits == self._primary_bits:
            # For format-based mixed precision (FP8 vs INT8), use original ratio
            target_ratio = self._ratio if self._ratio is not None else 1.0
        else:
            # Convert avg_bits to ratio of weights in primary precision
            # avg_bits = ratio * primary_bits + (1 - ratio) * backup_bits
            # ratio = (backup_bits - avg_bits) / (backup_bits - primary_bits)
            target_ratio = (self._backup_bits - self._avg_bits) / (self._backup_bits - self._primary_bits)
        target_ratio = max(0.0, min(1.0, target_ratio))  # Clamp to [0, 1]

        primary_precision_weight_params = []
        indexes_of_layers_in_ascending_order_of_scores = [
            i[0] for i in sorted(enumerate(scores), reverse=False, key=lambda x: x[1])
        ]
        num_weights_in_primary = 0
        for index in indexes_of_layers_in_ascending_order_of_scores:
            weight_param = weight_params[index]
            current_ratio = (num_weights_in_primary + weight_param.num_weights) / num_all_weights
            if current_ratio >= target_ratio:
                break
            primary_precision_weight_params.append(weight_param)
            num_weights_in_primary += weight_param.num_weights

        return primary_precision_weight_params

    def _apply_dp_selection(
        self,
        weight_params: list[WeightCompressionParameters],
        scores: list[float],
    ) -> tuple[dict[str, int], list[WeightCompressionParameters]]:
        """
        Dynamic programming selection for multi-bit-option cases (3+ bit-widths).

        Optimizes bit-width assignment to minimize sensitivity-weighted loss within a bit budget.
        The target bit budget is: avg_bits * total_weights.

        :param weight_params: List of weight parameters.
        :param scores: Sensitivity scores for each layer.
        :return: Tuple of (layer_bits_map, primary_precision_weight_params).
        """
        total_weights = sum(int(wp.num_weights) for wp in weight_params)
        total_backup_bits = total_weights * self._backup_bits
        target_bits = int(self._avg_bits * total_weights)

        nncf_logger.info(f"Mixed precision (DP): Total weights: {total_weights:,}")
        nncf_logger.info(
            f"Mixed precision (DP): Target bit budget: {target_bits:,} bits out of {total_backup_bits:,} bits"
        )
        nncf_logger.info(f"Mixed precision (DP): Target avg bits per weight: {self._avg_bits:.2f}")

        # Build options for DP
        layer_options = self._build_layer_options(weight_params, scores)

        # Run dynamic programming to find optimal bit assignment
        best_loss, best_path = choose_bits_per_layer_with_path(layer_options, target_bits)

        if best_path is None:
            msg = (
                f"Mixed precision: Could not find feasible solution with target bit budget {target_bits}. "
                f"Minimum required bits: {sum(int(wp.num_weights) * min(self._available_bits) for wp in weight_params)}"
            )
            raise ValueError(msg)

        # Create mapping from layer name to selected bits
        layer_bits_map = {layer_name: bits for layer_name, bits in best_path}

        # Calculate statistics and collect per-layer info
        actual_bits_used = 0
        bits_distribution = {bits: 0 for bits in self._available_bits}
        weights_per_bits = {bits: 0 for bits in self._available_bits}
        per_layer_info = []  # List of (num_weights, precision) tuples per layer

        for wp in weight_params:
            layer_name = wp.node_with_weight.node_name
            selected_bits = layer_bits_map.get(layer_name, self._backup_bits)
            actual_bits_used += int(wp.num_weights) * selected_bits
            bits_distribution[selected_bits] += 1
            weights_per_bits[selected_bits] += int(wp.num_weights)
            per_layer_info.append((int(wp.num_weights), selected_bits))

            # Assign compression config based on selected bits
            mode = self._get_compression_mode_for_bits(selected_bits, is_asym=True)
            wp.compression_config = WeightCompressionConfig(mode=mode, group_size=wp.compression_config.group_size)

        avg_bits = actual_bits_used / total_weights if total_weights > 0 else 0
        compression_ratio = actual_bits_used / total_backup_bits if total_backup_bits > 0 else 0

        nncf_logger.info(f"Mixed precision (DP): Actual avg bits per weight: {avg_bits:.3f}")
        nncf_logger.info(f"Mixed precision (DP): Bit budget utilization: {compression_ratio:.2%}")
        nncf_logger.info(f"Mixed precision (DP): Layer distribution by bits: {bits_distribution}")
        nncf_logger.debug(f"Mixed precision (DP): Weights per bit-width: {weights_per_bits}")
        nncf_logger.debug(f"Mixed precision (DP): Per-layer (num_weights, bits): {per_layer_info}")

        # Return all weight_params since DP sets compression config for all layers
        # This prevents the caller from overwriting configs with backup precision
        return layer_bits_map, weight_params

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
        weight_params: list[WeightCompressionParameters] = None,
    ) -> list[WeightCompressionParameters]:
        """
        Selects bit-widths for each layer based on sensitivity scores.

        For 2 bit-width options (e.g., 4-bit and 8-bit): Uses the original greedy algorithm
        for backward compatibility. The ratio is interpreted as the maximum fraction of
        weights in primary precision.

        For 3+ bit-width options (e.g., 2-bit, 4-bit, 8-bit): Uses dynamic programming to
        minimize loss within a bit budget. The ratio is interpreted as target_bits / backup_bits.

        :return: List of weight parameters for layers that should use primary precision
                 (for backward compatibility).
        """
        self._set_backend_entity(model)

        scores = self._calc_sensitivity(model, graph, weight_params, statistic_points)

        # Use greedy selection when:
        # - Only 1-2 unique bit options AND use_dp is False (backward compatibility)
        # - Same bit-width for primary and backup (format-based mixed precision like FP8 vs INT8)
        # Use DP when:
        # - 3+ bit options
        # - use_dp is True (available_bits explicitly provided by user)
        use_greedy = (len(self._available_bits) <= 2 and not self._use_dp) or self._primary_bits == self._backup_bits
        if use_greedy:
            return self._apply_greedy_selection(weight_params, scores)

        # Use DP for explicit available_bits or 3+ bit options
        _, primary_precision_weight_params = self._apply_dp_selection(weight_params, scores)
        return primary_precision_weight_params

    @abstractmethod
    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """

    @abstractmethod
    def get_statistic_points(
        self,
        model: TModel,
        graph: NNCFGraph,
        nodes_and_port_ids: Iterable[tuple[NNCFNode, int]],
    ) -> StatisticPointsContainer:
        """
        Returns statistic points, for which StatisticsCollector should collect statistics.

        :param model: Model for statistics collection.
        :param graph: Model graph.
        :param nodes_and_port_ids: Nodes and port ids for which statistics should be collected.
        :return: Statistic points, for which StatisticsCollector should collect statistics.
        """


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.WEIGHT_QUANTIZATION_ERROR)
class DataFreeCriterion(MixedPrecisionCriterion):
    """
    A baseline mixed precision criterion that is based on quantization noise of weights only.
    """

    @property
    def available_backends(self) -> list[BackendType]:
        return [BackendType.OPENVINO, BackendType.TORCH, BackendType.TORCH_FX, BackendType.ONNX]

    def _set_backend_entity(self, model: TModel) -> None:
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.weight_compression.openvino_backend import (
                OVTensorWeightCompressionAlgoBackend,
            )

            self._backend_entity = OVTensorWeightCompressionAlgoBackend(model)
        elif model_backend == BackendType.TORCH:
            from nncf.quantization.algorithms.weight_compression.torch_backend import PTWeightCompressionAlgoBackend

            self._backend_entity = PTWeightCompressionAlgoBackend()
        elif model_backend == BackendType.TORCH_FX:
            from nncf.quantization.algorithms.weight_compression.torch_fx_backend import FXWeightCompressionAlgoBackend

            self._backend_entity = FXWeightCompressionAlgoBackend()
        elif model_backend == BackendType.ONNX:
            from nncf.quantization.algorithms.weight_compression.onnx_backend import ONNXWeightCompressionAlgoBackend

            self._backend_entity = ONNXWeightCompressionAlgoBackend(model)
        else:
            msg = f"Cannot return backend-specific entity because {model_backend.value} is not supported!"
            raise nncf.UnsupportedBackendError(msg)

    def _calc_weight_sensitivity(
        self,
        weight_param: WeightCompressionParameters,
        model: TModel,
        graph: NNCFGraph,
    ) -> float:
        weight = self._backend_entity.get_weight(
            weight_param.node_with_weight,
            weight_param.weight_port_id,
            model,
            graph,
        )
        backup_config = WeightCompressionConfig()
        reduction_axes = weight_param.reduction_axes
        int_error = get_integer_quantization_error(weight, reduction_axes, backup_config, reduction="max_mean")
        eps = fns.finfo(weight).eps
        return 1 / (int_error + eps)

    def _calc_score_per_node(
        self,
        weight_param: WeightCompressionParameters,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
    ) -> float:
        weight_score = self._calc_weight_sensitivity(weight_param, model, graph)
        return weight_score

    def _calc_sensitivity(
        self,
        model: TModel,
        graph: NNCFGraph,
        weight_params: list[WeightCompressionParameters],
        statistic_points: Optional[StatisticPointsContainer] = None,
    ) -> list[float]:
        scores = []
        for weight_param in track(weight_params, description="Mixed-Precision assignment"):
            scores.append(self._calc_score_per_node(weight_param, model, graph, statistic_points))
        return scores

    def get_statistic_points(
        self,
        model: TModel,
        graph: NNCFGraph,
        nodes_and_port_ids: Iterable[tuple[NNCFNode, int]],
    ) -> StatisticPointsContainer:
        msg = "No statistics collection intended for data-free mixed precision criterion"
        raise RuntimeError(msg)


class DataBasedCriterion(DataFreeCriterion, ABC):
    """
    Data-based mixed precision criterion that takes into account outliers in the input statistics.
    Expecting statistics of the following shape: [hidden_dim]
    """

    STAT_KEY = None

    @property
    def available_backends(self) -> list[BackendType]:
        return [BackendType.OPENVINO, BackendType.TORCH, BackendType.ONNX]

    def _set_backend_entity(self, model: TModel) -> None:
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.weight_compression.openvino_backend import OVMixedPrecisionAlgoBackend

            self._backend_entity = OVMixedPrecisionAlgoBackend(model)
        elif model_backend == BackendType.TORCH:
            from nncf.quantization.algorithms.weight_compression.torch_backend import PTMixedPrecisionAlgoBackend

            self._backend_entity = PTMixedPrecisionAlgoBackend()
        elif model_backend == BackendType.TORCH_FX:
            from nncf.quantization.algorithms.weight_compression.torch_fx_backend import FXMixedPrecisionAlgoBackend

            self._backend_entity = FXMixedPrecisionAlgoBackend()
        elif model_backend == BackendType.ONNX:
            from nncf.quantization.algorithms.weight_compression.onnx_backend import ONNXMixedPrecisionAlgoBackend

            self._backend_entity = ONNXMixedPrecisionAlgoBackend(model)
        else:
            msg = f"Cannot return backend-specific entity because {model_backend.value} is not supported!"
            raise nncf.UnsupportedBackendError(msg)

    def _calc_activation_sensitivity(
        self,
        weight_param: WeightCompressionParameters,
        graph: NNCFGraph,
        statistic_points: StatisticPointsContainer,
    ) -> float:
        stats = self._get_statistics_for_node(statistic_points, weight_param.node_with_weight, graph, self.STAT_KEY)
        return stats[0].item()

    def _calc_score_per_node(
        self,
        weight_param: WeightCompressionParameters,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
    ):
        """
        NOTE: Data-based criteria for assigning 4-bit/8-bit precisions are valid for Matmul operations only.
        However, in some cases it can be beneficial to quantize Gather layers to 4-bit.
        Since there's no data-aware estimation of sensitivity in these layers, they receive the lowest sensitivity.
        It allows assigning Gather operation 4-bit in the first place.
        """
        if weight_param.node_with_weight.metatype in self._backend_entity.embedding_metatypes:
            return THE_LOWEST_SENSITIVITY
        weight_score = self._calc_weight_sensitivity(weight_param, model, graph)
        activation_score = self._calc_activation_sensitivity(weight_param, graph, statistic_points)
        return weight_score * activation_score

    def get_statistic_points(
        self,
        model: TModel,
        graph: NNCFGraph,
        nodes_and_port_ids: Iterable[tuple[NNCFNode, int]],
    ) -> StatisticPointsContainer:
        self._set_backend_entity(model)

        statistic_container = StatisticPointsContainer()
        for act_node, output_port_id, _ in nodes_and_port_ids:
            n_dims = len(graph.get_output_edges_by_port_id(act_node, output_port_id)[0].tensor_shape)
            if n_dims < 2:
                msg = (
                    f"Data-aware mixed precision criteria are not supported for MatMuls with 1D inputs. "
                    f"Node: {act_node.node_name}, number of dimensions: {n_dims}."
                )
                raise RuntimeError(msg)
            statistic_point = self._backend_entity.target_point(
                TargetType.POST_LAYER_OPERATION, act_node.node_name, port_id=output_port_id
            )
            stat_collector = self._get_statistic_collector()
            statistic_container.add_statistic_point(
                StatisticPoint(
                    target_point=statistic_point, tensor_collector=stat_collector, algorithm=self._algorithm_key
                )
            )

        return statistic_container

    @abstractmethod
    def _get_statistic_collector(self):
        """
        Get statistic collector
        """

    def _get_activation_node_and_port(self, node: NNCFNode, nncf_graph: NNCFGraph) -> tuple[NNCFNode, int]:
        """
        This method returns the activation layer and corresponding port id for the node.

        :param node: NNCFGraph node for which the activation is sought.
        :param nncf_graph: NNCFGraph instance with the node.
        :return: Tuple with the activation node and port id.
        """
        activation_port = self._backend_entity.get_activation_port_id(node, nncf_graph)
        activation_edge = nncf_graph.get_input_edge_by_port_id(node, activation_port)
        activation_node = activation_edge.from_node
        port_id = activation_edge.output_port_id
        return activation_node, port_id

    def _get_statistics_for_node(
        self, statistic_points: StatisticPointsContainer, node: NNCFNode, nncf_graph: NNCFGraph, stat_key: str
    ) -> list[Tensor]:
        act_node, act_port_id = self._get_activation_node_and_port(node, nncf_graph)
        stats = []
        for tensor_collector in statistic_points.get_algo_statistics_for_node(
            act_node.node_name,
            self._backend_entity.get_filter_fn_for_statistics(act_port_id, self._algorithm_key),
            self._algorithm_key,
        ):
            statistics = tensor_collector.get_statistics()
            for data in statistics.get_data().values():
                if isinstance(data, Tensor):
                    stats.append(data)
                else:
                    stats.extend(data)
        return stats


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.HESSIAN_INPUT_ACTIVATION)
class HAWQCriterion(DataBasedCriterion):
    """
    Calculates the average Hessian trace of weights with respect to the layer-wise quantization error
    multiplied by L2 norm of 8-bit quantization noise.
    """

    STAT_KEY = SensitivityMetric.HESSIAN_INPUT_ACTIVATION.value

    def _calc_weight_sensitivity(
        self,
        weight_param: WeightCompressionParameters,
        model: TModel,
        graph: NNCFGraph,
    ) -> float:
        weight = self._backend_entity.get_weight(
            weight_param.node_with_weight, weight_param.weight_port_id, model, graph
        )
        backup_config = WeightCompressionConfig()
        reduction_axes = weight_param.reduction_axes
        return get_integer_quantization_error(weight, reduction_axes, backup_config, reduction="frobenius")

    def _get_statistic_collector(self):
        return self._backend_entity.hawq_statistic_collector(self._subset_size)


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.MEAN_ACTIVATION_VARIANCE)
class MeanVarianceCriterion(DataBasedCriterion):
    """
    The mean variance of the layers' inputs multiplied by inverted 8-bit quantization noise.
    """

    STAT_KEY = SensitivityMetric.MEAN_ACTIVATION_VARIANCE.value

    def _get_statistic_collector(self):
        # Reducing across the second-last dimension, assuming it is the sequence length dimension
        return self._backend_entity.mean_variance_statistic_collector(
            reduction_axes=(-2,), subset_size=self._subset_size
        )


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.MAX_ACTIVATION_VARIANCE)
class MaxVarianceCriterion(DataBasedCriterion):
    """
    The maximum variance of the layers' inputs multiplied by inverted 8-bit quantization noise.
    """

    STAT_KEY = SensitivityMetric.MAX_ACTIVATION_VARIANCE.value

    def _get_statistic_collector(self):
        # Reducing across the second-last dimension, assuming it is the sequence length dimension
        return self._backend_entity.max_variance_statistic_collector(
            reduction_axes=(-2,), subset_size=self._subset_size
        )


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE)
class MeanMaxCriterion(DataBasedCriterion):
    """
    The mean magnitude of the layers' inputs multiplied by inverted 8-bit quantization noise.
    """

    STAT_KEY = SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE.value

    def _get_statistic_collector(self):
        # Reducing across the second-last dimension, assuming it is the sequence length dimension
        return self._backend_entity.mean_abs_max_statistic_collector(
            reduction_axes=(-2,), subset_size=self._subset_size
        )
