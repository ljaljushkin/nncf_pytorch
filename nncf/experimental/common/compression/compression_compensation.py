from typing import Dict, List, Optional

import numpy as np
import openvino.runtime as ov
from openvino.runtime import opset9 as opset
from nncf.openvino.statistics.aggregator import OVStatisticsAggregator
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.openvino.graph.transformations.commands import OVTargetPoint, TargetType
from nncf.openvino.statistics.collectors import get_mean_statistic_collector


def mean_statistic_collector(
    channel_axis: int,
    inplace: bool,
    num_samples: Optional[int] = None,
    window_size: Optional[int] = None,
):
    return get_mean_statistic_collector(num_samples, channel_axis, window_size, inplace)


def get_statistic_points(model, graph, nodes, subset_size) -> StatisticPointsContainer:
    statistic_container = StatisticPointsContainer()
    OUTPUT_PORT_OF_NODE = 0

    # Collection of statistics after layers where biases will be corrected.
    for node in nodes:
        node_name = node.node_name
        channel_axis = node.metatype.output_channel_axis

        # For layers with weights, there is only one output port - 0.
        statistic_point = OVTargetPoint(
            TargetType.POST_LAYER_OPERATION, node_name, port_id=OUTPUT_PORT_OF_NODE
        )
        stat_collector = mean_statistic_collector(
            channel_axis=channel_axis, num_samples=subset_size, inplace=True
        )
        statistic_container.add_statistic_point(
            StatisticPoint(
                target_point=statistic_point, tensor_collector=stat_collector, algorithm="self._algorithm_key"
            )
        )

    return statistic_container


def compression_compensation(model, dataset, compression_algorithm, subset_size=128):
    # find last MatMul
    matmuls = [op for op in model.get_ordered_ops() if op.get_type_name() == "MatMul"]
    assert len(matmuls) != 0
    last_matmul = matmuls[-1]

    # insert identity Multiply and Add
    port_id = 0 # ????
    target_node_output = last_matmul.input_value(port_id)

    sz = target_node_output.partial_shape.get_dimension(2).max_length
    
    mul = opset.multiply(target_node_output, np.ones(sz, dtype=np.float32), name="NormalizerMul")
    add = opset.add(mul, np.zeros(sz, dtype=np.float32), name="NormalizerAdd")
    last_matmul.input(port_id).replace_source_output(add.output(0))

    graph = GraphConverter.create_nncf_graph(model)
    statistics_aggregator = OVStatisticsAggregator(dataset)
    
    node_keys = graph.get_all_node_keys()
    target_name = [val for val in node_keys if 'NormalizerMul' in val][0]
    target_mul = graph.get_node_by_key(target_name)

    statistic_points = get_statistic_points(model, graph, [target_mul], subset_size)
    statistics_aggregator.register_statistic_points(statistic_points)
    statistics_aggregator.collect_statistics(model, graph)
    
    # statistics_aggregator.statistic_points ???

    compressed_model = compression_algorithm(model)
    
    graph = GraphConverter.create_nncf_graph(compressed_model)
    statistics_aggregator = OVStatisticsAggregator(dataset)
    statistic_points = [mul]
    statistics_aggregator.register_statistic_points(statistic_points)
    statistics_aggregator.collect_statistics(compressed_model, graph)
    
    
    
    return compressed_model
