"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from dataclasses import dataclass
from functools import reduce
import operator
from pathlib import Path
from typing import (
    Any,
    Optional,
    Dict,
    List,
    Set
)

from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.experimental.common.graph.netron import save_for_netron
from nncf.experimental.common.pruning.propagation_data import (
    ConsumerInfo,
    MaskProducer,
    PropagationBlock,
    PropagationGroup,
    PropagationMask,
)
from nncf.experimental.common.pruning.block_hierarchy import BlockHierarchy


@dataclass
class PruningBlock:
    """
    Final and minimal representation of PropagationBlock after mask propagation.
    By analogy, it defines how much and which particular channels are supposed to be pruned for the node
    when a single element of pruning mask is 0.
    We assume that pruning mask is a vector with 1's and 0's. 1 retains the corresponding channel in weights,
    0 prunes it.
    """
    size: int
    offset: int
    producer_id: int
    pruning_dimension: int

    @classmethod
    def from_propagation_block(cls, block: PropagationBlock) -> 'PruningBlock':
        """
        Creates an object by taking all necessary information from PropagationBlock.
        """
        return cls(block.size, block.offset, block._producer.node_id, block.pruning_dimension)

    def __str__(self) -> str:
        return f'S{self.size}_O{self.offset}_PID{self.producer_id}'

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: 'PruningBlock') -> bool:
        return str(self) == str(other)

@dataclass
class PruningGroup:
    """
    Group of pruning blocks that is obtained after propagation.
    """
    # TODO: group should have:
    #   1) pruning block that is common for producers and consumers??? always True?
    #   2) producer ids
    #   3) consumer ids
    #   4) symbolic channel mapping
    # producer_info: Set[ProducerInfo]
    # consumer_info: Set[ConsumerInfo] # node_id & pruning dimension...
    # block: PruningBlock

    producers: Set[PruningBlock]
    consumers: Set[PruningBlock]


    def __eq__(self, other: 'PruningGroup'):
        return self.producers == other.producers and self.consumers == other.consumers

    @classmethod
    def from_propagation_group(cls, group: PropagationGroup) -> 'PruningGroup':
        """
        Creates an object by taking all necessary information from PropagationGroup.
        """
        producers = {PruningBlock.from_propagation_block(block) for block in group.get_blocks()}
        first_producer = next(iter(producers))
        size = first_producer.size
        offset = first_producer.offset
        assert all(b.size == size for b in producers) == 1 and all(b.offset == offset for b in producers)
        # TODO: consumer should not be a pruning block, as well as producer.
        consumers = {
            PruningBlock(size, offset, c.node_id, c.pruning_dimension) for c in group.get_consumers()
        }
        return cls(producers, consumers)


def get_pruning_groups(graph: NNCFGraph,
                       pruning_operations_metatypes: PruningOperationsMetatypeRegistry,
                       prune_operations_types: List[str],
                       dump_dir: Optional[Path] = None) -> List[PruningGroup]:
    """
    Determines how nodes of the given types should be pruned: which nodes should be pruned together, along which
    dimension, how many sequent channels with which offset. It's done by initializing PropagationMask's on the
    operations with prunable parameters (producers of pruning) and propagating them through the execution graph.

    :param graph: nncf graph to initialize and propagate masks.
    :param pruning_operations_metatypes: registry with operation metatypes pruning algorithm is aware of, i.e.
        metatypes describing operations with common pruning mask application and propagation properties, e.g.
        IdentityMaskForwardOps unifies operations that propagate pruning masks as is (relu, swish etc.), whereas
        Convolution unifies different convolution operations (conv1d, conv2d, conv3d) which accepts some input masks and
        provide some output masks.
    :param prune_operations_types: types of operations with prunable parameters.
    :param dump_dir: path to the directory for dumping debug files.
    :return: list of groups with parameters of pruning.
    """
    # 1. Initialize masks for producing nodes
    all_nodes_to_prune = graph.get_nodes_by_types(prune_operations_types)  # type: List[NNCFNode]
    roots = {}
    for node in all_nodes_to_prune:
        assert isinstance(node.layer_attributes, (LinearLayerAttributes, ConvolutionLayerAttributes))
        pruning_dim = node.layer_attributes.get_target_dim_for_compression()
        output_tensors_shapes = [x.tensor_shape for x in graph.get_output_edges(node)]
        assert not len(set(output_tensors_shapes)) > 1, node.node_name
        output_tensors_shape = output_tensors_shapes[0]
        # TODO: (107663) generalize by introducing get_output_dim_affected_by_compression for layer_attributes
        target_output_dim_for_compression = len(output_tensors_shape) - 1
        root_group = PropagationGroup(
            blocks=[
                PropagationBlock(
                    producer=MaskProducer(node.node_id),
                    pruning_dimension=pruning_dim
                )
            ]
        )
        mask = PropagationMask(
            dim_groups_map={
                target_output_dim_for_compression: [root_group]
            }
        )
        roots[node.node_id] = root_group
        node.data['output_mask'] = mask

    def get_attributes_fn(node: NNCFNode) -> Dict[str, Any]:
        result = {'metatype': str(node.metatype.name), 'node_id': str(node.node_id)}
        if node.layer_attributes:
            result.update(map(lambda pair: (pair[0], str(pair[1])), node.layer_attributes.__dict__.items()))
        if 'output_mask' in node.data:
            output_mask = node.data['output_mask']
            if output_mask:
                result['output_mask'] = str(output_mask)
        return result

    try:
        # 2. Propagate masks
        MaskPropagationAlgorithm(graph, pruning_operations_metatypes).mask_propagation()
    finally:
        block_hierarchy = BlockHierarchy(list(roots.values()))
        if dump_dir is not None:
            block_hierarchy.visualize_graph(dump_dir / 'latest_block_group_hierarchy.dot')
            save_for_netron(graph, str(dump_dir / 'latest_propagated_graph.xml'), get_attributes_fn=get_attributes_fn)

    propagation_groups = block_hierarchy.get_groups_on_leaves()
    not_invalid_groups = filter(lambda group: not group.is_invalid, propagation_groups)
    return [PruningGroup.from_propagation_group(pg) for pg in not_invalid_groups]


def select_largest_groups(pruning_groups: List[PruningGroup]) -> List[PruningGroup]:
    """
    Selects largest pruning groups with larger number of pruning blocks.
    """
    finished_producers = set()
    sorted_groups = sorted(pruning_groups, key=lambda group: len(group.producers), reverse=True)
    result = []
    for group in sorted_groups:
        producers = {block.producer_id for block in group.producers}
        if not producers.intersection(finished_producers):
            finished_producers.update(producers)
            result.append(group)
    return result