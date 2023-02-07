from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Set

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm


class MaskProducer:
    def __init__(self, id_) -> None:
        self.id = id_


class DimensionBlock:
    def __init__(self,
                 producer: MaskProducer,
                 size: int = 1, offset: int = 0,
                 pruning_dimension: int = 0,
                 opened_branches: int = 0, closed_branches: int = 0) -> None:
        self.size = size
        self.offset = offset
        self.pruning_dimension = pruning_dimension
        self._producer = producer
        self._opened_branches = opened_branches
        self._closed_branches = closed_branches
        self._group = None
        self._is_invalid = False

    def get_state(self):
        return f"S:{self.size}__O:{self.offset}__ID:{self._producer.id}",

    def split_by_reshape(self, shape_map) -> List['DimensionBlock']:
        # TODO: make it common !!!
        if len(shape_map[1]) == 1:
            raise RuntimeError
        if len(shape_map[1]) > 2:
            raise NotImplementedError

        a = DimensionBlock(size=shape_map[1][-1], offset=0,
                           pruning_dimension=self.pruning_dimension,
                           producer=self._producer,
                           opened_branches=self._opened_branches,
                           closed_branches=self._closed_branches)
        b = DimensionBlock(size=1, offset=shape_map[1][-1],
                           pruning_dimension=self.pruning_dimension,
                           producer=self._producer,
                           opened_branches=self._opened_branches,
                           closed_branches=self._closed_branches)
        return [a, b]

    def open_branch(self):
        self._opened_branches += 1

    def close_branch(self):
        self._closed_branches += 1

    def set_group(self, group):
        self._group = group


class BlockGroup:
    def __init__(self, blocks: List[DimensionBlock]) -> None:
        self._blocks = blocks
        for block in blocks:
            block.set_group(self)  # TODO: circle dependency, bad smell
        self._childs: List['BlockGroup'] = []
        self.is_invalid = False

    def get_state(self):
        return list(map(lambda x: x.get_state(), self._blocks))

    def invalidate(self):
        for block in self._blocks:
            block.is_invalid = True
        # TODO(nlyalyus): are you sure that all should be affected?
        for child in self._childs:
            child.invalidate()

    def get_actual_groups(self) -> List[List[DimensionBlock]]:
        if not self._childs:
            return [self._blocks]
        retval = []
        for child in self._childs:
            groups = child.get_actual_groups()
            retval.append(groups[0] if len(groups) == 1 else groups)
        return retval

    def has_childs(self):
        return bool(self._childs)

    def add_childs(self, childs: List['BlockGroup']):
        self._childs.extend(childs)

    def add_block(self, block: DimensionBlock):
        self._blocks.append(block)
        block.set_group(self)

    def split_blocks_by_reshape(self, shape_map):
        if self._childs:
            raise NotImplementedError('Splitting BlockGroup with childs isn\'t implemented yet')

        new_blocks: List[List[DimensionBlock]] = []
        for block in self._blocks:
            new_blocks.append(block.split_by_reshape(shape_map))
        self._childs = []
        for group in zip(*new_blocks):
            self._childs.append(BlockGroup(blocks=list(group)))
        return self._childs.copy()

    # TODO: work on open branches
    def close_branch(self):
        for block in self._blocks:
            block.close_branch()

    def get_blocks(self):
        return self._blocks.copy()

    @staticmethod
    def join_groups(*args) -> 'BlockGroup':
        for groups in args:
            assert isinstance(groups, BlockGroup) or isinstance(groups, list), \
                f'Couldn\'t join args {args}, all elements should be BlockGroup instances'

        retval = BlockGroup([])
        for groups in args:
            for group in groups:
                group.add_childs([retval])
                for block in group.get_blocks():
                    retval._blocks.append(block)
        return retval


class PropagationMask:
    def __init__(self,
                 dim_groups_map: Dict[int, List[BlockGroup]] = None):
        self.dim_groups_map = dim_groups_map if dim_groups_map is not None else {}

    def invalidate_groups(self):
        for groups in self.dim_groups_map.values():
            for group in groups:
                group.invalidate()

    # TODO: is propagation mask invalid if any internal group is invalid? is this method needed?
    # def is_invalid(self):
    #     return any(group.is_invalid for groups in self.dim_groups_map.values() for group in groups)

    def get_state(self):
        result = {}
        for dim, groups in self.dim_groups_map.items():
            groups_state = [group.get_state() for group in groups]
            result[dim] = groups_state
        return result


@dataclass
class MinimalDimensionBlock:
    size: int
    offset: int
    producer_id: int
    pruning_dimension: int

    # TODO: output shape dimension? needed for propagation for only?
    @classmethod
    def from_dimension_block(cls, dim_block: DimensionBlock):
        return cls(dim_block.size, dim_block.offset, dim_block._producer.id, dim_block.pruning_dimension)

    def __str__(self):
        return f'S{self.size}_O{self.offset}_PID{self.producer_id}'

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other: 'MinimalDimensionBlock'):
        return str(self) == str(other)


@dataclass
class PruningNodeGroup:
    dim_blocks: Set[MinimalDimensionBlock]
    # -producing_nodes: Set[MinimalDimensionBlock]
    # adjusted_nodes: List[int] = []
    # consumer_nodes_id: List[int] = None

    def __eq__(self, other: 'PruningNodeGroup'):
        return self.dim_blocks == other.dim_blocks


def get_pruning_groups(graph: NNCFGraph,
                       pruning_operations_metatypes,
                       prune_operations_types) -> List[PruningNodeGroup]:
    # 1. Initialize masks for producing nodes
    # TODO: clarify that all possibly pruned nodes will be collected here
    all_nodes_to_prune = graph.get_nodes_by_types(prune_operations_types)  # type: List[NNCFNode]
    roots = {}
    for node in all_nodes_to_prune:
        assert isinstance(node.layer_attributes, (LinearLayerAttributes, ConvolutionLayerAttributes))
        pruning_dim = node.layer_attributes.get_target_dim_for_compression()
        root_group = BlockGroup([DimensionBlock(MaskProducer(node.node_id), pruning_dimension=pruning_dim)])
        roots[node.node_id] = root_group

        output_tensors_shapes = [x.tensor_shape for x in graph.get_output_edges(node)]
        # TODO: SWIN has linear lauer with 2 identical outputs
        assert len(output_tensors_shapes) == 1 or len(set(output_tensors_shapes)) <= 1, node.node_name
        output_tensors_shape = output_tensors_shapes[0]
        # TODO: make dimension map common here
        #  Usually it's module.target_weight_dim_for_compression [Torch] or get_filter_axis(layer, weight_attr) [TF]
        #  Ideally, node should know this. Is it part of layer attributes??
        #  NO: it's not weight dim for compression, it's output shape dimension that is affected by pruning
        target_output_dim_for_compression = len(output_tensors_shape) - 1
        mask = PropagationMask({target_output_dim_for_compression: [root_group]})
        node.data['output_mask'] = mask

    # 2. Propagate masks
    MaskPropagationAlgorithm(graph, pruning_operations_metatypes).mask_propagation()

    # 3. Collect groups from producers
    blocks_map: Dict[int, List[List[DimensionBlock]]] = {}
    for id, group in roots.items():
        # TODO: need to filter groups that didn't reach consumers
        blocks_map[id] = group.get_actual_groups()

    # Filter non closing and duplicated groups
    pruning_groups = []  # type: List[PruningNodeGroup]
    finished_producers = []
    for producer_id, groups in blocks_map.items():
        # TODO: choose block based on other strategies (blocks that leads to the biggest sparsity rate or ???)
        # TODO: iterate and choose first valid and not finished
        group = groups[0]
        # TODO: verify that all leaves are closed, did_reach_consumer: boolean
        if all(block._closed_branches == 1 for block in group):
            for block in group:
                assert not block._is_invalid, 'invalid groups are not handled'
            min_group = set(map(MinimalDimensionBlock.from_dimension_block, group))
            all_not_finished = all(g.producer_id not in finished_producers for g in min_group)
            candidate_group = PruningNodeGroup(min_group)
            if candidate_group not in pruning_groups and all_not_finished:
                pruning_groups.append(candidate_group)
                finished_producers.extend(g.producer_id for g in min_group)

    return pruning_groups
