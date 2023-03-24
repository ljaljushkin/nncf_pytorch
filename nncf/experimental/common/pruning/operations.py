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

from collections import defaultdict
from typing import List
from typing import Optional
from typing import Type

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import GetItemLayerAttributes
from nncf.common.graph.layer_attributes import GroupNormLayerAttributes
from nncf.common.graph.layer_attributes import MultipleOutputLayerAttributes
from nncf.common.graph.layer_attributes import PermuteLayerAttributes
from nncf.common.graph.layer_attributes import TransposeLayerAttributes
from nncf.common.logging import nncf_logger
from nncf.common.pruning.tensor_processor import NNCFPruningBaseTensorProcessor
from nncf.common.pruning.utils import get_input_masks
from nncf.common.pruning.utils import identity_mask_propagation
from nncf.common.pruning.utils import is_grouped_conv
from nncf.common.pruning.utils import is_prunable_depthwise_conv
from nncf.common.tensor import NNCFTensor
from nncf.experimental.common.pruning.nodes_grouping import BlockGroup
from nncf.experimental.common.pruning.nodes_grouping import DimensionBlock
from nncf.experimental.common.pruning.nodes_grouping import MaskProducer
from nncf.experimental.common.pruning.nodes_grouping import PropagationMask


class BasePruningOp:
    """
    Determines meta operations which aggregate operations having common
    properties of interaction with pruning masks
    """

    subtypes = []
    additional_types = []

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        """
        Propagates the pruning mask through a node using pruning masks of all inputs and the current node (if any).

        :param node: The graph node to propagate mask through it
        :param graph: The model graph to prune
        :param tensor_processor: Interface with tensor processing methods
        """
        cls.mask_propagation_impl(node, graph, tensor_processor)
        cls.common_post_process(node, graph)

    @classmethod
    def mask_propagation_impl(cls, node: NNCFNode, graph: NNCFGraph,
                              tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        raise NotImplementedError

    @classmethod
    def common_post_process(cls, node: NNCFNode, graph: NNCFGraph):
        output_mask = node.data.get('output_mask', None)
        if output_mask is not None:
            num_output_nodes = len(graph.get_next_nodes(node))
            if num_output_nodes > 1:
                for groups in output_mask.dim_groups_map.values():
                    for group in groups:
                        for block in group.get_blocks():
                            block.add_open_branch(num_output_nodes - 1)

    @classmethod
    def get_all_op_aliases(cls) -> List[str]:
        """
        :return: list of all aliases of types in metatype
        """
        op_types = []
        for subtype in cls.subtypes:
            op_types.extend(subtype.get_all_aliases())
        op_types = list(set(op_types)) + cls.additional_types
        return op_types


class InputPruningOp(BasePruningOp):
    @classmethod
    def mask_propagation_impl(cls, node: NNCFNode, graph: NNCFGraph,
                              tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        node.data['output_mask'] = None


class OutputPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        return True

    @classmethod
    def mask_propagation_impl(cls, node: NNCFNode, graph: NNCFGraph,
                              tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        node.data['output_mask'] = None


class IdentityMaskForwardPruningOp(BasePruningOp):
    @classmethod
    def mask_propagation_impl(cls, node: NNCFNode, graph: NNCFGraph,
                              tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        identity_mask_propagation(node, graph)


class ConvolutionPruningOp(BasePruningOp):
    @classmethod
    def mask_propagation_impl(cls, node: NNCFNode, graph: NNCFGraph,
                              tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        input_masks = get_input_masks(node, graph)
        output_mask = node.data.get('output_mask', None)

        if is_grouped_conv(node):
            output_mask = None
            if is_prunable_depthwise_conv(node):
                output_mask = input_masks[0]

        node.data['output_mask'] = output_mask


class TransposeConvolutionPruningOp(BasePruningOp):
    @classmethod
    def mask_propagation_impl(cls, node: NNCFNode, graph: NNCFGraph,
                              tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        input_masks = get_input_masks(node, graph)
        output_mask = node.data.get('output_mask', None)

        if is_grouped_conv(node):
            output_mask = None
            if is_prunable_depthwise_conv(node):
                output_mask = input_masks[0]

        node.data['output_mask'] = output_mask


class LinearPruningOp(BasePruningOp):
    @classmethod
    def mask_propagation_impl(cls, node: NNCFNode, graph: NNCFGraph,
                              tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        input_masks = get_input_masks(node, graph)
        # length=3 in case of input, and apply_mask for weight and bias. E.g. input: [1,128,4] weight: [4,4] bias: [4]
        assert len(input_masks) in [1, 2, 3]
        input_masks = list(filter(lambda mask: mask is not None, input_masks))
        # single non-empty input
        if (len(input_masks) == 1 and input_masks[0] is not None) or (
                len(input_masks) > 1 and not all(input_masks[1:])):
            output_mask = node.data['output_mask']
            # Propagating batch dims
            input_tensors_shapes = [x.tensor_shape for x in graph.get_input_edges(node)]
            input_shape_len = len(input_tensors_shapes[0])
            for dim, groups in input_masks[0].dim_groups_map.items():
                if dim == input_shape_len - 1:
                    for group in groups:
                        group.close_branch()
                        cls.add_consumer_block(group, node)
                else:
                    output_mask.dim_groups_map[dim] = groups

        elif len(input_masks) == 2 and all(input_masks):
            input_tensors_shapes = [x.tensor_shape for x in graph.get_input_edges(node)]
            assert len(input_tensors_shapes[0]) == len(input_tensors_shapes[1])
            input_shape_len = len(input_tensors_shapes[0])
            # Join consumed masks
            left_dim_groups, right_dim_groups = [input_masks[i].dim_groups_map for i in range(2)]  # ignore the 3rd bias

            def _both_dim_blocks_exist(left_idx, right_idx):
                if left_idx in left_dim_groups or \
                        right_idx in right_dim_groups:
                    assert left_idx in left_dim_groups and \
                           right_idx in right_dim_groups
                    return True
                return False

            output_mask = PropagationMask()
            # Propagating batch dims
            for dim in range(input_shape_len - 2):
                if _both_dim_blocks_exist(dim, dim):
                    assert len(left_dim_groups[dim]) == 1 and len(
                        right_dim_groups[dim]) == 1, "multiple groups is not supported"
                    output_mask.dim_groups_map[dim] = [BlockGroup.join_groups(left_dim_groups[dim][0],
                                                                              right_dim_groups[dim][0])]
            # Propagating left rows / right cols
            for idx, dim in enumerate(range(input_shape_len - 2, input_shape_len)):
                if dim in input_masks[idx].dim_groups_map:
                    output_mask.dim_groups_map[dim] = input_masks[idx].dim_groups_map[dim]

            # Close branch
            if _both_dim_blocks_exist(input_shape_len - 1, input_shape_len - 2):
                left = left_dim_groups[input_shape_len - 1]
                right = right_dim_groups[input_shape_len - 2]
                assert len(left) == 1 and len(right) == 1, "multiple groups is not supported"
                group = BlockGroup.join_groups(left[0], right[0])
                group.close_branch()
                cls.add_consumer_block(group, node)
        else:
            output_mask = node.data.get('output_mask', None)

        node.data['output_mask'] = output_mask

    @classmethod
    def add_consumer_block(cls, group: BlockGroup, node):
        if node.layer_attributes is not None:
            first_block: DimensionBlock = group.get_blocks()[0]
            consumer_block = DimensionBlock(
                MaskProducer(node.node_id),
                size=first_block.size,
                offset=first_block.offset,
                pruning_dimension=1,
                closed_branches=1
            )
            group.add_block(consumer_block)


class BatchNormPruningOp(BasePruningOp):
    @classmethod
    def mask_propagation_impl(cls, node: NNCFNode, graph: NNCFGraph,
                              tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        identity_mask_propagation(node, graph)


class GroupNormPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        # For Instance Normalization
        return isinstance(node.layer_attributes, GroupNormLayerAttributes) \
               and node.layer_attributes.num_groups == node.layer_attributes.num_channels

    @classmethod
    def mask_propagation_impl(cls, node: NNCFNode, graph: NNCFGraph,
                              tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        if cls.accept_pruned_input(node):
            identity_mask_propagation(node, graph)
        else:
            node.data['output_mask'] = None


class ConcatPruningOp(BasePruningOp):
    @classmethod
    def generate_output_mask(cls, node: NNCFNode, graph: NNCFGraph,
                             tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> Optional[NNCFTensor]:
        """
        Generate output mask from input masks with all None replaced by identity masks.
        If all input masks is None return None.

        :param node: Node to determine it's sources.
        :param graph: NNCF graph to work with.
        :param tensor_processor: Interface with tensor processing methods.
        :return: Filled input masks.
        """
        input_edges = graph.get_input_edges(node)
        previous_nodes = [edge.from_node for edge in input_edges]
        input_masks = [input_node.data['output_mask'] for input_node in previous_nodes]

        not_empty_masks = [mask for mask in input_masks if mask is not None]  # type: List[NNCFTensor]
        if not not_empty_masks:
            return None


    @classmethod
    def mask_propagation_impl(cls, node: NNCFNode, graph: NNCFGraph,
                              tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        result_mask = cls.generate_output_mask(node, graph, tensor_processor)
        node.data['output_mask'] = result_mask


class ElementwisePruningOp(BasePruningOp):
    @classmethod
    def mask_propagation_impl(cls, node: NNCFNode, graph: NNCFGraph,
                              tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        input_masks = get_input_masks(node, graph)
        if not input_masks:
            input_masks = [None]

        output_mask = input_masks[0]  # elif all(m is None for m in input_masks):
        if len(input_masks) == 1:
            nncf_logger.warning(f"ElementWise with a single input is not properly supported. node_name={node.node_name}"
                                "The second input might be a constant without node in the graph. "
                                "The constant should be in the graph or in the node attributes."
                                "It's also should be pruned in accordance with an input mask")

        if any(m is None for m in input_masks):
            output_mask = None
            for m in input_masks:
                if m:
                    m.invalidate_groups()

        node.data['output_mask'] = output_mask


class GatherPruningOp(BasePruningOp):
    @classmethod
    def mask_propagation_impl(cls, node: NNCFNode, graph: NNCFGraph,
                              tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        input_masks = get_input_masks(node, graph)
        assert len(input_masks) == 1
        input_mask = input_masks[0]
        if input_mask is None:
            node.data['output_mask'] = None
            return

        def is_dim_removed_by_splitting() -> Optional[int]:
            """
            Determines whether the operations going from parent's op is equivalent to split
            Currently, limited to the case of simple __getitem__ with axis=0 and integer key rather than more general
            `gather` operation or __getitem__ with slice or tuple.
            q, k, v = qkv[0], qkv[1], qkv[2] (like in official SwinTransformer implementation)

            - look at all consumer of parent node
            - all of them should
                - be getitem
                - slice along the same axis
                - combine the whole input (the keys should contain all dimension along axis. size of first dim=3 => [0,1,2])
            :return : axis that is removed by split, or None otherwise
            """
            split_axis = None
            if isinstance(node.layer_attributes, GetItemLayerAttributes):
                input_edge = graph.get_input_edges(node)[0]
                input_shape = input_edge.tensor_shape
                parent_node = input_edge.from_node
                child_nodes = graph.get_next_nodes(parent_node)
                child_attributes = [cnode.layer_attributes for cnode in child_nodes]
                all_getitem = all(isinstance(ca, GetItemLayerAttributes) for ca in child_attributes)
                assert all_getitem, "currently supported only case with all  __getitem__ on branches"
                all_int_keys = all(isinstance(ca.key, int) for ca in child_attributes)
                # currently supported only case __getitem__ with single int, no slices
                if not all_int_keys:
                    return None
                all_keys = set(ca.key for ca in child_attributes)
                split_dim = input_shape[0]
                if all_keys == set(range(split_dim)):
                    split_axis = 0
            return split_axis

        removed_axis = is_dim_removed_by_splitting()
        if removed_axis is not None:
            output_mask = PropagationMask()
            for input_mask in input_masks:
                for dim, groups in input_mask.dim_groups_map.items():
                    if dim != removed_axis:
                        shifted_dim = dim - 1
                        output_mask.dim_groups_map[shifted_dim] = groups
                        # other groups propagated further
            node.data['output_mask'] = output_mask
        else:
            for m in input_masks:
                if m:
                    m.invalidate_groups()
            node.data['output_mask'] = None


class SplitPruningOp(BasePruningOp):
    @classmethod
    def mask_propagation_impl(cls, node: NNCFNode, graph: NNCFGraph,
                              tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        input_masks = get_input_masks(node, graph)
        if not input_masks:
            input_masks = [None]

        assert len(input_masks) == 1
        output_mask = input_masks[0]

        assert isinstance(node.layer_attributes, MultipleOutputLayerAttributes)
        chunk_axis = node.layer_attributes.axis
        chunks = node.layer_attributes.chunks
        print(node.layer_attributes)

        input_edge = graph.get_input_edges(node)[0]
        input_shape = input_edge.tensor_shape
        is_chunk_axis_removed = chunks == input_shape[chunk_axis]
        if is_chunk_axis_removed:
            output_mask = PropagationMask()
            for dim, groups in input_masks[0].dim_groups_map.items():
                if dim != chunk_axis:
                    output_mask.dim_groups_map[dim] = groups
                    # other groups propagated further
        else:
            raise NotImplementedError("symbolic mask propagation for split by prune dimension is not implemented")

        node.data['output_mask'] = output_mask


class ReshapePruningOp(BasePruningOp):
    @staticmethod
    def _is_flatten(node: NNCFNode) -> bool:
        return len(node.layer_attributes.output_shape) == 2

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        if node.layer_attributes is None:
            return False
        return True

    @staticmethod
    def _map_dims(node: NNCFNode) -> bool:
        def _check_dim_splitted(dim_from: int, dims_to: List[int],
                                dims_to_start_idx: int):
            idx = dims_to_start_idx
            accum = dims_to[idx]
            while accum < dim_from:
                idx += 1
                accum *= dims_to[idx]
            if accum > dim_from:
                return (False, idx)
            return (True, idx)

        def _map_dims_(dims_from: List[int], dims_to: List[int],
                       from_idx: int, to_idx: int, from_map, to_map):
            res, to_idx_next = _check_dim_splitted(dims_from[from_idx], dims_to, to_idx)
            if not res:
                return (res, to_idx_next)
            from_map[from_idx] = list(range(to_idx, to_idx_next + 1))
            for idx in range(to_idx, to_idx_next + 1):
                to_map[idx] = from_idx
            return (res, to_idx_next)

        input_shape = node.layer_attributes.input_shape
        output_shape = node.layer_attributes.output_shape

        inp_idx = 0
        out_idx = 0
        inp_map = {}
        out_map = {}

        mode = 'default'

        while (inp_idx < len(input_shape) and out_idx < len(output_shape)):
            if input_shape[inp_idx] == output_shape[out_idx]:
                inp_map[inp_idx] = out_idx
                out_map[out_idx] = inp_idx
            elif input_shape[inp_idx] > output_shape[out_idx]:
                res, out_idx = _map_dims_(input_shape, output_shape,
                                          inp_idx, out_idx, inp_map, out_map)
                if not res or mode == 'shrink':
                    return None
                mode = 'extend'
            else:
                res, inp_idx = _map_dims_(output_shape, input_shape, out_idx, inp_idx, out_map, inp_map)
                if not res or mode == 'extend':
                    return None
                mode = 'shrink'
            inp_idx += 1
            out_idx += 1
        return inp_map, out_map, mode

    @classmethod
    def _get_propagated_mask(cls, node: NNCFNode, graph: NNCFGraph):
        masks = get_input_masks(node, graph)
        assert len(masks) == 1
        mask = masks[0]
        if mask is None or not node.layer_attributes:
            return None

        map = cls._map_dims(node)
        if not map:
            return None

        inp_map, out_map, mode = map
        input_shape = node.layer_attributes.input_shape
        output_shape = node.layer_attributes.output_shape

        if mode == 'default':
            return mask

        output_mask = PropagationMask()
        if mode == 'extend':
            for dim, groups in mask.dim_groups_map.items():
                if len(groups) > 1:
                    raise NotImplementedError('Extend reshape for several groups is not supported yet')
                if not isinstance(inp_map[dim], list):
                    # pruning dimension is not affected, change pruning dimension only
                    shifted_dim = inp_map[dim]
                    output_mask.dim_groups_map[shifted_dim] = groups
                else:
                    mapped_input_shape = input_shape[dim]  # assume a single int by definition of extend
                    mapped_output_shape = [output_shape[x] for x in inp_map[dim]]
                    # check that it simply adds 1 to the shape (Unsqueeze)
                    if mapped_input_shape in mapped_output_shape:
                        index_in_mapped = mapped_output_shape.index(mapped_input_shape)
                        shifted_dim = inp_map[dim][index_in_mapped]
                        output_mask.dim_groups_map[shifted_dim] = groups
                    else:
                        shape_map = [mapped_input_shape, mapped_output_shape]
                        assert len(groups) == 1
                        group = groups[0]
                        new_groups = group.split_blocks_by_reshape(shape_map)
                        for new_group, in_dim in zip(new_groups, inp_map[dim]):
                            output_mask.dim_groups_map[in_dim] = [new_group]
            return output_mask

        if mode == 'shrink':
            grouping = defaultdict(list)
            for inp_idx, groups in mask.dim_groups_map.items():
                grouping[inp_map[inp_idx]].extend(groups)
            output_mask.dim_groups_map = dict(grouping)
            return output_mask

    @classmethod
    def mask_propagation_impl(cls, node: NNCFNode, graph: NNCFGraph,
                              tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        node.data['output_mask'] = cls._get_propagated_mask(node, graph)


class TransposePruningOp(BasePruningOp):
    @classmethod
    def mask_propagation_impl(cls, node: NNCFNode, graph: NNCFGraph,
                              tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        input_masks = get_input_masks(node, graph)
        assert len(input_masks) == 1
        input_mask = input_masks[0]
        if input_mask is None:
            node.data['output_mask'] = None
            return

        if isinstance(node.layer_attributes, TransposeLayerAttributes):
            input_tensors_shapes = [x.tensor_shape for x in graph.get_input_edges(node)]
            assert len(input_tensors_shapes) == 1
            new_order = list(range(len(input_tensors_shapes[0])))
            dim0 = node.layer_attributes.dim0
            dim1 = node.layer_attributes.dim1
            new_order[dim1], new_order[dim0] = new_order[dim0], new_order[dim1]
        elif isinstance(node.layer_attributes, PermuteLayerAttributes):
            new_order = node.layer_attributes.permutation

        idx_map = [(old_idx, new_idx) for new_idx, old_idx in enumerate(new_order) if
                   old_idx in input_mask.dim_groups_map]
        output_mask = PropagationMask({new_idx: input_mask.dim_groups_map[old_idx] for old_idx, new_idx in idx_map})

        node.data['output_mask'] = output_mask


class FlattenPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        if node.layer_attributes is not None:
            return True
        return False

    @classmethod
    def mask_propagation_impl(cls, node: NNCFNode, graph: NNCFGraph,
                              tensor_processor: Type[NNCFPruningBaseTensorProcessor]):
        output_mask = None
        input_masks = get_input_masks(node, graph)
        assert len(input_masks) == 1
        input_mask = input_masks[0]
        if input_mask is not None and node.layer_attributes is not None:
            # We assume all layers known by the mask propagation algo except
            # StopMaskForwardOp have input/output batch dim == 0.
            # Besides, since input_mask is not None thus no StopMaskForwardOp operations
            # was in the path from mask producer node to this node. As all
            # known nodes have input/output batch dim == 0 previous has too.
            flatten_channels = node.layer_attributes.output_shape[1]
            mask_len = input_mask.shape[0]
            assert flatten_channels % mask_len == 0
            output_mask = tensor_processor.repeat(input_mask, repeats=flatten_channels // mask_len)

        node.data['output_mask'] = output_mask


class StopMaskForwardPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        return False

    @classmethod
    def mask_propagation_impl(cls, node: NNCFNode, graph: NNCFGraph,
                              tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        input_masks = get_input_masks(node, graph)
        for m in input_masks:
            if m:
                m.invalidate_groups()
        node.data['output_mask'] = None