"""
 Copyright (c) 2019-2020 Intel Corporation
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
from collections import OrderedDict
from copy import deepcopy

from nncf.algo_selector import COMPRESSION_ALGORITHMS
from nncf.algo_selector import ZeroCompressionLoss
from nncf.api.compression import CompressionLevel
from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.utils.logger import logger as nncf_logger
from nncf.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.compression_method_api import PTCompressionAlgorithmController
from nncf.graph.graph import NNCFNodeExpression as N
from nncf.graph.patterns import BN
from nncf.graph.transformations.commands import PTInsertionCommand
from nncf.graph.transformations.commands import PTTargetPoint
from nncf.graph.transformations.layout import PTTransformationLayout
from nncf.module_operations import UpdatePadding
from nncf.nas.bootstrapNAS.layers import ElasticBatchNormOp
from nncf.nas.bootstrapNAS.layers import ElasticConv2DKernelOp
from nncf.nas.bootstrapNAS.layers import ElasticConv2DWidthOp
from nncf.nas.bootstrapNAS.layers import ElasticKernelPaddingAdjustment
from nncf.nncf_network import NNCFNetwork


@COMPRESSION_ALGORITHMS.register('bootstrapNAS')
class BootstrapNASBuilder(PTCompressionAlgorithmBuilder):
    def __init__(self, config, should_init: bool = True, is_elastic_kernel=False, is_elastic_width=False):
        super().__init__(config, should_init)
        self._weight_dynamic = OrderedDict()
        self._processed_insertion_points = set()  # type: Set[PTTargetPoint]
        self.is_elastic_kernel = is_elastic_kernel
        self.is_elastic_width = is_elastic_width

        self.elastic_kernel_ops = []
        self.scope_vs_elastic_width_op_map = {}

    def build_controller(self, target_model: NNCFNetwork) -> PTCompressionAlgorithmController:
        return BootstrapNASController(target_model, self.config, self.scope_vs_elastic_width_op_map,
                                      self.elastic_kernel_ops)

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        layout = PTTransformationLayout()
        commands = []
        if self.is_elastic_kernel:
            commands.extend(self._elastic_kernel(target_model))
        if self.is_elastic_width:
            commands.extend(self._elastic_width(target_model))
        for command in commands:
            layout.register(command)
        return layout

    @staticmethod
    def create_elastic_kernel_operation(module, scope):
        return ElasticConv2DKernelOp(module.kernel_size[0], scope)

    @staticmethod
    def create_elastic_bn_operation(module, scope):
        return ElasticBatchNormOp(module.out_channels, scope)

    @staticmethod
    def create_elastic_width_operation(module, scope):
        return ElasticConv2DWidthOp(module.in_channels, module.out_channels, scope)

    def _elastic_kernel(self, target_model: NNCFNetwork):
        graph = target_model.get_original_graph()
        device = next(target_model.parameters()).device
        insertion_commands = []
        pad_commands = []
        conv2d_nodes = graph.get_nodes_by_types(['conv2d'])
        conv2d_elastic_ids = []
        for node in conv2d_nodes:
            module_scope = node.ia_op_exec_context.scope_in_model
            module = target_model.get_module_by_scope(module_scope)
            module_scope_str = str(module_scope)
            if module.kernel_size[0] <= 3:
                continue
            nncf_logger.info("Adding Dynamic Conv2D Layer in scope: {}".format(module_scope_str))
            operation = self.create_elastic_kernel_operation(module, module_scope)
            hook = operation.to(device)
            insertion_commands.append(
                PTInsertionCommand(
                    PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS,
                                  module_scope=module_scope),
                    hook
                )
            )
            # Padding
            ap = ElasticKernelPaddingAdjustment(operation)
            pad_op = UpdatePadding(ap).to(device)
            insertion_point = PTTargetPoint(target_type=TargetType.PRE_LAYER_OPERATION,
                                            module_scope=module_scope)
            nncf_logger.warning('Padding will be adjusted for {}'.format(module_scope_str))
            pad_commands.append(PTInsertionCommand(insertion_point, pad_op, TransformationPriority.DEFAULT_PRIORITY))
            self.elastic_kernel_ops.append(operation)
            conv2d_elastic_ids.append(node.node_id + 1)
        if pad_commands:
            insertion_commands += pad_commands
        # BatchNorm
        bn_nodes = graph.get_nodes_by_types(['batch_norm'])
        for node in bn_nodes:
            if node.node_id in conv2d_elastic_ids:
                module_scope = node.ia_op_exec_context.scope_in_model
                module = target_model.get_module_by_scope(module_scope)
                module_scope_str = str(module_scope)
                nncf_logger.info("Adding Dynamic BatchNorm Layer in scope: {}".format(module_scope_str))
                operation = self.create_elastic_bn_operation(module, module_scope)
                hook = operation.to(device)
                insertion_commands.append(
                    PTInsertionCommand(
                        PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS,
                                      module_scope=module_scope),
                        hook
                    )
                )

        return insertion_commands

    def _elastic_width(self, target_model: NNCFNetwork):
        insertion_commands = []

        nncf_graph = target_model.get_original_graph()
        conv_bn_pattern = N('conv2d') + BN
        nx_graph = deepcopy(nncf_graph.get_nx_graph_copy())
        from nncf.graph.graph_matching import search_all
        matches = search_all(nx_graph, conv_bn_pattern)
        conv2d_bn_node_pairs = []
        for match in matches:
            input_node_key = match[0]
            output_node_key = match[-1]
            conv_node = nncf_graph.get_node_by_key(input_node_key)
            bn_node = nncf_graph.get_node_by_key(output_node_key)
            conv2d_bn_node_pairs.append((conv_node, bn_node))

        for conv_node, bn_node in conv2d_bn_node_pairs:
            conv_module_scope = conv_node.ia_op_exec_context.scope_in_model
            conv_module = target_model.get_module_by_scope(conv_module_scope)
            nncf_logger.info("Adding Elastic Width Op for Conv in scope: {}".format(str(conv_module_scope)))
            elastic_conv_width_op = self.create_elastic_width_operation(conv_module, conv_module_scope)
            insertion_commands.append(
                PTInsertionCommand(
                    PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS,
                                  module_scope=conv_module_scope),
                    elastic_conv_width_op
                )
            )

            bn_module_scope = bn_node.ia_op_exec_context.scope_in_model
            nncf_logger.info("Adding Elastic Width Op for BN in scope: {}".format(str(bn_module_scope)))

            elastic_bn_width_op = self.create_elastic_bn_operation(conv_module, bn_module_scope)
            insertion_commands.append(
                PTInsertionCommand(
                    PTTargetPoint(TargetType.OPERATION_WITH_BN_PARAMS,
                                  module_scope=bn_module_scope),
                    elastic_bn_width_op
                )
            )
            self.scope_vs_elastic_width_op_map[str(conv_module_scope)] = elastic_conv_width_op
        return insertion_commands


class BootstrapNASController(PTCompressionAlgorithmController):
    def __init__(self, target_model: NNCFNetwork, config, scope_vs_elastic_width_op_map,
                 elastic_kernel_ops):
        super().__init__(target_model)
        self._loss = ZeroCompressionLoss(next(target_model.parameters()).device)
        self._scheduler = BaseCompressionScheduler()
        self.scope_vs_elastic_width_op_map = scope_vs_elastic_width_op_map
        self.elastic_kernel_ops = elastic_kernel_ops

        from nncf.utils import is_main_process
        if is_main_process():
            print("Created BootstrapNAS controller")

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    def compression_level(self) -> CompressionLevel:
        return CompressionLevel.FULL

    def progressive_shrinking(self, stage, phase):
        if stage == 'elastic_kernel':
            pass
        # 1. Check stage and phase. More info needed. Config.

        # 2. sample random subnetwork based on stage and base

        # 3. train / finetune network stage / phase
        #   train
        #       train one epoch
        #       validate
