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
from typing import List, Callable

import torch
from texttable import Texttable
from torch import nn

from nncf.algo_selector import COMPRESSION_ALGORITHMS, ZeroCompressionLoss
from nncf.api.compression import CompressionLevel
from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.compression_method_api import PTCompressionAlgorithmController
from nncf.config import NNCFConfig
from nncf.graph.transformations.layout import PTTransformationLayout
from nncf.layers import NNCFConv2d
from nncf.module_operations import UpdateInputs
from nncf.common.utils.logger import logger as nncf_logger
from nncf.graph.transformations.commands import PTTargetPoint
from nncf.graph.transformations.commands import PTInsertionCommand
from nncf.nncf_network import NNCFNetwork

from nncf.nas.bootstrapNAS.layers import ElasticConv2DWidthOp, ElasticConv2DKernelOp

@COMPRESSION_ALGORITHMS.register('bootstrapNAS')
class BootstrapNASBuilder(PTCompressionAlgorithmBuilder):
    def __init__(self, config, should_init: bool = True):
        super().__init__(config, should_init)
        self._weight_dynamic = OrderedDict()
        self._processed_insertion_points = set() # type: Set[PTTargetPoint]
        # if self.should_init: # TODO

        self.elastic_kernel_ops = []
        self.elastic_width_ops = []

    def build_controller(self, target_model: NNCFNetwork) -> PTCompressionAlgorithmController:
        return BootstrapNASController(target_model, self.config, self.elastic_width_ops, self.elastic_kernel_ops)

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        layout = PTTransformationLayout()
        commands = self._elastic_width(target_model)
        for command in commands:
            layout.register(command)
        return layout

    def create_elastic_kernel_operation(self, module, scope):
        device = module.weight.device
        return ElasticConv2DKernelOp(module.in_channels, module.out_channels, scope).to(device)

    def create_elastic_width_operation(self, module, scope):
        device = module.weight.device
        return ElasticConv2DWidthOp(module.in_channels, module.out_channels, scope).to(device)

    def _elastic_kernel(self, target_model: NNCFNetwork):
        graph = target_model.get_original_graph()
        device = next(target_model.parameters()).device
        insertion_commands = []
        conv2d_nodes = graph.get_nodes_by_types(['conv2d'])
        for node in conv2d_nodes:
            module_scope = node.ia_op_exec_context.scope_in_model
            module = target_model.get_module_by_scope(module_scope)
            module_scope_str = str(module_scope)
            nncf_logger.info("Adding Dynamic Layer in scope: {}".format(module_scope_str))
            operation = self.create_elastic_kernel_operation(module, module_scope_str)
            self.elastic_kernel_ops.append(operation)
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
        graph = target_model.get_original_graph()
        device = next(target_model.parameters()).device
        insertion_commands = []
        conv2d_nodes = graph.get_nodes_by_types(['conv2d'])
        #
        for node in conv2d_nodes:
            module_scope = node.ia_op_exec_context.scope_in_model
            module = target_model.get_module_by_scope(module_scope)
            module_scope_str = str(module_scope)
            nncf_logger.info("Adding Dynamic Layer in scope: {}".format(module_scope_str))
            operation = self.create_elastic_width_operation(module, module_scope_str)
            self.elastic_width_ops.append(operation)
            hook = operation.to(device)
            insertion_commands.append(
                PTInsertionCommand(
                    PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS,
                    module_scope=module_scope),
                    hook
                )
            )
        return insertion_commands

class BootstrapNASController(PTCompressionAlgorithmController):
    def __init__(self, target_model: NNCFNetwork, config, elastic_width_ops, elastic_kernel_ops): # elastic_width_ops might be able to be extracted from config.
        super().__init__(target_model)
        self._loss = ZeroCompressionLoss(next(target_model.parameters()).device)
        self._scheduler = BaseCompressionScheduler()
        self.elastic_width_ops = elastic_width_ops
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