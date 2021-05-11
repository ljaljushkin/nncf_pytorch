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

from typing import List

from torch import nn
from torch.nn import Conv2d

from nncf.algo_selector import ZeroCompressionLoss
from nncf.api.compression import CompressionAlgorithmController
from nncf.api.compression import CompressionLevel
from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.api.compression import ModelType
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.utils.logger import logger as nncf_logger
from nncf.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.compression_method_api import PTCompressionAlgorithmController
from nncf.graph.transformations.commands import PTInsertionCommand
from nncf.graph.transformations.commands import PTTargetPoint
from nncf.graph.transformations.layout import PTTransformationLayout
from nncf.nncf_network import NNCFNetwork


class ElasticConv2dWidthOp(nn.Module):
    def __init__(self, max_in_channels, max_out_channels, scope):
        super().__init__()
        self.max_in_channels = max_in_channels
        self.scope = scope
        self.max_out_channels = max_out_channels
        self.active_out_channels = self.max_in_channels

    def set_active_out_channels(self, num_channels):
        nncf_logger.info('set active out channels={} for scope={}'.format(num_channels, self.scope))
        if 0 > num_channels > self.max_out_channels:
            raise ValueError(
                'invalid number of output channels to set. Should be within [{}, {}]'.format(0, self.max_in_channels))
        self.active_out_channels = num_channels

    def forward(self, weight, inputs):
        in_channels = inputs.size(1)
        return weight[:self.active_out_channels, :in_channels, :, :].contiguous()


class NASAlgoBuilder(PTCompressionAlgorithmBuilder):
    def build_controller(self, model: ModelType) -> CompressionAlgorithmController:
        return NASAlgoController(model, self.elastic_width_ops)

    def __init__(self, config, should_init: bool = True):
        super().__init__(config, should_init)
        self.elastic_width_ops = []

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        layout = PTTransformationLayout()
        commands = self._make_elastic_width(target_model)
        for command in commands:
            layout.register(command)
        return layout

    def _make_elastic_width(self, target_model: NNCFNetwork) -> List[PTInsertionCommand]:
        device = next(target_model.parameters()).device
        # TODO: copy paste from sparsity, probably other way to get modules
        candidates_for_progressive_shrinking = target_model.get_nncf_modules_by_module_names(
            self.compressed_nncf_module_names)
        insertion_commands = []
        for module_scope, module in candidates_for_progressive_shrinking.items():
            scope_str = str(module_scope)

            if not self._should_consider_scope(scope_str):
                nncf_logger.info("Ignored adding Elastic Width in scope: {}".format(scope_str))
                continue

            # TODO: probably should be handled more properly, like list of supported layers: Conv2d, Linear...
            if not isinstance(module, Conv2d):
                nncf_logger.info("Not adding Elastic Width for non Conv2d modules in scope: {}".format(scope_str))
                continue
            operation = self.create_elastic_width_operation_for_conv2d(module, scope_str)
            nncf_logger.info("Adding Elastic Width in scope: {}".format(scope_str))
            self.elastic_width_ops.append(operation)
            hook = operation.to(device)
            # TODO: UpdateWeightByUsingInputs
            insertion_commands.append(PTInsertionCommand(PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS,
                                                                       module_scope=module_scope),
                                                         hook))
        return insertion_commands

    @staticmethod
    def create_elastic_width_operation_for_conv2d(module, scope_str):
        # TODO: probably should be handled more properly, like list of supported layers: Conv2d, Linear...
        if not hasattr(module, 'in_channels') or not hasattr(module, 'out_channels'):
            raise RuntimeError('module should have in_channels and out_channels attributes')
        device = module.weight.device
        return ElasticConv2dWidthOp(module.in_channels, module.out_channels, scope_str).to(device)


class NASAlgoController(PTCompressionAlgorithmController):
    def __init__(self, target_model: NNCFNetwork, elastic_width_ops):
        super().__init__(target_model)
        self._loss = ZeroCompressionLoss(next(target_model.parameters()).device)
        self._scheduler = BaseCompressionScheduler()
        self.elastic_width_ops = elastic_width_ops

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    def compression_level(self) -> CompressionLevel:
        return CompressionLevel.FULL
