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

import torch.nn
from copy import deepcopy

from nncf.compression_method_api import CompressionLoss, CompressionScheduler, \
    CompressionAlgorithmController, CompressionLevel, CompressionAlgorithmBuilder
from nncf.hw_config import HWConfigType, HW_CONFIG_TYPE_TARGET_DEVICE_MAP
from nncf.nncf_network import NNCFNetwork
from nncf.pruning.base_algo import BasePruningAlgoController


class CompositeCompressionLoss(CompressionLoss):
    def __init__(self):
        super().__init__()
        self._child_losses = torch.nn.ModuleList()

    @property
    def child_losses(self):
        return self._child_losses

    def add(self, child_loss):
        self._child_losses.append(child_loss)

    def forward(self):
        result_loss = 0
        for loss in self._child_losses:
            result_loss += loss()
        return result_loss

    def statistics(self, quickly_collected_only=False):
        stats = {}
        for loss in self._child_losses:
            stats.update(loss.statistics())
        return stats


class CompositeCompressionScheduler(CompressionScheduler):
    def __init__(self):
        super().__init__()
        self._child_schedulers = []

    @property
    def child_schedulers(self):
        return self._child_schedulers

    def add(self, child_scheduler):
        self._child_schedulers.append(child_scheduler)

    def step(self, next_step=None):
        super().step(next_step)
        for scheduler in self._child_schedulers:
            scheduler.step(next_step)

    def epoch_step(self, next_epoch=None):
        super().epoch_step(next_epoch)
        for scheduler in self._child_schedulers:
            scheduler.epoch_step(next_epoch)

    def state_dict(self):
        result = {}
        for child_scheduler in self._child_schedulers:
            result.update(child_scheduler.state_dict())
        return result

    def load_state_dict(self, state_dict):
        for child_scheduler in self._child_schedulers:
            child_scheduler.load_state_dict(state_dict)


class CompositeCompressionAlgorithmBuilder(CompressionAlgorithmBuilder):
    def __init__(self, config: 'NNCFConfig', should_init: bool = True):
        from nncf import NNCFConfig
        from nncf.quantization.structs import QuantizerSetupType
        from nncf.model_creation import get_compression_algorithm

        super().__init__(config, should_init)
        self._child_builders = []  # type: List[CompressionAlgorithmBuilder]

        compression_config_json_section = config.get('compression', {})
        compression_config_json_section = deepcopy(compression_config_json_section)

        hw_config_type = None
        quantizer_setup_type_str = config.get("quantizer_setup_type", "propagation_based")
        quantizer_setup_type = QuantizerSetupType.from_str(quantizer_setup_type_str)
        if quantizer_setup_type == QuantizerSetupType.PROPAGATION_BASED:
            target_device = config.get("target_device", "ANY")
            if target_device != 'TRIAL':
                hw_config_type = HWConfigType.from_str(HW_CONFIG_TYPE_TARGET_DEVICE_MAP[target_device])

        if isinstance(compression_config_json_section, dict):
            compression_config = NNCFConfig(compression_config_json_section)
            compression_config.register_extra_structs(config.get_all_extra_structs_for_copy())
            compression_config["hw_config_type"] = hw_config_type
            compression_config['quantizer_setup_type'] = quantizer_setup_type
            self._child_builders = [
                get_compression_algorithm(compression_config)(compression_config, should_init=should_init), ]
        else:
            for algo_config in compression_config_json_section:
                algo_config = NNCFConfig(algo_config)
                algo_config.register_extra_structs(config.get_all_extra_structs_for_copy())
                algo_config["hw_config_type"] = hw_config_type
                algo_config['quantizer_setup_type'] = quantizer_setup_type
                self._child_builders.append(
                    get_compression_algorithm(algo_config)(algo_config, should_init=should_init))

    def __bool__(self):
        return bool(self.child_builders)

    @property
    def child_builders(self):
        return self._child_builders

    def apply_to(self, target_model: NNCFNetwork) -> NNCFNetwork:
        for ctrl in self._child_builders:
            target_model = ctrl.apply_to(target_model)
        return target_model


class CompositeCompressionAlgorithmController(CompressionAlgorithmController):
    def __init__(self, target_model: NNCFNetwork):
        super().__init__(target_model)
        self._child_ctrls = []  # type: List[CompressionAlgorithmController]
        self._loss = CompositeCompressionLoss()
        self._scheduler = CompositeCompressionScheduler()

    @property
    def child_ctrls(self):
        return self._child_ctrls

    def add(self, child_ctrl: CompressionAlgorithmController):
        # pylint: disable=protected-access
        assert child_ctrl._model is self._model, "Cannot create a composite controller " \
                                                 "from controllers belonging to different models!"
        self.child_ctrls.append(child_ctrl)
        self._loss.add(child_ctrl.loss)
        self._scheduler.add(child_ctrl.scheduler)
        self._model = child_ctrl._model

    def distributed(self):
        for ctrl in self.child_ctrls:
            ctrl.distributed()

    def statistics(self, quickly_collected_only=False):
        stats = {}
        for ctrl in self.child_ctrls:
            stats.update(ctrl.statistics())
        return stats

    def prepare_for_export(self):
        if len(self.child_ctrls) > 1 and any(isinstance(x, BasePruningAlgoController) for x in self.child_ctrls):
            # Waiting for the implementation of the related function in OpenVINO
            raise NotImplementedError("Exporting a model that was compressed by filter pruning algorithm "
                                      "in combination with another compression algorithm is not yet supporting")

        for child_ctrl in self.child_ctrls:
            child_ctrl.prepare_for_export()

    def apply_to(self, target_model: NNCFNetwork) -> NNCFNetwork:
        for ctrl in self.child_ctrls:
            target_model = ctrl.apply_to(target_model)
        return target_model

    def compression_level(self) -> CompressionLevel:
        if not self.child_ctrls:
            return CompressionLevel.NONE
        result = None
        for ctrl in self.child_ctrls:
            current_level = ctrl.compression_level()
            if not result:
                result = current_level
            else:
                result += current_level
        return result
