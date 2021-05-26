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

from typing import Dict
from typing import TypeVar

import torch.nn
from copy import deepcopy

from nncf.api.composite_compression import CompositeCompressionAlgorithmBuilder
from nncf.api.composite_compression import CompositeCompressionAlgorithmController
from nncf.api.composite_compression import CompositeCompressionLoss
from nncf.common.hardware.config import HWConfigType
from nncf.common.hardware.config import HW_CONFIG_TYPE_TARGET_DEVICE_MAP
from nncf.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.compression_method_api import PTCompressionAlgorithmController
from nncf.compression_method_api import PTCompressionLoss
from nncf.graph.transformations.layout import PTTransformationLayout
from nncf.nncf_network import NNCFNetwork
from nncf.nncf_network import PTModelTransformer
from nncf.pruning.base_algo import BasePruningAlgoController

ModelType = TypeVar('ModelType')


class PTCompositeCompressionLoss(CompositeCompressionLoss, PTCompressionLoss):
    def __init__(self):
        super().__init__()
        self._child_losses = torch.nn.ModuleList()

    @property
    def child_losses(self) -> torch.nn.ModuleList:
        return self._child_losses


class PTCompositeCompressionAlgorithmBuilder(
        CompositeCompressionAlgorithmBuilder, PTCompressionAlgorithmBuilder):
    _ALGO_NAME_VS_STATE_MAP_STATE_ATTR = 'algo_name_vs_state_map'
    CONFIG_STATE_ATTR = 'config'

    def __init__(self, config: 'NNCFConfig', should_init: bool = True,
                 should_init_per_builder: Dict[str, bool] = None):
        from nncf import NNCFConfig
        from nncf.model_creation import get_compression_algorithm

        super().__init__(config, should_init)
        global_should_init = should_init
        compression_config_json_section = config.get('compression', {})
        compression_config_json_section = deepcopy(compression_config_json_section)

        hw_config_type = None
        target_device = config.get("target_device", "ANY")
        global_compression_lr_multiplier = config.get("compression_lr_multiplier", None)
        if target_device != 'TRIAL':
            hw_config_type = HWConfigType.from_str(HW_CONFIG_TYPE_TARGET_DEVICE_MAP[target_device])

        if isinstance(compression_config_json_section, dict):
            compression_config = NNCFConfig(compression_config_json_section)
            compression_config.register_extra_structs(config.get_all_extra_structs_for_copy())
            compression_config["hw_config_type"] = hw_config_type
            if "compression_lr_multiplier" not in compression_config:
                compression_config["compression_lr_multiplier"] = global_compression_lr_multiplier
            compression_algorithm_class = get_compression_algorithm(compression_config)
            should_init = global_should_init
            if should_init_per_builder:
                should_init = should_init_per_builder[compression_algorithm_class]
            self._child_builders = [compression_algorithm_class(compression_config, should_init=should_init), ]
        else:
            for algo_config in compression_config_json_section:
                algo_config = NNCFConfig(algo_config)
                algo_config.register_extra_structs(config.get_all_extra_structs_for_copy())
                algo_config["hw_config_type"] = hw_config_type
                if "compression_lr_multiplier" not in algo_config:
                    algo_config["compression_lr_multiplier"] = global_compression_lr_multiplier
                compression_algorithm_class = get_compression_algorithm(algo_config)
                should_init = global_should_init
                if should_init_per_builder:
                    should_init = should_init_per_builder[compression_algorithm_class]
                self._child_builders.append(
                    compression_algorithm_class(algo_config, should_init=should_init))

    def __bool__(self):
        return bool(self.child_builders)

    def apply_to(self, target_model: NNCFNetwork) -> NNCFNetwork:
        layout = self.get_transformation_layout(target_model)
        transformer = PTModelTransformer(target_model, layout)
        transformed_model = transformer.transform()
        target_model.builder_state = self.get_state()
        return transformed_model

    def get_state(self) -> Dict[str, object]:
        """
        Returns a JSON-compatible dictionary containing a state of the object
        """
        algo_name_vs_state_map = {}
        for builder in self.child_builders:
            builder_state = builder.get_state()
            if builder_state:
                algo_name = builder.registered_name.replace('_', ' ')
                algo_name_vs_state_map[algo_name] = builder_state
        return {self.CONFIG_STATE_ATTR: dict(self.config.items()),
                self._ALGO_NAME_VS_STATE_MAP_STATE_ATTR:  algo_name_vs_state_map}

    def load_state(self, state: Dict[str, object]):
        """
        Initializes object from the state
        :param state: Output of `get_state()` method.
        """
        algo_name_vs_state_map = state[self._ALGO_NAME_VS_STATE_MAP_STATE_ATTR]
        if algo_name_vs_state_map:
            for builder in self.child_builders:
                algo_name = builder.registered_name.replace('_', ' ')
                if algo_name in algo_name_vs_state_map:
                    builder.load_state(algo_name_vs_state_map[algo_name])

    def build_controller(self, model: ModelType) -> PTCompressionAlgorithmController:
        """
        Builds `PTCompositeCompressionAlgorithmController` to handle the additional
        modules, parameters, and hooks inserted into the model to enable
        algorithm-specific compression.

        :param model: The model with additional modifications necessary to enable
         algorithm-specific compression during fine-tuning.
        :return: The instance of the `PTCompositeCompressionAlgorithmController`.
        """
        if len(self._child_builders) == 1:
            return self._child_builders[0].build_controller(model)
        composite_ctrl = PTCompositeCompressionAlgorithmController(model)
        for builder in self.child_builders:
            composite_ctrl.add(builder.build_controller(model))
        return composite_ctrl

    def get_transformation_layout(self, model: ModelType) -> PTTransformationLayout:
        """
        Computes necessary model transformations to enable algorithm-specific
        compression.

        :param model: The original uncompressed model.
        :return: The instance of the `PTTransformationLayout` class containing
            a list of algorithm-specific modifications.
        """
        transformations = PTTransformationLayout()
        for builder in self.child_builders:
            transformations.update(builder.get_transformation_layout(model))
        return transformations

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        pass  # Higher-level get_transformation_layout is overridden, no need to define this


class PTCompositeCompressionAlgorithmController(
    CompositeCompressionAlgorithmController, PTCompressionAlgorithmController):
    def __init__(self, target_model: ModelType):
        super().__init__(target_model)
        self._loss = PTCompositeCompressionLoss()

    def get_nncf_checkpoint(self) -> Dict[str, object]:
        """
        Returns entire compression state - a NNCF checkpoint, containing nncf_network state, builder state
         and composite controller state, containing the state of all children controllers.
         This checkpoint can be used to resume compression via nncf_checkpoint of create_compressed_model
         met
        :return: The entire compression state.
        """
        return self._nncf_checkpoint_helper(super().get_state())

    def load_state(self, state: Dict) -> None:
        """
        Calls `load_state()` method for all children.

        :param state: Output of `get_state()` method.
        """
        for ctrl in self.child_ctrls:
            algo_key = self._get_algo_key()
            if algo_key in state:
                ctrl.load_state(state[algo_key])

    def get_state(self) -> Dict[str, object]:
        """
        Returns the composite compression controller state. This state contains
        the state of all children.

        :return: The composite compression controller state.
        """
        algo_key_vs_state_map = {}
        for ctrl in self.child_ctrls:
            ctrl_state = ctrl.get_state()
            if ctrl_state:
                algo_key = ctrl.__class__.__name__
                algo_key_vs_state_map[algo_key] = ctrl_state
        return algo_key_vs_state_map

    def distributed(self):
        for ctrl in self.child_ctrls:
            ctrl.distributed()

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
