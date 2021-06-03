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

from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional
from typing import TypeVar

import torch.nn

from nncf.api.compression import CompressionSetup
from nncf.common.composite_compression import CompositeCompressionAlgorithmBuilder
from nncf.common.composite_compression import CompositeCompressionAlgorithmController
from nncf.common.composite_compression import CompositeCompressionLoss
from nncf.common.hardware.config import HWConfigType
from nncf.common.hardware.config import HW_CONFIG_TYPE_TARGET_DEVICE_MAP
from nncf.torch.algo_selector import COMPRESSION_ALGORITHMS
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.compression_method_api import PTCompressionLoss
from nncf.torch.compression_method_api import PTCompressionState
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.nncf_network import PTModelTransformer
from nncf.torch.pruning.base_algo import BasePruningAlgoController

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
    def __init__(self, config: 'NNCFConfig', should_init: bool = True,
                 compression_setups: Optional[List[CompressionSetup]] = None):
        from nncf import NNCFConfig
        from nncf.torch.model_creation import get_compression_algorithm
        saved_builder_classes = []
        if compression_setups is not None:
            saved_builder_classes = map(lambda x: COMPRESSION_ALGORITHMS.get(x.name), compression_setups)

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
            if compression_algorithm_class not in saved_builder_classes:
                should_init = True
            self._child_builders = [compression_algorithm_class(compression_config, should_init=should_init,
                                                                compression_setups=compression_setups), ]
        else:
            for algo_config in compression_config_json_section:
                algo_config = NNCFConfig(algo_config)
                algo_config.register_extra_structs(config.get_all_extra_structs_for_copy())
                algo_config["hw_config_type"] = hw_config_type
                if "compression_lr_multiplier" not in algo_config:
                    algo_config["compression_lr_multiplier"] = global_compression_lr_multiplier
                compression_algorithm_class = get_compression_algorithm(algo_config)
                should_init = global_should_init
                if compression_algorithm_class not in saved_builder_classes:
                    should_init = True
                self._child_builders.append(compression_algorithm_class(algo_config, should_init=should_init,
                                                                        compression_setups=compression_setups))

    def __bool__(self):
        return bool(self.child_builders)

    def apply_to(self, target_model: NNCFNetwork) -> NNCFNetwork:
        layout = self.get_transformation_layout(target_model)
        transformer = PTModelTransformer(target_model, layout)
        transformed_model = transformer.transform()
        return transformed_model

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

    def get_compression_state(self) -> Dict:
        """
        Returns PT-specific representatino of entire compression state, containing nncf_network state
        (with state of the builder) and composite controller state, containing the state of all children controllers.
        This checkpoint can be used to resume compression via compression_state of create_compressed_model
        :return: The entire compression state.
        """
        model_state = self.model.state_dict()
        setups = [child_ctrl.get_compression_setup() for child_ctrl in self._child_ctrls]
        return PTCompressionState(setups, model_state).get_state()

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
