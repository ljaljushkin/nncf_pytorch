#
#  Copyright (c) 2019-2020 Intel Corporation
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
@package docstring
This package defines the API for the NNCF compression methods so that the user could
extend the existing algorithms.
"""
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar

import numpy
import torch
from torch import nn

from nncf.api.compression import CompressionAlgorithmController
from nncf.api.compression import CompressionSetup
from nncf.api.compression import CompressionState
from nncf.common.utils.registry import Registry
from nncf.config import NNCFConfig
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.initialization import DataLoaderBNAdaptationRunner
from nncf.torch.layers import NNCF_MODULES_DICT, NNCF_WRAPPED_USER_MODULES_DICT
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.compression import BaseCompressionAlgorithmController
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.nncf_network import PTModelTransformer
from nncf.torch.structures import BNAdaptationInitArgs
from nncf.torch.utils import should_consider_scope
from nncf.api.compression import CompressionAlgorithmBuilder
from nncf.api.compression import CompressionLoss

ModelType = TypeVar('ModelType')

DOMAIN_CUSTOM_OPS_NAME = "org.openvinotoolkit"


class PTCompressionLoss(nn.Module, CompressionLoss):
    """
    Used to calculate additional loss to be added to the base loss during the
    training process. It uses the model graph to measure variables and activations
    values of the layers during the loss construction. For example, the $L_0$-based
    sparsity algorithm calculates the number of non-zero weights in convolutional
    and fully-connected layers to construct the loss function.
    """

    def calculate(self) -> torch.Tensor:
        """
        Calculates the compression loss value.

        :return: The compression loss value.
        """
        return torch.zeros([])

    def forward(self) -> torch.Tensor:
        """
        Overriding  forward function of the base nn.Module class

        :return: The compression loss value.
        """
        return self.calculate()

    def load_state(self, state: Dict[str, object]) -> None:
        """
        Loads the compression loss state.

        :param state: Output of `get_state()` method.
        """

    def get_state(self) -> None:
        """
        Returns the compression loss state.

        :return: The compression loss state.
        """


class PTCompressionState(CompressionState):
    """
    Entire compression state, containing state of `NNCFNetwork`, and builder and controller states.
    This state can be used to resume compression via `compression_state` parameter of `create_compressed_model`
    """
    MODEL_STATE_ATTR = 'nncf_model_state'

    def __init__(self):
        super().__init__()
        self._model_state = None  # type: Optional[Dict[str, torch.Tensor]]

    @property
    def model_state(self) -> Dict[str, torch.Tensor]:
        return self._model_state

    def get_state(self) -> Dict:
        """
        :return: PT-specific representation of the object for saving to the checkpoint
        """
        state = super().get_state()
        state[self.MODEL_STATE_ATTR] = self._model_state
        return state

    def load_state(self, state: Dict):
        """
        Loads state to the object
        :param state: Output of `get_state()` method.
        """
        super().load_state(state)
        self._model_state = state[self.MODEL_STATE_ATTR]


class PTCompressionAlgorithmController(BaseCompressionAlgorithmController):
    """Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model in order to enable algorithm-specific compression.
    Hosts entities that are to be used during the training process, such as compression scheduler and
    compression loss."""

    def __init__(self, target_model: ModelType):
        super().__init__(target_model)
        self._builder_state_with_name = None

    def distributed(self):
        """
        Should be called when distributed training with multiple training processes
        is going to be used (i.e. after the model is wrapped with DistributedDataParallel).
        Any special preparations for the algorithm to properly support distributed training
        should be made inside this function.
        """

    def set_builder_state_with_name(self, name: str, builder_state: Dict):
        self._builder_state_with_name = (name, builder_state)

    def load_state(self, state: Dict) -> None:
        """
        Loads the compression controller state.

        :param state: Output of `get_state()` method.
        """
        self.scheduler.load_state(state)

    def get_state(self) -> Dict[str, object]:
        """
        Returns the compression controller state.

        :return: The compression controller state.
        """
        return self.scheduler.get_state()

    def get_compression_setup(self) -> CompressionSetup:
        ctrl_state = self.get_state()
        if self._builder_state_with_name:
            raise RuntimeError('Internal error: builder state is not set for the controller')
        name, builder_state = self._builder_state_with_name
        return CompressionSetup(name, builder_state, ctrl_state)

    def get_compression_state(self) -> Dict:
        """
        Returns entire compression state, containing nncf_network state (with state of the builder)
        and composite controller state, containing the state of all children controllers.
        This checkpoint can be used to resume compression via compression_state of create_compressed_model
        :return: The entire compression state.
        """
        model_state = self.model.state_dict()
        setups = [self.get_compression_setup()]
        return PTCompressionState(setups, model_state).get_state()

    def run_batchnorm_adaptation(self, config):
        initializer_params = config.get("initializer", {})
        init_bn_adapt_config = initializer_params.get('batchnorm_adaptation', {})
        num_bn_adaptation_samples = init_bn_adapt_config.get('num_bn_adaptation_samples', 0)
        num_bn_forget_samples = init_bn_adapt_config.get('num_bn_forget_samples', 0)
        try:
            bn_adaptation_args = config.get_extra_struct(BNAdaptationInitArgs)
            has_bn_adapt_init_args = True
        except KeyError:
            has_bn_adapt_init_args = False

        if not init_bn_adapt_config:
            if has_bn_adapt_init_args:
                nncf_logger.warning("Enabling quantization batch norm adaptation with default parameters.")
                num_bn_adaptation_samples = 2000
                num_bn_forget_samples = 1000

        if num_bn_adaptation_samples < 0:
            raise AttributeError('Number of adaptation samples must be >= 0')
        if num_bn_adaptation_samples > 0:
            if not has_bn_adapt_init_args:
                nncf_logger.info(
                    'Could not run batchnorm adaptation '
                    'as the adaptation data loader is not provided as an extra struct. '
                    'Refer to `NNCFConfig.register_extra_structs` and the `BNAdaptationInitArgs` class')
                return
            batch_size = bn_adaptation_args.data_loader.batch_size
            num_bn_forget_steps = numpy.ceil(num_bn_forget_samples / batch_size)
            num_bn_adaptation_steps = numpy.ceil(num_bn_adaptation_samples / batch_size)
            bn_adaptation_runner = DataLoaderBNAdaptationRunner(self._model, bn_adaptation_args.device,
                                                                num_bn_forget_steps)
            bn_adaptation_runner.run(bn_adaptation_args.data_loader, num_bn_adaptation_steps)


class PTCompressionAlgorithmBuilder(CompressionAlgorithmBuilder):
    """
    Determines which modifications should be made to the original FP32 model in
    order to enable algorithm-specific compression during fine-tuning. Operates
    on an NNCFNetwork object wrapping a target PyTorch model (torch.nn.Module).
    """

    def __init__(self, config: NNCFConfig, should_init: bool = True,
                 compression_setups: Optional[List[CompressionSetup]] = None):
        """
        Arguments:
          `config` - a dictionary that contains parameters of compression method
          `should_init` - if False, trainable parameter initialization will be skipped during building
        """
        super().__init__(config, should_init)
        self.ignored_scopes = None
        self.target_scopes = None
        if not isinstance(self.config, list):
            self.ignored_scopes = self.config.get('ignored_scopes')
            self.target_scopes = self.config.get('target_scopes')
        self.compressed_nncf_module_names = self._nncf_module_types_to_compress()
        self._ctrl_state = None

        for compression_setup in compression_setups:
            name, builder_state, ctrl_state = compression_setup
            if self.registered_name == name:
                self.load_state(builder_state)
                self._ctrl_state = ctrl_state

    def apply_to(self, model: NNCFNetwork) -> NNCFNetwork:
        transformation_layout = self.get_transformation_layout(model)
        transformer = PTModelTransformer(model, transformation_layout)
        transformed_model = transformer.transform()
        return transformed_model

    def get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        """
        Applies algorithm-specific modifications to the model. Hooks to be executed during model
        forward operation may be registered using NNCFNetwork command insertion methods. Additional
        compression modules that are expected to be saved along with the network via torch.save should also be
        registered and added to the model here.
        :param target_model: An instance of NNCFNetwork for the algorithm to be applied to.
        :return: NNCFNetwork with algorithm-specific modifications applied
        """
        layout = self._get_transformation_layout(target_model)
        self._handle_frozen_layers(target_model)
        return layout

    def build_controller(self, model: ModelType) -> PTCompressionAlgorithmController:
        ctrl = super().build_controller(model)
        if not isinstance(ctrl, PTCompressionAlgorithmController):
            raise RuntimeError('Internal error: builder must create controller inherited from '
                               '`PTCompressionAlgorithmController` class')
        ctrl.set_builder_state_with_name(self.registered_name, self.get_state())
        if self._ctrl_state is not None:
            ctrl.load_state(self._ctrl_state)
        return ctrl

    def get_state(self) -> Dict[str, object]:
        """
        Returns a JSON-compatible dictionary containing a state of the object
        """
        return {}

    def load_state(self, state: Dict[str, object]):
        """
        Initializes object from the state
        :param state: Output of `get_state()` method.
        """
        pass

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        raise NotImplementedError()

    def _handle_frozen_layers(self, target_model: NNCFNetwork):
        scopes_of_frozen_layers = []
        for scope, module in target_model.get_nncf_modules().items():
            if not module.weight.requires_grad:
                if should_consider_scope(str(scope), self.target_scopes, self.ignored_scopes):
                    scopes_of_frozen_layers.append(str(scope))
        scopes_to_print = '\n'.join(scopes_of_frozen_layers)
        if len(scopes_of_frozen_layers) > 0:
            is_allowed, reason = self._are_frozen_layers_allowed()
            if is_allowed:
                nncf_logger.warning('{}, compressing them without tuning weights.\n'
                                    'Frozen layers:\n'
                                    '{}'.format(reason, scopes_to_print))
            else:
                raise RuntimeError(f'{reason}.\n'
                                   f'Please unfreeze them or put into the Ignored Scope.\n'
                                   f'Frozen Layers:\n'
                                   f'{scopes_to_print}')

    def _should_consider_scope(self, scope_str: str) -> bool:
        return should_consider_scope(scope_str, self.target_scopes, self.ignored_scopes)

    def _nncf_module_types_to_compress(self) -> List[str]:
        """
        Return list of NNCF module types which should be compressed by specific algorithm.
        As name of algorithm used the value set by decorator @Registry.register() or default one.
        :return: List of names of modules
        """
        filtered_nncf_module_names_list = list()
        for module in list(NNCF_MODULES_DICT) + list(NNCF_WRAPPED_USER_MODULES_DICT.values()):
            if self.registered_name not in module.ignored_algorithms:
                filtered_nncf_module_names_list.append(module.__name__)
        return filtered_nncf_module_names_list

    def _are_frozen_layers_allowed(self) -> Tuple[bool, str]:
        algo_name = self.registered_name.replace('_', ' ')
        return False, f'Frozen layers are not allowed for {algo_name}'

    @property
    def registered_name(self) -> str:
        return getattr(self, Registry.REGISTERED_NAME_ATTR, 'NOT_REGISTERED_' + self.__class__.__name__)
