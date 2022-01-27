"""
 Copyright (c) 2019-2021 Intel Corporation
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
from typing import Any
from typing import Dict

from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.api.compression import CompressionStage
from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.statistics import NNCFStatistics
from nncf.torch.algo_selector import ZeroCompressionLoss
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import MultiElasticityHandler
from nncf.torch.nncf_network import NNCFNetwork


class EControllerStateNames:
    MULTI_ELASTICITY_HANDLER_STATE = 'multi_elasticity_handler_state'


class ElasticityController(PTCompressionAlgorithmController):
    _ec_state_names = EControllerStateNames

    def __init__(self, target_model: NNCFNetwork, algo_config: Dict, multi_elasticity_handler: MultiElasticityHandler):
        super().__init__(target_model)
        self.target_model = target_model
        self._algo_config = algo_config
        self._loss = ZeroCompressionLoss(next(target_model.parameters()).device)
        self._scheduler = BaseCompressionScheduler()

        self.multi_elasticity_handler = multi_elasticity_handler
        # Handlers deactivated at init
        self.multi_elasticity_handler.deactivate()

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        return NNCFStatistics()

    def compression_stage(self) -> CompressionStage:
        # TODO(nlyalyus): should return FULLY_COMPRESSED when search algorithm activates final subnet
        return CompressionStage.UNCOMPRESSED

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Loads the compression controller state from the map of algorithm name to the dictionary with state attributes.

        :param state: map of the algorithm name to the dictionary with the corresponding state attributes.
        """
        super().load_state(state)
        self.multi_elasticity_handler.load_state(state[self._ec_state_names.MULTI_ELASTICITY_HANDLER_STATE])

    def get_state(self) -> Dict[str, Any]:
        """
        Returns compression controller state, which is the map of the algorithm name to the dictionary with the
        corresponding state attributes.

        :return: The compression controller state.
        """
        state = super().get_state()
        state[self._ec_state_names.MULTI_ELASTICITY_HANDLER_STATE] = self.multi_elasticity_handler.get_state()
        return state


class ElasticNet:
    def __init__(self, nncf_network: NNCFNetwork, elasticity_controller: ElasticityController):
        self.nncf_network = nncf_network
        self.elasticity_controller = elasticity_controller