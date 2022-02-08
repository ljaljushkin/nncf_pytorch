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
import inspect
from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import OrderedDict as OrderedDictType

from nncf.common.pruning.utils import count_flops_and_weights_per_node
from nncf.torch.nas.bootstrapNAS.elasticity.base_handler import ElasticityConfig
from nncf.torch.nas.bootstrapNAS.elasticity.base_handler import ElasticityHandler
from nncf.torch.nas.bootstrapNAS.elasticity.base_handler import SingleElasticityHandler
from nncf.torch.nas.bootstrapNAS.elasticity.elastic_depth import ElasticDepthHandler
from nncf.torch.nas.bootstrapNAS.elasticity.elastic_depth import ElasticDepthSearchSpace
from nncf.torch.nas.bootstrapNAS.elasticity.elastic_kernel import ElasticKernelHandler
from nncf.torch.nas.bootstrapNAS.elasticity.elastic_kernel import ElasticKernelSearchSpace
from nncf.torch.nas.bootstrapNAS.elasticity.elastic_width import ElasticWidthHandler
from nncf.torch.nas.bootstrapNAS.elasticity.elastic_width import ElasticWidthSearchSpace
from nncf.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim

SubnetConfig = OrderedDictType[ElasticityDim, ElasticityConfig]


class MEHandlerStateNames:
    IS_HANDLER_ENABLED_MAP = 'is_handler_enabled_map'
    STATES_OF_HANDLERS = 'states_of_handlers'


class MultiElasticityHandler(ElasticityHandler):
    _state_names = MEHandlerStateNames

    def __init__(self, handlers: OrderedDictType[ElasticityDim, SingleElasticityHandler]):
        self._handlers = handlers
        self._is_handler_enabled_map = {elasticity_dim: True for elasticity_dim in handlers}
        self.activate_supernet()

    @property
    def width_search_space(self) -> ElasticWidthSearchSpace:
        return self.width_handler.get_search_space()

    @property
    def kernel_search_space(self) -> ElasticKernelSearchSpace:
        return self.kernel_handler.get_search_space()

    @property
    def depth_search_space(self) -> ElasticDepthSearchSpace:
        return self.depth_handler.get_search_space()

    @property
    def width_handler(self) -> Optional[ElasticWidthHandler]:
        return self._get_handler_by_elasticity_dim(ElasticityDim.WIDTH)

    @property
    def kernel_handler(self) -> Optional[ElasticKernelHandler]:
        return self._get_handler_by_elasticity_dim(ElasticityDim.KERNEL)

    @property
    def depth_handler(self) -> Optional[ElasticDepthHandler]:
        return self._get_handler_by_elasticity_dim(ElasticityDim.DEPTH)

    def get_available_elasticity_dims(self) -> List[ElasticityDim]:
        return list(self._handlers)

    def get_active_config(self) -> SubnetConfig:
        """
        Forms an elasticity configuration that describes currently activated Subnet across all elasticities.

        :return: elasticity configuration
        """
        return self._collect_handler_data_by_method_name(self._get_current_method_name())

    def get_random_config(self) -> SubnetConfig:
        """
        Forms an elasticity configuration that describes a Subnet with randomly chosen elastic values across all
        elasticities.

        :return: elasticity configuration
        """
        return self._collect_handler_data_by_method_name(self._get_current_method_name())

    def get_minimum_config(self) -> SubnetConfig:
        """
        Forms an elasticity configuration that describes a Subnet with minimum elastic values across all elasticities.

        :return: elasticity configuration
        """
        return self._collect_handler_data_by_method_name(self._get_current_method_name())

    def get_maximum_config(self) -> SubnetConfig:
        """
        Forms an elasticity configuration that describes a Subnet with maximum elastic values across all elasticities.

        :return: elasticity configuration
        """
        return self._collect_handler_data_by_method_name(self._get_current_method_name())

    def activate_supernet(self) -> None:
        """
        Activates the Supernet - the original network to which elasticity was applied.
        """
        self._collect_handler_data_by_method_name(self._get_current_method_name())

    def set_config(self, config: SubnetConfig) -> None:
        """
        Activates a Subnet that corresponds to the given elasticity configuration

        :param config: elasticity configuration
        """
        active_handlers = {
            dim: self._handlers[dim] for dim in self._handlers if self._is_handler_enabled_map[dim]
        }
        for handler_id, handler in self._handlers.items():
            if handler_id in config:
                sub_config = config[handler_id]
                other_active_handlers = dict(filter(lambda pair: pair[0] != handler_id, active_handlers.items()))
                resolved_config = handler.resolve_conflicts_with_other_elasticities(sub_config, other_active_handlers)
                handler.set_config(resolved_config)

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Initializes object from the state.

        :param state: Output of `get_state()` method.
        """
        states_of_handlers = state[self._state_names.STATES_OF_HANDLERS]
        is_handler_enabled_map = state[self._state_names.IS_HANDLER_ENABLED_MAP]

        for dim_str, handler_state in states_of_handlers.items():
            dim = ElasticityDim.from_str(dim_str)
            if dim in self._handlers:
                self._handlers[dim].load_state(handler_state)

        for dim_str, is_enabled in is_handler_enabled_map.items():
            dim = ElasticityDim.from_str(dim_str)
            self._is_handler_enabled_map[dim] = is_enabled

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        states_of_handlers = {dim.value: handler.get_state() for dim, handler in self._handlers.items()}
        is_handler_enabled_map = {dim.value: is_enabled for dim, is_enabled in self._is_handler_enabled_map.items()}
        return {
            self._state_names.STATES_OF_HANDLERS: states_of_handlers,
            self._state_names.IS_HANDLER_ENABLED_MAP: is_handler_enabled_map
        }

    def enable_all(self):
        self._is_handler_enabled_map = {elasticity_dim: True for elasticity_dim in self._is_handler_enabled_map}

    def disable_all(self):
        self._is_handler_enabled_map = {elasticity_dim: False for elasticity_dim in self._is_handler_enabled_map}

    def enable_elasticity(self, dim: ElasticityDim):
        self._is_handler_enabled_map[dim] = True

    def disable_elasticity(self, dim: ElasticityDim):
        self._is_handler_enabled_map[dim] = False

    def count_flops_and_weights_for_active_subnet(self):
        kwargs = {}
        for handler in self._handlers.values():
            kwargs.update(handler.get_kwargs_for_flops_counting())

        flops_pers_node, num_weights_per_node = count_flops_and_weights_per_node(**kwargs)

        flops = sum(flops_pers_node.values())
        num_weights = sum(num_weights_per_node.values())
        return flops, num_weights

    def _get_handler_by_elasticity_dim(self, dim: ElasticityDim) -> Optional[SingleElasticityHandler]:
        result = None
        if dim in self._handlers:
            result = self._handlers[dim]
        return result

    def _collect_handler_data_by_method_name(self, method_name) -> OrderedDictType[ElasticityDim, Any]:
        result = OrderedDict()
        for elasticity_dim, handler in self._handlers.items():
            if self._is_handler_enabled_map[elasticity_dim]:
                handler_method = getattr(handler, method_name)
                data = handler_method()
                result[elasticity_dim] = data
        return result

    @staticmethod
    def _get_current_method_name() -> str:
        return inspect.stack()[1].function
