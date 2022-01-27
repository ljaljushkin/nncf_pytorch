"""
 Copyright (c) 2021 Intel Corporation
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
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple

from nncf.torch.model_creation import create_nncf_network
from tests.torch.helpers import get_empty_config
from tests.torch.nas.helpers import move_model_to_cuda_if_available
from tests.torch.nas.models.synthetic import ThreeConvModel


class ElasticityDesc:
    def __init__(self,
                 elastic_builder_cls: Callable,
                 model_cls: Optional[Callable] = None,
                 ref_state: Optional[Dict[str, Any]] = None,
                 name: Optional[str] = None,
                 params: Dict[str, Any] = None,
                 input_size: Optional[List[int]] = None,
                 ref_search_space: Optional[Any] = None,
                 ref_output_fn: Optional[Callable] = None):
        self.elastic_builder_cls = elastic_builder_cls
        self.model_cls = model_cls
        self.ref_state = ref_state
        self.name = name
        self.params = dict() if params is None else params
        self.input_size = input_size
        self.ref_search_space = ref_search_space
        self.ref_output_fn = ref_output_fn

    def __str__(self):
        if self.name:
            return self.name
        result = self.elastic_builder_cls.__name__
        if hasattr(self.model_cls, '__name__'):
            result += '_' + self.model_cls.__name__
        return result

    def build_handler(self):
        model = self.model_cls()
        move_model_to_cuda_if_available(model)
        input_size = self.input_size
        if not input_size:
            input_size = model.INPUT_SIZE
        config = get_empty_config(input_sample_sizes=input_size)
        nncf_network = create_nncf_network(model, config)
        builder = self.elastic_builder_cls(self.params)
        handler = builder.build(nncf_network)
        return handler, builder


class WidthElasticityDesc:
    def __init__(self, desc: ElasticityDesc,
                 width_num_params_indicator: Optional[int] = -1):
        self._elasticity_desc = desc
        self._width_num_params_indicator = width_num_params_indicator

    @property
    def ref_search_space(self):
        return self._elasticity_desc.ref_search_space

    def build_handler(self):
        handler, builder = self._elasticity_desc.build_handler()
        handler.width_num_params_indicator = self._width_num_params_indicator
        return handler, builder

    def __str__(self):
        return str(self._elasticity_desc) + '_wi:' + str(self._width_num_params_indicator)


class ModelStats(NamedTuple):
    flops: int
    num_weights: int

    def __eq__(self, other: Tuple[int, int]):
        return other[0] == self.flops and other[1] == self.num_weights


class RefModelStats(NamedTuple):
    supernet: ModelStats
    kernel_stage: ModelStats
    width_stage: ModelStats
    depth_stage: ModelStats


# TODO(nlyalyus): any way to combine with ElasticityDesc?
class MultiElasticityTestDesc(NamedTuple):
    model_creator: Any
    ref_model_stats: RefModelStats = None
    blocks_to_skip: List[List[str]] = None
    blocks_dependencies: Dict[int, List[int]] = {0: [0]}
    input_sizes: List[int] = [1, 3, 32, 32]
    algo_params: Dict = {}
    name: str = None
    is_auto_skipped_blocks: bool = False
    ordinal_ids: List = None

    def __str__(self):
        if hasattr(self.model_creator, '__name__'):
            name = self.model_creator.__name__
        elif self.name is not None:
            name = self.name
        else:
            name = 'NOT_DEFINED'
        return name


THREE_CONV_TEST_DESC = MultiElasticityTestDesc(model_creator=ThreeConvModel,
                                               ref_model_stats=RefModelStats(supernet=ModelStats(17400, 87),
                                                                             kernel_stage=ModelStats(7800, 39),
                                                                             depth_stage=ModelStats(5400, 27),
                                                                             width_stage=ModelStats(1800, 9)),
                                               blocks_to_skip=[
                                                   # TODO(nlyalyus): no need to delete last_layer, as mask propagation
                                                   #  should fail on newly artificially created last layer?
                                                   ['ThreeConvModel/NNCFConv2d[conv1]/conv2d_0',
                                                    '/nncf_model_output_0']],
                                               algo_params={'width': {'min_out_channels': 1, 'width_step': 1}},
                                               input_sizes=ThreeConvModel.INPUT_SIZE,
                                               )