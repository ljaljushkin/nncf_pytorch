"""
 Copyright (c) 2022 Intel Corporation
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

import pytest

from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple

from examples.torch.common.models.classification.resnet_cifar10 import resnet50_cifar10
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.nas.creators import create_bnas_model_and_ctrl_by_test_desc
from tests.torch.nas.descriptors import MultiElasticityTestDesc
from tests.torch.nas.models.synthetic import ThreeConvModel

from nncf import NNCFConfig
from nncf.config.structures import BNAdaptationInitArgs
from nncf.experimental.torch.nas.bootstrapNAS.search import SearchAlgorithm

class SearchTestDesc(NamedTuple):
    model_creator: Any
    # ref_model_stats: RefModelStats = None
    blocks_to_skip: List[List[str]] = None
    input_sizes: List[int] = [1, 3, 32, 32]
    algo_params: Dict = {}
    name: str = None
    # is_auto_skipped_blocks: bool = False
    mode: str = "auto"

    def __str__(self):
        if hasattr(self.model_creator, '__name__'):
            name = self.model_creator.__name__
        elif self.name is not None:
            name = self.name
        else:
            name = 'NOT_DEFINED'
        return name

search_desc = SearchTestDesc(model_creator=ThreeConvModel,
                          algo_params={'width': {'min_out_channels': 1, 'width_step': 1}},
                          input_sizes=ThreeConvModel.INPUT_SIZE,
                          )

def create_test_model(search_desc):
    model, ctrl = create_bnas_model_and_ctrl_by_test_desc(search_desc)
    elasticity_ctrl = ctrl.elasticity_controller
    config = {
        "input_info": {"sample_size": search_desc.input_sizes},
        "bootstrapNAS": {
            "training": {
                "batchnorm_adaptation": {
                    "num_bn_adaptation_samples": 2
                },
            },
            "search": {
                "algorithm": "NSGA2",
                "num_evals": 2,
                "population": 1
            }
        }
    }
    nncf_config = NNCFConfig.from_dict(config)
    bn_adapt_args = BNAdaptationInitArgs(data_loader=create_ones_mock_dataloader(nncf_config))
    nncf_config.register_extra_structs([bn_adapt_args])
    return model, elasticity_ctrl, nncf_config

def test_activate_maximum_subnet_at_init():
    model, elasticity_ctrl, nncf_config = create_test_model(search_desc)
    SearchAlgorithm(model, elasticity_ctrl, nncf_config)

    config_init = elasticity_ctrl.multi_elasticity_handler.get_active_config()
    elasticity_ctrl.multi_elasticity_handler.activate_maximum_subnet()
    assert config_init == elasticity_ctrl.multi_elasticity_handler.get_active_config()


def test_upper_bounds():
    model, elasticity_ctrl, nncf_config = create_test_model(search_desc)
    nncf_config = NNCFConfig.from_dict(nncf_config)
    bn_adapt_args = BNAdaptationInitArgs(data_loader=create_ones_mock_dataloader(nncf_config))
    nncf_config.register_extra_structs([bn_adapt_args])

    search = SearchAlgorithm(model, elasticity_ctrl, nncf_config)
    assert search._xu == [2, 1, 0, 0, 0]

