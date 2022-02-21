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

from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.helpers import DummyDataLoader
from tests.torch.helpers import get_empty_config
from tests.torch.nas.creators import create_bnas_model_and_ctrl_by_test_desc
from tests.torch.nas.creators import create_bootstrap_nas_training_algo
from tests.torch.nas.creators import NAS_MODEL_DESCS
from tests.torch.nas.models.synthetic import ThreeConvModel
from tests.torch.nas.test_all_elasticity import NAS_MODELS_SCOPE, fixture_nas_model_name
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
    mode: str = "auto"

    def __str__(self):
        if hasattr(self.model_creator, '__name__'):
            name = self.model_creator.__name__
        elif self.name is not None:
            name = self.name
        else:
            name = 'NOT_DEFINED'
        return name


def prepare_test_model(search_desc):
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
    search_desc = SearchTestDesc(model_creator=ThreeConvModel,
                                 algo_params={'width': {'min_out_channels': 1, 'width_step': 1}},
                                 input_sizes=ThreeConvModel.INPUT_SIZE,
                                 )
    model, elasticity_ctrl, nncf_config = prepare_test_model(search_desc)
    SearchAlgorithm(model, elasticity_ctrl, nncf_config)
    config_init = elasticity_ctrl.multi_elasticity_handler.get_active_config()
    elasticity_ctrl.multi_elasticity_handler.activate_maximum_subnet()
    assert config_init == elasticity_ctrl.multi_elasticity_handler.get_active_config()


NAS_MODELS_SEARCH_ENCODING = {
    'resnet18': [1, 3, 7, 1, 1, 3, 3, 7, 7, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15],
    'resnet50': [7, 15, 31, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 15, 15, 15,
                 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15],
    'densenet_121': [1, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
                     0, 3, 0, 3, 0, 7, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
                     0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                     3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7],
    'mobilenet_v2': [0, 0, 1, 2, 4, 0, 0, 2, 3, 3, 5, 5, 5, 11, 11, 11, 11, 17, 17, 17, 29, 29, 29, 9, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31],
    'vgg11': [1, 3, 7, 7, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'vgg11_k7': [1, 3, 7, 7, 15, 15, 15, 2, 2, 2, 2, 2, 2, 2, 2, 0],
    'unet': [1, 3, 7, 15, 31, 15, 7, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'squeezenet1_0': [2, 0, 1, 1, 0, 1, 1, 0, 3, 3, 0, 3, 3, 0, 5, 5, 0, 5, 5, 1, 7, 7, 1, 7, 7, 2, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15],
    'resnext29_32x4d': [7, 15, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 7],
    'pnasnetb': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1,
                 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0,
                 2, 0, 1, 0, 2, 0, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 7],
    'ssd_mobilenet': [0, 1, 3, 3, 7, 7, 15, 15, 15, 15, 15, 15, 31, 31, 7, 15, 3, 7, 3, 7, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 32767],
    'ssd_vgg': [1, 1, 3, 3, 7, 7, 7, 15, 15, 15, 15, 15, 31, 31, 7, 15, 3, 7, 3, 7, 3, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7],
    'mobilenet_v3_small': [0, 0, 2, 6, 6, 0, 2, 3, 0, 8, 17, 17, 2, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 3, 3, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                           0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 32767]
}


def prepare_search_algorithm(nas_model_name):
    if 'inception_v3' in nas_model_name:
        pytest.skip(
            f'Skip test for {nas_model_name} as it fails because of 2 issues: '
            'not able to set DynamicInputOp to train-only layers (ticket 60976) and '
            'invalid padding update in elastic kernel (ticket 60990)')

    elif nas_model_name in ['efficient_net_b0', 'shufflenetv2']:
        pytest.skip(
            f'Skip test for {nas_model_name} as exploration is underway to better manage its search space'
        )
    model, ctrl, _ = create_bootstrap_nas_training_algo(nas_model_name)

    nncf_config = get_empty_config(input_sample_sizes=NAS_MODEL_DESCS[nas_model_name][1])
    nncf_config['bootstrapNAS'] = {'training': {'algorithm': 'progressive_shrinking'},
                                   'search': {"algorithm": "NSGA2", "num_evals": 2, "population": 1}}
    nncf_config['input_info'][0].update({'filler': 'random'})
    if nas_model_name == 'densenet_121':
        nncf_config['bootstrapNAS']['training'] = {
            'elasticity': {'depth': {'min_block_size': 10, 'max_block_size': 117}}}
    elasticity_ctrl = ctrl.elasticity_controller
    nncf_config = NNCFConfig.from_dict(nncf_config)
    nncf_config.register_extra_structs([BNAdaptationInitArgs(data_loader=DummyDataLoader(), device=None)])
    return SearchAlgorithm(model, elasticity_ctrl, nncf_config)


def test_design_upper_bounds(nas_model_name):
    search = prepare_search_algorithm(nas_model_name)
    assert search.vars_upper == NAS_MODELS_SEARCH_ENCODING[nas_model_name]


def test_num_variables(nas_model_name):
    search = prepare_search_algorithm(nas_model_name)
    assert search.num_vars == len(NAS_MODELS_SEARCH_ENCODING[nas_model_name])


def test_create_default_evaluators(nas_model_name):
    search = prepare_search_algorithm(nas_model_name)

    def fake_acc_validate_fn(model, val_loader):
        return 0, 0, 0
    search.run(fake_acc_validate_fn, None)
    evaluators = search.evaluators
    assert len(evaluators) == 2
    assert not evaluators[0].use_model_for_evaluation
    assert evaluators[0].name == 'flops'
    assert evaluators[1].use_model_for_evaluation
    assert evaluators[1].name == 'top1_acc'