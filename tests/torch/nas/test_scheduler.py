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
from collections import OrderedDict
from functools import partial
from typing import Dict
from typing import List

import pytest

from nncf.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.torch.nas.bootstrapNAS.elasticity.base_handler import SingleElasticHandler
from nncf.torch.nas.bootstrapNAS.elasticity.elastic_depth import ElasticDepthHandler
from nncf.torch.nas.bootstrapNAS.elasticity.elastic_width import ElasticWidthHandler
from nncf.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import MultiElasticityHandler
from nncf.torch.nas.bootstrapNAS.training.progressive_shrinking_builder import ProgressiveShrinkingBuilder
from nncf.torch.nas.bootstrapNAS.training.progressive_shrinking_controller import ProgressiveShrinkingController
from nncf.torch.nas.bootstrapNAS.training.scheduler import BootstrapNASScheduler
from nncf.torch.nas.bootstrapNAS.training.stage_descriptor import StageDescriptor
from tests.torch.helpers import MockModel

LIST_STAGES__K_KW_KWD = [
    [ElasticityDim.KERNEL],
    [ElasticityDim.KERNEL, ElasticityDim.WIDTH],
    [ElasticityDim.KERNEL, ElasticityDim.WIDTH, ElasticityDim.DEPTH]
]

LIST_STAGES__K_KD_KDW = [
    [ElasticityDim.KERNEL],
    [ElasticityDim.KERNEL, ElasticityDim.DEPTH],
    [ElasticityDim.KERNEL, ElasticityDim.DEPTH, ElasticityDim.WIDTH]
]

SIMPLE_LIST_STAGE_DESCRIPTORS = [
    {"train_dims": ["kernel"], "epochs": 1},
    {"train_dims": ["kernel", "depth"], "epochs": 1, "depth_indicator": 1},
    {"train_dims": ["kernel", "depth"], "epochs": 1, "depth_indicator": 2},
    {"train_dims": ["kernel", "depth", "width"], "epochs": 1, "depth_indicator": 2, "reorg_weights": True,
     "width_indicator": 2},
    {"train_dims": ["kernel", "depth", "width"], "epochs": 1, "depth_indicator": 2, "reorg_weights": True,
     "width_indicator": 3}
]


@pytest.fixture(params=[SIMPLE_LIST_STAGE_DESCRIPTORS], ids=['simple_desc'])
def schedule_params(request):
    list_descriptors = request.param
    return {"list_stage_descriptions": list_descriptors}


LIST_DIMS__KDW = [ElasticityDim.KERNEL, ElasticityDim.DEPTH, ElasticityDim.WIDTH]


class TestScheduler:
    def test_get_stage(self, schedule_params, mocker):
        scheduler = BootstrapNASScheduler(mocker.stub(), schedule_params, LIST_DIMS__KDW,
                                          LIST_DIMS__KDW)

        scheduler.current_epoch = 0
        ref_desc = StageDescriptor(train_dims=[ElasticityDim.KERNEL], epochs=1)
        act_desc, act_idx = scheduler.get_train_dims_for_epoch()
        assert ref_desc == act_desc
        assert act_idx == 0

        scheduler.current_epoch = 2
        ref_desc.train_dims.append(ElasticityDim.DEPTH)
        ref_desc.depth_indicator = 2
        act_desc, act_idx = scheduler.get_train_dims_for_epoch()
        assert ref_desc == act_desc
        assert act_idx == 2

        scheduler.current_epoch = 3
        ref_desc.train_dims.append(ElasticityDim.WIDTH)
        ref_desc.reorg_weights = True
        ref_desc.width_indicator = 2
        act_desc, act_idx = scheduler.get_train_dims_for_epoch()
        assert ref_desc == act_desc
        assert act_idx == 3

        scheduler.current_epoch = 4
        ref_desc.width_indicator = 3
        act_desc, act_idx = scheduler.get_train_dims_for_epoch()
        assert ref_desc == act_desc
        assert act_idx == 4

    def test_epoch_step(self, schedule_params, mocker):
        mock_model = MockModel()
        mock_width_handler = mocker.MagicMock(spec=ElasticWidthHandler)
        mock_depth_handler = mocker.MagicMock(spec=ElasticDepthHandler)
        mock_kernel_handler = mocker.MagicMock(spec=SingleElasticHandler)
        handlers = OrderedDict({
            ElasticityDim.WIDTH: mock_width_handler,
            ElasticityDim.KERNEL: mock_kernel_handler,
            ElasticityDim.DEPTH: mock_depth_handler,
        })
        mock_handler = MultiElasticityHandler(handlers)
        mock_elasticity_ctrl = mocker.stub()
        mock_elasticity_ctrl.multi_elasticity_handler = mock_handler
        training_algo = ProgressiveShrinkingController(mock_model, mock_elasticity_ctrl, mocker.stub(),
                                                       ProgressiveShrinkingBuilder.DEFAULT_PROGRESSIVITY,
                                                       schedule_params)
        scheduler = training_algo.scheduler
        scheduler.epoch_step()
        mock_width_handler.activate.assert_not_called()
        mock_depth_handler.activate.assert_not_called()
        mock_kernel_handler.activate.assert_called()

        scheduler.epoch_step()
        mock_width_handler.activate.assert_not_called()
        mock_depth_handler.activate.assert_called()
        mock_kernel_handler.activate.assert_called()
        assert mock_depth_handler.depth_indicator == 1

        scheduler.epoch_step()
        mock_width_handler.activate.assert_not_called()
        mock_depth_handler.activate.assert_called()
        mock_kernel_handler.activate.assert_called()
        assert mock_depth_handler.depth_indicator == 2

        scheduler.epoch_step()
        mock_width_handler.activate.assert_called()
        mock_width_handler.reorganize_weights.assert_called()
        assert mock_width_handler.width_num_params_indicator == 2

        scheduler.epoch_step()
        mock_width_handler.activate.assert_called()
        mock_width_handler.reorganize_weights.assert_called()
        assert mock_width_handler.width_num_params_indicator == 3

    def test_get_total_training_epochs(self, schedule_params, mocker):
        scheduler = BootstrapNASScheduler(mocker.stub(), schedule_params,
                                          enabled_elasticity_dims=LIST_DIMS__KDW,
                                          progressivity_of_elasticity=LIST_DIMS__KDW)
        assert scheduler.get_total_training_epochs() == 5


class SchedulerTestDesc:
    def __init__(self, list_stage_dims: List[List[ElasticityDim]],
                 progressivity_of_elasticity: List[ElasticityDim],
                 enabled_elasticity_dims: List[ElasticityDim],
                 name: str = '',
                 error_in_scheduler: bool = False,
                 error_in_builder: bool = False):
        self.list_stage_dims = list_stage_dims
        self.progressivity_of_elasticity = progressivity_of_elasticity
        self.enabled_elasticity_dims = enabled_elasticity_dims
        self.error_in_scheduler = error_in_scheduler
        self.error_in_builder = error_in_builder
        self.name = name

    def __str__(self):
        return self.name

    @property
    def scheduler_params(self) -> Dict[str, List[Dict]]:
        list_stage_descs = [{"train_dims": list(map(lambda x: x.value, stage_dims))} for stage_dims in
                            self.list_stage_dims]
        return {"list_stage_descriptions": list_stage_descs}


LIST_SCHEDULER_DESCS = [
    SchedulerTestDesc(
        name='default',
        list_stage_dims=LIST_STAGES__K_KD_KDW,
        progressivity_of_elasticity=LIST_DIMS__KDW,
        enabled_elasticity_dims=LIST_DIMS__KDW,
    ),
    SchedulerTestDesc(
        name='wrong order in progressivity',
        list_stage_dims=LIST_STAGES__K_KW_KWD,
        progressivity_of_elasticity=LIST_DIMS__KDW,
        enabled_elasticity_dims=LIST_DIMS__KDW,
        error_in_scheduler=True
    ),
    SchedulerTestDesc(
        name='limited progressivity',
        list_stage_dims=LIST_STAGES__K_KW_KWD,
        progressivity_of_elasticity=[ElasticityDim.KERNEL],
        enabled_elasticity_dims=LIST_DIMS__KDW,
        error_in_builder=True,
        error_in_scheduler=True,
    ),
    SchedulerTestDesc(
        name='limited enabled dims',
        list_stage_dims=LIST_STAGES__K_KW_KWD,
        progressivity_of_elasticity=LIST_DIMS__KDW,
        enabled_elasticity_dims=[ElasticityDim.KERNEL],
        error_in_scheduler=True
    ),
    SchedulerTestDesc(
        name='limited progressivity and enabled dims',
        list_stage_dims=LIST_STAGES__K_KW_KWD,
        progressivity_of_elasticity=[ElasticityDim.KERNEL],
        enabled_elasticity_dims=[ElasticityDim.KERNEL],
        error_in_scheduler=True,
    ),
    SchedulerTestDesc(
        name='limited list stages',
        list_stage_dims=[[ElasticityDim.KERNEL]],
        progressivity_of_elasticity=LIST_DIMS__KDW,
        enabled_elasticity_dims=LIST_DIMS__KDW,
    ),
    SchedulerTestDesc(
        name='violated progressivity',
        list_stage_dims=LIST_STAGES__K_KW_KWD,
        progressivity_of_elasticity=[ElasticityDim.KERNEL, ElasticityDim.DEPTH, ElasticityDim.WIDTH],
        enabled_elasticity_dims=LIST_DIMS__KDW,
        error_in_scheduler=True,
    ),
    SchedulerTestDesc(
        name='order within stage doesn\'t matter',
        list_stage_dims=[
            [ElasticityDim.KERNEL],
            [ElasticityDim.DEPTH, ElasticityDim.KERNEL],
            [ElasticityDim.DEPTH, ElasticityDim.WIDTH, ElasticityDim.KERNEL]
        ],
        progressivity_of_elasticity=LIST_DIMS__KDW,
        enabled_elasticity_dims=LIST_DIMS__KDW,
    ),
    SchedulerTestDesc(
        name='new single dim on each stage',
        list_stage_dims=[
            [ElasticityDim.KERNEL],
            [ElasticityDim.DEPTH],
            [ElasticityDim.WIDTH],
        ],
        progressivity_of_elasticity=LIST_DIMS__KDW,
        enabled_elasticity_dims=LIST_DIMS__KDW,
        error_in_scheduler=True,
    ),
    SchedulerTestDesc(
        name='intermediate dim is not enabled',
        list_stage_dims=[
            [ElasticityDim.KERNEL],
            [ElasticityDim.DEPTH, ElasticityDim.KERNEL],
        ],
        progressivity_of_elasticity=[ElasticityDim.KERNEL, ElasticityDim.WIDTH, ElasticityDim.DEPTH],
        enabled_elasticity_dims=[ElasticityDim.KERNEL, ElasticityDim.DEPTH],
    ),
]


@pytest.mark.parametrize('desc', LIST_SCHEDULER_DESCS, ids=map(lambda x: str(x), LIST_SCHEDULER_DESCS))
class TestElasticityConsistency:
    def test_checks_on_scheduler_init(self, mocker, desc: SchedulerTestDesc):
        scheduler_fn = partial(BootstrapNASScheduler,
                               mocker.stub(),
                               desc.scheduler_params,
                               progressivity_of_elasticity=desc.progressivity_of_elasticity,
                               enabled_elasticity_dims=desc.enabled_elasticity_dims)
        scheduler = scheduler_fn()
        if desc.error_in_scheduler:
            with pytest.raises(ValueError):
                _ = scheduler.list_stage_descriptors
        else:
            _ = scheduler.list_stage_descriptors

    def test_progressivity_vs_enabled_dims(self, desc: SchedulerTestDesc):
        builder_fn = partial(ProgressiveShrinkingBuilder.check_elasticity_dims_consistency,
                             desc.enabled_elasticity_dims, desc.progressivity_of_elasticity)
        if desc.error_in_builder:
            with pytest.raises(ValueError):
                builder_fn()
        else:
            builder_fn()
