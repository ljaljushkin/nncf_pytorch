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

from nncf.experimental.torch.nas.bootstrapNAS.training.lr_scheduler import GlobalLRScheduler
from nncf.experimental.torch.nas.bootstrapNAS.training.lr_scheduler import StageLRScheduler
from nncf.experimental.torch.nas.bootstrapNAS.training.lr_scheduler import LRSchedulerParams

LR_SCHEDULER_PARAMS = LRSchedulerParams.from_dict({
            'num_steps_in_epoch': 10,
            'base_lr': 2.5e-6,
            'num_epochs': 5,
            'warmup_epochs': 2,
        })


class TestLRScheduler:
    def test_warmup_lr(self, mocker):
        warmup_lr = mocker.patch(
            'nncf.experimental.torch.nas.bootstrapNAS.training.lr_scheduler.warmup_adjust_learning_rate')
        optimizer = mocker.stub()
        optimizer.param_groups = [{'lr': 0}]
        lr_scheduler = GlobalLRScheduler.from_config(optimizer, LR_SCHEDULER_PARAMS)
        lr_scheduler.epoch_step(0)
        lr_scheduler.step()
        warmup_lr.assert_called()

    def test_adjust_lr(self, mocker):
        adjust_lr = mocker.patch(
            'nncf.experimental.torch.nas.bootstrapNAS.training.lr_scheduler.adjust_learning_rate')
        optimizer = mocker.stub()
        optimizer.param_groups = [{'lr': 0}]
        params = LR_SCHEDULER_PARAMS
        params.warmup_epochs = 0
        lr_scheduler = GlobalLRScheduler.from_config(optimizer, params)
        lr_scheduler.epoch_step(0)
        lr_scheduler.step()
        adjust_lr.assert_called()

    def test_warmup_to_regular(self, mocker):
        warmup_lr = mocker.patch(
            'nncf.experimental.torch.nas.bootstrapNAS.training.lr_scheduler.warmup_adjust_learning_rate')
        adjust_lr = mocker.patch(
            'nncf.experimental.torch.nas.bootstrapNAS.training.lr_scheduler.adjust_learning_rate')
        optimizer = mocker.stub()
        optimizer.param_groups = [{'lr': 0}]
        params = LR_SCHEDULER_PARAMS
        params.warmup_epochs = 1
        lr_scheduler = GlobalLRScheduler.from_config(optimizer, params)
        lr_scheduler.epoch_step(0)
        lr_scheduler.step()
        warmup_lr.assert_called()
        lr_scheduler.epoch_step()
        lr_scheduler.step()
        adjust_lr.assert_called()

    def test_reset(self, mocker):
        optimizer = mocker.stub()
        optimizer.param_groups = [{'lr': 1}]
        lr_scheduler = StageLRScheduler.from_config(optimizer, LR_SCHEDULER_PARAMS)
        lr_scheduler._init_lr = 2.5e-4
        lr_scheduler._num_epochs = 10
        assert lr_scheduler.current_epoch == -1
        assert lr_scheduler.current_step == -1
        lr_scheduler.epoch_step()
        lr_scheduler.step()
        assert lr_scheduler.current_epoch == 0
        assert lr_scheduler.current_step == 0
        lr_scheduler.epoch_step()
        lr_scheduler.step()
        assert lr_scheduler.current_epoch == 1
        assert lr_scheduler.current_step == 1
        lr_scheduler.reset(LR_SCHEDULER_PARAMS.base_lr, LR_SCHEDULER_PARAMS.num_epochs)
        assert lr_scheduler.current_epoch == 0
        assert lr_scheduler.current_step == 0

    STAGE_LR_UPDATE = [0.00025, 0.0002450367107096179, 0.00023054099068775188,
                      0.00020766398316545648, 0.0001782224114456341, 0.00014455430813002887,
                      0.00010933334580446199, 7.535651367065242e-05, 4.532200128141378e-05,
                      2.1614928215679784e-05]

    GLOBAL_LR_UPDATE = [0.00030625000000000004, 2.4975334105353397e-06, 2.3453833500548297e-06,
                        1.8521920926271442e-06, 1.1715118505883583e-06]

    def test_lr_value_update(self, mocker):
        optimizer = mocker.stub()
        optimizer.param_groups = [{'lr': 3.4e-4}]
        lr_scheduler = StageLRScheduler.from_config(optimizer, LR_SCHEDULER_PARAMS)
        lr_scheduler._init_lr = 2.5e-4
        lr_scheduler._num_epochs = 10
        for i in range(lr_scheduler._num_epochs):
            lr_scheduler.epoch_step()
            for j in range(lr_scheduler._num_steps_in_epoch):
                lr_scheduler.step()
                if j == 0:
                    assert optimizer.param_groups[0]['lr'] == pytest.approx(TestLRScheduler.STAGE_LR_UPDATE[i])

        optimizer.param_groups = [{'lr': 3.4e-4}]
        lr_scheduler = GlobalLRScheduler.from_config(optimizer, LR_SCHEDULER_PARAMS)
        for i in range(lr_scheduler._num_epochs):
            lr_scheduler.epoch_step()
            for j in range(lr_scheduler._num_steps_in_epoch):
                lr_scheduler.step()
                if j == 0:
                    assert optimizer.param_groups[0]['lr'] == pytest.approx(TestLRScheduler.GLOBAL_LR_UPDATE[i])