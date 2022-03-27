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

LR_SCHEDULER_PARAMS = {
            'base_lr': 2.5e-6,
            'num_epochs': 5,
            'warmup_epochs': 2,
            'warmup_lr': 1
        }

class TestCosineLRScheduler:
    def test_warmup_lr(self, mocker):
        warmup_lr = mocker.patch(
            'nncf.experimental.torch.nas.bootstrapNAS.training.lr_scheduler.warmup_adjust_learning_rate')
        optimizer = mocker.stub()
        optimizer.param_groups = [{'lr': 0}]
        num_steps_in_epoch = 10
        params = LR_SCHEDULER_PARAMS
        lr_scheduler = GlobalLRScheduler(optimizer, num_steps_in_epoch, **params)
        lr_scheduler.epoch_step(0)
        lr_scheduler.step()
        warmup_lr.assert_called()

    def test_adjust_lr(self, mocker):
        adjust_lr = mocker.patch(
            'nncf.experimental.torch.nas.bootstrapNAS.training.lr_scheduler.adjust_learning_rate')
        optimizer = mocker.stub()
        optimizer.param_groups = [{'lr': 0}]
        num_steps_in_epoch = 10
        params = LR_SCHEDULER_PARAMS
        params['warmup_epochs'] = 0
        lr_scheduler = GlobalLRScheduler(optimizer, num_steps_in_epoch, **params)
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
        num_steps_in_epoch = 10
        params = LR_SCHEDULER_PARAMS
        params['warmup_epochs'] = 1
        lr_scheduler = GlobalLRScheduler(optimizer, num_steps_in_epoch, **params)
        lr_scheduler.epoch_step(0)
        lr_scheduler.step()
        warmup_lr.assert_called()
        lr_scheduler.epoch_step()
        lr_scheduler.step()
        adjust_lr.assert_called()

    def test_reset(self, mocker):
        optimizer = mocker.stub()
        optimizer.param_groups = [{'lr': 0}]
        params = LR_SCHEDULER_PARAMS
        lr_scheduler = GlobalLRScheduler(optimizer, 10, **params)
        assert lr_scheduler.current_epoch == -1
        assert lr_scheduler.current_step == -1
        lr_scheduler.epoch_step()
        lr_scheduler.step()
        assert lr_scheduler.current_epoch == 0
        assert lr_scheduler.current_step == 0
        lr_scheduler.epoch_step()
        lr_scheduler.step()
        assert lr_scheduler.epoch_step(1)
        assert lr_scheduler.step(1)
        lr_scheduler.reset(LR_SCHEDULER_PARAMS['base_lr'], LR_SCHEDULER_PARAMS['num_epochs'])
        assert lr_scheduler.epoch_step(0)
        assert lr_scheduler.step(0)