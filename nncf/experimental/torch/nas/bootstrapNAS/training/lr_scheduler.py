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

import math
from typing import Any
from typing import Dict
from typing import NoReturn
from typing import Optional

from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.utils.logger import logger as nncf_logger
from nncf.experimental.torch.nas.bootstrapNAS.training.stage_descriptor import StageDescriptor


def adjust_learning_rate(optimizer, epoch, init_lr, epochs, batch=0, nBatch=None, lr_schedule_type='cosine'):
    new_lr = calc_learning_rate(epoch, init_lr, epochs, batch, nBatch, lr_schedule_type)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


def warmup_adjust_learning_rate(optimizer, init_lr, T_total, nBatch, epoch, batch=0, warmup_lr=0):
    T_cur = epoch * nBatch + batch + 1
    new_lr = T_cur / T_total * (init_lr - warmup_lr) + warmup_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


def calc_learning_rate(epoch, init_lr, n_epochs, batch=0, nBatch=None, lr_schedule_type='cosine'):
    if lr_schedule_type == 'cosine':
        t_total = n_epochs * nBatch
        t_cur = epoch * nBatch + batch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
    elif lr_schedule_type is None:
        lr = init_lr
    else:
        raise ValueError('do not support: %s' % lr_schedule_type)
    return lr


class BaseLRScheduler(BaseCompressionScheduler):
    def __init__(self, optimizer, num_steps_in_epoch, *, base_lr, num_epochs, warmup_epochs=0, warmup_lr=3.4e-4):
        super().__init__()
        self._base_lr = base_lr
        self._num_epochs = num_epochs
        self._warmup_epochs = warmup_epochs
        self._warmup_lr = warmup_lr
        self._optimizer = optimizer
        self._num_steps_in_epoch = num_steps_in_epoch

    def step(self, next_step: Optional[int] = None) -> None:
        super().step(next_step)
        step_from_epoch_start = self.current_step - (self.current_epoch*(self._num_steps_in_epoch+1))
        if self.current_epoch < self._warmup_epochs and self.current_epoch != -1:
            warmup_adjust_learning_rate(optimizer=self._optimizer,
                                                 init_lr=self._base_lr,
                                                 T_total=self._warmup_epochs * self._num_steps_in_epoch,
                                                 nBatch=self._num_steps_in_epoch,
                                                 epoch=self.current_epoch,
                                                 batch=step_from_epoch_start,
                                                 warmup_lr=self._warmup_lr)
        else:
            adjust_learning_rate(optimizer=self._optimizer,
                                          epoch=self.current_epoch - self._warmup_epochs,
                                          init_lr=self._base_lr,
                                          epochs=self._num_epochs,
                                          batch=step_from_epoch_start,
                                          nBatch=self._num_steps_in_epoch,
                                          lr_schedule_type='cosine')

    def stage_step(self, stage_desc: StageDescriptor):
        pass

    @classmethod
    def from_state(cls, state: Dict[str, Any], optimizer):
        return cls(optimizer, **state)

    def get_state(self) -> Dict[str, Any]:
        state_dict = {
            'num_steps_in_epoch': self._num_steps_in_epoch,
            'base_lr': self._base_lr,
            'num_epochs': self._num_epochs,
            'warmup_epochs': self._warmup_epochs,
            'warmup_lr': self._warmup_lr
        }
        return state_dict


class GlobalLRScheduler(BaseLRScheduler):
    def __init__(self, optimizer, num_steps_in_epoch, *, base_lr, num_epochs, warmup_epochs=0, warmup_lr=3.4e-4):
        super().__init__(optimizer, num_steps_in_epoch, **{'base_lr': base_lr, 'num_epochs': num_epochs, 'warmup_epochs': warmup_epochs, 'warmup_lr': warmup_lr})

    def stage_step(self, stage_desc: StageDescriptor):
        # do nothing
        pass


class StageLRScheduler(BaseLRScheduler):
    def __init__(self, optimizer, num_steps_in_epoch):
        super().__init__(optimizer, num_steps_in_epoch, **{'base_lr': None, 'num_epochs': None})

    def stage_step(self, stage_desc: StageDescriptor):
        self.reset(stage_desc.init_lr, stage_desc.epochs_lr)

    def reset(self, base_lr, num_epochs):
        self._num_epochs = num_epochs
        self._base_lr = base_lr
        self.epoch_step(0)
        self.step(0)
