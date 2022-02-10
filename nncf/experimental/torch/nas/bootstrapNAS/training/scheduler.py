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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.utils.logger import logger as nncf_logger
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.nas.bootstrapNAS.training.base_training import BNASTrainingAlgorithm
from nncf.experimental.torch.nas.bootstrapNAS.training.cosine_lr_scheduler import CosineLRScheduler
from nncf.experimental.torch.nas.bootstrapNAS.training.stage_descriptor import DEFAULT_STAGE_LR_RATE
from nncf.experimental.torch.nas.bootstrapNAS.training.stage_descriptor import StageDescriptor


class BNASSchedulerStateNames:
    LIST_STAGE_DESCRIPTIONS = 'list_stage_descriptions'


class BootstrapNASScheduler(BaseCompressionScheduler):
    """
    The cornerstone of supernet training within a NAS algorithm. The `step()` and `epoch_step()` methods of the
    compression scheduler must be called in the beginning of each training step and epoch, respectively.
    These methods trigger a subnet activations, elasticity configuration during the training.
    """
    _state_names = BNASSchedulerStateNames

    def __init__(self, training_ctrl: BNASTrainingAlgorithm,
                 params: Dict[str, List[Dict]],
                 available_elasticity_dims: List[ElasticityDim],
                 progressivity_of_elasticity: List[ElasticityDim]):
        super().__init__()
        self._training_ctrl = training_ctrl
        self._params = params if params else self._get_default_params()
        self._available_elasticity_dims = available_elasticity_dims
        self._progressivity_of_elasticity = progressivity_of_elasticity

        list_stage_descriptions = self._params.get('list_stage_descriptions', [])
        self.current_stage_idx = -1
        # Property setter with validation is not used intentionally for the resume case. When the actual list stage
        #  descriptors are loaded after creation of the scheduler. Scheduler is resumed without config = with empty
        #  params = default stage descriptors, that could lead to inconsistency with progressivity and enabled dims.
        #  The validation will happen in the first usage of list_stage_descriptors property.
        self._list_stage_descriptors = [StageDescriptor.from_state(d) for d in list_stage_descriptions]
        self._is_elasticity_dims_validated = False
        self._global_lr_scheduler = None
        self._stage_lr_scheduler = None

    def set_global_lr_scheduler(self, lr_scheduler: CosineLRScheduler):
        self._global_lr_scheduler = lr_scheduler

    def set_stage_lr_scheduler(self, lr_scheduler: CosineLRScheduler):
        self._stage_lr_scheduler = lr_scheduler

    @property
    def list_stage_descriptors(self) -> List[StageDescriptor]:
        """
        :return: a list of stage descriptors (parameters of the training stage).
        """
        if not self._is_elasticity_dims_validated:
            self._validate_elasticity_dims(self._available_elasticity_dims, self._progressivity_of_elasticity)
        self._is_elasticity_dims_validated = True
        self._validate_lr()
        return self._list_stage_descriptors

    @list_stage_descriptors.setter
    def list_stage_descriptors(self, stage_descriptors: List[StageDescriptor]) -> None:
        """
        Sets a given stage descriptors to the schedule. Can be used on loading state from a checkpoint.

        :param stage_descriptors: list of stage descriptors
        """
        self._list_stage_descriptors = stage_descriptors
        self._validate_elasticity_dims(self._available_elasticity_dims, self._progressivity_of_elasticity)
        self._is_elasticity_dims_validated = True
        self._validate_lr()

    def step(self, next_step: Optional[int] = None) -> None:
        """
        Should be called at the beginning of each training step to prepare
        the compression method to continue training the model in the `next_step`.

        :param next_step: The global step index for which the compression scheduler
            will update the state of the compression method.
        """
        self._training_ctrl.step()
        if self._global_lr_scheduler is not None:
            self._global_lr_scheduler.step(next_step)
        else:
            self._stage_lr_scheduler.step(next_step)
        # self._lr_scheduler.step(next_step)

    def epoch_step(self, next_epoch: Optional[int] = None) -> None:
        """
        Should be called at the beginning of each training epoch to prepare
        the compression method to continue training the model in the `next_epoch`.

        :param next_epoch: The epoch index for which the compression scheduler
            will update the state of the compression method.
        """
        super().epoch_step(next_epoch)
        if self._global_lr_scheduler is not None:
            self._global_lr_scheduler.epoch_step(next_epoch)
        else:
            self._stage_lr_scheduler.epoch_step(next_epoch)
        stage_desc, stage_desc_idx = self.get_current_stage_desc()
        if stage_desc is not None:
            if stage_desc_idx != self.current_stage_idx:
                if self._global_lr_scheduler is None:
                    self._stage_lr_scheduler.reset(stage_desc.init_lr, stage_desc.epochs_lr)
                self._training_ctrl.set_stage(stage_desc)
                self.current_stage_idx = stage_desc_idx

    def is_final_stage(self) -> bool:
        """
        :return: True, if final stage has been reached, False - otherwise
        """
        return self.current_stage_idx == len(self.list_stage_descriptors) - 1

    def get_current_stage_desc(self) -> Tuple[Optional[StageDescriptor], int]:
        """
        :return: current stage descriptor and its index in the list of all descriptors
        """
        partial_epochs = 0
        stage_desc_idx = 0
        for stage_desc in self.list_stage_descriptors:
            partial_epochs += stage_desc.epochs
            if self.current_epoch < partial_epochs:
                return stage_desc, stage_desc_idx
            stage_desc_idx += 1
        return None, -1

    def get_total_training_epochs(self) -> int:
        """
        Returns total number of epochs required for the supernet training.

        :return: number of epochs
        """
        total_epochs = 0
        for stage_desc in self.list_stage_descriptors:
            total_epochs += stage_desc.epochs
        return total_epochs

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Loads the compression scheduler state, but does not update the state of the
        compression method.

        :param state: Output of `get_state()` method.
        """
        super().load_state(state)
        list_stage_descriptors = state[self._state_names.LIST_STAGE_DESCRIPTIONS]
        # No conflict resolving with the related config options, parameters are overridden by compression state
        self.list_stage_descriptors = list(map(StageDescriptor.from_state, list_stage_descriptors))

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the compression scheduler state.

        :return: The compression scheduler state.
        """
        state = super().get_state()
        state[self._state_names.LIST_STAGE_DESCRIPTIONS] = [desc.get_state() for desc in self.list_stage_descriptors]
        return state

    def _validate_elasticity_dims(self, available_elasticity_dims, progressivity_of_elasticity):
        last_stage = -1
        first_stage = len(progressivity_of_elasticity)
        for desc in self._list_stage_descriptors:
            high_priority_dim_idx = -1
            low_priority_dim_idx = len(progressivity_of_elasticity)
            stages_covered = []
            for train_dim in desc.train_dims:
                if train_dim not in available_elasticity_dims:
                    raise ValueError(
                        f"Invalid training elasticity dimension {train_dim} in the scheduler.\n"
                        f"The elasticity for this dimension is not enabled.\n"
                        f"It can be enabled by specifying `available_elasticity_dims` param in the `elasticity` "
                        f"section of config.\n"
                        f"List of currently available dimensions: {[dim.value for dim in available_elasticity_dims]}")
                dim_idx = progressivity_of_elasticity.index(train_dim)
                if dim_idx not in stages_covered:
                    stages_covered.append(dim_idx)
                if dim_idx > high_priority_dim_idx:
                    high_priority_dim_idx = dim_idx
                if dim_idx < low_priority_dim_idx:
                    low_priority_dim_idx = dim_idx
            if high_priority_dim_idx < last_stage or low_priority_dim_idx > first_stage:
                raise ValueError(
                    f"stage {progressivity_of_elasticity[high_priority_dim_idx]} violates progressivity of elasticity")
            for i in range(low_priority_dim_idx, high_priority_dim_idx):
                if i not in stages_covered and progressivity_of_elasticity[i] in available_elasticity_dims:
                    raise ValueError(
                        f"Missed to call {progressivity_of_elasticity[i]} in {desc.train_dims} which violates "
                        f"progressivity of elasticity {progressivity_of_elasticity}")
            last_stage = high_priority_dim_idx
            first_stage = low_priority_dim_idx

    def _validate_lr(self):
        for desc in self._list_stage_descriptors:
            # Check if global learning rate has been set
            if desc.init_lr is not None and bool(self._training_ctrl._lr_schedule_config):
                print(desc.init_lr, self._training_ctrl._lr_schedule_config)
                raise ValueError(
                    f"Global learning rate scheduler is in use. Cannot set stage learning rate: {desc.init_lr}"
                )
            # Check if stage learning rate has been set
            elif desc.init_lr is None and not bool(self._training_ctrl._lr_schedule_config):
                nncf_logger.warning(
                    "Stage learning rate in use but init_lr value for stage wasn't set. Using default value of 3.5e-6")
                desc.init_lr = DEFAULT_STAGE_LR_RATE

            if desc.init_lr is not None and desc.epochs_lr is None:
                nncf_logger.warning(
                    f"Stage learning rate in use but epochs_lr value for stage wasn't set. Using number of epochs for stage {desc.epochs}")
                desc.epochs_lr = desc.epochs

    @staticmethod
    def _get_default_params() -> Dict[str, List[Dict]]:
        # TODO(nlyalyus): Perform some studies to determine default params (ticket 76938)
        return {
            "list_stage_descriptions": [
                {"train_dims": ["kernel"], "epochs": 1},
                {"train_dims": ["kernel", "depth"], "epochs": 1},
                {"train_dims": ["kernel", "depth"], "epochs": 1},
                {"train_dims": ["kernel", "depth", "width"], "epochs": 1},
                {"train_dims": ["kernel", "depth", "width"], "epochs": 1, "reorg_weights": True, "bn_adapt": True}
            ]
        }
