"""
 Copyright (c) 2020 Intel Corporation
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
import json
from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Optional
from typing import TypeVar
from typing import Union

import tensorflow as tf

from beta.nncf.tensorflow.graph.model_transformer import TFModelTransformer
from nncf.api.compression import CompressionAlgorithmBuilder
from nncf.api.compression import CompressionSetup
from nncf.api.compression import CompressionState
from nncf.common.compression import BaseCompressionAlgorithmController

ModelType = TypeVar('ModelType')
DatasetType = TypeVar('DatasetType')
LossType = TypeVar('LossType')


class TFCompressionAlgorithmInitializer(ABC):
    @abstractmethod
    def call(self,
             model: ModelType,
             dataset: Optional[DatasetType] = None,
             loss: Optional[LossType] = None) -> None:
        """
        Initializes minimum and maximum quantization ranges.
        """

    def __call__(self, *args, **kwargs) -> None:
        self.call(*args, **kwargs)


class TFCompressionState(CompressionState, tf.train.experimental.PythonState):
    def __init__(self, compression_setups: List[CompressionSetup]):
        super().__init__(compression_setups)

    def serialize(self) -> str:
        """
        Callback to serialize the object by tf.train.experimental.PythonState.
        """
        json_compatible_setups = self.get_state()
        string_value = json.dumps(json_compatible_setups)
        return string_value

    def deserialize(self, string_value: str) -> None:
        """
        Callback to deserialize the object by tf.train.experimental.PythonState.
        """
        json_compatible_setups = json.loads(string_value)
        self.load_state(json_compatible_setups)


class TFCompressionAlgorithmController(BaseCompressionAlgorithmController, tf.train.experimental.PythonState):
    """
    Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model to enable algorithm-specific compression.
    Hosts entities that are to be used during the training process, such as
    compression scheduler and compression loss.
    """

    def initialize(self,
                   dataset: Optional[DatasetType] = None,
                   loss: Optional[LossType] = None) -> None:
        pass

    def load_state(self, state: Dict[str, object]) -> None:
        """
        Loads the compression controller state.

        :param state: Output of `get_state()` method.
        """
        self.scheduler.load_state(state['scheduler_state'])
        self.loss.load_state(state['loss_state'])

    def get_state(self) -> Dict[str, object]:
        """
        Returns the compression controller state.

        :return: The compression controller state.
        """
        return {
            'scheduler_state': self.scheduler.get_state(),
            'loss_state': self.loss.get_state()
        }

    def get_compression_state(self) -> Union[CompressionState, Dict]:
        # TODO(nlyalyus) add support for TF
        return {}

    def serialize(self) -> str:
        """
        Callback to serialize the object by tf.train.experimental.PythonState.

        :return: State of the compression controller.
        """
        string_value = json.dumps(self.get_state())
        return string_value

    def deserialize(self, state: str) -> None:
        """
        Callback to deserialize the object by tf.train.experimental.PythonState.

        :param state: State of the compression controller.
        """
        state = json.loads(state)
        self.load_state(state)


class TFCompressionAlgorithmBuilder(CompressionAlgorithmBuilder):
    """
    Determines which modifications should be made to the original model in
    order to enable algorithm-specific compression during fine-tuning.
    """

    def get_state(self) -> Dict[str, object]:
        """
        Returns a JSON-compatible dictionary containing a state of the object.
        """
        return {}

    def load_state(self, state: Dict[str, object]):
        """
        Initializes object from the state.
        :param state: Output of `get_state()` method.
        """

    def apply_to(self, model: ModelType) -> ModelType:
        """
        Applies algorithm-specific modifications to the model.

        :param model: The original uncompressed model.
        :return: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        """
        transformation_layout = self.get_transformation_layout(model)
        transformer = TFModelTransformer(model, transformation_layout)
        transformed_model = transformer.transform()
        return transformed_model
