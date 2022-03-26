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
from typing import Callable
from typing import Dict
from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import TypeVar
import csv

from nncf.common.utils.logger import logger as nncf_logger
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_controller import ElasticityController
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import MultiElasticityHandler
from nncf.torch.nncf_network import NNCFNetwork

DataLoaderType = TypeVar('DataLoaderType')
ModelType = TypeVar('ModelType')
ValFnType = Callable[
    [
        ModelType,
        DataLoaderType
    ],
    float
]

class BNASEvaluatorStateNames:
    BNAS_EVALUATOR_STAGE = 'evaluator_state'

def get_macs_for_active_subnet(elasticity_handler: MultiElasticityHandler) -> float:
    """
    Gets the MACs for the active sub-network

    :param elasticity_handler: Interface for handling super-network elasticity.
    :return: Multiply-accumulate operations for active sub-network
    """
    flops, _ = elasticity_handler.count_flops_and_weights_for_active_subnet()
    return flops/2000000   # MACs

def get_weights_for_active_subnet(elasticity_handler: MultiElasticityHandler) -> float:
    """
    Gets the number of weights in the active sub-network for convolution and fully connected layers.

    :param elasticity_handler: Interface for handling super-network elasticity.
    :return: Number of weights for active sub-network
    """
    _, num_weights = elasticity_handler.count_flops_and_weights_for_active_subnet()
    return num_weights

class Evaluator:
    """
    An interface for handling measurements collected on a target device. Evaluators make use
    of functions provided by the users to measure a particular property, e.g., accuracy, latency, etc.
    """
    def __init__(self, name: str, eval_func: ValFnType, ideal_val: float, elasticity_ctrl: ElasticityController):
        """
        Initializes evaluator

        :param name: Name of the evaluator
        :param eval_func: Function used to obtain the associated metric from the sub-network
        :param ideal_val: Ideal value for the metric computed by the evaluator
        :param elasticity_ctrl: Interface for handling super-network elasticity.
        """
        self.name = name
        self._eval_func = eval_func
        self._current_value = 0
        self._ideal_value = ideal_val
        self._elasticity_ctrl = elasticity_ctrl
        self._use_model_for_evaluation = True
        self.cache = {}
        self.input_model_value = None
        #TODO(pablo): Here we should store some super-network signature that is associated with this evaluator

    @property
    def current_value(self):
        """
        :return: current value
        """
        return self._current_value

    @current_value.setter
    def current_value(self, val: float) -> NoReturn:
        """
        :param val: value to update the current value of the evaluator
        :return:
        """
        self._current_value = val

    @property
    def use_model_for_evaluation(self):
        """
        :return: whether the model (or the elasticity controlled) is used for evaluation.
        """
        return self._use_model_for_evaluation

    @use_model_for_evaluation.setter
    def use_model_for_evaluation(self, val: bool) -> NoReturn:
        """
        :param val: set whether model or elasticity controller is used for evaluation.
        :return:
        """
        self._use_model_for_evaluation = val

    def evaluate_from_pymoo(self, model: NNCFNetwork, pymoo_repr: Tuple[float, ...]):
        """
        Evaluates active sub-network and uses Pymoo representation for insertion in cache.

        :param model: Active sub-network
        :param pymoo_repr: tuple representing the associated values for the design variables
                            in Pymoo.
        :return: the value obtained from the model evaluation.
        """
        if self._use_model_for_evaluation:
            value = self.evaluate_model(model)
        else:
            value = self.evaluate_with_elasticity_handler()
        self.add_to_cache(pymoo_repr, value)
        return value

    def evaluate_model(self, model: NNCFNetwork) -> float:
        """
        Evaluates metric using model

        :param model: Active sub-network
        :return: value obtained from evaluation.
        """
        self._curr_value = self._eval_func(model)
        return self._curr_value

    def evaluate_with_elasticity_handler(self) -> Tuple[float, ...]:
        """
        Evaluates metric for active sub-network using elasticity handler

        :return: value obtained from evaluation
        """
        if self._use_model_for_evaluation:
            raise RuntimeError("Evaluator set to evaluate with model but elasticity handler was requested.")
        self._curr_value = self._eval_func(self._elasticity_ctrl.multi_elasticity_handler)
        return self._curr_value

    def add_to_cache(self, subnet_config_repr: Tuple[float, ...], measurement: float) -> NoReturn:
        """
        Adds evaluation result to cache

        :param subnet_config_repr: tuple containing the values for the associated design variables.
        :param measurement: value for the evaluator's metric.
        :return:
        """
        nncf_logger.info("Add to evaluator {name}: {subnet_config_repr}, {measurement}".format(
                         name=self.name, subnet_config_repr=subnet_config_repr, measurement=measurement))
        self.cache[subnet_config_repr] = measurement

    def retrieve_from_cache(self, subnet_config_repr: Tuple[float, ...]) -> Tuple[bool, float]:
        """
        Checks if sub-network info is in cache and returns the corresponding value.
        :param subnet_config_repr: tuple representing the values for the associated design variables.
        :return: (True if the information is in cache, and corresponding value stored in cache, 0 otherwise)
        """
        if subnet_config_repr in self.cache.keys():
            return True, self.cache[subnet_config_repr]
        return False, 0

    def get_state(self) -> Dict[str, Any]:
        """
        Returns state of the evaluatar

        :return: Dict with the state of the evaluator
        """
        state_dict = {
            'name': self.name,
            'eval_func': self._eval_func,
            'curr_value': self._curr_value,
            'ideal_value': self._ideal_value,
            'elasticity_controller_compression_state': self._elasticity_ctrl.get_state(),
            'use_model_for_evaluation': self._use_model_for_evaluation,
            'cache': self.cache,
            'input_model_value': self.input_model_value
        }
        return state_dict

    @classmethod
    def from_state(cls, state: Dict[str, Any], elasticity_ctrl: ElasticityController) -> 'Evaluator':
        """
        Constructs evaluator from existing state information.

        :param state: Dictionary with information to create evaluator
        :param elasticity_ctrl: Interface for handling super-network elasticity.
        :return:
        """
        new_dict = state.copy()
        evaluator = cls(new_dict['name'], new_dict['eval_func'], new_dict['ideal_val'], elasticity_ctrl)
        evaluator._curr_value = new_dict['curr_value']
        evaluator._use_model_for_evaluation = new_dict['use_model_for_evaluation']
        evaluator.cache = new_dict['cache']
        evaluator.input_model_value = new_dict['input_model_value']
        return evaluator

    def load_cache_from_csv(self, cache_file_path: str) -> NoReturn:
        """
        Loads cache from CSV file.

        :param cache_file_path: Path to CSV file containing the cache information.
        :return:
        """
        with open(f"{cache_file_path}", 'r', encoding='utf8') as cache_file:
            reader = csv.reader(cache_file)
            for row in reader:
                rep_tuple = tuple(map(int, row[0][1:len(row[0])-1].split(',')))
                self.add_to_cache(rep_tuple, float(row[1]))

    def export_cache_to_csv(self, cache_file_path: str) -> NoReturn:
        """
        Exports cache information to CSV.

        :param cache_file_path: Path to export a CSV file with the cache information.
        :return:
        """
        with open(f'{cache_file_path}/cache_{self.name}.csv', 'w', encoding='utf8') as cache_dump:
            writer = csv.writer(cache_dump)
            for key in self.cache:
                row = [key, self.cache[key]]
                writer.writerow(row)


class AccuracyEvaluator(Evaluator):
    """
    A particular kind of evaluator for collecting model's accuracy measurements
    """

    def __init__(self, eval_func: ValFnType, val_loader: DataLoaderType, is_top1: Optional[bool] = True, ref_acc: Optional[float] = 100):
        """
        Initializes Accuracy operator

        :param eval_func: function used to validate a sub-network
        :param val_loader: Datq loader used by the validation function
        :param is_top1: Whether is top 1 accuracy or top 5.
        :param ref_acc: Accuracy from a model that is used as input to BootstrapNAS
        """
        if is_top1:
            name = "top1_acc"
        super().__init__(name, eval_func, 100, None)
        self._is_top1 = is_top1
        self._val_loader = val_loader
        self._use_model_for_evaluation = True
        self._ideal_value = 100
        self._ref_acc = ref_acc

    @property
    def ref_acc(self) -> float:
        """
        :return: reference accuracy
        """
        return self._ref_acc

    @ref_acc.setter
    def ref_acc(self, val: float) -> NoReturn:
        """
        :param val: value to update the reference accuracy value.
        :return:
        """
        self._ref_acc = val

    def evaluate_model(self, model: NNCFNetwork) -> float:
        """
        Obtain accuracy from evaluating the model.
        :param model: Active sub-network
        :return: accuracy from active sub-network.
        """
        self._curr_value = self._eval_func(model, self._val_loader) * -1.0
        return self._curr_value

    def evaluate_from_pymoo(self, model: NNCFNetwork, pymoo_repr: Tuple[float, ...]) -> NoReturn:
        """
        Obtain accuracy for active sub-network and stores value to cache.

        :param model: Active sub-network
        :param pymoo_repr: Sub-network represented by the values of the design variables of
        the search algorithm.
        :return:
        """
        value = self.evaluate_model(model)
        self._curr_value = value
        self.add_to_cache(pymoo_repr, value)
        return value

    def get_state(self) -> Dict[str, Any]:
        """
        Get state of Accuracy evaluator.

        :return: Dict with state of evaluator
        """
        state = super.get_state()
        state['is_top1'] = self._is_top1
        state['ref_acc'] = self._ref_acc
        return state

    @classmethod
    def from_state(cls, state: Dict[str, Any], val_loader) -> 'AccuracyEvaluator':
        """

        :param state: dict with state that should be used for resuming this evaluator
        :param val_loader: Data loader used when evaluating the model.
        :return: An instance of the Accuracy Evaluator.
        """
        new_dict = state.copy()
        evaluator = cls(new_dict['eval_func'], val_loader, new_dict['is_top1'], new_dict['ref_acc'])
        evaluator._curr_value = new_dict['curr_value']
        evaluator._use_model_for_evaluation = new_dict['use_model_for_evaluation']
        evaluator.cache = new_dict['cache']
        evaluator.input_model_value = new_dict['input_model_value']
        return evaluator
