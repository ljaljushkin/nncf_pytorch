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
from typing import Any, Tuple, NoReturn
from typing import Callable
from typing import Dict
import csv

from nncf.common.utils.logger import logger as nncf_logger
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_controller import ElasticityController
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import MultiElasticityHandler
from nncf.torch.nncf_network import NNCFNetwork

class BNASEvaluatorStateNames:
    BNAS_EVALUATOR_STAGE = 'evaluator_state'

def get_flops_for_active_subnet(elasticity_handler: MultiElasticityHandler) -> float:
    flops, _ = elasticity_handler.count_flops_and_weights_for_active_subnet()
    return flops /2000000   # MACs

def get_weights_for_active_subnet(elasticity_handler: MultiElasticityHandler) -> float:
    _, num_weights = elasticity_handler.count_flops_and_weights_for_active_subnet()
    return num_weights

class Evaluator:
    """
    An interface for handling measurements collected on a target device. Evaluators make use of functions provided by the users to measure a particular property, e.g., accuracy, latency, etc.
    """
    def __init__(self, name: str, eval_func: Callable, ideal_val: float, elasticity_ctrl: ElasticityController):
        self.name = name
        self._eval_func = eval_func
        self._curr_value = 0
        self._ideal_value = ideal_val
        self._elasticity_ctrl = elasticity_ctrl
        self._use_model_for_evaluation = True
        self.cache = {}
        self.input_model_value = None
        #TODO(pablo): Here we should store some super-network signature that is associated with this evaluator

    def evaluate_from_pymoo(self, model: NNCFNetwork, pymoo_repr):
        if self._use_model_for_evaluation:
            value = self.evaluate_model(model)
        else:
            value = self.evaluate_with_elasticity_handler()
        self.add_to_cache(pymoo_repr, value)
        return value

    def evaluate_model(self, model: NNCFNetwork) -> float:
        self._curr_value = self._eval_func(model)
        return self._curr_value

    def evaluate_with_elasticity_handler(self) -> Tuple[float, ...]:
        if self._use_model_for_evaluation:
            raise RuntimeError("Evaluator set to evaluate with model but elasticity handler was requested.")
        self._curr_value = self._eval_func(self._elasticity_ctrl.multi_elasticity_handler)
        return self._curr_value

    def add_to_cache(self, subnet_config_repr, measurement: float) -> NoReturn:
        nncf_logger.info(f"Add to evaluator {self.name}: {subnet_config_repr}, {measurement}")
        self.cache[subnet_config_repr] = measurement

    def retrieve_from_cache(self, subnet_config_repr: Tuple[float, ...]) -> Tuple[bool, float]:
        if subnet_config_repr in self.cache.keys():
            return True, self.cache[subnet_config_repr]
        return False, 0

    def get_state(self) -> Dict[str, Any]:
        state_dict = {
            'name': self.name,
            'eval_func': self._eval_func,
            'curr_value': self._curr_value,
            'ideal_value': self._ideal_value,
            'elasticity_controller_compression_state': self.elasticity_ctrl.get_state(),
            'use_model_for_evaluation': self._use_model_for_evaluation,
            'cache': self.cache,
            'input_model_value': self.input_model_value
        }
        return state_dict

    @classmethod
    def from_state(cls, state: Dict[str, Any], elasticity_ctrl: ElasticityController) -> 'Evaluator':
        new_dict = state.copy()
        evaluator = cls(new_dict['name'], new_dict['eval_func'], new_dict['ideal_val'], elasticity_ctrl)
        evaluator._curr_value = new_dict['curr_value']
        evaluator._use_model_for_evaluation = new_dict['use_model_for_evaluation']
        evaluator.cache = new_dict['cache']
        evaluator.input_model_value = new_dict['input_model_value']
        return evaluator

    def load_cache_from_csv(self, cache_file_path: str) -> NoReturn:
        with open(f"{cache_file_path}", 'r') as cache_file:
            reader = csv.reader(cache_file)
            for row in reader:
                rep_tuple = tuple(map(int, row[0][1:len(row[0])-1].split(',')))
                self.add_to_cache(rep_tuple, float(row[1]))

    def export_cache_to_csv(self, cache_file_path: str) -> NoReturn:
        with open(f'{cache_file_path}/cache_{self.name}.csv', 'w') as cache_dump:
            writer = csv.writer(cache_dump)
            for key in self.cache:
                row = [key, self.cache[key]]
                writer.writerow(row)


class AccuracyEvaluator(Evaluator):
    """
    A particular kind of evaluator n interface for collecting model's accuracy measurements
    """

    def __init__(self, eval_func, val_loader, is_top1=True, ref_acc=100):
        if is_top1:
            name = "top1_acc"
        super(AccuracyEvaluator, self).__init__(name, eval_func, 100, None)
        self._is_top1 = is_top1
        self._val_loader = val_loader
        self._use_model_for_evaluation = True
        self._ideal_value = 100
        self._ref_acc = ref_acc

    def evaluate_model(self, model: NNCFNetwork) -> Tuple[float, ...]:
        self._curr_value = self._eval_func(model, self._val_loader) * -1.0
        return self._curr_value

    def evaluate_from_pymoo(self, model: NNCFNetwork, pymoo_repr):
        value = self.evaluate_model(model)
        self._curr_value = value
        self.add_to_cache(pymoo_repr, value)
        return value

    def get_state(self) -> Dict[str, Any]:
        state = super.get_state()
        state['is_top1'] = self._is_top1
        state['ref_acc'] = self._ref_acc
        return state

    @classmethod
    def from_state(cls, state: Dict[str, Any], val_loader) -> 'AccuracyEvaluator':
        new_dict = state.copy()
        evaluator = cls(new_dict['eval_func'], val_loader, new_dict['is_top1'], new_dict['ref_acc'])
        evaluator._curr_value = new_dict['curr_value']
        evaluator._use_model_for_evaluation = new_dict['use_model_for_evaluation']
        evaluator.cache = new_dict['cache']
        evaluator.input_model_value = new_dict['input_model_value']
        return evaluator