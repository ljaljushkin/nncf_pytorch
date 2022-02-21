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

import csv

def get_flops_for_active_subnet(elasticity_handler):
    flops, _ = elasticity_handler.count_flops_and_weights_for_active_subnet()
    return flops /2000000   # MACs

def get_weights_for_active_subnet(elasticity_handler):
    _, num_weights = elasticity_handler.count_flops_and_weights_for_active_subnet()
    return num_weights

class Evaluator:
    def __init__(self, name, eval_func, ideal_val, elasticity_ctrl):
        print(name, eval_func)
        self.name = name
        self._eval_func = eval_func
        self._curr_value = 0
        self._ideal_value = ideal_val
        self._elasticity_ctrl = elasticity_ctrl
        self.use_model_for_evaluation = True
        self.cache = {}
        self.input_model_value = None
        #TODO(pablo): Here we should store some super-network signature that is associated with this evaluator

    def evaluate_model(self, model):
        return self._eval_func(model)

    def evaluate_with_elasticity_handler(self):
        if self.use_model_for_evaluation:
            raise RuntimeError("Evaluator set to evaluate with model but elasticity handler was requested.")
        return self._eval_func(self._elasticity_ctrl.multi_elasticity_handler)

    def add_to_cache(self, subnet_config_repr, measurement: float):
        print("add", subnet_config_repr, measurement)
        self.cache[subnet_config_repr] = measurement

    def retrieve_from_cache(self, subnet_config_repr):
        if subnet_config_repr in self.cache.keys():
            return True, self.cache[subnet_config_repr]
        return False, False

    def get_state(self): # Returns a dictionary with Python data structures
        raise NotImplementedError

    def from_state(self, eval_state):
        raise NotImplementedError

    def load_cache_from_csv(self, cache_file_path):
        with open(f"{cache_file_path}", 'r') as cache_file:
            reader = csv.reader(cache_file)
            for row in reader:
                print(type(row[0]), row[0], row[1])
                rep_tuple = tuple(map(int, row[0][1:len(row[0])-1].split(',')))
                print(type(rep_tuple), rep_tuple)
                self.add_to_cache(rep_tuple, float(row[1]))

    def export_cache_to_csv(self, cache_file_path):
        with open(f'{cache_file_path}/cache_{self.name}.csv', 'w') as cache_dump:
            writer = csv.writer(cache_dump)
            for key in self.cache:
                row = [key, self.cache[key]]
                writer.writerow(row)


class AccuracyEvaluator(Evaluator):
    def __init__(self, eval_func, val_loader, is_top1=True):
        if is_top1:
            name = "top1_acc"
        super(AccuracyEvaluator, self).__init__(name, eval_func, 100, None)
        self._val_loader = val_loader
        self._use_model_for_evaluation = True
        self._ideal_value = 100

    def evaluate_model(self, model):
        return self._eval_func(model, self._val_loader)
