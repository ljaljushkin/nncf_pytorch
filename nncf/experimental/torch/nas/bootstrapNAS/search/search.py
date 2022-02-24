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
from abc import abstractmethod

import autograd.numpy as anp
import numpy as np
from pathlib import Path
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.factory import get_crossover
from pymoo.factory import get_mutation
from pymoo.factory import get_sampling
from pymoo.optimize import minimize
import torch

from nncf.experimental.torch.nas.bootstrapNAS.search.evaluator import AccuracyEvaluator
from nncf.experimental.torch.nas.bootstrapNAS.search.evaluator import Evaluator
from nncf.experimental.torch.nas.bootstrapNAS.search.evaluator import get_flops_for_active_subnet
from nncf import NNCFConfig
from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.common.utils.logger import logger as nncf_logger
from nncf.config.extractors import get_bn_adapt_algo_kwargs
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_controller import ElasticityController
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import SubnetConfig
from nncf.torch.nncf_network import NNCFNetwork


class BaseSearchAlgorithm:
    """
    Base class for search algorithms. It contains the evaluators used by search approches.
    """
    def __init__(self):
        self._use_default_evaluators = True
        self._evaluators = None
        self._bad_requests = []

    @abstractmethod
    def run(self):
        pass


class SearchParams:
    """
    Storage class for search parameters.
    """
    def __init__(self, num_evals, num_contstraints, population,
                 seed, crossover_prob, crossover_eta,
                 mutation_prob, mutation_eta, acc_delta):
        self._num_evals = num_evals
        self._num_constraints = num_contstraints
        self._population = population
        if population > num_evals:
            raise ValueError("Population size must not be greater than number of evaluations.")
        self._seed = seed
        self._crossover_prob = crossover_prob
        self._crossover_eta = crossover_eta
        self._mutation_prob = mutation_prob
        self._mutation_eta = mutation_eta
        self._acc_delta = acc_delta

    @classmethod
    def from_dict(cls, search_config):
        num_evals = search_config.get('num_evals', 3000)
        num_constraints = search_config.get('num_constraints', 0)
        population = search_config.get('population', 40)
        seed = search_config.get('seed', 0)
        crossover_prob = search_config.get('crossover_prob', 0.9)
        crossover_eta = search_config.get('crossover_eta', 10.0)
        mutation_prob = search_config.get('mutation_prob', 0.02)
        mutation_eta = search_config.get('mutation_eta', 3.0)
        acc_delta = search_config.get('acc_delta', 1)

        return cls(num_evals, num_constraints, population,
                 seed, crossover_prob, crossover_eta,
                 mutation_prob, mutation_eta, acc_delta)

class SearchAlgorithm(BaseSearchAlgorithm):
    def __init__(self,
                 model: NNCFNetwork,
                 elasticity_ctrl: ElasticityController,
                 nncf_config: NNCFConfig,
                 verbose=True):
        super(SearchAlgorithm, self).__init__()
        self._model = model
        self._elasticity_ctrl = elasticity_ctrl
        self._elasticity_ctrl.multi_elasticity_handler.activate_maximum_subnet()
        search_config = nncf_config.get('bootstrapNAS', {}).get('search', {})
        self._num_obj = None
        self.search_params = SearchParams.from_dict(search_config)
        self._log_dir = nncf_config.get("log_dir", ".")
        self._tb = None
        self._verbose = verbose
        self._top1_accuracy_validation_fn = None
        self._val_loader = None
        evo_algo = search_config['algorithm']
        if evo_algo == 'NSGA2':
            self._algorithm = NSGA2(pop_size=self.search_params._population,
                                    sampling=get_sampling("int_lhs"),
                                    crossover=get_crossover("int_sbx", prob=self.search_params._crossover_prob,
                                                            eta=self.search_params._crossover_eta),
                                    mutation=get_mutation("int_pm", prob=self.search_params._mutation_prob, eta=self.search_params._mutation_eta),
                                    eliminate_duplicates=True,
                                    save_history=True,
                                    )
        else:
            raise NotImplementedError(f"Evolutionary Search Algorithm {evo_algo} not implemented")
        self._num_vars = 0
        self._vars_lower = 0
        self._vars_upper = []

        for dim in elasticity_ctrl.multi_elasticity_handler.get_available_elasticity_dims():
            if dim == ElasticityDim.KERNEL:
                self._kernel_search_space = self._elasticity_ctrl.multi_elasticity_handler.kernel_search_space
                self._num_vars += len(self._kernel_search_space)
                self._vars_upper += [len(self._kernel_search_space[i]) - 1 for i in
                                     range(len(self._kernel_search_space))]
            elif dim == ElasticityDim.WIDTH:
                self._width_search_space = self._elasticity_ctrl.multi_elasticity_handler.width_search_space
                self._num_vars += len(self._width_search_space)
                self._vars_upper += [len(self._width_search_space[i]) - 1 for i in range(len(self._width_search_space))]
            elif dim == ElasticityDim.DEPTH:
                self._valid_depth_configs = self._elasticity_ctrl.multi_elasticity_handler.depth_search_space
                if [] not in self._valid_depth_configs:
                    self._valid_depth_configs.append([])
                self._num_vars += 1
                self._vars_upper.append(len(self._valid_depth_configs) - 1)

        self._type_var = np.int
        self._result = None

        bn_adapt_params = nncf_config.get('compression', {}).get('initializer', {}).get('batchnorm_adaptation', {})
        bn_adapt_algo_kwargs = get_bn_adapt_algo_kwargs(nncf_config, bn_adapt_params)
        self._bn_adaptation = BatchnormAdaptationAlgorithm(**bn_adapt_algo_kwargs)
        self._search_records = []
        self._problem = None
        self._best_config = None
        self._best_vals = None
        self._ref_acc = None
        self._best_pair_objective = float('inf')
        self.checkpoint_save_dir = None

    @property
    def evaluators(self):
        if self._evaluators is not []:
            return self._evaluators
        else:
            raise RuntimeError("Evaluators haven't been defined")

    @property
    def acc_delta(self):
        return self.search_params._acc_delta

    @acc_delta.setter
    def acc_delta(self, val):
        if val > 50:
            val = 50
        if val > 5:
            nncf_logger.warning("Accuracy delta was set to a value greater than 5")
        self.search_params._acc_delta = val

    @property
    def vars_lower(self):
        """
        Gets access to design variables lower bounds.
        :return: lower bounds for design variables
        """
        return self._vars_lower

    @property
    def vars_upper(self):
        """
        Gets access to design variables upper bounds.
        :return: upper bounds for design variables
        """
        return self._vars_upper

    @property
    def num_vars(self):
        return self._num_vars

    def run(self, validate_fn, val_loader, checkpoint_save_dir, evaluators=None, ref_acc=100, tensorboard_writer=None):
        nncf_logger.info("Searching for optimal subnet.")
        self._ref_acc = ref_acc
        self._tb = tensorboard_writer
        self.checkpoint_save_dir = checkpoint_save_dir
        if evaluators is not None:
            self._use_default_evaluators = False
            self._num_obj = len(evaluators)
            self._evaluators = evaluators
        else:
            self._add_default_evaluators(validate_fn, val_loader)

        self.update_evaluators_for_input_model()
        self._problem = SearchProblem(self)
        self._result = minimize(self._problem, self._algorithm,
                                ('n_gen', int(self.search_params._num_evals / self.search_params._population)),
                                seed=self.search_params._seed,
                                # save_history=True,
                                verbose=self._verbose)

        if self._best_config is not None:
            self._elasticity_ctrl.multi_elasticity_handler.set_config(self._best_config)
            self._bn_adaptation.run(self._model)
        return self._elasticity_ctrl, self._best_config, self._best_vals

    def update_evaluators_for_input_model(self):
        self._elasticity_ctrl.multi_elasticity_handler.activate_maximum_subnet()
        for evaluator in self._evaluators:
            if evaluator.type_of_measurement == 'accuracy':
                value, _, _ = evaluator.evaluate_model(self._model)
                evaluator.input_model_value = value
            else:
                if evaluator.use_model_for_evaluation:
                    value = evaluator.evaluate_model(self._model)
                else:
                    value = evaluator.evaluate_with_elasticity_handler()
                evaluator.input_model_value = value

    def search_progression_to_csv(self):
        with open(f'{self._log_dir}/search_progression.csv', 'w') as progression:
            writer = csv.writer(progression)
            for record in self._search_records:
                writer.writerow(record)

    def evaluators_to_csv(self):
        for evaluator in self._evaluators:
            evaluator.export_cache_to_csv(self._log_dir)

    def _add_default_evaluators(self, validate_fn, val_loader):
        self._use_default_evaluators = True
        self._num_obj = 2
        self._evaluators = []
        flop_evaluator = Evaluator("flops", get_flops_for_active_subnet, 0, self._elasticity_ctrl)
        flop_evaluator.use_model_for_evaluation = False
        self._evaluators.append(flop_evaluator)
        self._evaluators.append(AccuracyEvaluator(validate_fn, val_loader))


class SearchProblem(Problem):
    def __init__(self, search):
        super().__init__(n_var=search.num_vars,
                         n_obj=search._num_obj,
                         n_constr=search.search_params._num_constraints,
                         xl=search.vars_lower,
                         xu=search.vars_upper,
                         type_var=search._type_var)
        self._search = search
        self._elasticity_handler = self._search._elasticity_ctrl.multi_elasticity_handler
        self._dims_enabled = self._elasticity_handler.get_available_elasticity_dims()
        self._best_dist_ideal = float('inf')
        self._iter = 0

    def _evaluate(self, x, out, *args, **kargs):
        evaluators_arr = [[] for i in range(len(self._search._evaluators))]

        for i in range(len(x)):
            sample = SubnetConfig()

            start_index = 0
            for dim in self._dims_enabled:
                if dim == ElasticityDim.KERNEL:
                    sample[dim] = [self._search._kernel_search_space[j - start_index][x[i][j]] for j in
                                   range(start_index, start_index + len(self._search._kernel_search_space))]
                    start_index += len(self._search._kernel_search_space)
                elif dim == ElasticityDim.WIDTH:
                    sample[dim] = {key - start_index: self._search._width_search_space[key - start_index][x[i][key]] for
                                   key in range(start_index, start_index + len(self._search._width_search_space))}
                    start_index += len(self._search._width_search_space)
                elif dim == ElasticityDim.DEPTH:
                    sample[dim] = self._search._valid_depth_configs[x[i][start_index]]
                    start_index += 1

            self._elasticity_handler.set_config(sample)

            if sample != self._elasticity_handler.get_active_config():
                nncf_logger.warning("Requested configuration was invalid")
                nncf_logger.warning(f"Requested: {sample}")
                nncf_logger.warning(f"Provided: {self._elasticity_handler.get_active_config()}")
                self._search._bad_requests.append((sample, self._elasticity_handler.get_active_config()))

            result = []
            for key in sample.keys():
                result.append(key.value)
                result.append(sample[key])

            eval_idx = 0
            bn_adaption_executed = False
            acc_within_tolerance = 0
            pair_objective = None
            in_cache = False
            for evaluator in self._search._evaluators:
                in_cache, value = evaluator.retrieve_from_cache(tuple(x[i]))
                if not bn_adaption_executed and not in_cache:
                    self._search._bn_adaptation.run(self._search._model)
                    bn_adaption_executed = True
                if evaluator.type_of_measurement == 'accuracy':
                    if not in_cache:
                        value, _, _ = evaluator.evaluate_model(self._search._model)
                        evaluator.add_to_cache(tuple(x[i]), value)
                    evaluators_arr[eval_idx].append(value * -1.0)
                    upper_bound = self._search._ref_acc + self._search.acc_delta
                    lower_bound = self._search._ref_acc - self._search.acc_delta
                    temp_acc = (value * -1.0) if value < 0 else value
                    if temp_acc < upper_bound and temp_acc > lower_bound:
                        acc_within_tolerance = temp_acc
                else:
                    if not in_cache:
                        if evaluator.use_model_for_evaluation:
                            value = evaluator.evaluate_model(self._search._model)
                        else:
                            value = evaluator.evaluate_with_elasticity_handler()
                        evaluator.add_to_cache(tuple(x[i]), value)
                    evaluators_arr[eval_idx].append(value)
                    pair_objective = value
                eval_idx += 1
                evaluator.curr_value = value
                result.append(evaluator.name)
                result.append(value)

            if acc_within_tolerance > 0:
                if pair_objective < self._search._best_pair_objective:
                    self._search._best_pair_objective = pair_objective
                    self._search._best_config = sample
                    self._search._best_vals = [evaluator.curr_value for evaluator in self._search._evaluators]
                    print(f"Best: {acc_within_tolerance}, {pair_objective}")
                    checkpoint_path = Path(self._search.checkpoint_save_dir, 'subnetwork_best.pth')
                    checkpoint = {
                        'best_acc1': acc_within_tolerance,
                        'subnet_config': sample
                    }
                    torch.save(checkpoint, checkpoint_path)

            self._search._search_records.append(result)

        self._iter += 1
        out["F"] = anp.column_stack([arr for arr in evaluators_arr])
