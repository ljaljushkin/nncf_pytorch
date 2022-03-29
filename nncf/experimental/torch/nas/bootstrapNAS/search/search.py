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
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import TypeVar

import numpy as np
from pathlib import Path
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.factory import get_crossover
from pymoo.factory import get_mutation
from pymoo.factory import get_sampling
from pymoo.optimize import minimize
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from nncf.experimental.torch.nas.bootstrapNAS.search.evaluator import AccuracyEvaluator
from nncf.experimental.torch.nas.bootstrapNAS.search.evaluator import Evaluator
from nncf.experimental.torch.nas.bootstrapNAS.search.evaluator import get_macs_for_active_subnet
from nncf import NNCFConfig
from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.common.utils.logger import logger as nncf_logger
from nncf.config.extractors import get_bn_adapt_algo_kwargs
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_controller import ElasticityController
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import SubnetConfig
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


class EvolutionaryAlgorithms(Enum):
    NSGA2 = 'NSGA2'


class SearchParams:
    """
    Storage class for search parameters.
    """
    def __init__(self, num_evals: float, num_constraints: float, population: float,
                 seed: float, crossover_prob: float, crossover_eta: float,
                 mutation_prob: float, mutation_eta: float, acc_delta: float, ref_acc: float):
        """
        Initializes storage class for search parameters.

        :param num_evals: Number of evaluations for the search algorithm.
        :param num_constraints: Number of constraints in search problem
        :param population: Population size
        :param seed: Seed used by the search algorithm.
        :param crossover_prob: Crossover probability used by evolutionary algorithm
        :param crossover_eta: Crossover eta
        :param mutation_prob: Mutation probability
        :param mutation_eta: Mutation eta
        :param acc_delta: Tolerated accuracy delta to select a single sub-network from Pareto front.
        :param ref_acc: Accuracy of input model or reference.
        """
        self.num_evals = num_evals
        self.num_constraints = num_constraints
        self.population = population
        if population > num_evals:
            raise ValueError("Population size must not be greater than number of evaluations.")
        self.seed = seed
        self.crossover_prob = crossover_prob
        self.crossover_eta = crossover_eta
        self.mutation_prob = mutation_prob
        self.mutation_eta = mutation_eta
        self.acc_delta = acc_delta
        self.ref_acc = ref_acc

    @classmethod
    def from_dict(cls, search_config: Dict[str, Any]) -> 'SearchParams':
        """
        Initializes params storage class from Dict.

        :param search_config: Dictionary with search configuration.
        :return: Instance of the storage class
        """
        num_evals = search_config.get('num_evals', 3000)
        num_constraints = search_config.get('num_constraints', 0)
        population = search_config.get('population', 40)
        seed = search_config.get('seed', 0)
        crossover_prob = search_config.get('crossover_prob', 0.9)
        crossover_eta = search_config.get('crossover_eta', 10.0)
        mutation_prob = search_config.get('mutation_prob', 0.02)
        mutation_eta = search_config.get('mutation_eta', 3.0)
        acc_delta = search_config.get('acc_delta', 1)
        ref_acc = search_config.get('ref_acc', 100)

        return cls(num_evals, num_constraints, population,
                   seed, crossover_prob, crossover_eta,
                   mutation_prob, mutation_eta, acc_delta, ref_acc)


class BaseSearchAlgorithm:
    """
    Base class for search algorithms. It contains the evaluators used by search approches.
    """
    def __init__(self):
        """
        Initialize BaseSearchAlgorithm class.

        """
        self._use_default_evaluators = True
        self._evaluators = None
        self._log_dir = None
        self._search_records = []
        self.bad_requests = []
        self.best_config = None
        self.best_vals = None
        self.best_pair_objective = float('inf')
        self._tb = None

    @abstractmethod
    def run(self, validate_fn: Callable, val_loader: DataLoader, checkpoint_save_dir: str,
            evaluators: Optional[List[Evaluator]] = None, ref_acc: Optional[float] = 100,
            tensorboard_writer: Optional[SummaryWriter] = None) -> Tuple[ElasticityController,
                                                                         SubnetConfig, Tuple[float, ...]]:
        """This method should implement how to run the search algorithm."""

    def search_progression_to_csv(self, filename='search_progression.csv') -> NoReturn:
        """
        Exports search progression to CSV file

        :param filename: path to save the CSV file.
        :return:
        """
        with open(f'{self._log_dir}/{filename}', 'w', encoding='utf8') as progression:
            writer = csv.writer(progression)
            for record in self._search_records:
                writer.writerow(record)


class SearchAlgorithm(BaseSearchAlgorithm):
    def __init__(self,
                 model: NNCFNetwork,
                 elasticity_ctrl: ElasticityController,
                 nncf_config: NNCFConfig,
                 verbose=True):
        """
        Initializes search algorithm

        :param model: Super-network
        :param elasticity_ctrl: interface to manage the elasticity of the super-network.
        :param nncf_config: Configuration file.
        :param verbose:
        """
        super().__init__()
        self._model = model
        self._elasticity_ctrl = elasticity_ctrl
        self._elasticity_ctrl.multi_elasticity_handler.activate_maximum_subnet()
        search_config = nncf_config.get('bootstrapNAS', {}).get('search', {})
        self.num_obj = None
        self.search_params = SearchParams.from_dict(search_config)
        self._log_dir = nncf_config.get("log_dir", ".")
        self._verbose = verbose
        self._top1_accuracy_validation_fn = None
        self._val_loader = None
        evo_algo = search_config['algorithm']
        if evo_algo == EvolutionaryAlgorithms.NSGA2.value:
            self._algorithm = NSGA2(pop_size=self.search_params.population,
                                    sampling=get_sampling("int_lhs"),
                                    crossover=get_crossover("int_sbx", prob=self.search_params.crossover_prob,
                                                            eta=self.search_params.crossover_eta),
                                    mutation=get_mutation("int_pm", prob=self.search_params.mutation_prob,
                                                          eta=self.search_params.mutation_eta),
                                    eliminate_duplicates=True,
                                    save_history=True,
                                    )
        else:
            raise NotImplementedError(f"Evolutionary Search Algorithm {evo_algo} not implemented")
        self._num_vars = 0
        self._vars_lower = 0
        self._vars_upper = []

        self._num_vars, self._vars_upper = self._elasticity_ctrl.multi_elasticity_handler.get_design_vars_info()
        self._result = None
        bn_adapt_params = nncf_config.get('compression', {}).get('initializer', {}).get('batchnorm_adaptation', {})
        bn_adapt_algo_kwargs = get_bn_adapt_algo_kwargs(nncf_config, bn_adapt_params)
        self.bn_adaptation = BatchnormAdaptationAlgorithm(**bn_adapt_algo_kwargs)
        self._problem = None
        self.checkpoint_save_dir = None
        self.type_var = np.int

    @property
    def evaluators(self) -> List[Evaluator]:
        """
        Gets a list of the evaluators used by the search algorithm.

        :return: List of available evaluators.
        """
        if self._evaluators:
            return self._evaluators
        raise RuntimeError("Evaluators haven't been defined")

    @property
    def acc_delta(self) -> float:
        """
        :return: Value allowed to deviate from when selecting a sub-network in the pareto front.
        """
        return self.search_params.acc_delta

    @acc_delta.setter
    def acc_delta(self, val: float) -> NoReturn:
        """
        :param val: value to update the allowed accuracy delta.
        :return:
        """
        val = min(val, 50)
        if val > 5:
            nncf_logger.warning("Accuracy delta was set to a value greater than 5")
        self.search_params.acc_delta = val

    @property
    def vars_lower(self) -> List[float]:
        """
        Gets access to design variables lower bounds.
        :return: lower bounds for design variables
        """
        return self._vars_lower

    @property
    def vars_upper(self) -> List[float]:
        """
        Gets access to design variables upper bounds.
        :return: upper bounds for design variables
        """
        return self._vars_upper

    @property
    def num_vars(self) -> float:
        """
        Number of design variables used by the search algorithm.
        :return:
        """
        return self._num_vars

    @classmethod
    def from_config(cls, model, elasticity_ctrl, nncf_config):
        """
        Construct a search algorithm from a
        :param model: Super-Network
        :param elasticity_ctrl: Interface to manage elasticity of super-network
        :param nncf_config: Dict with configuration for search algorithm
        :return: instance of the search algorithm.
        """
        return cls(model, elasticity_ctrl, nncf_config)

    @classmethod
    def from_checkpoint(cls, model: NNCFNetwork, elasticity_ctrl: ElasticityController, bn_adapt_args,
                        resuming_checkpoint_path: str) -> 'SearchAlgorithm':
        raise NotImplementedError

    def run(self, validate_fn: ValFnType, val_loader: DataLoaderType, checkpoint_save_dir: str,
            evaluators: Optional[List[Evaluator]] = None, ref_acc: Optional[float] = 100,
            tensorboard_writer: Optional[SummaryWriter] = None) -> Tuple[
        ElasticityController, SubnetConfig, Tuple[float, ...]]:
        """
        Runs the search algorithm

        :param validate_fn: Function used to validate the accuracy of the model.
        :param val_loader: Data loader used by the validation function.
        :param checkpoint_save_dir: Path to save checkpoints.
        :param evaluators: External evaluators.
        :param ref_acc: Reference Accuracy for sub-network selection.
        :param tensorboard_writer: Tensorboard writer to log data points.
        :return: Elasticity controller with discovered sub-network, sub-network configuration and
                its performance metrics.
        """
        nncf_logger.info("Searching for optimal subnet.")
        if ref_acc != 100:
            self.search_params.ref_acc = ref_acc
        self._tb = tensorboard_writer
        self.checkpoint_save_dir = checkpoint_save_dir
        if evaluators is not None:
            self._use_default_evaluators = False
            self._evaluators = evaluators
            self.num_obj = len(evaluators)
        else:
            self._add_default_evaluators(validate_fn, val_loader)
        self.update_evaluators_for_input_model()
        self._problem = SearchProblem(self)
        self._result = minimize(self._problem, self._algorithm,
                                ('n_gen', int(self.search_params.num_evals / self.search_params.population)),
                                seed=self.search_params.seed,
                                # save_history=True,
                                verbose=self._verbose)

        if self.best_config is not None:
            self._elasticity_ctrl.multi_elasticity_handler.activate_subnet_for_config(self.best_config)
            self.bn_adaptation.run(self._model)
        else:
            nncf_logger.warning("Couldn't find a subnet that satisfies the requirements. Returning maximum subnet.")
            self._elasticity_ctrl.multi_elasticity_handler.activate_maximum_subnet()
            self.bn_adaptation.run(self._model)
            self.best_config = self._elasticity_ctrl.multi_elasticity_handler.get_active_config()
            self.best_vals = [None, None]

        return self._elasticity_ctrl, self.best_config, self.best_vals

    def update_evaluators_for_input_model(self) -> NoReturn:
        """
        Update evaluators information based on input model.

        :return:
        """
        self._elasticity_ctrl.multi_elasticity_handler.activate_maximum_subnet()
        for evaluator in self._evaluators:
            if hasattr(evaluator, 'ref_acc'):
                evaluator.ref_acc = self.search_params.ref_acc
                top1_acc = evaluator.evaluate_model(self._model)
                evaluator.input_model_value = top1_acc
                if top1_acc > evaluator.ref_acc - 0.01 or top1_acc < evaluator.ref_acc - 0.01:
                    nncf_logger.warning("Accuracy obtained from evaluation {value} differs from "
                                        "reference accuracy {ref_acc}".format(value=top1_acc,
                                                                              ref_acc=evaluator.ref_acc))
                    if evaluator.ref_acc == 100:
                        nncf_logger.info("Adjusting reference accuracy to accuracy obtained from evaluation")
                        evaluator.ref_acc = top1_acc
                    elif evaluator.ref_acc < 100:
                        nncf_logger.info("Using reference accuracy.")
                        evaluator.input_model_value = evaluator.ref_acc
                self.search_params.ref_acc = evaluator.ref_acc
            else:
                if evaluator.use_model_for_evaluation:
                    value = evaluator.evaluate_model(self._model)
                else:
                    value = evaluator.evaluate_with_elasticity_handler()
                evaluator.input_model_value = value

    def visualize_search_progression(self, filename='search_progression.pth') -> NoReturn:
        # TODO(pablo): Plot and save image
        pass

    def save_evaluators_state(self) -> NoReturn:
        """
        Save state of evaluators used in search.
        :return:
        """
        evaluators_state = []
        for evaluator in self._evaluators:
            eval_state = evaluator.get_state()
            evaluators_state.append(eval_state)
        torch.save(Path(self.checkpoint_save_dir), 'evaluators_state.pth')

    def load_evaluators_state(self) -> NoReturn:
        # TODO(pablo)
        pass

    def evaluators_to_csv(self) -> NoReturn:
        """
        Export evaluators' information used by search algorithm to CSV
        :return:
        """
        for evaluator in self.evaluators:
            evaluator.export_cache_to_csv(self._log_dir)

    def _add_default_evaluators(self, validate_fn: Callable, val_loader: DataLoader) -> NoReturn:
        """
        Instantiate default evaluators

        :param validate_fn: Function used to obtain accuracy of a sub-network
        :param val_loader: Data loader used by the validation function.
        :return:
        """
        self.num_obj = 2
        self._use_default_evaluators = True
        self._evaluators = []
        macs_evaluator = Evaluator("MACs", get_macs_for_active_subnet, 0, self._elasticity_ctrl)
        macs_evaluator.use_model_for_evaluation = False
        self._evaluators.append(macs_evaluator)
        self._evaluators.append(AccuracyEvaluator(validate_fn, val_loader))


class SearchProblem(Problem):
    """
    Pymoo problem with design variables and evaluation methods.
    """
    def __init__(self, search: SearchAlgorithm):
        """
        Initializes search problem

        :param search: search algorithm.
        """
        super().__init__(n_var=search.num_vars,
                         n_obj=search.num_obj,
                         n_constr=search.search_params.num_constraints,
                         xl=search.vars_lower,
                         xu=search.vars_upper,
                         type_var=search.type_var)
        self._search = search
        self._search_records = search._search_records
        self._elasticity_handler = self._search._elasticity_ctrl.multi_elasticity_handler
        self._dims_enabled = self._elasticity_handler.get_available_elasticity_dims()
        self._iter = 0
        self._evaluators = search.evaluators
        self._model = search._model
        self._lower_bound_acc = search.search_params.ref_acc - search.acc_delta

    def _evaluate(self, x: List[float], out: Dict[str, Any], *args, **kargs) -> NoReturn:
        """
        Evaluates a population of sub-networks.

        :param x: set of sub-networks to evaluate.
        :param out: measurements obtained by evaluating sub-networks.
        :param args:
        :param kargs:
        :return:
        """
        evaluators_arr = [[] for i in range(len(self._search.evaluators))]

        for _, x_i in enumerate(x):
            sample = self._elasticity_handler.get_config_from_pymoo(x_i)
            self._elasticity_handler.activate_subnet_for_config(sample)
            if sample != self._elasticity_handler.get_active_config():
                nncf_logger.warning("Requested configuration was invalid")
                nncf_logger.warning("Requested: {sample}".format(sample=sample))
                nncf_logger.warning("Provided: {config}".format(
                                    config=self._elasticity_handler.get_active_config()))
                self._search.bad_requests.append((sample, self._elasticity_handler.get_active_config()))

            result = [sample]

            eval_idx = 0
            bn_adaption_executed = False
            for evaluator in self._evaluators:
                in_cache, value = evaluator.retrieve_from_cache(tuple(x_i))
                if not in_cache:
                    if not bn_adaption_executed:
                        self._search.bn_adaptation.run(self._model)
                        bn_adaption_executed = True
                    value = evaluator.evaluate_from_pymoo(self._model, tuple(x_i))
                evaluators_arr[eval_idx].append(value)
                eval_idx += 1

                result.append(evaluator.name)
                result.append(value)

            self._save_checkpoint_best_subnetwork(sample)
            self._search_records.append(result)

        self._iter += 1
        out["F"] = np.column_stack(list(evaluators_arr))

    def _save_checkpoint_best_subnetwork(self, config: SubnetConfig) -> NoReturn:
        """
        Saves information of current best sub-network discovered by the search algorithm.

        :param config: Best sub-network configuration
        :return:
        """
        acc_within_tolerance = 0
        pair_objective = None
        for evaluator in self._evaluators:
            if hasattr(evaluator, '_ref_acc'):
                acc_within_tolerance = evaluator.current_value
            else:
                pair_objective = evaluator.current_value
        if acc_within_tolerance < (self._lower_bound_acc * -1.0):
            if pair_objective < self._search.best_pair_objective:
                self._search.best_pair_objective = pair_objective
                self._search.best_config = config
                self._search.best_vals = [evaluator.current_value for evaluator in self._evaluators]
                checkpoint_path = Path(self._search.checkpoint_save_dir, 'subnetwork_best.pth')
                checkpoint = {
                    'best_acc1': acc_within_tolerance * -1.0,
                    'subnet_config': config
                }
                torch.save(checkpoint, checkpoint_path)