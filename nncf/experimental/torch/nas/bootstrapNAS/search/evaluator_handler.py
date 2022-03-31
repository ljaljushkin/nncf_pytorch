from typing import Tuple, NoReturn, Optional

from nncf.common.utils.logger import logger as nncf_logger

class BaseEvaluatorHandler:
    def __init__(self, evaluator, elasticity_ctr):
        self.evaluator = evaluator
        self.elasticity_ctrl = elasticity_ctr
        self.elasticity_ctrl.multi_elasticity_handler.activate_maximum_subnet()
        self.input_model_value = evaluator.evaluate_subnet()

    def retrieve_from_cache(self, subnet_config_repr: Tuple[float, ...]) -> Tuple[bool, float]:
        return self.evaluator.retrieve_from_cache(subnet_config_repr)

    def evaluate_and_add_to_cache_from_pymoo(self, pymoo_repr: Tuple[float, ...]) -> float:
        return self.evaluator.evaluate_and_add_to_cache_from_pymoo(pymoo_repr)

    @property
    def name(self):
        return self.evaluator.name

    @property
    def current_value(self):
        return self.evaluator.current_value

class EfficiencyEvaluatorHandler(BaseEvaluatorHandler):
    def __init__(self, efficiency_evaluator, elasticity_ctrl):
        super().__init__(efficiency_evaluator, elasticity_ctrl)


class AccuracyEvaluatorHandler(BaseEvaluatorHandler):
    def __init__(self, accuracy_evaluator, elasticity_ctrl, ref_acc: Optional[float] = 100):
        super().__init__(accuracy_evaluator, elasticity_ctrl)
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

    def update_reference_accuracy(self, search_params):
        self.ref_acc = search_params.ref_acc
        if self.input_model_value > self.ref_acc - 0.01 or self.input_model_value < self.ref_acc + 0.01:
            nncf_logger.warning("Accuracy obtained from evaluation {value} differs from "
                                        "reference accuracy {ref_acc}".format(value=self.input_model_value,
                                                                              ref_acc=self.ref_acc))
            if self.ref_acc == 100:  # TODO: Use -1 here. or is None
                nncf_logger.info("Adjusting reference accuracy to accuracy obtained from evaluation")
                self.ref_acc = self.input_model_value
            elif self.ref_acc < 100:
                nncf_logger.info("Using reference accuracy.")
                self.input_model_value = self.ref_acc
        search_params.ref_acc = self.ref_acc
