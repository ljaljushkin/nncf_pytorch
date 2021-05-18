from typing import Any
from typing import Callable
from typing import Dict

from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.commands import TransformationType
from nncf.graph.graph import InputAgnosticOperationExecutionContext


class PTTargetPoint(TargetPoint):
    _OPERATION_TYPES = [TargetType.PRE_LAYER_OPERATION,
                        TargetType.POST_LAYER_OPERATION,
                        TargetType.OPERATION_WITH_WEIGHTS]
    _HOOK_TYPES = [TargetType.OPERATOR_PRE_HOOK,
                   TargetType.OPERATOR_POST_HOOK]

    _IA_OP_EXEC_CONTEXT_STATE_ATTR = 'ia_op_exec_context'
    _MODULE_SCOPE_STATE_ATTR = 'module_scope'
    _INPUT_PORT_STATE_ATTR = 'input_port_id'
    _TARGET_TYPE_STATE_ATTR = 'target_type'

    def __init__(self, target_type: TargetType, *,
                 ia_op_exec_context: InputAgnosticOperationExecutionContext = None,
                 module_scope: 'Scope' = None,
                 input_port_id: int = None):
        super().__init__(target_type)
        self.target_type = target_type
        if self.target_type in self._OPERATION_TYPES:
            if module_scope is None:
                raise ValueError("Should specify module scope for module pre- and post-op insertion points!")

        elif self.target_type in self._HOOK_TYPES:
            if ia_op_exec_context is None:
                raise ValueError("Should specify an operator's InputAgnosticOperationExecutionContext "
                                 "for operator pre- and post-hook insertion points!")
        else:
            raise NotImplementedError("Unsupported target type: {}".format(target_type))

        self.module_scope = module_scope
        self.ia_op_exec_context = ia_op_exec_context
        self.input_port_id = input_port_id

    def __eq__(self, other: 'PTTargetPoint'):
        return self.target_type == other.target_type and self.ia_op_exec_context == other.ia_op_exec_context \
               and self.input_port_id == other.input_port_id and self.module_scope == other.module_scope

    def __str__(self):
        prefix = str(self.target_type)
        retval = prefix
        if self.target_type in self._OPERATION_TYPES:
            retval += " {}".format(self.module_scope)
        elif self.target_type in self._HOOK_TYPES:
            if self.input_port_id is not None:
                retval += " {}".format(self.input_port_id)
            retval += " " + str(self.ia_op_exec_context)
        return retval

    def __hash__(self):
        return hash(str(self))

    def get_state(self) -> Dict[str, object]:
        """
        Returns a JSON-compatible dictionary containing a state of the object
        """
        state = {self._TARGET_TYPE_STATE_ATTR: self.target_type.get_state(),
                 self._INPUT_PORT_STATE_ATTR: self.input_port_id}
        if self.target_type in self._OPERATION_TYPES:
            state[self._MODULE_SCOPE_STATE_ATTR] = str(self.module_scope)
        elif self.target_type in self._HOOK_TYPES:
            state[self._IA_OP_EXEC_CONTEXT_STATE_ATTR] = str(self.ia_op_exec_context)
        return state

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> 'PTTargetPoint':
        """
        Creates the object from its state
        :param state: Output of `get_state()` method.
        """
        kwargs = {cls._TARGET_TYPE_STATE_ATTR: TargetType.from_state(state[cls._TARGET_TYPE_STATE_ATTR]),
                  cls._INPUT_PORT_STATE_ATTR: state[cls._INPUT_PORT_STATE_ATTR]}
        if cls._MODULE_SCOPE_STATE_ATTR in state:
            from nncf.dynamic_graph.context import Scope
            kwargs[cls._MODULE_SCOPE_STATE_ATTR] = Scope.from_str(state[cls._MODULE_SCOPE_STATE_ATTR])
        if cls._IA_OP_EXEC_CONTEXT_STATE_ATTR in state:
            ia_op_exec_ctx_str = state[cls._IA_OP_EXEC_CONTEXT_STATE_ATTR]
            kwargs[cls._IA_OP_EXEC_CONTEXT_STATE_ATTR] = \
                InputAgnosticOperationExecutionContext.from_str(ia_op_exec_ctx_str)
        return cls(**kwargs)


class PTInsertionCommand(TransformationCommand):
    def __init__(self, point: PTTargetPoint, fn: Callable,
                 priority: TransformationPriority = TransformationPriority.DEFAULT_PRIORITY):
        super().__init__(TransformationType.INSERT, point)
        self.fn = fn  # type: Callable
        self.priority = priority  # type: TransformationPriority

    def union(self, other: 'TransformationCommand') -> 'TransformationCommand':
        # TODO: keep all TransformationCommands atomic, refactor TransformationLayout instead
        raise NotImplementedError()
