import json
from abc import ABC
from abc import abstractmethod
from typing import Dict

import pytest
import torch

from examples.common import restricted_pickle_module
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.quantization.structs import QuantizerConfig

from nncf.dynamic_graph.context import Scope
from nncf.dynamic_graph.graph import InputAgnosticOperationExecutionContext
from nncf.dynamic_graph.transformations.commands import PTTargetPoint
# from nncf.json_serialization import JSONSerializable
from nncf.json_serialization import deserialize
from nncf.json_serialization import serialize
from nncf.quantization.quantizer_setup import SingleConfigQuantizationPoint
from nncf.quantization.quantizer_setup import SingleConfigQuantizerSetup
from tests.helpers import MockModel


def test_json_dump(tmp_path):
    target_type = TargetType.OPERATOR_POST_HOOK
    assert target_type == TargetType.from_str(str(target_type))
    assert target_type == TargetType.from_str('OPCompositeBuilderStateERATOR_POST_HOOK')
    assert target_type == TargetType.from_str('TargetType.OPERATOR_POST_HOOK')

    scope = Scope.from_str('MyConv/1[2]/3[4]/5')
    ia_op_exec_context = InputAgnosticOperationExecutionContext(operator_name='MyConv',
                                                                scope_in_model=scope,
                                                                call_order=1)
    assert ia_op_exec_context == InputAgnosticOperationExecutionContext.from_str(str(ia_op_exec_context))

    pttp = PTTargetPoint(target_type,
                         ia_op_exec_context=ia_op_exec_context,
                         input_port_id=7)
    assert pttp == PTTargetPoint.from_str(str(pttp))

    qc = QuantizerConfig()
    assert qc == QuantizerConfig.from_str(str(qc))

    scqp = SingleConfigQuantizationPoint(pttp, qc, scopes_of_directly_quantized_operators=[scope])

    # encoded = SingleConfigQuantizationPointEncoder().encode(scqp)
    # # assert scqp == SingleConfigQuantizationPoint.from_str(str(scqp))
    # assert scqp == SingleConfigQuantizationPointDecoder().decode(encoded)
    #
    # with open(tmp_path / 'data.json', 'w') as fp:
    #     json.dump(scqp, fp, cls=SingleConfigQuantizationPointEncoder)
    # with open(tmp_path / 'data.json', 'r') as fp:
    #     assert scqp == json.load(fp, cls=SingleConfigQuantizationPointDecoder)

    scqs = SingleConfigQuantizerSetup()
    scqs.quantization_points = {0: scqp, 1: scqp}
    scqs.unified_scale_groups = {2: {0, 1}}
    scqs.shared_input_operation_set_groups = {2: {0, 1}}

    with open(tmp_path / 'data.json', 'w') as fw:
        # if 0:
        #     json.dump(scqs, fw, cls=Encoder, sort_keys=True, indent=4)
        #     json.dump(scqs, fw)
        # elif 0:
        #     scqs_str = json.dumps(scqs, sort_keys=True, indent=4, cls=Encoder)
        #     print(scqs_str)
        #     fw.write(scqs_str)
        # else:
        scqs_str = scqs.serialize()
        print(scqs_str)
        fw.write(scqs_str)

    with open(tmp_path / 'data.json', 'r') as fr:
        json_dict = json.load(fr)
        scqs2 = SingleConfigQuantizerSetup.from_dict(json_dict)
        assert scqs == scqs2
        # assert scqs == json.load(fr, cls=SingleConfigQuantizerSetupDecoder)


def test_ckpt(tmp_path):
    ckpt_path = tmp_path / 'model.pth'
    json_like_dict = {
        'q_points': {
            '0': ['0', '0'],
            '1': ['0', '0']
        }
    }
    model = MockModel()
    saved_sd = model.state_dict()
    saved_sd['builder_state'] = json_like_dict
    scope = Scope.from_str('MyConv/1[2]/3[4]/5')
    saved_sd['scope'] = scope
    torch.save(saved_sd, ckpt_path)
    loaded_sd = torch.load(ckpt_path, pickle_module=restricted_pickle_module)
    assert saved_sd == loaded_sd
    assert loaded_sd['builder_state'] == json_like_dict
    assert loaded_sd['scope'] == scope


path_to_file = '/tmp/pytest-of-nlyalyus/not_deleted/pickled_scope.pth'


def test_save_pickle():
    scope = Scope.from_str('MyConv/1[2]/3[4]/5')
    torch.save(scope, path_to_file)
    assert torch.load(path_to_file) == scope


def test_load_pickle():
    scope = Scope.from_str('MyConv/1[2]/3[4]/5')
    assert torch.load(path_to_file) == scope


# # TODO: how to test and not influence library by importing tests???
# def test_subclass():
#     class A1(JSONSerializable, reg_name='A', prefix='TF'):
#         pass
#
#     assert 'TF>A' in JSONSerializable.REGISTERED_CLASSES
#     assert A1 in JSONSerializable.REGISTERED_NAMES
#
#     class A2(JSONSerializable, reg_name='A', prefix='PT'):
#         pass
#
#     assert 'PT>A' in JSONSerializable.REGISTERED_CLASSES
#     assert A2 in JSONSerializable.REGISTERED_NAMES
#
#     with pytest.raises(ValueError):
#         class A3(JSONSerializable, reg_name='A', prefix='PT'):
#             pass


def test_simple():
    qc = QuantizerConfig()
    json_str = serialize(qc)
    print(json_str)
    assert qc == deserialize(json_str)


def test_enum():
    tt = TargetType.OPERATOR_POST_HOOK
    json_str = serialize(tt)
    print(json_str)
    assert tt == deserialize(json_str)


def test_setup():
    target_type = TargetType.OPERATOR_POST_HOOK
    assert target_type == deserialize(serialize(target_type))

    scope = Scope.from_str('MyConv/1[2]/3[4]/5')
    assert scope == deserialize(serialize(scope))

    ia_op_exec_context = InputAgnosticOperationExecutionContext(operator_name='MyConv',
                                                                scope_in_model=scope,
                                                                call_order=1)
    assert ia_op_exec_context == deserialize(serialize(ia_op_exec_context))

    pttp = PTTargetPoint(target_type,
                         ia_op_exec_context=ia_op_exec_context,
                         input_port_id=7)
    assert pttp == deserialize(serialize(pttp))

    qc = QuantizerConfig()
    assert qc == deserialize(serialize(qc))

    scqp = SingleConfigQuantizationPoint(pttp, qc, scopes_of_directly_quantized_operators=[scope])
    assert scqp == deserialize(serialize(scqp))

    scqs = SingleConfigQuantizerSetup()
    scqs.quantization_points = {0: scqp, 1: scqp}
    scqs.unified_scale_groups = {2: {0, 1}}
    scqs.shared_input_operation_set_groups = {2: {0, 1}}
    json_str = serialize(scqs)
    print(json_str)
    restored = deserialize(json_str)
    assert scqs == restored