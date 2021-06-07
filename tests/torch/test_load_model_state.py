"""
 Copyright (c) 2019 Intel Corporation
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
import os
from typing import Dict, List, Set

import pytest
import torch

from examples.torch.common.model_loader import load_model
from nncf.torch.checkpoint_loading import KeyMatcher
from nncf.torch.checkpoint_loading import OPTIONAL_PARAMETERS_REGISTRY
from nncf.torch.checkpoint_loading import ProcessedKeyStatus
from nncf.torch.checkpoint_loading import ProcessedKeys
from nncf.torch.checkpoint_loading import load_state
from nncf.torch.dynamic_graph.transform_graph import replace_modules_by_nncf_modules
from nncf.torch.layers import NNCF_PADDING_VALUE_ATTR_NAME
from nncf.torch.nncf_network import EXTERNAL_QUANTIZERS_STORAGE_NAME
from nncf.torch.nncf_network import LEGACY_ACT_STORAGE_NAME
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import check_equal


def test_export_sq_11_is_ok(tmp_path):
    test_path = str(tmp_path.joinpath("test.onnx"))
    model = load_model('squeezenet1_1', pretrained=False)
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, test_path, verbose=False)
    os.remove(test_path)


def test_load_state_skips_not_matched_params__from_larger_to_smaller():
    ref_weights = BasicConvTestModel.default_weight()
    ref_bias = BasicConvTestModel.default_bias()
    model_save = BasicConvTestModel(out_channels=1, weight_init=2, bias_init=2)
    model_load = BasicConvTestModel(out_channels=2)

    num_loaded = load_state(model_load, model_save.state_dict())

    act_bias = model_load.conv.bias.data
    act_weights = model_load.conv.weight.data
    assert num_loaded == 0
    check_equal(act_bias, ref_bias)
    check_equal(act_weights, ref_weights)


def test_can_skip_padding_value():
    model = BasicConvTestModel(out_channels=2)
    state_dict = ({'conv.weight': model.default_weight(),
                   'conv.bias': model.default_bias()})
    model, _ = replace_modules_by_nncf_modules(model)

    num_loaded = load_state(model, state_dict, is_resume=True)

    assert num_loaded == 2


def test_can_load_padding_value():
    VALUE_TO_SET = 5
    model = BasicConvTestModel()
    state_dict = ({
        'conv.weight': model.default_weight(),
        'conv.bias': model.default_bias(),
        '.'.join(['conv', NNCF_PADDING_VALUE_ATTR_NAME]): torch.Tensor([VALUE_TO_SET])
    })
    model, _ = replace_modules_by_nncf_modules(model)
    assert model.conv.get_padding_value_ref().item() == 0

    num_loaded = load_state(model, state_dict, is_resume=True)

    assert num_loaded == 3
    assert model.conv.get_padding_value_ref().item() == VALUE_TO_SET


def test_load_state_skips_not_matched_params__from_smaller_to_larger():
    ref_weights = torch.tensor([[[[3, 2],
                                  [2, 3]]]])
    ref_bias = torch.tensor([2.])
    model_save = BasicConvTestModel(out_channels=2)
    model_load = BasicConvTestModel(out_channels=1, weight_init=2, bias_init=2)

    num_loaded = load_state(model_load, model_save.state_dict())

    assert num_loaded == 0
    act_bias = model_load.conv.bias.data
    act_weights = model_load.conv.weight.data
    check_equal(act_bias, ref_bias)
    check_equal(act_weights, ref_weights)


class MatchKeyDesc:
    MOCKED_VALUE = torch.zeros([1])

    def __init__(self, num_loaded=0, is_resume=True, expects_error=False,
                 state_dict_to_load: Dict[str, torch.Tensor] = None,
                 model_state_dict: Dict[str, torch.Tensor] = None):
        self.state_dict_to_load = state_dict_to_load if state_dict_to_load else {}
        self.model_state_dict = model_state_dict if model_state_dict else {}
        self.new_dict: Dict[str, torch.Tensor] = {}
        self.num_loaded = num_loaded
        self.processed_keys = ProcessedKeys()
        self.ignored_keys = []
        self.is_resume = is_resume
        self.expects_error = expects_error
        self.has_deprecation_warning = False

    def __str__(self):
        result = '-'.join(self.state_dict_to_load.keys()) + '__TO__' + '-'.join(self.model_state_dict.keys())
        if self.ignored_keys:
            result += '__IGNORE__' + '-'.join(self.ignored_keys)
        if self.is_resume:
            result += '__resume'
        return result

    def setup_test(self, mocker):
        pass

    def keys_to_load(self, keys: List[str]):
        for k in keys:
            self.state_dict_to_load[k] = self.MOCKED_VALUE
        return self

    def model_keys(self, keys: List[str]):
        for k in keys:
            self.model_state_dict[k] = self.MOCKED_VALUE
        return self

    def keys_to_ignore(self, keys: List[str]):
        self.ignored_keys = keys
        return self

    def missing(self, keys: List[str]):
        self.processed_keys.extend_keys(keys, ProcessedKeyStatus.MISSING)
        return self

    def unexpected(self, keys: List[str]):
        self.processed_keys.extend_keys(keys, ProcessedKeyStatus.UNEXPECTED)
        return self

    def size_mismatched(self, keys: List[str]):
        self.processed_keys.extend_keys(keys, ProcessedKeyStatus.SIZE_MISMATCHED)
        return self

    def matched(self, keys: List[str]):
        self.processed_keys.extend_keys(keys, ProcessedKeyStatus.MATCHED)
        return self

    def skipped(self, keys: List[str]):
        self.processed_keys.extend_keys(keys, ProcessedKeyStatus.SKIPPED)
        return self

    def all_not_matched(self):
        self.unexpected(list(self.state_dict_to_load))
        self.missing(list(self.model_state_dict))
        return self

    def all_matched(self):
        self.matched(list(self.model_state_dict))
        return self

    def with_deprecation_warning(self):
        self.has_deprecation_warning = True
        return self


OP1 = 'op1'
OP2 = 'op2'
PREFIX = 'prx'
SUFFIX = 'sfx'
OP1_NOT_PARAM = f'{PREFIX}_{OP1}'
OP1_SUFFIX = f'{PREFIX}.{OP1}'
OP1_PREFIX = f'{OP1}.{SUFFIX}'
OP2_SUFFIX = f'{PREFIX}.{OP2}'
OP2_NOT_PARAM = f'{PREFIX}_{OP2}'
OP2_MIDDLE = f'{PREFIX}.{OP2}.{SUFFIX}'


class OptionalMatchKeyDesc(MatchKeyDesc):
    def setup_test(self, mocker):
        def fn() -> Set['str']:
            return {OP1, OP2}

        mocked_registry_get = mocker.patch.object(OPTIONAL_PARAMETERS_REGISTRY, 'get_parameters_names')
        mocked_registry_get.side_effect = fn


MATCH_KEY_DESC_LIST = [
    # basic errors handling: mismatched size, unexpected and missing
    MatchKeyDesc(num_loaded=0, expects_error=True,
                 state_dict_to_load={'1': torch.zeros(1)},
                 model_state_dict={'1': torch.zeros(2)})
        .size_mismatched(['1']),
    MatchKeyDesc(num_loaded=0, is_resume=False,
                 state_dict_to_load={'1': torch.zeros(1)},
                 model_state_dict={'1': torch.zeros(2)})
        .size_mismatched(['1']),
    MatchKeyDesc(num_loaded=1, is_resume=False,
                 state_dict_to_load={'1': torch.zeros(1)},
                 model_state_dict={'1': torch.zeros(2)}).keys_to_load(['2']).model_keys(['2', '3'])
        .size_mismatched(['1']).missing(['3']).matched(['2']),
    MatchKeyDesc(num_loaded=1, is_resume=False,
                 state_dict_to_load={'1': torch.zeros(1)},
                 model_state_dict={'1': torch.zeros(2)}).keys_to_load(['2', '4']).model_keys(['2', '3'])
        .size_mismatched(['1']).missing(['3']).unexpected(['4']).matched(['2']),
    MatchKeyDesc(num_loaded=2).keys_to_load(['1', '2']).model_keys(['1', '2'])
        .all_matched(),
    MatchKeyDesc(num_loaded=1, expects_error=True).keys_to_load(['1', '2']).model_keys(['1'])
        .unexpected(['2']).matched(['1']),
    MatchKeyDesc(num_loaded=1, expects_error=True).keys_to_load(['1']).model_keys(['1', '2'])
        .missing(['2']).matched(['1']),
    MatchKeyDesc(num_loaded=1, is_resume=False).keys_to_load(['1']).model_keys(['1', '2'])
        .missing(['2']).matched(['1']),

    # wrapping by NNCFNetwork and DataParallel & DistributedDataParallel
    MatchKeyDesc(num_loaded=2).keys_to_load(['module.1', 'nncf_module.2']).model_keys(['1', '2'])
        .all_matched(),
    MatchKeyDesc(num_loaded=2).keys_to_load(['1', '2']).model_keys(['module.1', 'nncf_module.2'])
        .all_matched(),
    MatchKeyDesc(num_loaded=2).keys_to_load(['module.nncf_module.1', 'module.2']).model_keys(['1', 'nncf_module.2'])
        .all_matched(),
    MatchKeyDesc(num_loaded=0, expects_error=True)
        .keys_to_load(['module.nncf_module.1.1', 'module.2']).model_keys(['1', '2.2'])
        .all_not_matched(),

    # collisions after normalization of keys

    # different order of pre_ops
    MatchKeyDesc(num_loaded=2)
        .keys_to_load(['pre_ops.0.op.1', 'pre_ops.1.op.2'])
        .model_keys(['pre_ops.1.op.1', 'pre_ops.0.op.2'])
        .all_matched(),
    # binarization of activation and weight may have the identical parameter (e.g. enabled)
    MatchKeyDesc(num_loaded=2)
        .keys_to_load(['pre_ops.0.op.1', 'pre_ops.1.op.1'])
        .model_keys(['pre_ops.0.op.1', 'pre_ops.1.op.1'])
        .all_matched(),
    MatchKeyDesc(num_loaded=2)
        .keys_to_load(['nncf_module.pre_ops.1.op.1', 'nncf_module.pre_ops.0.op.1'])
        .model_keys(['module.nncf_module.pre_ops.1.op.1', 'module.nncf_module.pre_ops.0.op.1'])
        .all_matched(),
    # quantization -> quantization + sparsity: op.1 was first, than
    MatchKeyDesc(num_loaded=2)
        .keys_to_load(['pre_ops.0.op.1', 'pre_ops.1.op.2'])
        .model_keys(['pre_ops.1.op.1', 'pre_ops.1.op.2'])
        .all_matched(),
    MatchKeyDesc(num_loaded=2)
        .keys_to_load(['module.1', '1']).model_keys(['module.1', '1'])
        .all_matched(),
    MatchKeyDesc(num_loaded=1, expects_error=True)
        .keys_to_load(['module.1', '1']).model_keys(['module.1'])
        .matched(['module.1']).unexpected(['module.1']),
    MatchKeyDesc(num_loaded=2)
        .keys_to_load(['pre_ops.0.op.1', 'module.pre_ops.1.op.2'])
        .model_keys(['module.pre_ops.0.op.1|OUTPUT', 'pre_ops.6.op.2'])
        .all_matched(),
    MatchKeyDesc(num_loaded=1, expects_error=True)
        .keys_to_load(['module.1']).model_keys(['module.1', '1'])
        .matched(['1']).missing(['module.1']),
    MatchKeyDesc(num_loaded=1, expects_error=True)
        .keys_to_load(['1']).model_keys(['module.1', '1'])
        .matched(['1']).missing(['module.1']),
    MatchKeyDesc(num_loaded=1, expects_error=True)
        .keys_to_load(['1', 'module.1']).model_keys(['1'])
        .matched(['1']).unexpected(['module.1']),

    # can match legacy activation quantizer storage name
    MatchKeyDesc(num_loaded=2)
        .keys_to_load([LEGACY_ACT_STORAGE_NAME + '.RELU_0.' + OP1,
                       LEGACY_ACT_STORAGE_NAME + '.RELU_0.' + OP2])
        .model_keys([EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0.' + OP1,
                     EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0.' + OP2])
        .all_matched()
        .with_deprecation_warning(),

    # can match new format of activation quantizer with |INPUT and |OUTPUT
    MatchKeyDesc(num_loaded=2)
        .keys_to_load(['RELU_0.' + OP1, 'RELU_0.' + OP2]).model_keys(['RELU_0|OUTPUT.' + OP1, 'RELU_0|INPUT.' + OP2])
        .all_matched(),

    # can match legacy activation quantizer + new format with |INPUT and |OUTPUT
    MatchKeyDesc(num_loaded=2)
        .keys_to_load([LEGACY_ACT_STORAGE_NAME + '.RELU_0.' + OP1,
                       LEGACY_ACT_STORAGE_NAME + '.RELU_0.' + OP2])
        .model_keys([EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0|OUTPUT.' + OP1,
                     EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0|INPUT.' + OP2])
        .all_matched()
        .with_deprecation_warning(),

    # can match unified FQ
    MatchKeyDesc(num_loaded=1)
        .keys_to_load(['module.' + LEGACY_ACT_STORAGE_NAME + '.RELU_0.' + OP1,
                       'module.' + LEGACY_ACT_STORAGE_NAME + '.RELU_1.' + OP1])
        .model_keys([EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0|OUTPUT;RELU_1|OUTPUT.' + OP1])
        .all_matched()
        .with_deprecation_warning(),

    MatchKeyDesc(num_loaded=1)
        .keys_to_load(['module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0.' + OP1,
                       'module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_1.' + OP1])
        .model_keys([EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0|OUTPUT;RELU_1|OUTPUT.' + OP1])
        .all_matched()
        .with_deprecation_warning(),

    MatchKeyDesc(num_loaded=1)
        .keys_to_load(['module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0.' + OP1,
                       'module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_1.' + OP1,
                       'module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_2.' + OP1])
        .model_keys([EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0|OUTPUT;RELU_2|OUTPUT;RELU_1|OUTPUT.' + OP1])
        .all_matched()
        .with_deprecation_warning(),

    # not matched common operation
    MatchKeyDesc(num_loaded=1, expects_error=True)
        .keys_to_load(['module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0.' + OP1,
                       'module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_1.' + OP2,
                       'module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_2.' + OP1_NOT_PARAM])
        .model_keys([EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0|OUTPUT;RELU_2|OUTPUT;RELU_1|OUTPUT.' + OP1])
        .matched([EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0|OUTPUT;RELU_2|OUTPUT;RELU_1|OUTPUT.' + OP1])
        .unexpected(['module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_1.' + OP2,
                     'module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_2.' + OP1_NOT_PARAM]),

    # not all unified scopes are matched: RELU_3 vs RELU_1
    MatchKeyDesc(num_loaded=1, expects_error=True)
        .keys_to_load(['module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0.' + OP1,
                       'module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_3.' + OP1,
                       'module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_2.' + OP1])
        .model_keys([EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0|OUTPUT;RELU_2|OUTPUT;RELU_1|OUTPUT.' + OP1])
        .matched([EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0|OUTPUT;RELU_2|OUTPUT;RELU_1|OUTPUT.' + OP1])
        .unexpected(['module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_3.' + OP1]),

    # can skip ignored parameters
    MatchKeyDesc(num_loaded=1).keys_to_load(['1']).model_keys(['1', '2'])
        .keys_to_ignore(['2'])
        .skipped(['2']).matched(['1']),
    MatchKeyDesc(num_loaded=1).keys_to_load(['1', '2']).model_keys(['1'])
        .keys_to_ignore(['2'])
        .skipped(['2']).matched(['1']),
    MatchKeyDesc(num_loaded=0, state_dict_to_load={'1': torch.zeros(1)}, model_state_dict={'1': torch.zeros(2)})
        .keys_to_ignore(['1'])
        .skipped(['1']),
    MatchKeyDesc(num_loaded=0, expects_error=True)
        .keys_to_load(['module.nncf_module.1.1', '2.2']).model_keys(['module.1', 'module.2'])
        .keys_to_ignore(['1', '2.2'])
        .skipped(['module.1', '2.2']).missing(['module.2']).unexpected(['module.nncf_module.1.1']),

    # optional parameter - not necessary in checkpoint can be initialized by default in the model
    # can match legacy activation quantizer + new format with |INPUT and |OUTPUT
    MatchKeyDesc(num_loaded=2)
        .keys_to_load([LEGACY_ACT_STORAGE_NAME + '.RELU_0.' + OP1,
                       LEGACY_ACT_STORAGE_NAME + '.RELU_0.' + OP2])
        .model_keys([EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0|OUTPUT.' + OP1,
                     EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0|INPUT.' + OP2])
        .all_matched()
        .with_deprecation_warning(),

    # can match unified FQ
    MatchKeyDesc(num_loaded=1)
        .keys_to_load(['module.' + LEGACY_ACT_STORAGE_NAME + '.RELU_0.' + OP1,
                       'module.' + LEGACY_ACT_STORAGE_NAME + '.RELU_1.' + OP1])
        .model_keys([EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0|OUTPUT;RELU_1|OUTPUT.' + OP1])
        .all_matched()
        .with_deprecation_warning(),

    MatchKeyDesc(num_loaded=1)
        .keys_to_load(['module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0.' + OP1,
                       'module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_1.' + OP1])
        .model_keys([EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0|OUTPUT;RELU_1|OUTPUT.' + OP1])
        .all_matched()
        .with_deprecation_warning(),

    MatchKeyDesc(num_loaded=1)
        .keys_to_load(['module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0.' + OP1,
                       'module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_1.' + OP1,
                       'module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_2.' + OP1])
        .model_keys([EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0|OUTPUT;RELU_2|OUTPUT;RELU_1|OUTPUT.' + OP1])
        .all_matched()
        .with_deprecation_warning(),

    # not matched common operation
    MatchKeyDesc(num_loaded=1, expects_error=True)
        .keys_to_load(['module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0.' + OP1,
                       'module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_1.' + OP2,
                       'module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_2.' + OP1_NOT_PARAM])
        .model_keys([EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0|OUTPUT;RELU_2|OUTPUT;RELU_1|OUTPUT.' + OP1])
        .matched([EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0|OUTPUT;RELU_2|OUTPUT;RELU_1|OUTPUT.' + OP1])
        .unexpected(['module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_1.' + OP2,
                     'module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_2.' + OP1_NOT_PARAM]),

    # not all unified scopes are matched: RELU_3 vs RELU_1
    MatchKeyDesc(num_loaded=1, expects_error=True)
        .keys_to_load(['module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0.' + OP1,
                       'module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_3.' + OP1,
                       'module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_2.' + OP1])
        .model_keys([EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0|OUTPUT;RELU_2|OUTPUT;RELU_1|OUTPUT.' + OP1])
        .matched([EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_0|OUTPUT;RELU_2|OUTPUT;RELU_1|OUTPUT.' + OP1])
        .unexpected(['module.' + EXTERNAL_QUANTIZERS_STORAGE_NAME + '.RELU_3.' + OP1]),

    OptionalMatchKeyDesc(num_loaded=0)
        .keys_to_load([])
        .model_keys([OP1])
        .skipped([OP1]),
    OptionalMatchKeyDesc(num_loaded=1)
        .keys_to_load([OP1_PREFIX])
        .model_keys([OP1_PREFIX, OP1_SUFFIX, OP2_SUFFIX])
        .matched([OP1_PREFIX]).skipped([OP1_SUFFIX, OP2_SUFFIX]),
    OptionalMatchKeyDesc(num_loaded=1, expects_error=True)
        .keys_to_load([OP1_PREFIX, OP2_MIDDLE])
        .model_keys([OP1_PREFIX, OP1_SUFFIX, OP2_SUFFIX])
        .unexpected([OP2_MIDDLE]).matched([OP1_PREFIX]).skipped([OP1_SUFFIX, OP2_SUFFIX]),
    OptionalMatchKeyDesc(num_loaded=1, expects_error=True)
        .keys_to_load([OP1_PREFIX])
        .model_keys([OP1_PREFIX, OP1_SUFFIX, OP2_SUFFIX, OP2_MIDDLE])
        .missing([OP2_MIDDLE]).matched([OP1_PREFIX]).skipped([OP1_SUFFIX, OP2_SUFFIX]),
    OptionalMatchKeyDesc(num_loaded=2, expects_error=True)
        .keys_to_load([OP1_PREFIX, OP1_SUFFIX, OP2_SUFFIX])
        .model_keys([OP1_PREFIX, OP1_SUFFIX, OP2_MIDDLE])
        .missing([OP2_MIDDLE]).unexpected([OP2_SUFFIX]).matched([OP1_PREFIX, OP1_SUFFIX]),
    OptionalMatchKeyDesc(num_loaded=1, expects_error=True)
        .keys_to_load([OP1_PREFIX])
        .model_keys([OP1_PREFIX, OP1_NOT_PARAM, OP2_NOT_PARAM])
        .matched([OP1_PREFIX]).missing([OP1_NOT_PARAM, OP2_NOT_PARAM]),
    OptionalMatchKeyDesc(num_loaded=2, expects_error=True)
        .keys_to_load([OP1_PREFIX, OP1_NOT_PARAM, OP2_NOT_PARAM])
        .model_keys([OP1_PREFIX, OP1_NOT_PARAM, OP2_MIDDLE])
        .missing([OP2_MIDDLE]).unexpected([OP2_NOT_PARAM]).matched([OP1_PREFIX, OP1_NOT_PARAM]),
]


@pytest.mark.parametrize('desc', MATCH_KEY_DESC_LIST, ids=[str(d) for d in MATCH_KEY_DESC_LIST])
def test_match_key(desc: MatchKeyDesc, mocker):
    desc.setup_test(mocker)

    key_matcher = KeyMatcher(desc.is_resume, desc.state_dict_to_load, desc.model_state_dict, desc.ignored_keys)
    if desc.has_deprecation_warning:
        with pytest.deprecated_call():
            new_dict = key_matcher.run()
    else:
        new_dict = key_matcher.run()
    num_loaded_layers = len(new_dict)

    assert num_loaded_layers == desc.num_loaded
    # pylint: disable=protected-access
    assert key_matcher._processed_keys._keys == desc.processed_keys._keys
    if desc.expects_error:
        with pytest.raises(RuntimeError):
            key_matcher.handle_problematic_keys()
    else:
        key_matcher.handle_problematic_keys()
