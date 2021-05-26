"""
 Copyright (c) 2021 Intel Corporation
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
from typing import Dict
from typing import NamedTuple

import pytest

from nncf.torch.model_creation import _match_configs
from nncf.torch.model_creation import get_compression_algorithm
from tests.helpers import get_empty_config


class ConfigBuilder:
    def __init__(self):
        self.config = get_empty_config()
        self.config['compression'] = []
        self.algo_class_vs_should_init_map = {}
        self._algorithms = set()

    def __str__(self):
        if not self._algorithms:
            return 'no_compression'
        return '_'.join(self._algorithms)

    def _add_empty_algo(self, algo_name: str, should_init: bool):
        algo_section = {'algorithm': algo_name}
        self.config['compression'].append(algo_section)
        algo_class = get_compression_algorithm(algo_section)
        self.algo_class_vs_should_init_map[algo_class] = should_init
        self._algorithms.add(algo_name)

    def quantization(self, should_init: bool = False):
        self._add_empty_algo('quantization', should_init)
        return self

    def rb_sparsity(self, should_init: bool = False):
        self._add_empty_algo('rb_sparsity', should_init)
        return self

    def pruning(self, should_init: bool = False):
        self._add_empty_algo('filter_pruning', should_init)
        return self

    def const_sparsity(self, should_init: bool = False):
        self._add_empty_algo('const_sparsity', should_init)
        return self

    def add_algo_section(self, algo_section: Dict, should_init: bool = False):
        self.config['compression'].append(algo_section)
        algo_class = get_compression_algorithm(algo_section)
        self.algo_class_vs_should_init_map[algo_class] = should_init
        algo_name = algo_section.get('algorithm', '')
        self._algorithms.add(algo_name)
        return self

    def add_global_params(self, params: Dict):
        self.config.update(params)
        return self

    def __call__(self, *args, **kwargs):
        return self.config


class MatchingConfigTestDesc(NamedTuple):
    loaded_cfg_builder: ConfigBuilder = ConfigBuilder()
    saved_cfg_builder: ConfigBuilder = ConfigBuilder()
    is_error: bool = False
    is_strict: bool = False

    @property
    def algo_class_vs_should_init_map(self):
        return self.loaded_cfg_builder.algo_class_vs_should_init_map

    def __str__(self):
        msg = '{}strict'
        msg = msg.format('' if self.is_strict else 'not_')
        if self.is_error:
            msg = 'raises_error'
        return 'LOADED={}__SAVED={}__{}'.format(str(self.loaded_cfg_builder), str(self.saved_cfg_builder), msg)


PARAM_STUB = 'P1'
DIFF_PARAM_STUB = 'P2'

MATCHING_CONFIG_TEST_CASES = [
    # no compression cases
    # CHECKPOINT                 -> CREATE                     - STS       - INIT                          - STRICT
    # ---------------------------|-----------------------------|-----------|-------------------------------|-------
    # quantization              -> no compression              - OK        -  no init                      - False
    # sparsity + quantization   -> no compression              - OK        -  no init                      - False
    # no compression            -> quantization                - OK        -  init quantization            - False
    # no compression            -> sparsity + quantization     - OK        -  init sparsity + quantization - False
    MatchingConfigTestDesc(
        saved_cfg_builder=ConfigBuilder().quantization()
    ),
    MatchingConfigTestDesc(
        saved_cfg_builder=ConfigBuilder().quantization().rb_sparsity()
    ),
    MatchingConfigTestDesc(
        loaded_cfg_builder=ConfigBuilder().quantization(should_init=True)
    ),
    MatchingConfigTestDesc(
        loaded_cfg_builder=ConfigBuilder().quantization(should_init=True).rb_sparsity(should_init=True)
    ),
    # different algorithms
    # CHECKPOINT                -> CREATE                      - STS        - INIT                          - STRICT
    # ---------------------------|------------------------------|-----------|-------------------------------|--------
    # rb_sparsity + pruning     -> quantization                 - not OK    - n/a                           - n/a
    # rb_sparsity               -> quantization                 - not OK    - n/a                           - n/a
    MatchingConfigTestDesc(
        saved_cfg_builder=ConfigBuilder().rb_sparsity().pruning(),
        loaded_cfg_builder=ConfigBuilder().quantization(),
        is_error=True
    ),
    MatchingConfigTestDesc(
        saved_cfg_builder=ConfigBuilder().rb_sparsity(),
        loaded_cfg_builder=ConfigBuilder().quantization(),
        is_error=True
    ),
    # CHECKPOINT                -> CREATE                      - STS       - INIT                          - STRICT
    # ---------------------------|-----------------------------|-----------|-------------------------------|--------
    # RB spars. + quantization  -> quantization                - OK        - no init                       - False
    # RB spars.                 -> RB spars. + quantization    - OK        - init quantization             - False
    # RB spars.                 -> CONST spars. + quantization - OK        - init quantization             - False
    # RB spars. + quantization  -> quantization + pruning      - OK        - init pruning                  - False
    # RB spars. + quantization  -> CONST spars. + quantization - OK        - no init                       - False
    # RB spars. + quantization  -> RB spars. + quantization    - OK        - no init                       - True
    # quantization params1      -> quantization params1        - OK        - no init                       - True
    # quantization params1      -> quantization params2        - not OK    - n/a                           - n/a
    MatchingConfigTestDesc(
        saved_cfg_builder=ConfigBuilder().rb_sparsity().quantization(),
        loaded_cfg_builder=ConfigBuilder().quantization()
    ),
    MatchingConfigTestDesc(
        saved_cfg_builder=ConfigBuilder().rb_sparsity(),
        loaded_cfg_builder=ConfigBuilder().rb_sparsity().quantization(should_init=True),
    ),
    MatchingConfigTestDesc(
        saved_cfg_builder=ConfigBuilder().rb_sparsity(),
        loaded_cfg_builder=ConfigBuilder().const_sparsity().quantization(should_init=True),
    ),
    MatchingConfigTestDesc(
        saved_cfg_builder=ConfigBuilder().rb_sparsity().quantization(),
        loaded_cfg_builder=ConfigBuilder().pruning(should_init=True).quantization()
    ),
    MatchingConfigTestDesc(
        saved_cfg_builder=ConfigBuilder().rb_sparsity().quantization(),
        loaded_cfg_builder=ConfigBuilder().const_sparsity().quantization(),
    ),
    MatchingConfigTestDesc(
        saved_cfg_builder=ConfigBuilder().rb_sparsity().quantization(),
        loaded_cfg_builder=ConfigBuilder().rb_sparsity().quantization(),
        is_strict=True
    ),
    MatchingConfigTestDesc(
        saved_cfg_builder=ConfigBuilder().add_algo_section(
            {'algorithm': 'quantization', "ignored_scopes": [PARAM_STUB], PARAM_STUB: PARAM_STUB}),
        loaded_cfg_builder=ConfigBuilder().add_algo_section(
            {'algorithm': 'quantization', "ignored_scopes": [PARAM_STUB], PARAM_STUB: PARAM_STUB}),
        is_strict=True
    ),
    MatchingConfigTestDesc(
        saved_cfg_builder=ConfigBuilder().add_algo_section(
            {'algorithm': 'quantization', "ignored_scopes": [PARAM_STUB], PARAM_STUB: PARAM_STUB}),
        loaded_cfg_builder=ConfigBuilder().add_algo_section(
            {'algorithm': 'quantization', "ignored_scopes": [PARAM_STUB], DIFF_PARAM_STUB: PARAM_STUB}),
        is_error=True
    ),
    MatchingConfigTestDesc(
        saved_cfg_builder=ConfigBuilder().add_algo_section(
            {'algorithm': 'quantization', "ignored_scopes": [PARAM_STUB], PARAM_STUB: PARAM_STUB}),
        loaded_cfg_builder=ConfigBuilder().add_algo_section(
            {'algorithm': 'quantization', "ignored_scopes": [DIFF_PARAM_STUB], PARAM_STUB: PARAM_STUB}),
        is_error=True
    ),
    # sparsity algorithm's interoperability
    MatchingConfigTestDesc(
        saved_cfg_builder=ConfigBuilder().add_algo_section(
            {'algorithm': 'rb_sparsity', "ignored_scopes": [PARAM_STUB], PARAM_STUB: PARAM_STUB}),
        loaded_cfg_builder=ConfigBuilder().add_algo_section(
            {'algorithm': 'const_sparsity', "ignored_scopes": [DIFF_PARAM_STUB], PARAM_STUB: PARAM_STUB}),
        is_error=True
    ),
    MatchingConfigTestDesc(
        saved_cfg_builder=ConfigBuilder().add_algo_section(
            {'algorithm': 'rb_sparsity', "ignored_scopes": [PARAM_STUB], PARAM_STUB: PARAM_STUB}),
        loaded_cfg_builder=ConfigBuilder().add_algo_section(
            {'algorithm': 'const_sparsity', "ignored_scopes": [PARAM_STUB], DIFF_PARAM_STUB: DIFF_PARAM_STUB}),
    ),
    # can ignore init section
    MatchingConfigTestDesc(
        saved_cfg_builder=ConfigBuilder().add_algo_section(
            {'algorithm': 'quantization', "initializer": PARAM_STUB}),
        loaded_cfg_builder=ConfigBuilder().add_algo_section(
            {'algorithm': 'quantization', "initializer": DIFF_PARAM_STUB}),
        is_strict=True
    ),
    # can override global params except target_device, ignored_scopes, target_scopes
    MatchingConfigTestDesc(
        saved_cfg_builder=ConfigBuilder().quantization().add_global_params({PARAM_STUB: PARAM_STUB}),
        loaded_cfg_builder=ConfigBuilder().quantization(),
        is_strict=True
    ),
    MatchingConfigTestDesc(
        saved_cfg_builder=ConfigBuilder().quantization(),
        loaded_cfg_builder=ConfigBuilder().quantization().add_global_params({PARAM_STUB: PARAM_STUB}),
        is_strict=True
    ),
    MatchingConfigTestDesc(
        saved_cfg_builder=ConfigBuilder().quantization().add_global_params({'ignored_scopes': [PARAM_STUB]}),
        loaded_cfg_builder=ConfigBuilder().quantization(),
        is_error=True
    ),
    MatchingConfigTestDesc(
        saved_cfg_builder=ConfigBuilder().quantization(),
        loaded_cfg_builder=ConfigBuilder().quantization().add_global_params({'target_scopes': [PARAM_STUB]}),
        is_error=True
    ),
    MatchingConfigTestDesc(
        saved_cfg_builder=ConfigBuilder().quantization(),
        loaded_cfg_builder=ConfigBuilder().quantization().add_global_params({'target_device': PARAM_STUB}),
        is_error=True
    ),
]


@pytest.mark.parametrize('desc', MATCHING_CONFIG_TEST_CASES, ids=map(str, MATCHING_CONFIG_TEST_CASES))
def test_matching_configs(desc: MatchingConfigTestDesc):
    loaded_cfg = desc.loaded_cfg_builder()
    saved_cfg = desc.saved_cfg_builder()
    if desc.is_error:
        with pytest.raises(RuntimeError):
            _match_configs(loaded_cfg, saved_cfg)
    else:
        algo_class_vs_should_init_map, is_strict_loading = _match_configs(loaded_cfg, saved_cfg)
        assert algo_class_vs_should_init_map == desc.algo_class_vs_should_init_map
        assert is_strict_loading == desc.is_strict
