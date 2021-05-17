"""
 Copyright (c) 2020 Intel Corporation
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
from copy import deepcopy
from os import path as osp
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

from torch.nn import Module
from torch.distributed import barrier

from nncf.algo_selector import COMPRESSION_ALGORITHMS
from nncf.checkpoint_loading import load_state
from nncf.common.utils.logger import logger
from nncf.common.utils.logger import logger as nncf_logger
from nncf.composite_compression import PTCompositeCompressionAlgorithmBuilder
from nncf.compression_method_api import PTCompressionAlgorithmController
from nncf.config import NNCFConfig
from nncf.debug import set_debug_log_dir
from nncf.dynamic_graph.graph_tracer import create_dummy_forward_fn
from nncf.dynamic_graph.graph_tracer import create_input_infos
from nncf.graph.graph_builder import GraphBuilder
from nncf.nncf_network import NNCFNetwork
from nncf.utils import is_main_process
from nncf.utils import is_dist_avail_and_initialized
from nncf.algo_selector import COMPRESSION_ALGORITHMS

from nncf.common.utils.logger import logger


def get_compression_algorithm(config):
    algorithm_key = config.get('algorithm', 'NoCompressionAlgorithmBuilder')
    logger.info("Creating compression algorithm: {}".format(algorithm_key))
    return COMPRESSION_ALGORITHMS.get(algorithm_key)


def create_compressed_model(model: Module,
                            # TODO: API change - become a default parameter
                            config: NNCFConfig = None,
                            resuming_state_dict: dict = None,
                            dummy_forward_fn: Callable[[Module], Any] = None,
                            wrap_inputs_fn: Callable[[Tuple, Dict], Tuple[Tuple, Dict]] = None,
                            wrap_outputs_fn: Callable[[Tuple, Dict], Tuple[Tuple, Dict]] = None,
                            dump_graphs=True, ) \
    -> Tuple[PTCompressionAlgorithmController, NNCFNetwork]:
    """
    The main function used to produce a model ready for compression fine-tuning from an original PyTorch
    model and a configuration object.
    dummy_forward_fn
    :param model: The original model. Should have its parameters already loaded from a checkpoint or another
    source.
    :param config: A configuration object used to determine the exact compression modifications to be applied
    to the model
    :param resuming_state_dict: A PyTorch state dict object to load (strictly) into the compressed model after
    building.
    :param dummy_forward_fn: if supplied, will be used instead of a *forward* function call to build
    the internal graph representation via tracing. Specifying this is useful when the original training pipeline
    has special formats of data loader output or has additional *forward* arguments other than input tensors.
    Otherwise, the *forward* call of the model during graph tracing will be made with mock tensors according
    to the shape specified in the config object. The dummy_forward_fn code MUST contain calls to nncf.nncf_model_input
    functions made with each compressed model input tensor in the underlying model's args/kwargs tuple, and these
    calls should be exactly the same as in the wrap_inputs_fn function code (see below); if dummy_forward_fn is
    specified, then wrap_inputs_fn also must be specified.
    :param wrap_inputs_fn: if supplied, will be used on the module's input arguments during a regular, non-dummy
    forward call before passing the inputs to the underlying compressed model. This is required if the model's input
    tensors that are important for compression are not supplied as arguments to the model's forward call directly, but
    instead are located in a container (such as list), and the model receives the container as an argument.
    wrap_inputs_fn should take as input two arguments - the tuple of positional arguments to the underlying
    model's forward call, and a dict of keyword arguments to the same. The function should wrap each tensor among the
    supplied model's args and kwargs that is important for compression (e.g. quantization) with an nncf.nncf_model_input
    function, which is a no-operation function and marks the tensors as inputs to be traced by NNCF in the internal
    graph representation. Output is the tuple of (args, kwargs), where args and kwargs are the same as were supplied in
    input, but each tensor in the original input. Must be specified if dummy_forward_fn is specified.
    :param dump_graphs: Whether or not should also dump the internal graph representation of the
    original and compressed models in the .dot format into the log directory.
    :return: A controller for the compression algorithm (or algorithms, in which case the controller
    is an instance of CompositeCompressionController) and the model ready for compression parameter training wrapped
    as an object of NNCFNetwork."""

    if dummy_forward_fn is not None and wrap_inputs_fn is None:
        raise ValueError("A custom dummy forward function was specified, but the corresponding input wrapping function "
                         "was not. In case a custom dummy forward function is specified for purposes of NNCF graph "
                         "building, then the wrap_inputs_fn parameter MUST also be specified and be consistent with "
                         "the input wrapping done in dummy_forward_fn.")

    if config is None and resuming_state_dict is None:
        raise ValueError("Config and resuming checkpoint can not be empty at the same time.")

    # Compress model that will be deployed for the inference on target device. No need to compress parts of the
    # model that are used on training stage only (e.g. AuxLogits of Inception-v3 model) or unused modules with weights.
    # As a consequence, no need to care about spoiling BN statistics, as there're disabled in eval mode.
    model.eval()

    builder_state = NNCFNetwork.get_builder_state(resuming_state_dict)
    is_strict = True
    should_init_per_builder = None
    if builder_state:
        saved_config = NNCFConfig.from_dict(builder_state['config'])
        # TODO: simplify???
        if config is None:
            config = saved_config
        else:
            should_init_per_builder, is_strict = _match_configs(config, saved_config)
    elif config is None:
        # TODO: proper release version
        raise ValueError("No config to create compressed model. At can be specified as an argument to the "
                         "create_compressed_model or can be loaded from the resuming checkpoint, which was created "
                         "with NNCF release > 1.7.1")

    if dump_graphs:
        if dummy_forward_fn is None:
            input_info_list = create_input_infos(config)
            graph_builder = GraphBuilder(custom_forward_fn=
                                         create_dummy_forward_fn(input_info_list,
                                                                 with_input_tracing=True))
        else:
            graph_builder = GraphBuilder(custom_forward_fn=dummy_forward_fn)

        if is_main_process():
            graph = graph_builder.build_graph(model)
            graph.visualize_graph(osp.join(config.get("log_dir", "."), "original_graph.dot"))

    set_debug_log_dir(config.get("log_dir", "."))

    input_info_list = create_input_infos(config)
    scopes_without_shape_matching = config.get('scopes_without_shape_matching', [])
    ignored_scopes = config.get('ignored_scopes')
    target_scopes = config.get('target_scopes')

    compressed_model = NNCFNetwork(model, input_infos=input_info_list,
                                   dummy_forward_fn=dummy_forward_fn,
                                   wrap_inputs_fn=wrap_inputs_fn,
                                   wrap_outputs_fn=wrap_outputs_fn,
                                   ignored_scopes=ignored_scopes,
                                   target_scopes=target_scopes,
                                   scopes_without_shape_matching=scopes_without_shape_matching)

    should_init = resuming_state_dict is None

    composite_builder = PTCompositeCompressionAlgorithmBuilder(config, should_init, should_init_per_builder)
    if builder_state:
        composite_builder.load_state(builder_state)

    composite_builder.apply_to(compressed_model)
    compressed_model.set_builder_state(composite_builder.get_state())
    compression_ctrl = composite_builder.build_controller(compressed_model)

    # Required to ensure that the model leaving create_compressed_model has correct compressed graph.
    # In particular, this is currently required for correct functioning of RNNs.
    compressed_model.rebuild_graph()

    try:
        if resuming_state_dict is not None:
            load_state(compressed_model, resuming_state_dict, is_resume=is_strict,
                       ignored_keys=[NNCFNetwork.BUILDER_STATE_ATTR])
    finally:
        if dump_graphs and is_main_process() and composite_builder:
            if dummy_forward_fn is None:
                compressed_graph_builder = GraphBuilder(custom_forward_fn=
                                                        create_dummy_forward_fn(input_info_list,
                                                                                with_input_tracing=False,
                                                                                with_output_tracing=False))
            else:
                compressed_graph_builder = GraphBuilder(custom_forward_fn=dummy_forward_fn)

            graph = compressed_graph_builder.build_graph(compressed_model, compressed_model.get_tracing_context())
            graph.visualize_graph(osp.join(config.get("log_dir", "."), "compressed_graph.dot"))

    # Synchronize all processes if run in distributed mode
    if is_dist_avail_and_initialized():
        try:
            barrier()
        # Exception can be raised during running barrier
        # if the backend not in the supported list https://pytorch.org/docs/stable/distributed.html
        except RuntimeError as err:
            logger.warning(err)
            logger.warning(
                "NNCF continues work, while does not guarantee that "
                "the processes will finish model's compression at the same time. "
                "If your training pipeline demands the processes be synchronized, please, "
                "keep attention to that error")
            return compression_ctrl, compressed_model

    return compression_ctrl, compressed_model


# ignore init section
def _remove_init_section(cfg: Dict):
    was_found = False
    if 'initializer' in cfg:
        was_found = True
        cfg['initializer'] = {}
    return was_found


def _match_params(cfg1: Dict, cfg2: Dict, param_vs_default_pairs: List[Tuple[str, Any]]):
    is_matched = True
    for param_name, default in param_vs_default_pairs:
        if cfg1.get(param_name, default) != cfg2.get(param_name, default):
            is_matched = False
            break
    return is_matched


def _get_algo_configs(cfg) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
    """
        Returns mapping of the algorithm name and its config.
        Sparsity algorithms have unified name ('sparsity') since they are inter loadable
        No entry for  non compression case
    """

    algo_name_vs_algo_config_map = {}
    algo_name_vs_algo_class_map = {}
    if isinstance(cfg, dict):
        algo_name = cfg.get('algorithm')
        if algo_name:
            algo_name_vs_algo_config_map[algo_name] = cfg
            algo_class = get_compression_algorithm(cfg)
            algo_name_vs_algo_class_map[algo_name] = algo_class
    else:
        for algo_config in cfg:
            algo_name = algo_config.get('algorithm')
            if algo_name:
                algo_name_vs_algo_config_map[algo_name] = algo_config
                algo_class = get_compression_algorithm(algo_config)
                algo_name_vs_algo_class_map[algo_name] = algo_class

    return algo_name_vs_algo_config_map, algo_name_vs_algo_class_map


def _match_configs(loaded_config: NNCFConfig, saved_config: NNCFConfig) -> Tuple[Dict['str', bool], bool]:
    loaded_compression_config = deepcopy(loaded_config.get('compression', {}))
    saved_compression_config = deepcopy(saved_config.get('compression', {}))

    global_param_vs_default_pairs = [('target_device', 'ANY'), ('ignored_scopes', []), ('target_scopes', [])]
    global_params_matches = _match_params(saved_config, loaded_config, global_param_vs_default_pairs)
    if not global_params_matches:
        raise RuntimeError('Global parameters for the given config and for the config from checkpoint do not match: ')

    saved_algo_configs, _ = _get_algo_configs(saved_compression_config)
    loaded_algo_configs, algo_name_vs_algo_config_map = _get_algo_configs(loaded_compression_config)

    is_strict_loading = set(saved_algo_configs.keys()) == set(loaded_algo_configs.keys())

    # just need to disable init for common
    algo_class_vs_should_init_map = {algo_class: True for algo_class in algo_name_vs_algo_config_map.values()}

    if not saved_algo_configs or not loaded_algo_configs:
        # CHECKPOINT                 -> CREATE                     - STS       - INIT                          - STRICT
        # ---------------------------|-----------------------------|-----------|-------------------------------|--------
        # quantization              -> empty                       - OK        -  no init                      - False
        # sparsity + quantization   -> empty                       - OK        -  no init                      - False
        # empty                     -> quantization                - OK        -  init quantization            - False
        # empty                     -> sparsity + quantization     - OK        -  init sparsity + quantization - False
        return algo_class_vs_should_init_map, is_strict_loading

    at_least_one_matches = False
    loaded_sparsity_algo = [algo for algo in loaded_algo_configs.keys() if 'sparsity' in algo]
    for algo_name, saved_cfg in saved_algo_configs.items():
        matched = False
        loaded_algo_name = algo_name
        if 'sparsity' in algo_name and algo_name not in loaded_algo_configs and loaded_sparsity_algo:
            param_vs_default_pairs = [('ignored_scopes', []), ('target_scopes', [])]
            loaded_algo_name = loaded_sparsity_algo[0]
            matched = _match_params(saved_cfg, loaded_algo_configs[loaded_algo_name], param_vs_default_pairs)

        if algo_name in loaded_algo_configs:
            loaded_algo_section = loaded_algo_configs[algo_name]
            has_init = _remove_init_section(loaded_algo_section)
            _remove_init_section(saved_cfg)
            if saved_cfg != loaded_algo_section:
                raise RuntimeError(
                    'Failed to resume algorithm {} with config different from one with which the algorithm was '
                    'originally created and saved to checkpoint'.format(algo_name))
            if has_init:
                nncf_logger.warning('{} is not going to be initialized, since it\'s resumed from checkpoint'.format(algo_name))
            matched = True
        if matched:
            # CHECKPOINT                -> CREATE                      - STS       - INIT              - STRICT
            # ---------------------------|-----------------------------|-----------|-------------------|----------
            # RB spars. + quantization  -> quantization                - OK        - no init           - False
            # RB spars.                 -> RB spars. + quantization    - OK        - init quantization - False
            # RB spars.                 -> CONST spars. + quantization - OK        - init quantization - False
            # RB spars. + quantization  -> quantization + pruning      - OK        - init quantization - False
            # RB spars. + quantization  -> CONST spars. + quantization - OK        - no init           - False
            # RB spars. + quantization  -> RB spars. + quantization    - OK        - no init           - True
            # quantization params1      -> quantization params1        - OK        - no init           - True
            # quantization params1      -> quantization params2        - not OK    -                   -

            # OK, if common part is matched and new loaded algo should be initialized
            at_least_one_matches = True
            algo_class = algo_name_vs_algo_config_map[loaded_algo_name]
            algo_class_vs_should_init_map[algo_class] = False

    if not at_least_one_matches:
        # CHECKPOINT               -> CREATE                    - STS       - INIT
        # -------------------------|----------------------------|-----------|----------
        # rb_sparsity + pruning   -> quantization               - not OK    -
        # rb_sparsity             -> quantization               - not OK    -
        raise RuntimeError('Created algorithms don\'t match to algorithms in the checkpoint: {} vs {}'.format(
            list(loaded_algo_configs.keys()), list(saved_algo_configs.keys()),
        ))

    return algo_class_vs_should_init_map, is_strict_loading
