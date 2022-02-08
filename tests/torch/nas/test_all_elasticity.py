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

import pytest
import torch
from torch.backends import cudnn

from examples.torch.common.models.classification.resnet_cifar10 import resnet50_cifar10
from nncf import NNCFConfig
from nncf.api.compression import CompressionStage
from nncf.torch.model_creation import create_nncf_network
from nncf.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.torch.nas.bootstrapNAS.elasticity.visualization import SubnetGraph
from nncf.torch.nas.bootstrapNAS.training.model_creator_helpers import resume_compression_from_state
from nncf.torch.utils import manual_seed
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.nas.creators import create_bnas_model_and_ctrl_by_test_desc
from tests.torch.nas.creators import create_bootstrap_nas_training_algo
from tests.torch.nas.creators import create_bootstrap_training_model_and_ctrl
from tests.torch.nas.descriptors import MultiElasticityTestDesc
from tests.torch.nas.descriptors import THREE_CONV_TEST_DESC
from tests.torch.nas.helpers import do_training_step
from tests.torch.nas.models.synthetic import ThreeConvModel
from tests.torch.nas.models.synthetic import ThreeConvModelMode
from tests.torch.test_compressed_graph import check_nx_graph


###########################
# Helpers
###########################


@pytest.fixture
def _seed():
    cudnn.deterministic = True
    cudnn.benchmark = False
    manual_seed(0)


def check_subnet_visualization(multi_elasticity_handler, model, nas_model_name, stage):
    model.rebuild_graph()
    dot_file = f'{nas_model_name}_{stage}.dot'
    width_graph = SubnetGraph(model.get_graph(), multi_elasticity_handler).get()
    check_nx_graph(width_graph, dot_file, 'nas')


###########################
# Behavior
###########################

RESNET50_2_MANUAL_BLOCKS_DESC = MultiElasticityTestDesc(
    model_creator=resnet50_cifar10,
    blocks_to_skip=[
        ["ResNet/Sequential[layer1]/Bottleneck[1]/ReLU[relu]/relu__2",
         "ResNet/Sequential[layer1]/Bottleneck[2]/ReLU[relu]/relu__2"],
        ["ResNet/Sequential[layer4]/Bottleneck[1]/ReLU[relu]/relu__2",
         "ResNet/Sequential[layer4]/Bottleneck[2]/ReLU[relu]/relu__2"]
    ],
    blocks_dependencies={0: [0, 1]}
)


def test_bn_adaptation_on_minimal_subnet_width_stage():
    model, ctrl = create_bnas_model_and_ctrl_by_test_desc(RESNET50_2_MANUAL_BLOCKS_DESC, mode='manual')

    multi_elasticity_handler = ctrl.multi_elasticity_handler
    # pylint: disable=protected-access
    bn_adaptation = ctrl._bn_adaptation

    multi_elasticity_handler.enable_all()
    model.do_dummy_forward()

    multi_elasticity_handler.width_handler.reorganize_weights()
    bn_adaptation.run(model)
    model.do_dummy_forward()

    multi_elasticity_handler.activate_minimum_subnet()
    # ERROR HERE
    bn_adaptation.run(model)
    model.do_dummy_forward()


###########################
# Random sampling
###########################

NAS_MODELS_SCOPE = [
    'resnet18',
    'resnet50',
    'densenet_121',
    'mobilenet_v2',
    'vgg11',
    'vgg11_k7',
    'inception_v3',
    'efficient_net_b0',
    'shufflenetv2',
    'unet',
    'squeezenet1_0',
    'resnext29_32x4d',
    'pnasnetb',
    'ssd_mobilenet',
    'ssd_vgg',
    'mobilenet_v3_small'
]


@pytest.fixture(name='nas_model_name', scope='function', params=NAS_MODELS_SCOPE)
def fixture_nas_model_name(request):
    return request.param


def test_random_multi_elasticity(_seed, nas_model_name):
    if 'inception_v3' in nas_model_name:
        pytest.skip(
            f'Skip test for {nas_model_name} as it fails because of 2 issues: '
            'not able to set DynamicInputOp to train-only layers (ticket 60976) and '
            'invalid padding update in elastic kernel (ticket 60990)')

    model, ctrl, _ = create_bootstrap_nas_training_algo(nas_model_name)

    multi_elasticity_handler = ctrl.multi_elasticity_handler
    model.do_dummy_forward()

    multi_elasticity_handler.disable_all()
    multi_elasticity_handler.enable_elasticity(ElasticityDim.KERNEL)
    multi_elasticity_handler.activate_random_subnet()
    model.do_dummy_forward()
    check_subnet_visualization(multi_elasticity_handler, model, nas_model_name, stage='kernel')

    multi_elasticity_handler.enable_elasticity(ElasticityDim.DEPTH)
    multi_elasticity_handler.activate_random_subnet()
    if 'squeezenet1_0' in nas_model_name:
        pytest.skip(
            f'Skip test for {nas_model_name} as it fails with error: Given groups=1, weight of '
            'size [48, 256, 1, 1], expected input[1, 32, 4, 4] to have 256 channels, but got 32 channels instead')
    model.do_dummy_forward()
    check_subnet_visualization(multi_elasticity_handler, model, nas_model_name, stage='depth')

    multi_elasticity_handler.enable_elasticity(ElasticityDim.WIDTH)
    multi_elasticity_handler.width_handler.width_num_params_indicator = 1
    multi_elasticity_handler.activate_random_subnet()
    model.do_dummy_forward()
    # TODO(nlyalyus): some blocks are not skipped because they fail a conflict resolving with elastic width: start or
    #  end block can be out of pruning scope and can't retrieve width in the propagation graph: mask = None
    check_subnet_visualization(multi_elasticity_handler, model, nas_model_name, stage='width')


###########################
# Outputs
###########################

def test_multi_elasticity_outputs():
    model, ctrl = create_bnas_model_and_ctrl_by_test_desc(THREE_CONV_TEST_DESC, mode='manual')
    multi_elasticity_handler = ctrl.multi_elasticity_handler
    ref_model = THREE_CONV_TEST_DESC.model_creator()
    device = next(iter(model.parameters())).device
    ref_model.to(device)
    input_ = torch.ones(ref_model.INPUT_SIZE).to(device)
    ref_model.mode = ThreeConvModelMode.SUPERNET
    ref_output = ref_model(input_)
    actual_output = model(input_)
    assert torch.equal(actual_output, ref_output)

    multi_elasticity_handler.disable_all()
    multi_elasticity_handler.enable_elasticity(ElasticityDim.KERNEL)
    multi_elasticity_handler.activate_minimum_subnet()
    actual_output = model(input_)
    ref_model.mode = ThreeConvModelMode.KERNEL_STAGE
    ref_output = ref_model(input_)
    assert torch.equal(actual_output, ref_output)

    multi_elasticity_handler.enable_elasticity(ElasticityDim.DEPTH)
    multi_elasticity_handler.activate_minimum_subnet()
    actual_output = model(input_)
    ref_model.mode = ThreeConvModelMode.DEPTH_STAGE
    ref_output = ref_model(input_)
    assert torch.equal(actual_output, ref_output)

    multi_elasticity_handler.enable_all()
    multi_elasticity_handler.activate_minimum_subnet()
    print(multi_elasticity_handler.get_active_config())
    actual_output = model(input_)
    ref_model.mode = ThreeConvModelMode.WIDTH_STAGE
    ref_output = ref_model(input_)
    assert torch.equal(actual_output, ref_output)


def test_multi_elasticity_gradients():
    model, ctrl = create_bnas_model_and_ctrl_by_test_desc(THREE_CONV_TEST_DESC, mode='manual')
    multi_elasticity_handler = ctrl.multi_elasticity_handler
    ref_model = ThreeConvModel()
    if torch.cuda.is_available():
        ref_model.cuda()
    device = next(iter(ref_model.parameters())).device

    actual_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    ref_optimizer = torch.optim.Adam(ref_model.parameters(), lr=0.01)
    input_ = torch.ones(ThreeConvModel.INPUT_SIZE).to(device)

    ref_model.mode = ThreeConvModelMode.SUPERNET
    check_output_and_weights_after_training_step(model, ref_model, actual_optimizer, ref_optimizer, input_)

    multi_elasticity_handler.disable_all()
    multi_elasticity_handler.enable_elasticity(ElasticityDim.KERNEL)
    multi_elasticity_handler.activate_minimum_subnet()
    ref_model.mode = ThreeConvModelMode.KERNEL_STAGE
    check_output_and_weights_after_training_step(model, ref_model, actual_optimizer, ref_optimizer, input_)

    multi_elasticity_handler.enable_elasticity(ElasticityDim.DEPTH)
    multi_elasticity_handler.activate_minimum_subnet()
    ref_model.mode = ThreeConvModelMode.DEPTH_STAGE
    check_output_and_weights_after_training_step(model, ref_model, actual_optimizer, ref_optimizer, input_)

    multi_elasticity_handler.enable_elasticity(ElasticityDim.WIDTH)
    multi_elasticity_handler.activate_minimum_subnet()
    ref_model.mode = ThreeConvModelMode.WIDTH_STAGE
    check_output_and_weights_after_training_step(model, ref_model, actual_optimizer, ref_optimizer, input_)


def check_output_and_weights_after_training_step(model, ref_model, actual_optimizer, ref_optimizer, input_):
    ref_output = do_training_step(ref_model, ref_optimizer, input_)
    actual_output = do_training_step(model, actual_optimizer, input_)
    assert torch.equal(actual_output, ref_output)
    ref_model.assert_weights_equal(model.get_nncf_wrapped_model())
    transformation_matrix = [param.data for name, param in model.named_parameters() if name.endswith('5to3_matrix')]
    ref_model.assert_transition_matrix_equals(transformation_matrix[0])


###########################
# Outputs
###########################

REF_COMPRESSION_STATE_FOR_TWO_CONV = {
    'builder_state': {
        'progressive_shrinking': {
            'elasticity_builder_state': {
                'elasticity': {
                    'builder_states': {
                        'depth': {
                            'mode': 'manual',
                            'skipped_blocks': [
                                {
                                    'start_node': 'ThreeConvModel/NNCFConv2d[conv1]/conv2d_0',
                                    'end_node': '/nncf_model_output_0'
                                }
                            ],
                            'skipped_blocks_dependencies': {0: [0]},
                            'ordinal_ids': None,
                        },
                        'width': {
                            'elasticity_params': {'min_out_channels': 1, 'width_step': 1},
                            'grouped_node_names_to_prune': [['ThreeConvModel/NNCFConv2d[conv1]/conv2d_0',
                                                             'ThreeConvModel/NNCFConv2d[conv_to_skip]/conv2d_0']]
                        }
                    },
                    'available_elasticity_dims': ['width', 'depth']
                }
            },
            'progressivity_of_elasticity': ['kernel', 'width', 'depth'],
        }
    },
    'ctrl_state': {
        'elasticity_controller_compression_state': {
            'elasticity': {
                'compression_stage': CompressionStage.UNCOMPRESSED,
                'loss_state': None,
                'scheduler_state': {'current_epoch': -1, 'current_step': -1}
            },
            'multi_elasticity_handler_state': {
                'is_handler_enabled_map': {
                    'depth': True,
                    'width': True
                },
                'states_of_handlers': {
                    'depth': {
                        'active_config': [0]
                    },
                    'width': {
                        'active_config': {0: 1}
                    }
                }
            }
        },
        'progressive_shrinking': {
            'compression_stage': CompressionStage.PARTIALLY_COMPRESSED,
            'loss_state': None,
            'scheduler_state': {
                'current_epoch': 2, 'current_step': -1,
                'list_stage_descriptions': [
                    {
                        'bn_adapt': True, 'depth_indicator': 1, 'epochs': 1, 'reorg_weights': False,
                        'train_dims': ['width'], 'width_indicator': 2
                    },
                    {
                        'bn_adapt': False, 'depth_indicator': 2, 'epochs': 1, 'reorg_weights': True,
                        'train_dims': ['depth', 'width'], 'width_indicator': 1
                    }
                ]
            }
        }
    }
}

TWO_CONV_FULL_CONFIG = {
    "input_info": {"sample_size": THREE_CONV_TEST_DESC.input_sizes},
    "bootstrapNAS": {
        "training": {
            "batchnorm_adaptation": {
                "num_bn_adaptation_samples": 2
            },
            "progressivity_of_elasticity": ["kernel", "width", "depth"],
            "elasticity": {
                "available_elasticity_dims": ["width", "depth"],
                "depth": {
                    'mode': 'manual',
                    'skipped_blocks': THREE_CONV_TEST_DESC.blocks_to_skip,
                    'skipped_blocks_dependencies': THREE_CONV_TEST_DESC.blocks_dependencies
                },
                "kernel": {
                    "max_num_kernels": "3"
                },
                **THREE_CONV_TEST_DESC.algo_params
            },
            "schedule": {
                "list_stage_descriptions": [
                    {"train_dims": ["width"], "epochs": 1, "width_indicator": 2, "bn_adapt": True},
                    {"train_dims": ["depth", "width"], "epochs": 1, "depth_indicator": 2, "reorg_weights": True}
                ]
            },
        },
    }
}


def prepare_train_algo_for_resume(training_ctrl):
    multi_elasticity_handler = training_ctrl.multi_elasticity_handler
    multi_elasticity_handler.enable_all()
    multi_elasticity_handler.activate_minimum_subnet()
    training_ctrl.scheduler.epoch_step(next_epoch=2)


def test_multi_elasticity_state():
    nncf_config = NNCFConfig.from_dict(TWO_CONV_FULL_CONFIG)
    model = THREE_CONV_TEST_DESC.model_creator()
    _, training_ctrl = create_bootstrap_training_model_and_ctrl(model, nncf_config)

    prepare_train_algo_for_resume(training_ctrl)
    compression_state = training_ctrl.get_compression_state()

    assert compression_state == REF_COMPRESSION_STATE_FOR_TWO_CONV
    # TODO: check that compression is json serializable
    # check_serialization(training_ctrl)


def test_can_restore_from_state():
    nncf_config = NNCFConfig.from_dict(TWO_CONV_FULL_CONFIG)
    model = THREE_CONV_TEST_DESC.model_creator()
    _, training_ctrl = create_bootstrap_training_model_and_ctrl(model, nncf_config)
    prepare_train_algo_for_resume(training_ctrl)

    old_state = training_ctrl.get_compression_state()

    empty_nncf_config = NNCFConfig({"input_info": {"sample_size": THREE_CONV_TEST_DESC.input_sizes}})
    register_bn_adaptation_init_args(empty_nncf_config)
    clean_model = THREE_CONV_TEST_DESC.model_creator()
    nncf_network = create_nncf_network(clean_model, empty_nncf_config)
    _, training_ctrl = resume_compression_from_state(nncf_network, old_state, empty_nncf_config)

    new_state = training_ctrl.get_compression_state()
    assert new_state == old_state


def test_can_restore_and_get_the_same_output():
    ref_model = THREE_CONV_TEST_DESC.model_creator()
    if torch.cuda.is_available():
        ref_model.cuda()
    device = next(iter(ref_model.parameters())).device
    input_ = torch.ones(ThreeConvModel.INPUT_SIZE).to(device)

    model, training_ctrl = create_bnas_model_and_ctrl_by_test_desc(THREE_CONV_TEST_DESC, mode='manual')
    multi_elasticity_handler = training_ctrl.multi_elasticity_handler
    multi_elasticity_handler.enable_all()
    multi_elasticity_handler.activate_minimum_subnet()

    actual_output = training_ctrl.model(input_)
    ref_model.mode = ThreeConvModelMode.WIDTH_STAGE
    ref_output = ref_model(input_)
    assert torch.equal(actual_output, ref_output)

    old_state = training_ctrl.get_compression_state()

    empty_nncf_config = NNCFConfig({"input_info": {"sample_size": THREE_CONV_TEST_DESC.input_sizes}})
    register_bn_adaptation_init_args(empty_nncf_config)
    clean_model = THREE_CONV_TEST_DESC.model_creator()
    if torch.cuda.is_available():
        clean_model.cuda()
    nncf_network = create_nncf_network(clean_model, empty_nncf_config)
    model, _ = resume_compression_from_state(nncf_network, old_state, empty_nncf_config)

    actual_output = model(input_)
    ref_model.mode = ThreeConvModelMode.WIDTH_STAGE
    ref_output = ref_model(input_)
    assert torch.equal(actual_output, ref_output)
