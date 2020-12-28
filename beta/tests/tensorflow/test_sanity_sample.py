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

import json
import os
import tempfile
from operator import itemgetter
import pytest
import tensorflow as tf

from tests.conftest import TEST_ROOT
from tests.tensorflow.helpers import get_coco_dataset_builders
from tests.tensorflow.test_models import SequentialModel, SequentialModelNoInput

from examples.tensorflow.classification import main as cls_main
from examples.tensorflow.object_detection import main as od_main
from examples.tensorflow.common.model_loader import AVAILABLE_MODELS

od_main.get_dataset_builders = get_coco_dataset_builders
AVAILABLE_MODELS.update({
    'SequentialModel': SequentialModel,
    'SequentialModelNoInput': SequentialModelNoInput
})


class ConfigFactory:
    """Allows to modify config file before test run"""

    def __init__(self, base_config, config_path):
        self.config = base_config
        self.config_path = str(config_path)

    def serialize(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)
        return self.config_path

    def __getitem__(self, item):
        return self.config[item]

    def __setitem__(self, key, value):
        self.config[key] = value


def convert_to_argv(args):
    return ' '.join(key if val is None else '{} {}'.format(key, val) for key, val in args.items()).split()


SAMPLE_TYPES = ['classification', 'object_detection']

SAMPLES = {
    'classification': cls_main.main,
    'object_detection': od_main.main,
}

DATASETS = {
    'classification': [('cifar10', 'tfds'), ('cifar10', 'tfds'), ('cifar10', 'tfds')],
    'object_detection': [('coco2017', 'tfrecords')],
}

CONFIGS = {
    'classification': [
        TEST_ROOT.joinpath('tensorflow', 'data', 'configs',
                           'resnet50_cifar10_magnitude_sparsity_int8.json'),
        TEST_ROOT.joinpath('tensorflow', 'data', 'configs',
                           'sequential_model_cifar10_magnitude_sparsity_int8.json'),
        TEST_ROOT.joinpath('tensorflow', 'data', 'configs',
                           'sequential_model_no_input_cifar10_magnitude_sparsity_int8.json')
    ],
    'object_detection': [TEST_ROOT.joinpath('tensorflow', 'data', 'configs',
                                            'retinanet_coco2017_magnitude_sparsity_int8.json')]
}

BATCHSIZE_PER_GPU = {
    'classification': [256, 256, 256],
    'object_detection': [3, 3],
}

DATASET_PATHS = {
    'classification': {
        x: lambda dataset_root, dataset_name=x:
        os.path.join(dataset_root, dataset_name) if dataset_root else
        os.path.join(tempfile.gettempdir(), dataset_name)
        for x, _ in DATASETS['classification']
    },
    'object_detection': {
        'coco2017': lambda dataset_root: TEST_ROOT.joinpath('tensorflow', 'data', 'mock_datasets', 'coco2017')
    }
}

CONFIG_PARAMS = list()
for i, sample in enumerate(SAMPLE_TYPES):
    for idx, tpl in enumerate(list(zip(CONFIGS[sample],
                                      map(itemgetter(0), DATASETS[sample]),
                                      map(itemgetter(1), DATASETS[sample]),
                                      BATCHSIZE_PER_GPU[sample]))):
        CONFIG_PARAMS.append((sample,) + tpl + ('{}_{}'.format(i, idx),))


@pytest.fixture(params=CONFIG_PARAMS,
                ids=['-'.join([p[0], p[1].name, p[2], p[3], str(p[4])]) for p in CONFIG_PARAMS])
def _config(request, dataset_dir):
    sample_type, config_path, dataset_name, dataset_type, batch_size, tid = request.param
    dataset_path = DATASET_PATHS[sample_type][dataset_name](dataset_dir)

    with config_path.open() as f:
        jconfig = json.load(f)

    if 'checkpoint_save_dir' in jconfig.keys():
        del jconfig['checkpoint_save_dir']

    jconfig['dataset'] = dataset_name
    jconfig['dataset_type'] = dataset_type

    return {
        'sample_type': sample_type,
        'nncf_config': jconfig,
        'model_name': jconfig['model'],
        'dataset_path': dataset_path,
        'batch_size': batch_size,
        'tid': tid
    }


@pytest.fixture(scope='module')
def _case_common_dirs(tmp_path_factory):
    return {
        'checkpoint_save_dir': str(tmp_path_factory.mktemp('models'))
    }


def test_model_eval(_config, tmp_path):
    config_factory = ConfigFactory(_config['nncf_config'], tmp_path / 'config.json')
    args = {
        '--mode': 'test',
        '--data': _config['dataset_path'],
        '--config': config_factory.serialize(),
        '--log-dir': tmp_path,
        '--batch-size': _config['batch_size']
    }

    main = SAMPLES[_config['sample_type']]
    main(convert_to_argv(args))


@pytest.mark.dependency()
def test_model_train(_config, tmp_path, _case_common_dirs):
    checkpoint_save_dir = os.path.join(_case_common_dirs['checkpoint_save_dir'], _config['tid'])
    config_factory = ConfigFactory(_config['nncf_config'], tmp_path / 'config.json')
    args = {
        '--mode': 'train',
        '--data': _config['dataset_path'],
        '--config': config_factory.serialize(),
        '--log-dir': tmp_path,
        '--batch-size': _config['batch_size'],
        '--epochs': 1,
        '--checkpoint-save-dir': checkpoint_save_dir
    }

    main = SAMPLES[_config['sample_type']]
    main(convert_to_argv(args))

    assert tf.io.gfile.isdir(checkpoint_save_dir)
    assert tf.train.latest_checkpoint(checkpoint_save_dir)


@pytest.mark.dependency(depends=['test_model_train'])
def test_trained_model_eval(_config, tmp_path, _case_common_dirs):
    config_factory = ConfigFactory(_config['nncf_config'], tmp_path / 'config.json')
    ckpt_path = os.path.join(_case_common_dirs['checkpoint_save_dir'], _config['tid'])
    args = {
        '--mode': 'test',
        '--data': _config['dataset_path'],
        '--config': config_factory.serialize(),
        '--log-dir': tmp_path,
        '--batch-size': _config['batch_size'],
        '--resume': ckpt_path
    }

    main = SAMPLES[_config['sample_type']]
    main(convert_to_argv(args))


@pytest.mark.dependency(depends=['test_model_train'])
def test_resume(_config, tmp_path, _case_common_dirs):
    checkpoint_save_dir = os.path.join(str(tmp_path), 'models')
    config_factory = ConfigFactory(_config['nncf_config'], tmp_path / 'config.json')
    ckpt_path = os.path.join(_case_common_dirs['checkpoint_save_dir'], _config['tid'])

    args = {
        '--mode': 'train',
        '--data': _config['dataset_path'],
        '--config': config_factory.serialize(),
        '--log-dir': tmp_path,
        '--batch-size': _config['batch_size'],
        '--epochs': 2,
        '--checkpoint-save-dir': checkpoint_save_dir,
        '--resume': ckpt_path
    }

    main = SAMPLES[_config['sample_type']]
    main(convert_to_argv(args))

    assert tf.io.gfile.isdir(checkpoint_save_dir)
    assert tf.train.latest_checkpoint(checkpoint_save_dir)


@pytest.mark.dependency(depends=['test_model_train'])
def test_trained_model_resume_train_test_export_last_ckpt(_config, tmp_path, _case_common_dirs):
    checkpoint_save_dir = os.path.join(str(tmp_path), 'models')
    config_factory = ConfigFactory(_config['nncf_config'], tmp_path / 'config.json')
    ckpt_path = os.path.join(_case_common_dirs['checkpoint_save_dir'], _config['tid'])

    export_path = os.path.join(str(tmp_path), 'model.pb')
    args = {
        '--mode': 'train test export',
        '--data': _config['dataset_path'],
        '--config': config_factory.serialize(),
        '--log-dir': tmp_path,
        '--batch-size': _config['batch_size'],
        '--epochs': 2,
        '--checkpoint-save-dir': checkpoint_save_dir,
        '--resume': ckpt_path,
        '--to-frozen-graph': export_path
    }

    main = SAMPLES[_config['sample_type']]
    main(convert_to_argv(args))

    assert tf.io.gfile.isdir(checkpoint_save_dir)
    assert tf.train.latest_checkpoint(checkpoint_save_dir)
    assert os.path.exists(export_path)


FORMATS = [
    'frozen-graph',
    'saved-model',
    'h5'
]


def get_export_model_name(export_format):
    model_name = 'model'
    if export_format == 'frozen-graph':
        model_name = 'model.pb'
    elif export_format == 'h5':
        model_name = 'model.h5'
    return model_name


@pytest.mark.dependency(depends=['test_model_train'])
@pytest.mark.parametrize('export_format', FORMATS, ids=FORMATS)
def test_export_with_resume(_config, tmp_path, export_format, _case_common_dirs):
    config_factory = ConfigFactory(_config['nncf_config'], tmp_path / 'config.json')
    ckpt_path = os.path.join(_case_common_dirs['checkpoint_save_dir'], _config['tid'])

    if export_format == 'saved-model':
        compression_config = _config['nncf_config'].get('compression', {})
        if isinstance(compression_config, dict):
            compression_config = [compression_config]
        for config in compression_config:
            if config.get('algorithm', '') == 'quantization':
                pytest.skip()

    export_path = os.path.join(str(tmp_path), get_export_model_name(export_format))
    args = {
        '--mode': 'export',
        '--config': config_factory.serialize(),
        '--log-dir': tmp_path,
        '--resume': ckpt_path,
        '--to-{}'.format(export_format): export_path,
    }

    main = SAMPLES[_config['sample_type']]
    main(convert_to_argv(args))

    model_path = os.path.join(export_path, 'saved_model.pb') \
        if export_format == 'saved-model' else export_path
    assert os.path.exists(model_path)
