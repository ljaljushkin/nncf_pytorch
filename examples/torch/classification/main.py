"""
 Copyright (c) 2019-2023 Intel Corporation
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
import os.path as osp
import sys
import time
import warnings
from copy import deepcopy
from functools import partial
from pathlib import Path
from shutil import copyfile
from typing import Any

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
from torch.backends import cudnn
from torch.cuda.amp.autocast_mode import autocast
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets
from torchvision import models
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.models import InceptionOutputs

from examples.common.paths import configure_paths
from examples.common.sample_config import SampleConfig
from examples.common.sample_config import create_sample_config
from examples.torch.common.argparser import get_common_argument_parser
from examples.torch.common.argparser import parse_args
from examples.torch.common.example_logger import logger
from examples.torch.common.execution import ExecutionMode
from examples.torch.common.execution import get_execution_mode
from examples.torch.common.execution import prepare_model_for_execution
from examples.torch.common.execution import set_seed
from examples.torch.common.execution import start_worker
from examples.torch.common.export import export_model
from examples.torch.common.model_loader import COMPRESSION_STATE_ATTR
from examples.torch.common.model_loader import MODEL_STATE_ATTR
from examples.torch.common.model_loader import extract_model_and_compression_states
from examples.torch.common.model_loader import load_model
from examples.torch.common.model_loader import load_resuming_checkpoint
from examples.torch.common.optimizer import get_parameter_groups
from examples.torch.common.optimizer import make_optimizer
from examples.torch.common.utils import MockDataset
from examples.torch.common.utils import NullContextManager
from examples.torch.common.utils import configure_device
from examples.torch.common.utils import configure_logging
from examples.torch.common.utils import create_code_snapshot
from examples.torch.common.utils import get_run_name
from examples.torch.common.utils import is_pretrained_model_requested
from examples.torch.common.utils import is_staged_quantization
from examples.torch.common.utils import make_additional_checkpoints
from examples.torch.common.utils import print_args
from examples.torch.common.utils import write_metrics

model_names = sorted(name for name, val in models.__dict__.items()
                     if name.islower() and not name.startswith("__")
                     and callable(val))

def default_criterion_fn(outputs: Any, target: Any, criterion: Any) -> torch.Tensor:
    return criterion(outputs, target)

def get_argument_parser():
    parser = get_common_argument_parser()
    parser.add_argument(
        "--dataset",
        help="Dataset to use.",
        choices=["imagenet", "cifar100", "cifar10"],
        default=None
    )
    parser.add_argument('--test-every-n-epochs', default=1, type=int,
                        help='Enables running validation every given number of epochs')
    parser.add_argument('--mixed-precision',
                        dest='mixed_precision',
                        help='Enables torch.cuda.amp autocasting during training and'
                             ' validation steps',
                        action='store_true')
    return parser


def main(argv):
    parser = get_argument_parser()
    args = parse_args(parser, argv)
    config = create_sample_config(args, parser)

    if config.dist_url == "env://":
        config.update_from_env()

    configure_paths(config, get_run_name(config))
    copyfile(args.config, osp.join(config.log_dir, "config.json"))
    source_root = Path(__file__).absolute().parents[2]  # nncf root
    create_code_snapshot(source_root, osp.join(config.log_dir, "snapshot.tar.gz"))

    if config.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    config.execution_mode = get_execution_mode(config)

    if config.metrics_dump is not None:
        write_metrics(0, config.metrics_dump)

    main_worker(current_gpu=None, config=config)

from torch import distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

# pylint:disable=too-many-branches,too-many-statements
def main_worker(current_gpu, config: SampleConfig):
    configure_device(current_gpu, config)
    # config.mlflow = SafeMLFLow(config)
    if is_main_process():
        configure_logging(logger, config)
        print_args(config)
    else:
        config.tb = None

    set_seed(config)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(config.device)

    model_name = config['model']
    train_criterion_fn = default_criterion_fn

    train_loader = train_sampler = val_loader = None
    resuming_checkpoint_path = config.resuming_checkpoint_path
    nncf_config = config.nncf_config
    pretrained = is_pretrained_model_requested(config)
    is_export_only = 'export' in config.mode and ('train' not in config.mode and 'test' not in config.mode)

    train_dataset = create_datasets(config)
    train_loader, train_sampler = create_data_loaders(config, train_dataset)
    model = load_model(model_name,
                       pretrained=pretrained,
                       num_classes=config.get('num_classes', 1000),
                       model_params=config.get('model_params'),
                       weights_path=config.get('weights'))

    model.to(config.device)

    model, _ = prepare_model_for_execution(model, config)

    params_to_optimize = get_parameter_groups(model, config)
    optimizer, lr_scheduler = make_optimizer(params_to_optimize, config)

    best_acc1 = 0
    cudnn.benchmark = True


    train(config, model, criterion, train_criterion_fn, lr_scheduler, model_name, optimizer,
                train_loader, train_sampler, val_loader, best_acc1)


def train(config, model, criterion, criterion_fn, lr_scheduler, model_name, optimizer,
          train_loader, train_sampler, val_loader, best_acc1=0):
    for epoch in range(config.start_epoch, config.epochs):
        train_epoch(train_loader, model, criterion, criterion_fn, optimizer, epoch, config)


def get_dataset(dataset_config, config, transform, is_train):
    num_images = config.get('num_mock_images', 1000)
    return MockDataset(img_size=(32, 32), transform=transform, num_images=num_images)


def create_datasets(config):
    dataset_config = config.dataset if config.dataset is not None else 'imagenet'
    dataset_config = dataset_config.lower()
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                         std=(0.5, 0.5, 0.5))
    image_size = 32
    size = int(image_size / 0.875)
    train_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = get_dataset(dataset_config, config, train_transforms, is_train=True)

    return train_dataset


def create_data_loaders(config, train_dataset):
    pin_memory = config.execution_mode != ExecutionMode.CPU_ONLY
    batch_size = int(config.batch_size)
    workers = int(config.workers)
    batch_size_val = int(config.batch_size_val) if config.batch_size_val is not None else int(config.batch_size)
    train_sampler = None
    train_shuffle = train_sampler is None and config.seed is None

    def create_train_data_loader(batch_size_):
        return torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size_, shuffle=train_shuffle,
            num_workers=workers, pin_memory=pin_memory, sampler=train_sampler, drop_last=True)

    train_loader = create_train_data_loader(batch_size)

    return train_loader, train_sampler


def train_epoch(train_loader, model, criterion, criterion_fn, optimizer, epoch, config,
                train_iters=None, log_training_info=True):
    if train_iters is None:
        train_iters = len(train_loader)

    model.train()

    for i, (input_, target) in enumerate(train_loader):
        input_ = input_.to(config.device)
        target = target.to(config.device)

        output = model(input_)
        criterion_loss = criterion_fn(output, target, criterion)

        loss = criterion_loss# + compression_loss

        if isinstance(output, InceptionOutputs):
            output = output.logits
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main(sys.argv[1:])
