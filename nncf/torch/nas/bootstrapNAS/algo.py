"""
 Copyright (c) 2019-2020 Intel Corporation
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
import math
import os
import random
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Callable
from typing import NamedTuple

import torch
import torch.nn as nn
from nncf.torch.module_operations import UpdateWeight

from nncf.torch.layers import NNCFConv2d
import torch.nn.functional as F

from nncf.torch.graph.operator_metatypes import Conv2dMetatype

from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.graph_matching import find_subgraphs_matching_pattern
from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.pruning.clusterization import Cluster
from nncf.common.pruning.clusterization import Clusterization
from nncf.common.pruning.pruning_node_selector import PruningNodeSelector
from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.logger import logger as nncf_logger
from nncf.torch.algo_selector import COMPRESSION_ALGORITHMS
from nncf.torch.algo_selector import ZeroCompressionLoss
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.graph.operator_metatypes import BatchNormMetatype
from nncf.torch.graph.operator_metatypes import LinearMetatype
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.module_operations import UpdateBatchNormParams
from nncf.torch.module_operations import UpdatePadding
from nncf.torch.module_operations import UpdateWeightAndOptionalBias
from nncf.torch.nas.bootstrapNAS.layers import DynamicBatchNormInputOp
from nncf.torch.nas.bootstrapNAS.layers import DynamicConvInputOp
from nncf.torch.nas.bootstrapNAS.layers import DynamicLinearInputOp
from nncf.torch.nas.bootstrapNAS.layers import ElasticConv2DOp  # Unified Operator
from nncf.torch.nas.bootstrapNAS.layers import ElasticKernelPaddingAdjustment
from nncf.torch.layers import NNCFBatchNorm2d
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.pruning.export_helpers import PTElementwise
from nncf.torch.pruning.export_helpers import PT_PRUNING_OPERATOR_METATYPES
from nncf.torch.pruning.structs import PrunedModuleInfo
from nncf.torch.utils import is_main_process


class ElasticWidthInfo(NamedTuple):
    node_name: NNCFNodeName
    module: nn.Module
    layer_name: str
    elastic_op: Callable
    node_id: int


@COMPRESSION_ALGORITHMS.register('bootstrapNAS')
class BootstrapNASBuilder(PTCompressionAlgorithmBuilder):
    def __init__(self, config, should_init: bool = True):
        super().__init__(config, should_init)
        self._weight_dynamic = OrderedDict()
        self._processed_insertion_points = set()  # type: Set[PTTargetPoint]

        self.config = config
        self.elastic_kernel_ops = []
        self.elastic_width_ops = []
        self.scope_vs_elastic_kernel_op_map = OrderedDict()
        self.scope_vs_elastic_width_op_map = OrderedDict()

        # Unified op
        self.scope_vs_elastic_op_map = OrderedDict()
        self.pruned_module_groups_info = Clusterization[PrunedModuleInfo](id_fn=lambda x: x.node_name)

        # TODO: Read elastic parameters from previous checkpoints and decide the level of elasticity that should be applied to the model.


    def initialize(self, model: NNCFNetwork) -> None:
        pass

    def build_controller(self, target_model: NNCFNetwork) -> 'BootstrapNASController':
        return BootstrapNASController(target_model, self.config, self.scope_vs_elastic_op_map,
                                      self.elastic_kernel_ops, self.elastic_width_ops, self.pruned_module_groups_info)

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        layout = PTTransformationLayout()
        # TODO: Skip e_width and e_kernel if depth stage.
        commands = []
        commands.extend(self._elastic_k_w(target_model))
        for command in commands:
            layout.register(command)
        return layout

    # Unified operator
    @staticmethod
    def create_elastic_conv_k_w(module: nn.Conv2d, scope, config):
        return ElasticConv2DOp(module.kernel_size[0], module.out_channels, scope, config)

    @staticmethod
    def create_dynamic_conv_input_op(module: nn.Conv2d, scope, device):
        dynamic_conv_input_op = DynamicConvInputOp(module.in_channels, scope)
        return UpdateWeight(dynamic_conv_input_op).to(device)

    @staticmethod
    def create_dynamic_bn_input_op(module: nn.BatchNorm2d, scope, device):
        dynamic_bn_input_op = DynamicBatchNormInputOp(module.num_features, scope)
        return UpdateBatchNormParams(dynamic_bn_input_op).to(device)

    @staticmethod
    def create_dynamic_linear_input_op(module: nn.Linear, scope, device):
        dynamic_linear_input_op = DynamicLinearInputOp(module.in_features, scope)
        return UpdateWeight(dynamic_linear_input_op).to(device)

    def _elastic_k_w(self, target_model: NNCFNetwork):
        target_model_graph = target_model.get_original_graph()
        device = next(target_model.parameters()).device
        insertion_commands = []
        pad_commands = []

        prunable_types = [NNCFConv2d.op_func_name]
        types_of_grouping_ops = PTElementwise.get_all_op_aliases()

        pruning_node_selector = PruningNodeSelector(PT_PRUNING_OPERATOR_METATYPES,
                                                    prunable_types,
                                                    types_of_grouping_ops,
                                                    None,
                                                    # layers_with_elasticity,
                                                    target_scopes=self.config.get('target_scopes', None),
                                                    prune_first=True,
                                                    prune_last=True,
                                                    prune_downsample_convs=True)

        groups_of_nodes_to_prune = pruning_node_selector.create_pruning_groups(target_model_graph)
        for i, group in enumerate(groups_of_nodes_to_prune.get_all_clusters()):
            group_minfos = []
            print(f'Group #{i}')
            for conv_node in group.elements:
                conv_node_name = conv_node.node_name
                conv_module = target_model.get_containing_module(conv_node_name)

                # Currently prune only Convolutions
                assert isinstance(conv_module, nn.Conv2d), 'currently prune only 2D Convolutions'

                # if conv_module.kernel_size[0] <= 3:
                #     continue
                nncf_logger.info("Adding Dynamic Conv2D Layer in scope: {}".format(str(conv_node_name)))

                elastic_conv_operation = self.create_elastic_conv_k_w(conv_module, conv_node_name, self.config)
                update_conv_params_op = UpdateWeightAndOptionalBias(elastic_conv_operation)
                insertion_commands.append(
                    PTInsertionCommand(
                        PTTargetPoint(TargetType.PRE_LAYER_OPERATION,
                                      target_node_name=conv_node_name),
                        update_conv_params_op,
                        TransformationPriority.PRUNING_PRIORITY
                    )
                )
                # Padding
                ap = ElasticKernelPaddingAdjustment(elastic_conv_operation)
                pad_op = UpdatePadding(ap).to(device)
                nncf_logger.warning('Padding will be adjusted for {}'.format(conv_node_name))
                pad_commands.append(
                    PTInsertionCommand(
                        PTTargetPoint(target_type=TargetType.PRE_LAYER_OPERATION,
                                      target_node_name=conv_node_name),
                        pad_op,
                        TransformationPriority.DEFAULT_PRIORITY
                    )
                )
                self.elastic_kernel_ops.append(elastic_conv_operation)
                self.elastic_width_ops.append(elastic_conv_operation)

                group_minfos.append(ElasticWidthInfo(node_name=conv_node_name,
                                                     module=conv_module,
                                                     layer_name=conv_node.layer_name,
                                                     elastic_op=elastic_conv_operation,
                                                     node_id=conv_node.node_id))

                self.scope_vs_elastic_op_map[conv_node.layer_name] = elastic_conv_operation

            cluster = Cluster[PrunedModuleInfo](i, group_minfos, [n.node_id for n in group.elements])
            self.pruned_module_groups_info.add_cluster(cluster)

        if pad_commands:
            insertion_commands += pad_commands

        op_name_vs_op_creator = {
            Conv2dMetatype.name: self.create_dynamic_conv_input_op,
            BatchNormMetatype.name: self.create_dynamic_bn_input_op,
            LinearMetatype.name: self.create_dynamic_linear_input_op
        }

        for op_name, op_creator in op_name_vs_op_creator.items():
            nodes = target_model_graph.get_nodes_by_types([op_name])
            for node in nodes:
                node_name = node.node_name
                nncf_logger.info("Adding Dynamic Input Op for {} in scope: {}".format(op_name, node_name))
                module = target_model.get_containing_module(node_name)
                update_module_params = op_creator(module, node_name, device)
                insertion_commands.append(
                    PTInsertionCommand(
                        PTTargetPoint(TargetType.PRE_LAYER_OPERATION,
                                      target_node_name=node_name),
                        update_module_params,
                        priority=TransformationPriority.DEFAULT_PRIORITY
                    )
                )
        return insertion_commands


class BootstrapNASController(PTCompressionAlgorithmController):
    def __init__(self, target_model: NNCFNetwork, config, scope_vs_elastic_op_map,
                 elastic_kernel_ops, elastic_width_ops,
                 pruned_module_groups_info: Clusterization[ElasticWidthInfo]):
        super().__init__(target_model)
        self.target_model = target_model
        self.config = config
        self._loss = ZeroCompressionLoss(next(target_model.parameters()).device)
        self._scheduler = BaseCompressionScheduler()
        self.scope_vs_elastic_op_map = scope_vs_elastic_op_map
        self.elastic_kernel_ops = elastic_kernel_ops
        self.elastic_width_ops = elastic_width_ops
        self.pruned_module_groups_info = pruned_module_groups_info

        possible_skipped_blocks = self.config.get('skipped_blocks', [])
        self.num_possible_skipped_blocks = len(possible_skipped_blocks)

        # KD
        kd_ratio = config.get('kd_ratio', 0)
        if kd_ratio > 0:
            # TODO: Load teacher model
            self.teacher_model = None
        if is_main_process():
            test_mode = config.get('test_mode', False)
            if not test_mode:
                self._create_output_folder(config)
            print("Created BootstrapNAS controller")

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        return NNCFStatistics()

    def progressive_shrinking(self, optimizer, train_loader, val_loader, criterion):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.init_lr = optimizer.param_groups[0]['lr']

        config = self.config
        stage = config.get('fine_tuner_params', None)
        assert stage is not None, 'Missing fine_tuner_params in config'
        stage = stage.get('stage', 'depth')  # Default stage is depth.
        # stage = 'kernel' if stage is None else stage

        nncf_logger.info('Fine tuning stage = {}'.format(stage))

        self.best_acc = 0

        # 1. Check stage and phase.
        if stage == 'kernel':
            self.train()
        elif stage == 'depth':
            nncf_logger.info('Fine tuning depth dimension')
            ctx = self.target_model._compressed_context
            ctx._elastic_depth = True
            # Check accuracy of supernetwork
            # self.validate_single_setting(0)
            self.train()
        elif stage == 'expand':
            raise NotImplementedError
        elif stage == 'width':
            # TODO 1. Load previously finetuned network.
            # kernel_config = config.get('kernel_config')
            # depth_config =
            # TODO 2. Load elastic kernel and depth used in previously finetuned network
            # For now, we are using 'skipped blocks' taken into account once at initialization
            # TODO Elastic Depth
            if config.get('use_elastic_depth', False):
                ctx = self.model._compressed_context
                ctx._elastic_depth = True

            # Check accuracy of supernetwork
            self.validate_single_setting(0)
            # self.reorganize_weights() # TODO: Automation
            self.reorganize_weights_from_config() # TODO: Check issue #1
            # self.set_running_statistics()
            self.simple_set_running_statistics()
            # Check accuracy of supernetwork with reorg weights.
            # self.validate(0)
            self.validate_single_setting(0)
            self.train(validate_func=None)

    def activate_elastic_width_configuration_by_cluster_id(self, cluster_id_vs_width_map):
        for cluster_id, width in cluster_id_vs_width_map.items():
            cluster = self.pruned_module_groups_info.get_cluster_by_id(cluster_id)
            for elastic_width_info in cluster.elements:
                elastic_width_info.elastic_op.set_active_out_channels(width)

    def activate_elastic_width_configuration_by_layer_names(self, layer_name_vs_width_map):
        not_processed_layers = list(layer_name_vs_width_map.keys())
        for cluster in self.pruned_module_groups_info.get_all_clusters():
            width_vs_elastic_op_map = dict()
            grouped_layer_names = []
            for elastic_width_info in cluster.elements:
                name = elastic_width_info.layer_name
                if name in layer_name_vs_width_map:
                    width = layer_name_vs_width_map[name]
                    if width not in width_vs_elastic_op_map:
                        width_vs_elastic_op_map[width] = []
                    width_vs_elastic_op_map[width].append(elastic_width_info.elastic_op)
                    not_processed_layers.remove(name)
                    grouped_layer_names.append(name)
                else:
                    raise RuntimeError('cluster has layer={} which is not specified in the configuration'.format(name))
            if len(width_vs_elastic_op_map) > 1:
                raise AttributeError(
                    'Can set different width within the group of pruned modules:\n {}'.
                        format(*grouped_layer_names, sep='\n'))

            for width, list_ops in width_vs_elastic_op_map.items():
                for op in list_ops:
                    op.set_active_out_channels(width)

        if not_processed_layers:
            raise AttributeError('Can\'t set width for layers:\n {}'.format(*not_processed_layers, sep='\n'))

    def _get_random_kernel_conf(self):
        kernel_conf = []
        for op in self.elastic_kernel_ops:
            rand_i = random.randrange(0, len(op.kernel_size_list))
            # TODO: Slow decreasing of sizes?
            kernel_conf.append(op.kernel_size_list[rand_i])
        return kernel_conf

    def _get_random_depth_conf(self):
        depth_conf = []
        for i in range(0, self.num_possible_skipped_blocks):
            skip = bool(random.getrandbits(1)) #random.randint(0, 1)
            if skip:
                depth_conf.append(i)
        return depth_conf

    def _get_random_width_conf(self):
        cluster_id_vs_width_map = dict()
        for cluster in self.pruned_module_groups_info.get_all_clusters():
            all_max_out_channels = {el.elastic_op.max_out_channels for el in cluster.elements}
            if len(all_max_out_channels) != 1:
                raise RuntimeError('Invalid grouping of layers with different number of output channels')

            first_elastic_width_info = next(iter(cluster.elements))
            op = first_elastic_width_info.elastic_op
            rand_i = random.randrange(0, len(op.width_list))
            width_sel = op.width_list[rand_i]
            cluster_id_vs_width_map[cluster.id] = width_sel

            nncf_logger.debug('Select width={} for group #{}'.format(cluster.id, width_sel))
        return cluster_id_vs_width_map

    def get_active_config(self):
        subnet_config = {}
        for op in self.elastic_kernel_ops:
            subnet_config['kernel'].append(op.active_kernel_size)
        # width
        for op in self.elastic_width_ops:
            subnet_config['width'].append(op.active_num_out_channels)

        return subnet_config

    def sample_active_subnet(self, stage):
        # TODO: Handle to switch from elastic width and kernel
        subnet_config = {}
        if stage == 'kernel':
            subnet_config['kernel'] = self._get_random_kernel_conf()
        elif stage == 'depth':
            # Set random elastic kernel # TODO
            # Set random elastic depth
            subnet_config['depth'] = self._get_random_depth_conf() # Get blocks that will be skipped.
        elif stage == 'width':
            # Set random elastic kernel
            subnet_config['kernel'] = self._get_random_kernel_conf()
            # Set random elastic depth # TODO:
            # Set random elastic width
            subnet_config['width'] = self._get_random_width_conf()
        else:
            raise ValueError('Unsupported stage')
        # print(subnet_config)
        self.set_active_subnet(subnet_config)

    def set_active_subnet(self, subnet_config):
        # kernel
        if 'kernel' in subnet_config:
            for op, ks in zip(self.elastic_kernel_ops, subnet_config['kernel']):
                op.set_active_kernel_size(ks)

        # width
        if 'width' in subnet_config:
            for op, w in zip(self.elastic_width_ops, subnet_config['width']):
                op.set_active_out_channels(w)

        # depth
        if 'depth' in subnet_config:
            ctx = self.target_model._compressed_context
            print(subnet_config['depth'])
            ctx.set_active_skipped_block(subnet_config['depth'])

    def train(self, validate_func=None):
        config = self.config
        if validate_func is None:  # TODO
            validate_func = self.validate

        for epoch in range(config['fine_tuner_params']['start_epoch'],
                           config['fine_tuner_params']['epochs'] + config['fine_tuner_params']['warmup_epochs']):
            train_loss, (train_top1, train_top5) = self.train_one_epoch(epoch)
            if (epoch + 1) % config['fine_tuner_params']['validation_frequency'] == 0:
                # validate supernet
                self.reactivate_supernet()
                # val_loss, val_acc, val_acc5, _val_log = validate_func(epoch)
                self.simple_set_running_statistics()  # TODO
                val_loss, val_acc, val_acc5, _val_log = self.validate_single_setting(epoch)
                if is_main_process():
                    nncf_logger.info(
                        f"Supernet Val: [{epoch + 1 - config['fine_tuner_params']['warmup_epochs']}/{config['fine_tuner_params']['epochs'] + config['fine_tuner_params']['warmup_epochs']}], loss={val_loss:.3f}, top-1={val_acc:.3f}, top-5={val_acc5:.3f}")
                    self.export_model(f'{self.main_path}/checkpoint/checkpoint.onnx')
                    checkpoint_data = {
                        'epoch': epoch + 1 - config['fine_tuner_params']['warmup_epochs'],
                        'val_loss': val_loss.item(),
                        'val_acc': val_acc,
                        'val_acc5': val_acc5
                    }
                    json.dump(checkpoint_data, open(f'{self.main_path}/checkpoint/checkpoint.json', 'w'), indent=4)
                # validate minimal subnet
                self.get_minimal_subnet()
                # val_loss, val_acc, val_acc5, _val_log = validate_func(epoch)
                self.simple_set_running_statistics()  # TODO
                val_loss, val_acc, val_acc5, _val_log = self.validate_single_setting(epoch)
                if is_main_process():
                    nncf_logger.info(
                        f"Minimal Subnet Val: [{epoch + 1 - config['fine_tuner_params']['warmup_epochs']}/{config['fine_tuner_params']['epochs'] + config['fine_tuner_params']['warmup_epochs']}], loss={val_loss:.3f}, top-1={val_acc:.3f}, top-5={val_acc5:.3f}")

                    # best_acc for minimal subnet
                    is_best = val_acc > self.best_acc
                    self.best_acc = max(self.best_acc, val_acc)
                    if is_best:
                        nncf_logger.info(f"New best acc {self.best_acc} for minimal subnet")
                        self.export_model(f'{self.main_path}/checkpoint/min_subnet.onnx')
                        min_subnet_data = {
                            'epoch': epoch + 1 - config['fine_tuner_params']['warmup_epochs'],
                            'val_loss': val_loss.item(),
                            'val_acc': val_acc,
                            'val_acc5': val_acc5
                        }
                        json.dump(min_subnet_data, open(f'{self.main_path}/checkpoint/min_subnet.json', 'w'), indent=4)

    def train_one_epoch(self, epoch):
        config = self.config
        warmup_epochs = config['fine_tuner_params']['warmup_epochs']

        dynamic_net = self.target_model
        dynamic_net.to('cuda')
        dynamic_net.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        nBatch = len(self.train_loader)

        len_train_loader = len(self.train_loader)
        end = time.time()
        for i, (images, labels) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # TODO: Fix adjustment of learning rate.
            if epoch < warmup_epochs:
                new_lr = self.warmup_adjust_learning_rate(
                    self.optimizer, warmup_epochs * nBatch, nBatch, epoch, i, warmup_lr,
                )
            else:
                new_lr = self.adjust_learning_rate(
                    self.optimizer, epoch - warmup_epochs, i, nBatch
                )

            images, labels = images.cuda(), labels.cuda()
            target = labels

            # soft target
            if config['kd_ratio'] > 0:
                assert self.teacher_model is not None, 'Teacher model is None'
                # self.teacher_model.train()
                self.teacher_model.eval()
                with torch.no_grad():
                    soft_logits = self.teacher_model(images).detach()
                    soft_label = F.softmax(soft_logits, dim=1)

            # clean gradients
            dynamic_net.zero_grad()

            # compute output
            subnet_str = ''
            for _ in range(config['fine_tuner_params']['dynamic_batch_size']):
                subnet_seed = int('%d%.3d%.3d' % (epoch * nBatch + i, _, 0))
                random.seed(subnet_seed)
                if config['fine_tuner_params'].get('use_fixed_training_config', False):
                    subnet_config = config['fine_tuner_params']['fixed_training_config']
                    self.set_active_subnet(subnet_config)
                else:
                    self.sample_active_subnet(config['fine_tuner_params']['stage'])
                    nncf_logger.info(f'Fine tuning {self.get_active_config()}')

                output = dynamic_net(images)
                if config['kd_ratio'] == 0:
                    loss = self.criterion(output, labels)
                    loss_type = 'ce'
                else:
                    if config['kd_type'] == 'ce':
                        kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
                    else:
                        kd_loss = F.mse_loss(output, soft_logits)
                    loss = config['kd_ratio'] * kd_loss + self.criterion(output, labels)
                    loss_type = '%.1fkd-%s & ce' % (config['kd_ratio'], config['kd_type'])

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss, images.size(0))
                top1.update(acc1, images.size(0))
                top5.update(acc5, images.size(0))

                # self.optimizer.zero_grad()
                loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.config['print_freq'] == 0:
                nncf_logger.info(
                    '{rank}'
                    'Train: [{0}/{1}] '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Lr: {lr:.3} '
                    'Loss: {loss.val:.4f} ({loss.avg:.4f}) '
                    'Acc@1: {top1.val:.3f} ({top1.avg:.3f}) '
                    'Acc@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len_train_loader, batch_time=batch_time, lr=self.optimizer.param_groups[0]['lr'],
                        loss=losses,
                        top1=top1, top5=top5,
                        rank=''))  # {}:'.format(self.config['rank']) if self.config['multiprocessing_distributed'] else ''
                # )) # TODO: Fix getting multiprocessing info

            end = time.time()
        return losses.avg, (top1.avg, top5.avg)

    def validate(self, epoch):
        config = self.config
        dynamic_net = self.target_model
        dynamic_net.to('cuda')
        dynamic_net.eval()

        # TODO: Only validating active subnet for now setting for now.
        return self.validate_single_setting(epoch=epoch)

        # TODO Fix this function
        # depth_list = [5] # TODO
        # expand_ratio_list = [1] # TODO
        # ks_list = []
        # for op in self.elastic_kernel_ops:
        #     ks_list.append(op.kernel_size_list[0])
        # width_list = []
        # for op in self.elastic_width_ops:
        #     width_list.append(op.width_list[0])
        # image_size_list = [224] # TODO
        #
        # # TODO: Number of settings for validation is too big.
        #
        # # TODO: Fix dimensions that are not yet implemented.
        # subnet_settings = []
        # for d in depth_list:
        #     for e in expand_ratio_list:
        #         for k in ks_list:
        #             for w in width_list:
        #                 for img_size in image_size_list:
        #                     subnet_settings.append([{
        #                         # 'image_size': img_size,
        #                         # 'd': d,
        #                         # 'e': e,
        #                         'kernel': k,
        #                         'width': w,
        #                     }, 'R%s-D%s-E%s-K%s-W%s' % (img_size, d, e, k, w)])
        # # if additional_setting is not None:
        # #     subnet_settings += additional_setting
        #
        # print(subnet_settings)
        # # exit(0)
        #
        # losses_of_subnets, top1_of_subnets, top5_of_subnets = [], [], []
        #
        # valid_log = ''
        # for setting, name in subnet_settings:
        #     # self.write_log('-' * 30 + ' Validate %s ' % name + '-' * 30, 'train', should_print=False)
        #     # TODO: Elastic Resolution?
        #     # self.set_active_subnet(**setting) #TODO
        #     # self.write_log(dynamic_net.module_str, 'train', should_print=False) # TODO
        #
        #     # self.reset_running_statistics(dynamic_net) # TODO: JP Is there an utility function in NNCF for this already?
        #     loss, (top1, top5) = self.validate_single_setting(epoch=epoch) #, is_test=is_test, run_str=name, net=dynamic_net)
        #     losses_of_subnets.append(loss)
        #     top1_of_subnets.append(top1)
        #     top5_of_subnets.append(top5)
        #     valid_log += '%s (%.3f), ' % (name, top1)
        #
        # return np.array(losses_of_subnets).mean(), np.array(top1_of_subnets).mean(), np.array(top5_of_subnets).mean(), valid_log

    def validate_single_setting(self, epoch):
        dynamic_net = self.target_model
        dynamic_net.to('cuda')
        dynamic_net.eval()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        len_val_loader = len(self.val_loader)

        with torch.no_grad():
            end = time.time()
            for i, (images, labels) in enumerate(self.val_loader):  # data_loader):
                images, labels = images.cuda(), labels.cuda()
                # compute output
                output = dynamic_net(images)
                loss = self.criterion(output, labels)
                # measure accuracy and record loss
                losses.update(loss, images.size(0))
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                top1.update(acc1, images.size(0))
                top5.update(acc5, images.size(0))
                if i % self.config['print_freq'] == 0:
                    nncf_logger.debug(
                        '{rank}'
                        'Val: [{0}/{1}] '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss: {loss.val:.4f} ({loss.avg:.4f}) '
                        'Acc@1: {top1.val:.3f} ({top1.avg:.3f}) '
                        'Acc@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                            i, len_val_loader, batch_time=batch_time, loss=losses,
                            top1=top1, top5=top5,
                            rank=''))  # {}:'.format(self.config['rank']) if self.config['multiprocessing_distributed'] else ''
                    # )) # TODO: Fix getting multiprocessing info
            nncf_logger.info(
                'Val: '
                'Acc@1: {top1.val:.3f} ({top1.avg:.3f}) '
                'Acc@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len_val_loader,
                    top1=top1, top5=top5))  # ,
            # rank=''))  # {}:'.format(self.config['rank']) if self.config['multiprocessing_distributed'] else ''
            # )) # TODO: Fix getting multiprocessing info
            if is_main_process():
                # TODO: Tensorboard.
                pass  # TODO
        return losses.avg, top1.avg, top5.avg, 'TODO_log'

    def _create_output_folder(self, config):
        self.main_path = config.get('output_path', 'bootstrapNASOutput')
        stage = config.get('fine_tuner_params', None).get('stage', None)
        assert stage is not None, 'Config file does not specified stage'
        self.main_path = os.path.join(self.main_path, stage)
        if stage in ['width', 'depth']:
            phase = config.get('fine_tuner_params', None).get('phase', None)
            assert phase is not None, 'Config file does not specified phase'
            self.main_path = os.path.join(self.main_path, str(phase))
        os.makedirs(os.path.join(self.main_path, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(self.main_path, 'checkpoint'), exist_ok=True)

    def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
        """ adjust learning of a given optimizer and return the new learning rate """
        new_lr = self.calc_learning_rate(epoch, self.init_lr, self.config['fine_tuner_params']['epochs'], batch, nBatch,
                                         self.config['fine_tuner_params']['lr_schedule_type'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def warmup_adjust_learning_rate(self, optimizer, T_total, nBatch, epoch, batch=0, warmup_lr=0):
        T_cur = epoch * nBatch + batch + 1
        new_lr = T_cur / T_total * (self.init_lr - warmup_lr) + warmup_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    """ learning rate schedule """

    def calc_learning_rate(self, epoch, init_lr, n_epochs, batch=0, nBatch=None, lr_schedule_type='cosine'):
        if lr_schedule_type == 'cosine':
            t_total = n_epochs * nBatch
            t_cur = epoch * nBatch + batch
            lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
        elif lr_schedule_type is None:
            lr = init_lr
        else:
            raise ValueError('do not support: %s' % lr_schedule_type)
        return lr

    def reorganize_weights(self):
        pass

    @staticmethod
    def adjust_bn_according_to_idx(bn, idx):
        bn.weight.data = torch.index_select(bn.weight.data, 0, idx)
        bn.bias.data = torch.index_select(bn.bias.data, 0, idx)
        if type(bn) in [nn.BatchNorm1d, nn.BatchNorm2d]:
            bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, idx)
            bn.running_var.data = torch.index_select(bn.running_var.data, 0, idx)

    # TODO: Generalize
    def reorganize_weights_from_config(self):
        # TODO: Refactor to reorg from config.
        # Remove hardcoded Resnet50. Let advanced user pass this function.
        bottlenecks = [2, 3, 5, 2]  # TODO: Check
        for layer in range(1, 5):
            for bottleneck in range(1, bottlenecks[layer - 1] + 1):
                conv3 = self.target_model.get_module_by_scope(
                    Scope.from_str(f'ResNet/Sequential[layer{layer}]/Bottleneck[{bottleneck}]/NNCFConv2d[conv3]'))
                conv2 = self.target_model.get_module_by_scope(
                    Scope.from_str(f'ResNet/Sequential[layer{layer}]/Bottleneck[{bottleneck}]/NNCFConv2d[conv2]'))
                bn2 = self.target_model.get_module_by_scope(
                    Scope.from_str(f'ResNet/Sequential[layer{layer}]/Bottleneck[{bottleneck}]/NNCFConv2d[bn2]'))
                conv1 = self.target_model.get_module_by_scope(
                    Scope.from_str(f'ResNet/Sequential[layer{layer}]/Bottleneck[{bottleneck}]/NNCFConv2d[conv1]'))
                bn1 = self.target_model.get_module_by_scope(
                    Scope.from_str(f'ResNet/Sequential[layer{layer}]/Bottleneck[{bottleneck}]/NNCFConv2d[bn1]'))

                # conv3 -> conv2
                importance = torch.sum(torch.abs(conv3.weight.data), dim=(0, 2, 3))
                sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
                conv3.weight.data = torch.index_select(conv3.weight.data, 1, sorted_idx)
                self.adjust_bn_according_to_idx(bn2, sorted_idx)
                conv2.weight.data = torch.index_select(conv2.weight.data, 0, sorted_idx)

                # conv2 -> conv1
                importance = torch.sum(torch.abs(conv2.weight.data), dim=(0, 2, 3))
                sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)

                conv2.weight.data = torch.index_select(conv2.weight.data, 1, sorted_idx)
                self.adjust_bn_according_to_idx(bn1, sorted_idx)
                conv1.weight.data = torch.index_select(conv1.weight.data, 0, sorted_idx)

    def set_logger_level(self, level):
        nncf_logger.setLevel(level)
        for handler in nncf_logger.handlers:
            handler.setLevel(level)

    def reactivate_supernet(self):
        for op in self.elastic_kernel_ops:
            max_ks = max(op.kernel_size_list)
            op.set_active_kernel_size(max_ks)

        for op in self.elastic_width_ops:
            max_w = max(op.width_list)
            op.set_active_out_channels(max_w)

        ctx = self.target_model._compressed_context
        ctx.set_active_skipped_block([])

    def get_active_config(self, use_tuple_with_layer_name=False):
        config = {'kernel': [], 'width': [], 'depth': []}
        for op in self.elastic_kernel_ops:
            if use_tuple_with_layer_name:
                config['kernel'].append((str(op.scope), op.get_active_kernel_size()))
            else:
                config['kernel'].append(op.get_active_kernel_size())
        for op in self.elastic_width_ops:
            if use_tuple_with_layer_name:
                config['width'].append((str(op.scope), op.get_active_out_channels()))
            else:
                config['width'].append(op.get_active_out_channels())

        ctx = self.target_model._compressed_context
        config['depth'] = ctx.active_block_idxs
        return config

    def get_net_device(self, net):
        return net.parameters().__next__().device

    # def set_running_statistics(model, data_loader, distributed=False):
    # TODO: Fix.
    def set_running_statistics(self):
        bn_mean = {}
        bn_var = {}

        # forward_model = deepcopy(self.target_model)
        forward_model = self.target_model
        for name, m in forward_model.named_modules():
            print(type(m))
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, NNCFBatchNorm2d):
                bn_mean[name] = AverageMeter()
                bn_var[name] = AverageMeter()

                def new_forward(bn, mean_est, var_est):
                    def lambda_forward(x):
                        batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)

                        batch_mean = torch.squeeze(batch_mean)
                        batch_var = torch.squeeze(batch_var)

                        mean_est.update(batch_mean.data, x.size(0))
                        var_est.update(batch_var.data, x.size(0))

                        # bn forward using calculated mean & var
                        _feature_dim = batch_mean.size(0)
                        return F.batch_norm(
                            x, batch_mean, batch_var, bn.weight[:_feature_dim],
                            bn.bias[:_feature_dim], False,
                            0.0, bn.eps,
                        )

                    return lambda_forward

                m.forward = new_forward(m, bn_mean[name], bn_var[name])

        if len(bn_mean) == 0:
            # skip if there is no batch normalization layers in the network
            return

        with torch.no_grad():
            DynamicBatchNormInputOp.SET_RUNNING_STATISTICS = True
            for images, labels in self.train_loader:  # data_loader:
                images = images.to(self.get_net_device(forward_model))
                forward_model(images)
            DynamicBatchNormInputOp.SET_RUNNING_STATISTICS = False

        # for name, m in model.named_modules():
        for name, m in self.target_model.named_modules():
            if name in bn_mean and bn_mean[name].count > 0:
                feature_dim = bn_mean[name].avg.size(0)
                assert isinstance(m, nn.BatchNorm2d) or isinstance(m, NNCFBatchNorm2d)
                # m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
                # m.running_var.data[:feature_dim].copy_(bn_var[name].avg)
                m.running_mean.data[:feature_dim] = bn_mean[name].avg
                m.running_var.data[:feature_dim] = bn_var[name].avg

    def simple_set_running_statistics(self):
        # https://discuss.pytorch.org/t/how-to-run-the-model-to-only-update-the-batch-normalization-statistics/20626
        self.target_model.train()
        for _ in range(0, 2):
            for i, (images, _) in enumerate(self.train_loader):  # data_loader:
                # print(i)
                images = images.to(self.get_net_device(self.target_model))
                self.target_model(images)

    # Search API
    def get_elastic_parameters(self, use_tuple_with_layer_name=False):
        elastic_params = {'kernel': [], 'width': []}
        elastic_params = {'kernel': [], 'width': []}
        for op in self.elastic_kernel_ops:
            if use_tuple_with_layer_name:
                elastic_params['kernel'].append((str(op.scope), op.kernel_size_list))
            else:
                elastic_params['kernel'].append(op.kernel_size_list)
        for op in self.elastic_width_ops:
            if use_tuple_with_layer_name:
                elastic_params['width'].append((str(op.scope), op.width_list))
            else:
                elastic_params['width'].append(op.width_list)
        return elastic_params

    def get_supernet(self, save=False):
        self.reactivate_supernet()
        # TODO export supernet if save
        return self.target_model

    def get_subnet(self, config, save=False, filename='subnet.onnx'):
        self.set_active_subnet(config)
        if save:
            self.export_model(filename)
        return self.target_model

    def get_minimal_subnet(self, save=False):
        # TODO: If working on Depth, we are currently disabling elasticity in training. s
        # for op in self.elastic_kernel_ops:
        #     min_ks = min(op.kernel_size_list)
        #     op.set_active_kernel_size(min_ks)
        #
        # for op in self.elastic_width_ops:
        #     min_w = min(op.width_list)
        #     op.set_active_out_channels(min_w)

        ctx = self.target_model._compressed_context
        ctx.set_active_skipped_block(list(range(0, self.num_possible_skipped_blocks)))
        # TODO Save
        return self.target_model

    def find_subnets(self, req_config, save=False):
        pass


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res
