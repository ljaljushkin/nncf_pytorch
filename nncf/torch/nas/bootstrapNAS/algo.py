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
from collections import OrderedDict
from copy import deepcopy
import random
from tqdm import tqdm
import numpy as np
import os
import time
import torch

from nncf.torch.algo_selector import COMPRESSION_ALGORITHMS, ZeroCompressionLoss
from nncf.common.statistics import NNCFStatistics
# from nncf.api.compression import CompressionLevel
from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.graph.graph import NNCFNodeExpression as N
from nncf.torch.graph.patterns import BN
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.common.utils.logger import logger as nncf_logger
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.module_operations import UpdatePadding
from nncf.torch.nas.bootstrapNAS.layers import ElasticBatchNormOp
from nncf.torch.nas.bootstrapNAS.layers import ElasticConv2DOp # Unified Operator
from nncf.torch.nas.bootstrapNAS.layers import ElasticLinearOp
from nncf.torch.nas.bootstrapNAS.layers import ElasticKernelPaddingAdjustment
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.utils import is_main_process


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

    def build_controller(self, target_model: NNCFNetwork) -> PTCompressionAlgorithmController:
        return BootstrapNASController(target_model, self.config, self.scope_vs_elastic_op_map,
                                      self.elastic_kernel_ops, self.elastic_width_ops)

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        layout = PTTransformationLayout()
        commands = []
        commands.extend(self._elastic_k_w(target_model))
        for command in commands:
            layout.register(command)
        return layout

    # Unified operator
    @staticmethod
    def create_elastic_k_w(module, scope):
        return ElasticConv2DOp(module.kernel_size[0], module.in_channels, module.out_channels, scope)


    @staticmethod
    def create_elastic_bn_operation(module, scope):
        return ElasticBatchNormOp(module.out_channels, scope)

    @staticmethod
    def create_elastic_linear_op(module, scope):
        return ElasticLinearOp(module.in_features, module.out_features, module.bias, scope)


    def _elastic_k_w(self, target_model: NNCFNetwork):
        nncf_graph = target_model.get_original_graph()
        device = next(target_model.parameters()).device
        insertion_commands = []
        pad_commands = []
        conv_bn_pattern = N('conv2d') + BN
        conv2d_nodes = nncf_graph.get_nodes_by_types(['conv2d'])
        nx_graph = deepcopy(nncf_graph.get_nx_graph_copy())
        from nncf.torch.graph.graph_matching import search_all
        matches = search_all(nx_graph, conv_bn_pattern)
        conv2d_bn_node_pairs = []
        for match in matches:
            input_node_key = match[0]
            output_node_key = match[-1]
            conv_node = nncf_graph.get_node_by_key(input_node_key)
            bn_node = nncf_graph.get_node_by_key(output_node_key)
            conv2d_bn_node_pairs.append((conv_node, bn_node))

        for conv_node in conv2d_nodes:
            conv_module_scope = conv_node.ia_op_exec_context.scope_in_model
            conv_module = target_model.get_module_by_scope(conv_module_scope)
            # if conv_module.kernel_size[0] <= 3:
            #     continue
            nncf_logger.info("Adding Dynamic Conv2D Layer in scope: {}".format(str(conv_module_scope)))
            conv_operation = self.create_elastic_k_w(conv_module, conv_module_scope)
            insertion_commands.append(
                PTInsertionCommand(
                    PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS,
                                  module_scope=conv_module_scope),
                    conv_operation
                )
            )
            # Padding
            ap = ElasticKernelPaddingAdjustment(conv_operation)
            pad_op = UpdatePadding(ap).to(device)
            insertion_point = PTTargetPoint(target_type=TargetType.PRE_LAYER_OPERATION,
                                            module_scope=conv_module_scope)
            nncf_logger.warning('Padding will be adjusted for {}'.format(str(conv_module_scope)))
            pad_commands.append(PTInsertionCommand(insertion_point, pad_op, TransformationPriority.DEFAULT_PRIORITY))
            self.elastic_kernel_ops.append(conv_operation)
            self.elastic_width_ops.append(conv_operation)

            # Get BN in pair
            for conv_node_p, bn_node in conv2d_bn_node_pairs:
                if conv_node == conv_node_p:
                    bn_module_scope = bn_node.ia_op_exec_context.scope_in_model
                    nncf_logger.info("Adding Elastic Kernel Op for BN in scope: {}".format(str(bn_module_scope)))

                    elastic_bn_kernel_op = self.create_elastic_bn_operation(conv_module, bn_module_scope)
                    insertion_commands.append(
                        PTInsertionCommand(
                            PTTargetPoint(TargetType.OPERATION_WITH_BN_PARAMS,
                                          module_scope=bn_module_scope),
                            elastic_bn_kernel_op
                        )
                    )
                    self.scope_vs_elastic_op_map[str(conv_module_scope)] = conv_operation
                    break

        if pad_commands:
            insertion_commands += pad_commands

        linear_nodes = nncf_graph.get_nodes_by_types(['linear'])
        for node in linear_nodes:
            linear_module_scope = node.ia_op_exec_context.scope_in_model
            module = target_model.get_module_by_scope(linear_module_scope)
            nncf_logger.info("Adding Dynamic Linear Layer in scope: {}".format(str(linear_module_scope)))
            linear_operation = self.create_elastic_linear_op(module, linear_module_scope)
            insertion_commands.append(
                PTInsertionCommand(
                    PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS,
                                  module_scope=linear_module_scope),
                    linear_operation  # hook
                )
            )

        return insertion_commands



class BootstrapNASController(PTCompressionAlgorithmController):
    def __init__(self, target_model: NNCFNetwork, config, scope_vs_elastic_op_map,
                 elastic_kernel_ops, elastic_width_ops):
        super().__init__(target_model)
        self.target_model = target_model
        self.config = config
        self._loss = ZeroCompressionLoss(next(target_model.parameters()).device)
        self._scheduler = BaseCompressionScheduler()
        self.scope_vs_elastic_op_map = scope_vs_elastic_op_map
        self.elastic_kernel_ops = elastic_kernel_ops
        self.elastic_width_ops = elastic_width_ops

        # KD
        kd_ratio = config.get('kd_ratio', 0)
        if kd_ratio > 0:
            # TODO: Load teacher model
            self.teacher_model = None
        if is_main_process():
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

        config = self.config
        print(config)
        print(config.keys())
        stage = config['fine_tuner_params']['stage']
        stage = 'kernel' if stage is None else stage

        nncf_logger.info('Fine tuning stage = {}'.format(stage))

        self.best_acc = 0

        # 1. Check stage and phase.
        if stage == 'kernel':
            self.train()
        elif stage == 'depth':
            pass
        elif stage == 'expand':
            pass
        elif stage == 'width':
            # TODO Load already fine tuned supernet.
            # TODO 1. Load previously finetuned network.
            # TODO 2. Reorganize middle and outter weights. How to do it in NNCF? First use method used in OFA.
            self.train(validate_func=None)


    def _get_random_kernel_conf(self):
        kernel_conf = []
        for op in self.elastic_kernel_ops:
            rand_i = random.randrange(0, len(op.kernel_size_list))
            # TODO: Slow decreasing of sizes?
            kernel_conf.append(op.kernel_size_list[rand_i])
        return kernel_conf

    def _get_random_width_conf(self):
        #TODO: Generalize with clustering idea.
        dep = self.config.get("width_dependencies", [])
        dep_group_value_assigned = dict()

        width_conf = []
        for op in self.elastic_width_ops:
            rand_i = random.randrange(0, len(op.width_list))
            # TODO: Slow decreasing of number of channels?

            width_sel = op.width_list[rand_i]
            for g_i, dep_group in enumerate(dep):
                if str(op.scope) in dep_group:
                    print(f'{str(op.scope)} in dep list')
                    if g_i in dep_group_value_assigned.keys():
                        width_sel = dep_group_value_assigned[g_i]
                    else:
                        dep_group_value_assigned[g_i] = width_sel
            width_conf.append(width_sel)

            print(str(op.scope), width_sel)
        return width_conf


    def get_active_config(self):
        subnet_config = {}
        for op in self.elastic_kernel_ops:
            subnet_config['kernel'].append(op.active_kernel_size)
        # width
        for op in self.elastic_width_ops:
            subnet_config['width'].append(op.active_out_channels)

        return subnet_config

    def sample_active_subnet(self, stage):
        # TODO: Handle to switch from elastic width and kernel

        subnet_config = {}
        if stage == 'kernel':
            subnet_config['kernel'] = self._get_random_kernel_conf()
        elif stage == 'depth':
            raise ValueError('Depth dimension fine tuning has not been implemented, yet')
            # Set random elastic kernel
            # Set random elastic depth
        elif stage == 'width':
            # Set random elastic kernel
            subnet_config['kernel'] = self._get_random_kernel_conf()
            # Set random elastic depth # TODO:
            # Set random elastic width
            subnet_config['width'] = self._get_random_width_conf()
            # TODO: Nikolay's idea to satisfy other requirements
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

    def train(self, validate_func=None):

        config = self.config
        if validate_func is None:
            validate_func = self.validate

        for epoch in range(config['fine_tuner_params']['start_epoch'], config['fine_tuner_params']['epochs'] + config['fine_tuner_params']['warmup_epochs']):
            train_loss, (train_top1, train_top5) = self.train_one_epoch(epoch)

            if (epoch + 1) % config['fine_tuner_params']['validation_frequency'] == 0:
                val_loss, val_acc, val_acc5, _val_log = validate_func(epoch)
                # best_acc
                is_best = val_acc > self.best_acc
                self.best_acc = max(self.best_acc, val_acc)

                if is_main_process():
                    val_log = 'Valid [{0}/{1}] loss={2:.3f}, top-1={3:.3f} ({4:.3f})'. \
                        format(epoch + 1 - config['fine_tuner_params']['warmup_epochs'], config['fine_tuner_params']['epochs'], val_loss, val_acc,
                               self.best_acc)
                    val_log += ', Train top-1 {top1:.3f}, Train loss {loss:.3f}\t'.format(top1=train_top1, loss=train_loss)
                    val_log += _val_log
                    # self.write_log(val_log, 'valid', should_print=False)

                    torch.save({
                        'epoch': epoch,
                        'best_acc': self.best_acc,
                        'optimizer': self.optimizer.state_dict(),
                        'state_dict': self.target_model.state_dict(),
                    }, 'test_save_TODO_delete.pth')
                    # TODO: keep another copy if best.
                    if is_best:
                        pass

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


        with tqdm(total=nBatch,
                  desc='Train Epoch #{}'.format(epoch + 1),
                  disable=not is_main_process()) as t:
            end = time.time()
            for i, (images, labels) in enumerate(self.train_loader):
                # measure data loading time
                data_time.update(time.time() - end)

                # TODO: Fix adjustment of learning rate.
                # if epoch < warmup_epochs:
                #     new_lr = self.warmup_adjust_learning_rate(
                #         optimizer, warmup_epochs * nBatch, nBatch, epoch, i, warmup_lr,
                #     )
                # else:
                #     new_lr = self.adjust_learning_rate(
                #         optimizer, epoch - warmup_epochs, i, nBatch
                #     )

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
                    self.sample_active_subnet(config['fine_tuner_params']['stage'])

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
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1, images.size(0))
                    top5.update(acc5, images.size(0))

                    loss.backward()
                self.optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                t.set_postfix({
                    'loss': losses.avg,
                    'top1': top1.avg,
                    'top5': top5.avg,
                #     'R': images.size(2),
                #     'lr': new_lr,
                #     'loss_type': loss_type,
                #     'seed': str(subnet_seed),
                #     'str': subnet_str,
                #     'data_time': data_time.avg,
                })
                t.update(1)
                end = time.time()
        return losses.avg, (top1.avg, top5.avg)


    def validate(self, epoch):
        config = self.config
        dynamic_net = self.target_model

        dynamic_net.eval()

        # TODO: Only validating single setting for now.
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

        dynamic_net.eval()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        with torch.no_grad():
            end = time.time()
            with tqdm(total=len(self.val_loader),
                      desc='Validate Epoch #{}'.format(epoch + 1)) as t:
                for i, (images, labels) in enumerate(self.val_loader): #data_loader):
                    images, labels = images.cuda(), labels.cuda()
                    # compute output
                    output = dynamic_net(images)
                    # loss = self.test_criterion(output, labels)
                    loss = self.criterion(output, labels)
                    # measure accuracy and record loss
                    losses.update(loss, images.size(0))
                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                    top1.update(acc1, images.size(0))
                    top5.update(acc5, images.size(0))
                    t.set_postfix({
                        'loss': losses.avg,
                        'top1': top1.avg,
                        'top5': top5.avg,
                        'img_size': images.size(2),
                    })
                    t.update(1)
        return losses.avg, top1.avg, top5.avg, 'TODO_log'

    # TODO: Move to utils or use NNCF build in funcs
    def write_log(self, logs_path, log_str, prefix='valid', should_print=True, mode='a'):
        if not os.path.exists(logs_path):
            os.makedirs(logs_path, exist_ok=True)
        """ prefix: valid, train, test """
        if prefix in ['valid', 'test']:
            with open(os.path.join(logs_path, 'valid_console.txt'), mode) as fout:
                fout.write(log_str + '\n')
                fout.flush()
        if prefix in ['valid', 'test', 'train']:
            with open(os.path.join(logs_path, 'train_console.txt'), mode) as fout:
                if prefix in ['valid', 'test']:
                    fout.write('=' * 10)
                fout.write(log_str + '\n')
                fout.flush()
        else:
            with open(os.path.join(logs_path, '%s.txt' % prefix), mode) as fout:
                fout.write(log_str + '\n')
                fout.flush()
        if should_print:
            print(log_str)


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
