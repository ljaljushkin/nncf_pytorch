import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from nncf.common.utils.logger import logger as nncf_logger
from nncf.torch.layer_utils import COMPRESSION_MODULES
from nncf.torch.nas.bootstrapNAS.ofa_layers_utils import sub_filter_start_end

@COMPRESSION_MODULES.register()
class ElasticBypassOp(nn.Module):
    def __init__(self, scope):
        super().__init__()
        self.scope = scope
        self._is_active = False

    def activate_bypass(self, activate=False):
        self._is_active = activate

    def forward(self, weight, inputs):
        pass

@COMPRESSION_MODULES.register() # TODO: Remove?
class ElasticLinearOp(nn.Module):
    def __init__(self, max_in_features, max_out_features, bias, scope):
        super().__init__()
        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias
        self.scope = scope

        self.active_out_features = self.max_out_features

    # TODO: Remove
    def get_active_bias(self, out_features):
      return self.linear.bias[:out_features] if self.bias else None

    def forward(self, weight, inputs):
        nncf_logger.debug('Linear in scope={}'.format(self.scope))
        in_features = inputs.size(1)

        # TODO: Bias

        return weight[:self.active_out_features, :in_features].contiguous()


# Unified operator for elastic kernel and width
@COMPRESSION_MODULES.register() # TODO: Remove?
class ElasticConv2DOp(nn.Module):
    def __init__(self, max_kernel_size, max_in_channels, max_out_channels,scope): #, module_w):
        super().__init__()
        self.scope = scope
        # Create kernel_size_list based on max module kernel size
        self.kernel_size_list = self.generate_kernel_size_list(max_kernel_size)
        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.active_out_channels = self.max_out_channels

        # TODO: Add granularity for width changes from config
        self.width_list = self.generate_width_list(self.max_out_channels)

        scale_params = {}
        for i in range(len(self._ks_set) - 1):
            ks_small = self._ks_set[i]
            ks_larger = self._ks_set[i + 1]
            param_name = '%dto%d' % (ks_larger, ks_small)
            # noinspection PyArgumentList
            scale_params['%s_matrix' % param_name] = Parameter(torch.eye(ks_small ** 2))
        for name, param in scale_params.items():
            self.register_parameter(name, param)

        self.active_kernel_size = max_kernel_size

    def generate_kernel_size_list(self, max_kernel_size):
        assert max_kernel_size % 2 > 0, 'kernel size should be odd number'
        if max_kernel_size == 1:
            return [1]
        kernel = max_kernel_size
        ks_list = []
        while kernel > 1:
            ks_list.append(kernel)
            kernel -= 2
        return ks_list

    def generate_width_list(self, max_out_channels):
        width_list = []
        if max_out_channels <= 32:
            width_list.append(max_out_channels)
            return width_list

        width = 32*(max_out_channels // 32)
        while width >= 32:
            width_list.append(width)
            width -= 32
        return width_list

    def get_active_filter(self, out_channel, in_channel, kernel_size, weight):
        # out_channel = in_channel
        max_kernel_size = max(self.kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = weight[:out_channel, :in_channel, start:end, start:end]
        if kernel_size < max_kernel_size:
            start_filter = weight[:out_channel, :in_channel, :, :]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(_input_filter.size(0), _input_filter.size(1), -1)
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter, self.__getattr__('%dto%d_matrix' % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks ** 2)
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks, target_ks)
                start_filter = _input_filter
            filters = start_filter
        return filters

    def set_active_kernel_size(self, kernel_size):
        nncf_logger.debug('set active elastic_kernel={} for scope={}'.format(kernel_size, self.scope))
        assert kernel_size % 2 > 0, 'kernel size should be odd number'
        if kernel_size not in self.kernel_size_list:
            raise ValueError(
                'invalid kernel size to set. Should be a number in {}'.format(self.kernel_size_list))
        self.active_kernel_size = kernel_size

    def set_active_out_channels(self, num_channels):
        nncf_logger.debug('set active out channels={} for scope={}'.format(num_channels, self.scope))
        if 0 > num_channels > self.max_out_channels:
            raise ValueError(
                'invalid number of output channels to set. Should be within [{}, {}]'.format(0, self.max_out_channels))
        if num_channels not in self.width_list:
            raise ValueError(
                'invalid number of output channels to set. Should be a number in {}'.format(self.width_list))
        self.active_out_channels = num_channels

    def forward(self, weight, inputs):
        nncf_logger.debug('Conv2d with active kernel size={} and active number of out channels={} in scope={}'.format(self.active_kernel_size, self.active_out_channels, self.scope))
        kernel_size = self.active_kernel_size
        out_channels = self.active_out_channels
        in_channels = inputs.size(1)
        if kernel_size > 1:
            filters = self.get_active_filter(out_channels, in_channels, kernel_size, weight).contiguous()
            return filters
        else:
            return weight[:self.active_out_channels, :in_channels, :, :].contiguous()


class ElasticKernelPaddingAdjustment:
    # def __init__(self, elastic_kernel_op: ElasticConv2DKernelOp):
    def __init__(self, elastic_k_w_op: ElasticConv2DOp): # Using unified operator
        self._elastic_k_w_op = elastic_k_w_op
        self._is_enabled = True

    def __call__(self, previous_padding, _) -> int:
        if self._is_enabled:
            pad_v = self._elastic_k_w_op.active_kernel_size // 2
            return pad_v
        else:
            return previous_padding


@COMPRESSION_MODULES.register()  # TODO: Remove?
class ElasticBatchNormOp(nn.Module):
    def __init__(self, num_features, scope):
        super().__init__()
        self.num_features = num_features
        self.scope = scope

    def bn_forward(self, feature_dim, **bn_params):
        nncf_logger.debug('BN with active num_features={} in scope={}'.format(feature_dim, self.scope))
        if self.num_features == feature_dim:
            return list(bn_params.values())
        # TODO: training, track_running_stats.
        return [param[:feature_dim] for param in bn_params.values()]

    def forward(self, inputs, **bn_params):
        feature_dim = inputs.size(1)
        return self.bn_forward(feature_dim, **bn_params)

# ***************
# REMOVE OPS Below after confirming that unified op works correctly.
# ***************

@COMPRESSION_MODULES.register() # TODO: Remove?
class ElasticConv2DKernelOp(nn.Module):
    def __init__(self, max_kernel_size, scope):
        super().__init__()
        self.scope = scope
        # Create kernel_size_list based on max module kernel size
        self.kernel_size_list = self.generate_kernel_size_list(max_kernel_size)
        # self.stride = stride
        # self.dilation = dilation

        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()

        scale_params = {}
        for i in range(len(self._ks_set) - 1):
            ks_small = self._ks_set[i]
            ks_larger = self._ks_set[i + 1]
            param_name = '%dto%d' % (ks_larger, ks_small)
            # noinspection PyArgumentList
            scale_params['%s_matrix' % param_name] = Parameter(torch.eye(ks_small ** 2))
        for name, param in scale_params.items():
            self.register_parameter(name, param)

        self.active_kernel_size = max(self.kernel_size_list)

    def generate_kernel_size_list(self, max_kernel_size):
        assert max_kernel_size % 2 > 0, 'kernel size should be odd number'
        kernel = max_kernel_size
        ks_list = []
        while kernel > 1:
            ks_list.append(kernel)
            kernel -= 2
        return ks_list

    def get_active_filter(self, in_channel, kernel_size, weight):
        out_channel = in_channel
        max_kernel_size = max(self.kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = weight[:out_channel, :in_channel, start:end, start:end]
        if kernel_size < max_kernel_size:
            start_filter = weight[:out_channel, :in_channel, :, :]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(_input_filter.size(0), _input_filter.size(1), -1)
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter, self.__getattr__('%dto%d_matrix' % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks ** 2)
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks, target_ks)
                start_filter = _input_filter
            filters = start_filter
        return filters

    def set_active_kernel_size(self, kernel_size):
        nncf_logger.info('set active elastic_kernel={} for scope={}'.format(kernel_size, self.scope))
        assert kernel_size % 2 > 0, 'kernel size should be odd number'
        if kernel_size not in self.kernel_size_list:
            raise ValueError(
                'invalid kernel size to set. Should be a number in {}'.format(self.kernel_size_list))
        self.active_kernel_size = kernel_size

    def forward(self, weight, inputs):
        kernel_size = self.active_kernel_size
        in_channels = inputs.size(1)
        filters = self.get_active_filter(in_channels, kernel_size, weight).contiguous()
        return filters




@COMPRESSION_MODULES.register()  # TODO: Remove?
class ElasticConv2DWidthOp(nn.Module):
    def __init__(self, max_in_channels, max_out_channels, scope):
        super().__init__()
        self.max_in_channels = max_in_channels
        self.scope = scope
        self.max_out_channels = max_out_channels
        self.active_out_channels = self.max_out_channels
        # TODO: Add granularity for width changes from config
        self.width_list = self.generate_width_list(self.max_out_channels)

    def generate_width_list(self, max_out_channels):
        assert max_out_channels > 32, 'Max out channels should be greater than 32'
        width = 32*(max_out_channels // 32)
        width_list = []
        while width >= 32:
            width_list.append(width)
            width -= 32
        return width_list

    def set_active_out_channels(self, num_channels):
        nncf_logger.info('set active out channels={} for scope={}'.format(num_channels, self.scope))
        if 0 > num_channels > self.max_out_channels:
            raise ValueError(
                'invalid number of output channels to set. Should be within [{}, {}]'.format(0, self.max_out_channels))
        if num_channels not in self.width_list:
            raise ValueError(
                'invalid number of output channels to set. Should be a number in {}'.format(self.width_list))
        self.active_out_channels = num_channels

    def forward(self, weight, inputs):
        nncf_logger.info('Conv2d with active number of out channels={} in scope={}'.format(self.active_out_channels, self.scope))
        in_channels = inputs.size(1)
        return weight[:self.active_out_channels, :in_channels, :, :].contiguous()

