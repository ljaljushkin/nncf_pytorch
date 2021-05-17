import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from nncf.layer_utils import COMPRESSION_MODULES
from nncf.common.utils.logger import logger as nncf_logger

from nncf.nas.bootstrapNAS.ofa_layers_utils import sub_filter_start_end

@COMPRESSION_MODULES.register() # TODO: Remove?
class ElasticConv2DKernelOp(nn.Module):
    def __init__(self, max_kernel_size, scope): #, module_w):
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
        # filters = self.weight[:out_channel, :in_channel, start:end, start:end]
        filters = weight[:out_channel, :in_channel, start:end, start:end]
        if kernel_size < max_kernel_size:
            # start_filter = self.weight[:out_channel, :in_channel, :, :]  # start with max kernel
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

    def set_active_kernel_size(self,  kernel_size):
        nncf_logger.info('set active elastic_kernel={} for scope={}'.format(kernel_size, self.

            scope))
        assert kernel_size % 2 > 0, 'kernel size should be odd number'
        if kernel_size not in self.kernel_size_list:
            raise ValueError(
                'invalid kernel size to set. Should be a number in {}'.format(self.kernel_size_list))
        self.active_kernel_size = kernel_size

    def forward(self, weight, inputs):
        kernel_size = self.active_kernel_size
        in_channels = inputs.size(1)
        filters =self.get_active_filter(in_channels, kernel_size, weight).contiguous()

        return filters

class ElasticKernelPaddingAdjustment:
    def __init__(self, elastic_kernel_op: ElasticConv2DKernelOp) :
        self._elastic_kernel_op = elastic_kernel_op
        self._is_enabled = True

    def __call__(self, previous_padding, _) -> int:
        if self._is_enabled:
            pad_v = self._elastic_kernel_op.active_kernel_size // 2
            return pad_v
        else:
            return previous_padding

@COMPRESSION_MODULES.register() # TODO: Remove?
class ElasticConv2DWidthOp(nn.Module):
    def __init__(self, max_in_channels, max_out_channels, scope):
        super().__init__()
        self.max_in_channels = max_in_channels
        self.scope = scope
        self.max_out_channels = max_out_channels
        self.active_out_channels = self.max_in_channels

    def set_active_out_channels(self, num_channels):
        nncf_logger.info('set active out channels={} for scope={}'.format(num_channels, self.scope))
        if 0 > num_channels > self.max_out_channels:
            raise ValueError(
                'invalid number of output channels to set. Should be within [{}, {}]'.format(0, self.max_in_channels))
        self.active_out_channels = num_channels

    def forward(self, weight, inputs):
        in_channels = inputs.size(1)
        return weight[:self.active_out_channels, :in_channels, :, :].contiguous()


@COMPRESSION_MODULES.register() # TODO: Remove?
class ElasticBatchNormOp(nn.Module):
    def __init__(self, num_features, scope):
        super().__init__()
        self.num_features = num_features

    def bn_forward(self, feature_dim, weight):
        if self.num_features == feature_dim:
            return weight
        # TODO running_mean, running_var, bias, training, track_running_stats.
        return weight[:feature_dim]

    def forward(self, weight, inputs):
        feature_dim = inputs.size(1)
        return self.bn_forward(feature_dim, weight)