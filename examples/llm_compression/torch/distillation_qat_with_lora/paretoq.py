import math

import torch
import torch.nn as nn


class StretchedElasticQuant(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.num_bits = num_bits
        Qn = -(2 ** (num_bits - 1))
        Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=alpha.device).float()
        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = 1.0 / math.sqrt(input.numel()) if not Qp else 1.0 / math.sqrt(input.numel() * Qp)
        ctx.save_for_backward(input, alpha)
        clip_val = 1 - 1e-2
        n_levels = 2 ** (num_bits - 1)
        shift = 0.5
        Qp = (n_levels - shift) / n_levels
        Qn = -Qp
        ctx.other = grad_scale, Qn, Qp, layerwise
        q_w = (torch.round(torch.clamp(input / alpha, -clip_val, clip_val) * n_levels - shift) + shift) / n_levels
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        clip_val = 1 - 1e-2
        n_levels = 2 ** (ctx.num_bits - 1)
        shift = 0.5
        indicate_small = (q_w < -clip_val).float()
        indicate_big = (q_w > clip_val).float()
        indicate_middle = 1.0 - indicate_small - indicate_big
        if layerwise:
            grad_alpha = (
                (
                    (
                        indicate_small * Qn
                        + indicate_big * Qp
                        + indicate_middle
                        * (
                            -q_w
                            + (torch.round(torch.clamp(q_w, -clip_val, clip_val) * n_levels - shift) + shift) / n_levels
                        )
                    )
                    * grad_output
                    * grad_scale
                )
                .sum()
                .unsqueeze(dim=0)
            )
        else:
            grad_alpha = (
                (
                    indicate_small * Qn
                    + indicate_big * Qp
                    + indicate_middle
                    * (
                        -q_w
                        + (torch.round(torch.clamp(q_w, -clip_val, clip_val) * n_levels - shift) + shift) / n_levels
                    )
                )
                * grad_output
                * grad_scale
            )
            grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class QuantizeLinear(nn.Linear):
    def __init__(
        self,
        *kargs,
        symmetric=True,
        bias=False,
        w_bits=16,
        weight_layerwise=False,
    ):
        super().__init__(*kargs, bias=False)
        self.w_bits = w_bits
        self.weight_layerwise = weight_layerwise
        # params for weight quant
        if self.w_bits < 16:
            self.weight_clip_val = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))

    def forward(self, input_):
        # quantize weight
        assert len(self.weight.size()) == 2
        real_weights = self.weight

        if self.w_bits >= 16:
            weight = self.weight
        elif self.w_bits == 2 or self.w_bits == 0:
            weight = StretchedElasticQuant.apply(
                real_weights,
                self.weight_clip_val,
                self.w_bits,
                self.weight_layerwise,
            ).to(input_.dtype)
            raise NotImplementedError

        out = nn.functional.linear(input_, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out
