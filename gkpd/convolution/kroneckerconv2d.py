import copy
from abc import ABC
from typing import Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from gkpd.convolution.decomposedlayer import DecomposedLayer
from gkpd.gkpd import gkpd


class KroneckerConv2d(DecomposedLayer, ABC):
    """Kronecker factorized convolutional layer using GKPD (https://arxiv.org/abs/2109.14710)

    Usage:
        fact_conv = gkpd.KroneckerConv(a_dims, b_dims, rank='full')
        fact_conv = gkpd.KroneckerConv.from_conv(conv, rank=0.5, decompose_weights=True)

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Spatial dimensions of resulting implicit kernel.
        stride: Stride of resulting implicit kernel.
        padding: Padding of resulting implicit kernel.
        dilation: Dilation of resulting implicit kernel.
        a_dims: Dimensions of first Kronecker factor (second factor defined implicitly) [f, c, h, w].
        bias: Flag to randomly initialize bias.
        rank: Must be an integer or can pass the value 'full' to use rank upper bound
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, tuple],
                 stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 1, dilation: Union[int, tuple] = 1,
                 a_dims: tuple = (1, 1, 1, 1), bias: bool = False, rank: Union[int, str] = 'full'):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.a_dims = a_dims
        self.bias = bias
        self.rank = rank

        self.b_dims = self._get_b_dims(self.in_channels, self.out_channels, self.kernel_size, self.a_dims)

        self.a_padding, self.b_padding = [0, 0], [0, 0]
        self.a_stride, self.b_stride = [1, 1], [1, 1]

        # Distribute padding
        for i in range(2):
            self.a_padding[i] = self.padding[i] if self.a_dims[-2:][i] > 1 else 0
            self.b_padding[i] = self.padding[i] if self.b_dims[-2:][i] > 1 else 0

        # Distribute stride: for a 1x1 convolution apply stride only to factor `a`
        if (np.array(a_dims[-2:]) == 1).all():
            self.a_stride = self.stride

        # Else for separable filters we distribute the stride
        else:
            for i in range(2):
                self.a_stride[i] = self.stride[i] if self.a_dims[-2:][i] > 1 else 1
                self.b_stride[i] = self.stride[i] if self.b_dims[-2:][i] > 1 else 1

        # Define factorized layers (each individual kernel initialized `independently`)
        # Note: might be better to treat rank as fan_out value in kaiming init
        a_factors = [
            nn.Conv2d(self.c1, self.f1, kernel_size=(self.h1, self.w1),  bias=False, padding=self.a_padding,
                      stride=self.a_stride, requires_grad=False).weight for i in range(self.rank)
        ]

        b_factors = [
            nn.Conv2d(self.c2, self.f2, kernel_size=(self.h2, self.w2), bias=False, padding=self.b_padding,
                      stride=self.b_stride, requires_grad=False).weight for i in range(self.rank)
        ]

        self.a_kernels = nn.Parameter(torch.cat([wa_i for wa_i in a_factors],
                                                dim=0), requires_grad=True)  # [r*f1, c1, kh1, kw1]

        self.b_kernels = nn.Parameter(torch.cat([wb_i for wb_i in b_factors],
                                                dim=0), requires_grad=True)  # [r*f2, c2, kh2, kw2]

        self.f1, self.c1, self.h1, self.w1 = self.a_dims
        self.f2, self.c2, self.h2, self.w2 = self.b_dims

        if self.bias:
            sigma = 1. / math.sqrt(self.c1 * self.c2)
            self.bias = nn.Parameter(torch.randn(self.f1 * self.f2))
            self.bias.data.uniform_(-1 * sigma, sigma)

        # rank
        self.rank_upper_bound = min(np.prod(self.a_dims).item(), np.prod(self.b_dims).item())
        if self.rank == 'full':
            self.rank = self.rank_upper_bound
        elif not isinstance(self.rank, int):
            raise Exception("Invalid value for rank given ({})".format(self.rank))
        elif self.rank > self.rank_upper_bound:
            self.rank = self.rank_upper_bound

    @staticmethod
    def _get_b_dims(in_channels: int, out_channels: int, kernel_size: Union[int, tuple], a_dims: tuple):
        b_dims = np.array([out_channels, in_channels, *kernel_size]) // np.array(a_dims)
        if np.array(a_dims) * b_dims != np.array([out_channels, in_channels, *kernel_size]):
            raise Exception("Invalid factorization requested")
        return b_dims.to_list()

    def _update_weights(self, a_tensor: torch.Tensor, b_tensor: torch.Tensor, bias: torch.Tensor = None):
        r"""Fill factors `a` and `b` with pre-determined values

        Args:
            a_tensor: First factor of dimension [r, f1, c1, kh1, kw1]
            b_tensor: Second factor  of dimension [r, f2, c2, kh2, kw2]
            bias: bias vector of shape [f1*f2]
        """

        rank = a_tensor.size(0)
        if rank != self.rank:
            raise Exception("Rank mis-match occurred")

        a_dims, b_dims = a_tensor.shape[1:], b_tensor.shape[1:]
        self.a_kernels = nn.Parameter(a_tensor.reshape(rank * a_dims[0], *a_dims[1:]))
        self.b_kernels = nn.Parameter(b_tensor.reshape(rank * b_dims[0], *b_dims[1:]))

        if bias is not None:
            self.bias = nn.Parameter(bias)


    @classmethod
    def from_conv(cls, base_layer: torch.Tensor, a_dims: tuple = (1, 1, 1, 1), rank: Union[int, str] = 'full'):
        """Creates KroneckerConv2d object from a convolution layer"""

        if not isinstance(base_layer, nn.Conv2d):
            raise Exception("Invalid layer type given ({}). Only nn.Conv2d is supported".format(
                base_layer.__class__.__name__)
            )

        w = base_layer.weight
        out_channels, in_channels = w.shape[:2]
        kernel_size = w.shape[2:]
        b_dims = KroneckerConv2d._get_b_dims(in_channels, out_channels, kernel_size, a_dims)
        a_hat, b_hat = gkpd(w, base_layer.weight, a_dims, b_dims)  # [rank, *a_dims]

        obj = KroneckerConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, a_dims=a_dims, bias=bias, rank=rank)
        obj._update_weights(a_hat, b_hat)
        return obj

    def forward(self, x):
        """Kronecker convolution forward pass

        Note:
            For non-prime kernel dimensions this implementation isn't valid, a less efficient implementation using
            conv3d is needed in that case.

        Args:
            x (torch.tensor): Input with shape [n, c, h, w]

        Returns:
            (torch.tensor): Output with shape [n, f, h', w']
        """
        n, c, h, w = x.shape
        x = x.reshape(n * self.c1, self.c2, h, w)

        # First, apply  `b` factor
        x = F.conv2d(x, self.b_kernels, stride=self.b_stride, padding=self.b_padding)  # [n*c1, r*f2, h, w]

        # Reshaping: [n*c1, r*f2, h, w] => [n*f2, r*c1, h, w]
        h, w = x.shape[-2:]
        x = x.reshape(n, self.c1, self.rank, self.f2, h, w).permute(0, 3, 2, 1, 4, 5).reshape(
            n * self.f2, self.rank * self.c1, h, w)

        # Apply A Kernels
        x = F.conv2d(x, self.a_kernels, stride=self.a_stride, padding=self.a_padding,
                     groups=self.rank)  # [n*f2, r*f1, h, w]

        # Reshaping: [n*f2, r*f1, h, w] => [n, f1*f2, h, w]
        h, w = x.shape[-2:]
        x = x.reshape(n, self.f2, self.rank, self.f1, h, w).sum(dim=2)
        x = x.reshape(n, self.f2, self.f1, h, w).permute(0, 2, 1, 3, 4).reshape(n, self.f1 * self.f2, h, w)

        x = x + self.bias.view(1, self.f1 * self.f2, 1, 1) if self.use_bias else x  # [n, f, w, h]
        return x
