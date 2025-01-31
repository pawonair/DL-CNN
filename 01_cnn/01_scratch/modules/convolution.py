"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
Script ties all modules together to create the convolutional neural network.

@author: Sebastian Doerrich
@copyright: Copyright (c) 2022, Chair of Explainable Machine Learning (xAI), Otto-Friedrich University of Bamberg
@credits: [Christian Ledig, Sebastian Doerrich]
@license: CC BY-SA
@version: 1.0
@python: Python 3
@maintainer: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
@status: Production
"""

import numpy as np


class Conv2D:
    """
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution.

        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """

        out = None

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the convolution forward pass.                             #
        #                                                                           #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################

        N, C, H, W = x.shape
        pad = self.padding
        stride = self.stride
        F, _, HH, WW = self.weight.shape
        H_Out = (H + 2 * pad - HH) // stride + 1
        W_Out = (W + 2 * pad - WW) // stride + 1

        x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad)), mode='constant', constant_values=0)
        out = np.zeros((N, F, H_Out, W_Out))

        for i in range(H_Out):
            for j in range(W_Out):
                x_slice = x_padded[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
                for f in range(F):
                    out[:, f, i, j] = np.sum(x_slice * self.weight[f], axis=(1, 2, 3)) + self.bias[f]

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        """
        The backward pass of convolution

        :param dout: upstream gradients

        :return: nothing but dx, dw, and db of self should be updated
        """

        x = self.cache

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the convolution backward pass.                            #
        #                                                                           #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################

        N, C, H, W = x.shape
        F, _, HH, WW = self.weight.shape
        pad = self.padding
        stride = self.stride
        H_Out, W_Out = dout.shape[2], dout.shape[3]

        x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad)), mode='constant', constant_values=0)
        dx_padded = np.zeros_like(x_padded)
        self.dw = np.zeros_like(self.weight)
        self.db = np.zeros_like(self.bias)

        for i in range(H_Out):
            for j in range(W_Out):
                x_slice = x_padded[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
                for f in range(F):
                    self.dw[f] += np.sum(x_slice * dout[:, f, i, j][:, None, None, None], axis=0)
                for n in range(N):
                    dx_padded[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += np.sum(self.weight * dout[n, :, i, j][:, None, None, None], axis=0)

        self.db = np.sum(dout, axis=(0, 2, 3))

        if pad > 0:
            self.dx = dx_padded[:, :, pad:-pad, pad:-pad]
        else:
            self.dx = dx_padded

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
