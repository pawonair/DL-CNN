"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
Implementation of the max pooling operation.

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


class MaxPooling:
    """
    Max Pooling of input
    """
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling.

        :param x: input, (N, C, H, W)

        :return: The output by max pooling with kernel_size and stride
        """

        out = None
        H_out, W_out = None, None

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the max pooling forward pass.                             #
        #                                                                           #
        # Hint:                                                                     #
        #    1) You may implement the process with loops                            #
        #############################################################################

        N, C, H, W = x.shape
        HH, WW = self.kernel_size, self.kernel_size
        stride = self.stride
        H_out = (H - HH) // stride + 1
        W_out = (W - WW) // stride + 1

        out = np.zeros((N, C, H_out, W_out))

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start, w_start = i * stride, j * stride
                        h_end, w_end = h_start + HH, w_start + WW
                        out[n, c, i, j] = np.max(x[n, c, h_start:h_end, w_start:w_end])

        H_out, W_out = out.shape[2], out.shape[3]

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling.

        :param dout: Upstream derivatives
        """

        x, H_out, W_out = self.cache

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the max pooling backward pass.                            #
        #                                                                           #
        # Hint:                                                                     #
        #    1) You may implement the process with loops                            #
        #    2) You may find np.unravel_index useful                                #
        #############################################################################

        N, C, H, W = x.shape
        HH, WW = self.kernel_size, self.kernel_size
        stride = self.stride
        H_out, W_out = dout.shape[2], dout.shape[3]

        self.dx = np.zeros_like(x)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start, w_start = i * stride, j * stride
                        h_end, w_end = h_start + HH, w_start + WW
                        x_slice = x[n, c, h_start:h_end, w_start:w_end]
                        max_idx = np.unravel_index(np.argmax(x_slice), x_slice.shape)
                        self.dx[n, c, h_start + max_idx[0], w_start + max_idx[1]] += dout[n, c, i, j]

        return self.dx
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
