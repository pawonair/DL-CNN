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

        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
