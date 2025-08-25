"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
Implementation of the ReLU activation function.

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


class ReLU:
    def __init__(self):
        self.cache = None
        self.dx= None

    def forward(self, x):
        """
        The forward pass of ReLU. Save necessary variables for backward.

        :param x: input data

        :return: output of the ReLU function
        """

        out = None

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the ReLU forward pass.                                    #
        #############################################################################

        out = np.maximum(0, x)
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x

        return out

    def backward(self, dout):
        """
        The backward pass of ReLU.

        :param dout: the upstream gradients
        """

        dx, x = None, self.cache

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the ReLU backward pass.                                   #
        #############################################################################

        dx = dout * (x > 0)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        self.dx = dx
        return dx
