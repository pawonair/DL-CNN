"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
PyTorch implementation of 2-Layer Neural Network.

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

import torch
import torch.nn as nn


class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """

        super(TwoLayerNet, self).__init__()

        #############################################################################
        # TODO: Initialize the TwoLayerNet                                          #
        #           - use sigmoid activation between layers                         #
        #############################################################################
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        out = None

        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        # Flattening the input tensor to shape (batch_size, input_dim)
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out
