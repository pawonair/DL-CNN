"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
PyTorch implementation of a vanilla CNN.

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


class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()

        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #           - Conv: 7x7 kernel, stride 1 and padding 1                      #
        #           - Max Pooling: 2x2 kernel, stride 2                             #
        #############################################################################
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 14 * 14, 10)  # CIFAR-10: input 32x32, after conv+pool: 16x13x13
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        outs = self.conv(x)
        outs = self.relu(outs)
        outs = self.pool(outs)
        outs = self.flatten(outs)
        outs = self.fc(outs)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs
