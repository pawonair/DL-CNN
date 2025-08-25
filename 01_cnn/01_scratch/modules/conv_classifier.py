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

from .softmax_ce import SoftmaxCrossEntropy
from .relu import ReLU
from .max_pool import MaxPooling
from .convolution import Conv2D
from .linear import Linear


class ConvNet:
    '''
    Max Pooling of input
    '''
    def __init__(self, modules, criterion):
        self.modules = []
        for m in modules:
            if m['type'] == 'Conv2D':
                self.modules.append(
                    Conv2D(m['in_channels'],
                           m['out_channels'],
                           m['kernel_size'],
                           m['stride'],
                           m['padding'])
                )
            elif m['type'] == 'ReLU':
                self.modules.append(
                    ReLU()
                )
            elif m['type'] == 'MaxPooling':
                self.modules.append(
                    MaxPooling(m['kernel_size'],
                               m['stride'])
                )
            elif m['type'] == 'Linear':
                self.modules.append(
                    Linear(m['in_dim'],
                           m['out_dim'])
                )
        if criterion['type'] == 'SoftmaxCrossEntropy':
            self.criterion = SoftmaxCrossEntropy()
        else:
            raise ValueError("Wrong Criterion Passed")

    def forward(self, x, y):
        """
        The forward pass of the model.

        :param x: input data: (N, C, H, W)
        :param y: input label: (N, )

        :return:
            probs: the probabilities of all classes: (N, num_classes)
            loss: the cross entropy loss
        """

        probs = None
        loss = None

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement forward pass of the model                                 #
        #############################################################################

        # Forward pass through all modules
        for modules in self.modules:
            x = modules.forward(x)

        probs = x  # Set final output after the last layer
        loss = self.criterion.forward(probs, y)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return probs, loss

    def backward(self):
        """
        The backward pass of the model.

        :return: nothing but dx, dw, and db of all modules are updated
        """

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement backward pass of the model                                #
        #############################################################################

        dx = self.criterion.backward()  # Compute initial gradient
        for modules in reversed(self.modules):
            dx = modules.backward(dx)  # Back-propagate through all layers in reverse

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
