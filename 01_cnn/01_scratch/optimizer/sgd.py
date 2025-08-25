"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
Implementation of the SGD optimizer with momentum

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

from ._base_optimizer import _BaseOptimizer


class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum

        # initialize the velocity terms for each weight
        self.velocities = {}

        for idx, m in enumerate(model.modules):
            self.velocities[idx] = dict(vt_weight=0, vt_bias=0)

    def update(self, model):
        """
        Update model weights based on gradients.

        :param model: The model to be updated

        :return: None, but the model weights should be updated
        """

        self.apply_regularization(model)

        for idx, m in enumerate(model.modules):
            if hasattr(m, 'weight'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for weights                                        #
                #############################################################################

                vt = self.velocities[idx]['vt_weight']
                vt = self.momentum * vt - self.learning_rate * m.dweight
                self.velocities[idx]['vt_weight'] = vt
                m.weight += vt
                
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################

            if hasattr(m, 'bias'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for bias                                           #
                #############################################################################

                vt = self.velocities[idx]['vt_bias']
                vt = self.momentum * vt - self.learning_rate * m.dbias
                self.velocities[idx]['vt_bias'] = vt
                m.bias += vt
                
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
