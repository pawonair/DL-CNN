"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
Implementation of the basic optimizer.

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


class _BaseOptimizer:
    def __init__(self, model, learning_rate=1e-4, reg=1e-3):
        self.learning_rate = learning_rate
        self.reg = reg

        self.grad_tracker = {}

        for idx, m in enumerate(model.modules):
            self.grad_tracker[idx] = dict(dw=0, db=0)

    def update(self, model):
        pass

    def apply_regularization(self, model):
        """
        Apply L2 penalty to the model. Update the gradient dictionary in the model.

        :param model: The model with gradients

        :return: None, but the gradient dictionary of the model should be updated
        """

        for m in model.modules:
            if hasattr(m, 'weight'):
                m.dw += self.reg * m.weight
