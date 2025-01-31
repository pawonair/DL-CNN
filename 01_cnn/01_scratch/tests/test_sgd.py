"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
Testscript for the sgd optimizer.

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

import unittest
import numpy as np
from optimizer import SGD
from modules import ConvNet
from .utils import *


class TestSGD(unittest.TestCase):
    """ The class containing all test cases for this assignment"""

    def setUp(self):
        """Define the functions to be tested here."""
        pass

    def test_sgd(self):

        model_list = [dict(type='Linear', in_dim=128, out_dim=10)]
        criterion = dict(type='SoftmaxCrossEntropy')
        model = ConvNet(model_list, criterion)

        optimizer = SGD(model)

        # forward once
        np.random.seed(1024)
        x = np.random.randn(32, 128)
        np.random.seed(1024)
        y = np.random.randint(10, size=32)
        tmp = model.forward(x, y)
        model.backward()
        optimizer.update(model)
        # forward twice
        np.random.seed(512)
        x = np.random.randn(32, 128)
        np.random.seed(512)
        y = np.random.randint(10, size=32)
        tmp = model.forward(x, y)
        model.backward()
        optimizer.update(model)

        expected_weights = np.load('tests/sgd_weights/w.npy')
        expected_bias = np.load('tests/sgd_weights/b.npy')

        self.assertAlmostEqual(np.sum(np.abs(expected_weights - model.modules[0].weight)), 0, places=6)
        self.assertAlmostEqual(np.sum(np.abs(expected_bias - model.modules[0].bias)), 0)



