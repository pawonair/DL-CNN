"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
Testscript for the relu activation function.

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
from modules import ReLU
from .utils import *


class TestReLU(unittest.TestCase):
    """ The class containing all test cases for this assignment"""

    def setUp(self):
        """Define the functions to be tested here."""
        pass

    def _relu_forward(self, x):
        relu = ReLU()
        return relu.forward(x)

    def test_forward(self):
        x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
        relu = ReLU()
        out = relu.forward(x)
        correct_out = np.array([[0., 0., 0., 0., ],
                                [0., 0., 0.04545455, 0.13636364, ],
                                [0.22727273, 0.31818182, 0.40909091, 0.5, ]])
        diff = rel_error(out, correct_out)
        self.assertAlmostEqual(diff, 0, places=7)

    def test_backward(self):
        x = np.random.randn(10, 10)
        dout = np.random.randn(*x.shape)

        dx_num = eval_numerical_gradient_array(lambda x: self._relu_forward(x), x, dout)

        relu = ReLU()
        out = relu.forward(x)
        relu.backward(dout)
        dx = relu.dx

        self.assertAlmostEqual(rel_error(dx_num, dx), 0, places=7)


