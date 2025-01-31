#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
This script was written as a test module for the 'regularization.ipynb'-jupyter notebook.
Students may use the functions below to validate their solutions of the proposed tasks.

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

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
# Import packages
import numpy as np
import unittest
import argparse

# Import own files
import nbimporter  # Necessary to be able to use equations of ipynb-notebooks in python-files
import regularization


class TestCostL2Regularization(unittest.TestCase):
    """
    The class contains all test cases for task "7.1 - Cost Function".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the parameters
        np.random.seed(1)
        W1 = np.random.randn(2, 3)
        b1 = np.random.randn(2, 1)
        W2 = np.random.randn(3, 2)
        b2 = np.random.randn(3, 1)
        W3 = np.random.randn(1, 3)
        b3 = np.random.randn(1, 1)
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

        Y = np.array([[1, 1, 0, 1, 0]])
        A3 = np.array([[ 0.40682402,  0.01629284,  0.16722898,  0.10118111,  0.40682402]])
        lambd = 0.1

        # Create the student version
        self.stud_vers = regularization.compute_cost_with_regularization(A3, Y, parameters, lambd)

        # Load the references
        self.exp_vers = np.float64(1.7864859451590758)

    def test_cost(self):
        """ Test the cost 'cost'. """

        stud_vers = self.stud_vers
        exp_vers = self.exp_vers

        self.assertNotIsInstance(stud_vers, np.ndarray, msg="Type of 'cost' is not correct!")
        self.assertAlmostEqual(stud_vers, exp_vers, delta=0.0001, msg="'cost' is not correct!")


class TestBackpropagationWithL2Regularization(unittest.TestCase):
    """
    The class contains all test cases for task "7.2 - Backward Propagation with Regularization".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the parameters
        np.random.seed(1)
        X = np.random.randn(3, 5)
        Y = np.array([[1, 1, 0, 1, 0]])
        cache = (np.array([[-1.52855314,  3.32524635,  2.13994541,  2.60700654, -0.75942115],
                           [-1.98043538,  4.1600994 ,  0.79051021,  1.46493512, -0.45506242]]),
                 np.array([[ 0.        ,  3.32524635,  2.13994541,  2.60700654,  0.        ],
                           [ 0.        ,  4.1600994 ,  0.79051021,  1.46493512,  0.        ]]),
                 np.array([[-1.09989127, -0.17242821, -0.87785842],
                           [ 0.04221375,  0.58281521, -1.10061918]]),
                 np.array([[ 1.14472371],
                           [ 0.90159072]]),
                 np.array([[ 0.53035547,  5.94892323,  2.31780174,  3.16005701,  0.53035547],
                           [-0.69166075, -3.47645987, -2.25194702, -2.65416996, -0.69166075],
                           [-0.39675353, -4.62285846, -2.61101729, -3.22874921, -0.39675353]]),
                 np.array([[ 0.53035547,  5.94892323,  2.31780174,  3.16005701,  0.53035547],
                           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]]),
                 np.array([[ 0.50249434,  0.90085595],
                           [-0.68372786, -0.12289023],
                           [-0.93576943, -0.26788808]]),
                 np.array([[ 0.53035547],
                           [-0.69166075],
                           [-0.39675353]]),
                 np.array([[-0.3771104 , -4.10060224, -1.60539468, -2.18416951, -0.3771104 ]]),
                 np.array([[ 0.40682402,  0.01629284,  0.16722898,  0.10118111,  0.40682402]]),
                 np.array([[-0.6871727 , -0.84520564, -0.67124613]]),
                 np.array([[-0.0126646]]))
        lambd = 0.7

        # Create the student version
        self.stud_vers = regularization.backward_propagation_with_regularization(X, Y, cache, lambd)

        # Load the references
        self.exp_vers = {'dZ3': np.array([[-0.59317598, -0.98370716,  0.16722898, -0.89881889,  0.40682402]]),
                         'dW3': np.array([[-1.77691347, -0.11832879, -0.09397446]]),
                         'db3': np.array([[-0.38032981]]),
                         'dA2': np.array([[ 0.40761434,  0.67597671, -0.11491519,  0.6176438 , -0.27955836],
                                          [ 0.50135568,  0.83143484, -0.14134288,  0.7596868 , -0.34384996],
                                          [ 0.39816708,  0.66030962, -0.11225181,  0.6033287 , -0.27307905]]),
                         'dZ2': np.array([[ 0.40761434,  0.67597671, -0.11491519,  0.6176438 , -0.27955836],
                                          [ 0.        ,  0.        , -0.        ,  0.        , -0.        ],
                                          [ 0.        ,  0.        , -0.        ,  0.        , -0.        ]]),
                         'dW2': np.array([[ 0.79276486,  0.85133918],
                                          [-0.0957219 , -0.01720463],
                                          [-0.13100772, -0.03750433]]),
                         'db2': np.array([[0.26135226],
                                          [0.        ],
                                          [0.        ]]),
                         'dA1': np.array([[ 0.2048239 ,  0.33967447, -0.05774423,  0.31036252, -0.14047649],
                                          [ 0.3672018 ,  0.60895764, -0.10352203,  0.5564081 , -0.25184181]]),
                         'dZ1': np.array([[ 0.        ,  0.33967447, -0.05774423,  0.31036252, -0.        ],
                                          [ 0.        ,  0.60895764, -0.10352203,  0.5564081 , -0.        ]]),
                         'dW1': np.array([[-0.25604646,  0.12298827, -0.28297129],
                                          [-0.17706303,  0.34536094, -0.4410571 ]]),
                         'db1': np.array([[0.11845855],
                                          [0.21236874]])}

    def test_gradients(self):
        """
        Test the dictionary 'gradients' containing the gradients with respect to each parameter, activation and
        pre-activation variables.
        """

        stud_vers = self.stud_vers
        exp_vers = self.exp_vers

        self.assertIsInstance(stud_vers, dict, "Type of 'gradients' is not correct!")
        self.assertEqual(len(stud_vers.keys()), len(exp_vers.keys()), "Number of entries of 'gradients' is not correct!")

        for key in exp_vers.keys():
            self.assertIn(key, stud_vers.keys(), f"Key '{key}' is missing in gradients!")
            self.assertIsInstance(stud_vers[key], np.ndarray, f"Type of gradients['{key}'] is not correct!")
            self.assertEqual(stud_vers[key].shape, exp_vers[key].shape, f"Shape of gradients['{key}'] is not correct!")
            self.assertTrue(np.allclose(stud_vers[key], exp_vers[key], atol=0.0001), f"gradients['{key}'] of is not correct!")


class TestForwardPropagationWithDropout(unittest.TestCase):
    """
    The class contains all test cases for task "8.1 - Forward Propagation with Dropout".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the parameters
        np.random.seed(1)
        X = np.random.randn(3, 5)
        W1 = np.random.randn(2, 3)
        b1 = np.random.randn(2, 1)
        W2 = np.random.randn(3, 2)
        b2 = np.random.randn(3, 1)
        W3 = np.random.randn(1, 3)
        b3 = np.random.randn(1, 1)
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
        keep_prob = 0.7
        self.cache_structure = ('Z1', 'D1', 'A1', 'W1', 'b1', 'Z2', 'D2', 'A2', 'W2', 'b2', 'Z3', 'A3', 'W3', 'b3')

        # Create the student version
        stud_o1, stud_o2 = regularization.forward_propagation_with_dropout(X, parameters, keep_prob)

        self.stud_vers = {'A3': stud_o1,
                          'cache': stud_o2}

        # Load the references
        self.exp_vers = {'A3': np.array([[0.36974721, 0.00305176, 0.04565099, 0.49683389, 0.36974721]]),
                         'cache': (np.array([[-1.52855314,  3.32524635,  2.13994541,  2.60700654, -0.75942115],
                                             [-1.98043538,  4.1600994 ,  0.79051021,  1.46493512, -0.45506242]]),
                                   np.array([[ True, False,  True,  True,  True],
                                             [ True,  True,  True,  True,  True]]),
                                   np.array([[0.        , 0.        , 3.05706487, 3.72429505, 0.        ],
                                             [0.        , 5.94299915, 1.1293003 , 2.09276446, 0.        ]]),
                                   np.array([[-1.09989127, -0.17242821, -0.87785842],
                                             [ 0.04221375,  0.58281521, -1.10061918]]),
                                   np.array([[1.14472371],
                                             [0.90159072]]),
                                   np.array([[ 0.53035547,  5.88414161,  3.08385015,  4.28707196,  0.53035547],
                                             [-0.69166075, -1.42199726, -2.92064114, -3.49524533, -0.69166075],
                                             [-0.39675353, -1.98881216, -3.55998747, -4.44246165, -0.39675353]]),
                                   np.array([[ True,  True,  True, False,  True],
                                             [ True,  True,  True,  True,  True],
                                             [False, False,  True,  True, False]]),
                                   np.array([[0.75765067, 8.40591658, 4.40550021, 0.        , 0.75765067],
                                             [0.        , 0.        , 0.        , 0.        , 0.        ],
                                             [0.        , 0.        , 0.        , 0.        , 0.        ]]),
                                   np.array([[ 0.50249434,  0.90085595],
                                             [-0.68372786, -0.12289023],
                                             [-0.93576943, -0.26788808]]),
                                   np.array([[ 0.53035547],
                                             [-0.69166075],
                                             [-0.39675353]]),
                                   np.array([[-0.53330145, -5.78898099, -3.04000407, -0.0126646 , -0.53330145]]),
                                   np.array([[0.36974721, 0.00305176, 0.04565099, 0.49683389, 0.36974721]]),
                                   np.array([[-0.6871727 , -0.84520564, -0.67124613]]),
                                   np.array([[-0.0126646]]))}

    def test_output(self):
        """ Test the output of the 'forward_propagation_with_dropout' method. """

        stud_vers = self.stud_vers
        exp_vers = self.exp_vers

        self.assertEqual(len(stud_vers.keys()), len(exp_vers.keys()), "Number of output elements is not correct!")

    def test_cache(self):
        """ Test the output 'cache'. """

        stud_vers = self.stud_vers['cache']
        exp_vers = self.exp_vers['cache']

        self.assertIsInstance(stud_vers, tuple, "Type of 'gradients' is not correct!")
        self.assertEqual(len(stud_vers), len(exp_vers), "Number of entries of 'cache' is not correct!")

        for cache_entry, stud, exp in zip(self.cache_structure, stud_vers, exp_vers):
            self.assertIsInstance(stud, np.ndarray, f"Type of {cache_entry} in 'cache' is not correct!")
            self.assertEqual(stud.shape, exp.shape, f"Shape of {cache_entry} in 'cache' is not correct!")
            self.assertTrue(np.allclose(stud, exp, atol=0.0001), f"{cache_entry} in 'cache' is not correct!")

    def test_A3(self):
        """ Test the output 'A3'. """

        stud_vers = self.stud_vers['A3']
        exp_vers = self.exp_vers['A3']

        self.assertIsInstance(stud_vers, np.ndarray, "Type of 'A3' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'A3' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), f"'A3' is not correct!")


class TestBackpropagationWithDropout(unittest.TestCase):
    """
    The class contains all test cases for task "8.2 - Backward Propagation with Dropout".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the parameters
        np.random.seed(1)
        X = np.random.randn(3, 5)
        Y = np.array([[1, 1, 0, 1, 0]])
        cache = (np.array([[-1.52855314,  3.32524635,  2.13994541,  2.60700654, -0.75942115],
                           [-1.98043538,  4.1600994 ,  0.79051021,  1.46493512, -0.45506242]]),
                 np.array([[ True, False,  True,  True,  True],
                           [ True,  True,  True,  True, False]], dtype=bool),
                 np.array([[ 0.        ,  0.        ,  4.27989081,  5.21401307,  0.        ],
                           [ 0.        ,  8.32019881,  1.58102041,  2.92987024,  0.        ]]),
                 np.array([[-1.09989127, -0.17242821, -0.87785842],
                           [ 0.04221375,  0.58281521, -1.10061918]]),
                 np.array([[ 1.14472371],
                           [ 0.90159072]]),
                 np.array([[ 0.53035547,  8.02565606,  4.10524802,  5.78975856,  0.53035547],
                           [-0.69166075, -1.71413186, -3.81223329, -4.61667916, -0.69166075],
                           [-0.39675353, -2.62563561, -4.82528105, -6.0607449 , -0.39675353]]),
                 np.array([[ True, False,  True, False,  True],
                           [False,  True, False,  True,  True],
                           [False, False,  True, False, False]], dtype=bool),
                 np.array([[ 1.06071093,  0.        ,  8.21049603,  0.        ,  1.06071093],
                           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
                           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]]),
                 np.array([[ 0.50249434,  0.90085595],
                           [-0.68372786, -0.12289023],
                           [-0.93576943, -0.26788808]]),
                 np.array([[ 0.53035547],
                           [-0.69166075],
                           [-0.39675353]]),
                 np.array([[-0.7415562 , -0.0126646 , -5.65469333, -0.0126646 , -0.7415562 ]]),
                 np.array([[ 0.32266394,  0.49683389,  0.00348883,  0.49683389,  0.32266394]]),
                 np.array([[-0.6871727 , -0.84520564, -0.67124613]]),
                 np.array([[-0.0126646]]))
        keep_prob = 0.8

        # Create the student version
        self.stud_vers = regularization.backward_propagation_with_dropout(X, Y, cache, keep_prob)

        # Load the references
        self.exp_vers = {'dZ3': np.array([[-0.67733606, -0.50316611,  0.00348883, -0.50316611,  0.32266394]]),
                         'dW3': np.array([[-0.06951191,  0.        ,  0.        ]]),
                         'db3': np.array([[-0.2715031]]),
                         'dA2': np.array([[ 0.58180856,  0.        , -0.00299679,  0.        , -0.27715731],
                                          [ 0.        ,  0.53159854, -0.        ,  0.53159854, -0.34089673],
                                          [ 0.        ,  0.        , -0.00292733,  0.        , -0.        ]]),
                         'dZ2': np.array([[ 0.58180856,  0.        , -0.00299679,  0.        , -0.27715731],
                                          [ 0.        ,  0.        , -0.        ,  0.        , -0.        ],
                                          [ 0.        ,  0.        , -0.        ,  0.        , -0.        ]]),
                         'dW2': np.array([[-0.00256518, -0.0009476 ],
                                          [ 0.        ,  0.        ],
                                          [ 0.        ,  0.        ]]),
                         'db2': np.array([[0.06033089],
                                          [0.        ],
                                          [0.        ]]),
                         'dA1': np.array([[ 0.36544439,  0.        , -0.00188233,  0.        , -0.17408748],
                                          [ 0.65515713,  0.        , -0.00337459,  0.        , -0.        ]]),
                         'dZ1': np.array([[ 0.        ,  0.        , -0.00188233,  0.        , -0.        ],
                                          [ 0.        ,  0.        , -0.00337459,  0.        , -0.        ]]),
                         'dW1': np.array([[0.00019884, 0.00028657, 0.00012138],
                                          [0.00035647, 0.00051375, 0.00021761]]),
                         'db1': np.array([[-0.00037647],
                                          [-0.00067492]])}

    def test_gradients(self):
        """
        Test the dictionary 'gradients' containing the gradients with respect to each parameter, activation and
        pre-activation variables.
        """

        stud_vers = self.stud_vers
        exp_vers = self.exp_vers

        self.assertIsInstance(stud_vers, dict, "Type of 'gradients' is not correct!")
        self.assertEqual(len(stud_vers.keys()), len(exp_vers.keys()), "Number of entries of 'gradients' is not correct!")

        for key in exp_vers.keys():
            self.assertIn(key, stud_vers.keys(), f"Key '{key}' is missing in gradients!")
            self.assertIsInstance(stud_vers[key], np.ndarray, f"Type of gradients['{key}'] is not correct!")
            self.assertEqual(stud_vers[key].shape, exp_vers[key].shape, f"Shape of gradients['{key}'] is not correct!")
            self.assertTrue(np.allclose(stud_vers[key], exp_vers[key], atol=0.0001), f"gradients['{key}'] of is not correct!")




if __name__ == '__main__':
    # Instantiate the command line parser
    parser = argparse.ArgumentParser()

    # Add the option to run only a specific test case
    parser.add_argument('--test_case', help='Name of the test case you want to run')

    # Read the command line parameters
    args = parser.parse_args()

    # Run only a single test class
    if args.test_case:
        test_class = eval(args.test_case)
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        unittest.TextTestRunner().run(suite)

    # Run all test classes
    else:
        unittest.main(argv=[''], verbosity=1, exit=False)