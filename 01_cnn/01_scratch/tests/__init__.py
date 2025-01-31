import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .gradient_check import *
from .test_conv import TestConv
from .test_linear import TestLinear
from .test_maxpool import TestConv as TestMaxPool
from .test_relu import TestReLU
from .test_sgd import TestSGD