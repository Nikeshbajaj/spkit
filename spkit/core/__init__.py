from __future__ import absolute_import, division, print_function

name = "Signal Processing toolkit | Core funtions"
__version__ = '0.0.9.3'
__author__ = 'Nikesh Bajaj'


import sys, os

sys.path.append(os.path.dirname(__file__))

#from .infotheory import *
#from .matDecomposition import ICA, SVD #ICA
from .processing import *
import cwt

#__all__ = ['ICA', 'SVD','cwt']
