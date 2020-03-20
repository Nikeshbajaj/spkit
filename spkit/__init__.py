from __future__ import absolute_import, division, print_function

name = "Signal Processing toolkit"
__version__ = '0.0.9'
__author__ = 'Nikesh Bajaj'


import sys, os

sys.path.append(os.path.dirname(__file__))

from .infotheory import (entropy, entropy_joint, entropy_cond, mutual_Info,entropy_kld,entropy_cross,HistPlot,binSize_FD)
#import infotheory as it

#ICA
from .matDecomposition import ICA, SVD
import ml

#LFSR
#from .pylfsr import LFSR

__all__ = ['ICA', 'SVD','pylfsr', 'ml', 'example','data']
