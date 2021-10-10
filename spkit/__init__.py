from __future__ import absolute_import, division, print_function

name = "Signal Processing toolkit"
__version__ = '0.0.9.3'
__author__ = 'Nikesh Bajaj'


import sys, os

sys.path.append(os.path.dirname(__file__))

#from .core.infotheory import *
#import infotheory as it

#ICA
#from .core.matDecomposition import ICA, SVD
from core.infotheory import (entropy,entropy_joint,entropy_cond,mutual_Info,entropy_kld,entropy_cross,entropy_spectral,entropy_sample,entropy_approx,entropy_svd,entropy_permutation,HistPlot,binSize_FD,TD_Embed)
#from core.infotheory import *
from core.matDecomposition import ICA, SVD #ICA
from core.processing import (filterDC,filterDC_X,filterDC_sGolay,filter_X,Periodogram,getStats,getQuickStats,OutLiers)
from core.processing import (wavelet_filtering,wavelet_filtering_win,get_theta,WPA_coeff,WPA_temporal,WPA_plot)
from .core import cwt
from .core import *
import ml, data, utils
from .data import load_data
import eeg

#LFSR
#from .pylfsr import LFSR

__all__ = ['ICA', 'SVD','pylfsr', 'ml', 'example','data','load_data','cwt','utils']
