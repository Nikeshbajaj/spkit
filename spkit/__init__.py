from __future__ import absolute_import, division, print_function

name = "Signal Processing toolkit"
__version__ = '0.0.9.4'
__author__ = 'Nikesh Bajaj'


import sys, os

sys.path.append(os.path.dirname(__file__))

#from .core.infotheory import *
#import infotheory as it

#ICA
#from .core.matDecomposition import ICA, SVD
from core.infotheory import (entropy,entropy_joint,entropy_cond,mutual_Info,entropy_kld,entropy_cross,entropy_spectral,entropy_sample,entropy_approx,entropy_svd,entropy_permutation)
from core.infotheory import (TD_Embed,Mu_law,A_law,bin_width,binSize_FD,quantize_FD,Quantize,HistPlot,plotJointEntropyXY)
#from core.infotheory import *
from core.matDecomposition import ICA, SVD #ICA
from core.processing import (filterDC,filterDC_X,filterDC_sGolay,filter_X,Periodogram,getStats,getQuickStats,OutLiers)
from core.processing import (wavelet_filtering,wavelet_filtering_win,WPA_coeff,WPA_temporal,WPA_plot)
##wavelet_filtering, wavelet_filtering_win, WPA_coeff, WPA_temporal, WPA_plot

# Advanced
#---------
from core.infomation_theory_advance import (low_resolution,cdf_mapping,dispersion_entropy,dispersion_entropy_multiscale_refined)
from core.advance_techniques import (peak_detection, f0_detection,isPower2)
from core.advance_techniques import (dft_analysis, dft_synthesis, stft_analysis, stft_synthesis, sineModel_analysis, sineModel_synthesis)
from core.ramanujam_methods import (RFB, Create_Dictionary, PeriodStrength, RFB_example_1, RFB_example_2, RFB_prange)


from core.fractional_processes import (frft, ifrft, ffrft, iffrft)

#

from .core import cwt
from .core import *
import ml, data, utils
from .data import load_data
import eeg

#LFSR
#from .pylfsr import LFSR

__all__ = ['ICA', 'SVD','pylfsr', 'ml', 'example','data','load_data','cwt','utils']
