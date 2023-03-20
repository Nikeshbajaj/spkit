from __future__ import absolute_import, division, print_function

name = "Signal Processing toolkit"
__version__ = '0.0.9.5'
__author__ = 'Nikesh Bajaj'


import sys, os

sys.path.append(os.path.dirname(__file__))

#from .core.infotheory import *
#import infotheory as it

#ICA
#from .core.matDecomposition import ICA, SVD
#from core.infotheory import *
from .core.infotheory import (entropy,entropy_joint,entropy_cond,mutual_Info,entropy_kld,entropy_cross,entropy_spectral,entropy_sample,entropy_approx,entropy_svd,entropy_permutation)
from .core.infotheory import (TD_Embed,Mu_law,A_law,bin_width,binSize_FD,quantize_FD,quantize_signal,HistPlot,plotJointEntropyXY)
from .core.processing import (filterDC,filterDC_X,filterDC_sGolay,filter_X,Periodogram,getStats,getQuickStats,OutLiers)
from .core.processing import (wavelet_filtering,wavelet_filtering_win,WPA_coeff,WPA_temporal,WPA_plot)
from .core.processing import (conv_fft,conv2d_nan,conv1d_nan,conv1d_fb,denorm_kernel,fill_nans_1d,fill_nans_2d,sinc_interp)
from .core.processing import (gaussian_kernel, friedrichs_mollifier_kernel,add_noise)
from .core.processing import (filter_smooth_sGolay, filter_smooth_gauss,filter_smooth_mollifier, filter_with_kernel, filtering_pipeline, filter_powerline)
from .core.processing import (signal_diff, get_activation_time,get_repolarisation_time,agg_angles,show_compass,direction_flow_map)
from .core.processing import (phase_map,amplitude_equalizer,phase_map_reconstruction,dominent_freq,dominent_freq_win,clean_phase,mean_minSE,minMSE)
from .core.processing import (create_signal_1d, create_signal_2d)
from .core.matDecomposition import (ICA, SVD,infomax)
##wavelet_filtering, wavelet_filtering_win, WPA_coeff, WPA_temporal, WPA_plot

# Advanced
#---------
from .core.infomation_theory_advance import (low_resolution,cdf_mapping,dispersion_entropy,dispersion_entropy_multiscale_refined)
from .core.advance_techniques import (peak_detection, f0_detection,isPower2, peak_interp,TWM_algo,sinc_dirichlet,blackman_lobe,sine_spectrum,sinetracks_cleaning)
from .core.advance_techniques import (dft_analysis, dft_synthesis, stft_analysis, stft_synthesis, sineModel_analysis, sineModel_synthesis, simplify_signal)
from .core.ramanujam_methods import (RFB, Create_Dictionary, PeriodStrength, RFB_prange,RFB_example_1, RFB_example_2)
from .core.fractional_processes import (frft, ifrft, ffrft, iffrft)

#
#from .io
##from .io import read_hdf
#import io_utilis as io

from . import ml, data, utils, eeg, mea
from .stats import stats
from .core import cwt
#from .core import *
from .all_utils import io_utils as io
from .all_utils import tf_utils
from .all_utils import utils as utils_dev
from .data import load_data

#from .mea.mea_processing import *
from . import geometry
import pylfsr

#LFSR
#from .pylfsr import LFSR

__all__ = ['data','load_data','cwt','utils','io','geometry','eeg' ,'mea','stats','pylfsr', 'ml','tf_utils']
