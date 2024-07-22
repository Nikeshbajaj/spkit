from __future__ import absolute_import, division, print_function

name = "Signal Processing toolkit"
__version__ = '0.0.9.6.7'
__author__ = 'Nikesh Bajaj'



import sys, os

sys.path.append(os.path.dirname(__file__))

# try:
#     with open('../Version') as f:
#         __version__ = f.readline().strip()
# except:
#     pass

#from .core.infotheory import *
#import infotheory as it

#ICA
#from .core.matDecomposition import ICA, SVD
#from core.infotheory import *

# Basic Information Theory
from .core.infotheory import (entropy,entropy_joint,entropy_cond,mutual_Info,entropy_kld,entropy_cross,entropy_spectral,entropy_sample,entropy_approx,entropy_svd,entropy_permutation)
from .core.infotheory import (TD_Embed,Mu_law,A_law,bin_width,binSize_FD,quantize_FD,quantize_signal,HistPlot,plotJointEntropyXY)

# Basic Processings
from .core.processing import (filterDC,filterDC_X,filterDC_sGolay,filter_X,Periodogram,getStats,getQuickStats,OutLiers)
from .core.processing import (wavelet_filtering,wavelet_filtering_win,WPA_coeff,WPA_temporal,WPA_plot)
from .core.processing import (conv1d_fft,conv2d_nan,conv1d_nan,conv1d_fb,denorm_kernel,fill_nans_1d,fill_nans_2d)
from .core.processing import (gaussian_kernel, friedrichs_mollifier_kernel,add_noise,sinc_interp)
from .core.processing import (filter_smooth_sGolay, filter_smooth_gauss,filter_smooth_mollifier, filter_with_kernel, filtering_pipeline, filter_powerline)
from .core.processing import (signal_diff, get_activation_time,get_repolarisation_time,agg_angles,show_compass,direction_flow_map)
from .core.processing import (phase_map,amplitude_equalizer,phase_map_reconstruction,dominent_freq,dominent_freq_win,clean_phase,mean_minSE,minMSE)
from .core.processing import (create_signal_1d, create_signal_2d,spatial_filter_dist, spatial_filter_adj)
from .core.matDecomposition import (ICA, SVD,infomax)

from .core.processing import (Wavelet_decompositions)
##wavelet_filtering, wavelet_filtering_win, WPA_coeff, WPA_temporal, WPA_plot

# Advanced
#---------
from .core.infomation_theory_advance import (low_resolution,cdf_mapping,signal_embeddings,signal_delayed_space,create_multidim_space_signal)
from .core.infomation_theory_advance import (dispersion_entropy,dispersion_entropy_multiscale_refined)
from .core.infomation_theory_advance import (entropy_differential,entropy_diff_cond_self,entropy_diff_cond,entropy_diff_joint,entropy_diff_joint_cond,mutual_info_diff_self,mutual_info_diff)
from .core.infomation_theory_advance import (transfer_entropy, transfer_entropy_cond,partial_transfer_entropy,partial_transfer_entropy_,entropy_granger_causality,show_farmulas)

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
from .utils_misc import io_utils as io
from .utils_misc import tf_utils
from .utils_misc import utils as utils_dev
from .utils_misc.borrowed import resize
from .data import load_data

#from .mea.mea_processing import *
from . import geometry
import pylfsr

#LFSR
#from .pylfsr import LFSR

__all__ = ['data','load_data','cwt','utils','io','geometry','eeg' ,'mea','stats','pylfsr', 'ml','tf_utils']
