from __future__ import absolute_import, division, print_function

name = "Signal Processing toolkit"
__version__ = '0.0.9.7'
__author__ = 'Nikesh Bajaj'


import sys, os

sys.path.append(os.path.dirname(__file__))

# Basic Information Theory
# ========================
from .core.information_theory import (entropy,entropy_joint,entropy_cond,mutual_Info,mutual_info,entropy_kld,entropy_cross)
from .core.information_theory import (entropy_spectral,entropy_sample,entropy_approx,entropy_svd,entropy_permutation)
from .core.information_theory import (TD_Embed,Mu_law,A_law,bin_width,quantize_signal,hist_plot,plotJointEntropyXY)

# Basic Processings
# =================
from .core.processing import (filterDC,filterDC_X,filterDC_sGolay,filter_X)
from .core.processing import (wavelet_filtering,wavelet_filtering_win)
from .core.processing import (wpa_coeff,wpa_coeff_win,wpa_plot,periodogram,wavelet_decomposed_signals) #version 0.0.9.7
from .core.processing import (conv1d_fft,conv2d_nan,conv1d_nan,conv1d_fb,denorm_kernel,fill_nans_1d,fill_nans_2d)
from .core.processing import (gaussian_kernel, friedrichs_mollifier_kernel,add_noise,sinc_interp)
from .core.processing import (filter_smooth_sGolay, filter_smooth_gauss,filter_smooth_mollifier,filter_with_kernel,filtering_pipeline, filter_powerline)
from .core.processing import (signal_diff, get_activation_time,get_repolarisation_time,agg_angles,show_compass,direction_flow_map)
from .core.processing import (phase_map,amplitude_equalizer,phase_map_reconstruction,dominent_freq,dominent_freq_win,clean_phase,mean_minSE,minMSE)
from .core.processing import (create_signal_1d, create_signal_2d,spatial_filter_dist, spatial_filter_adj)
from .core.decomposition import (ICA, SVD,infomax, PCA)

# New
# =====
from .core.processing import (elbow_knee_point,total_variation, total_variation_win)
from .core.processing import (graph_filter_dist,graph_filter_adj)

##wavelet_filtering, wavelet_filtering_win, WPA_coeff, WPA_temporal, WPA_plot

# Advanced
# =========
from .core.information_theory_advance import (low_resolution,cdf_mapping,signal_embeddings,signal_delayed_space,create_multidim_space_signal)
from .core.information_theory_advance import (dispersion_entropy,dispersion_entropy_multiscale_refined)
from .core.information_theory_advance import (entropy_differential,entropy_diff_cond_self,entropy_diff_cond,entropy_diff_joint,
                                             entropy_diff_joint_cond,mutual_info_diff_self,mutual_info_diff)
from .core.information_theory_advance import (transfer_entropy, transfer_entropy_cond,partial_transfer_entropy,partial_transfer_entropy_,
                                              entropy_granger_causality,show_farmulas)

from .core.advance_techniques import (peak_detection, f0_detection, peak_interp,TWM_algo,sinc_dirichlet,blackman_lobe,sine_spectrum,sinetracks_cleaning)
from .core.advance_techniques import (dft_analysis, dft_synthesis, stft_analysis, stft_synthesis, sineModel_analysis, sineModel_synthesis, simplify_signal)
from .core.advance_techniques import (TWM_f0,is_power2,sine_tracking)

from .core.fractional_processes import (frft, ifrft, ffrft, iffrft)
from .core.ramanujan_methods import (RFB_example_1, RFB_example_2)
from .core.ramanujan_methods import (ramanujan_filter,ramanujan_filter_prange, create_dictionary,regularised_period_estimation)


# To Be REMOVED in future versions
# ================================
from .core.processing import (filterDC_X,Periodogram,getStats,getQuickStats,OutLiers)
from .core.information_theory import (quantize_FD,HistPlot,plotJointEntropyXY, Quantize)
from .core.processing import (WPA_coeff,WPA_temporal,WPA_plot)
from .core.processing import (Wavelet_decompositions)
from .core.advance_techniques import (isPower2)
from .core.ramanujan_methods import (RFB, Create_Dictionary, PeriodStrength, RFB_prange)


#from .io
##from .io import read_hdf
#import io_utilis as io

from . import ml, data, utils, eeg, mea, cwt, stats, io
from . import text
#from .stats import stats
#from .core import cwt
#from .core import *
#from .utils_misc import io_utils as io
#from .utils_misc import io_utils as io
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
