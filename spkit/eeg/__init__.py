from __future__ import absolute_import, division, print_function

name = "Signal Processing toolkit | EEG Processing"
__version__ = '0.0.9.5'
__author__ = 'Nikesh Bajaj'

import sys, os

sys.path.append(os.path.dirname(__file__))

from .artifact_correction import (ICA_filtering,CBIeye,ICAremoveArtifact,)
from .atar_algorithm import (ATAR,ATAR_1Ch,ATAR_mCh,Wfilter,SoftThresholding,LinearAttenuanating,Elimination,ATAR_mCh_noParallel)
from .eeg_map import (Gen_SSFI,GridInter,GridInterpolation,showTOPO,TopoMap,ch_names,pos,s1020_get_epos2d_)
from .eeg_processing import (RhythmicDecomposition, Periodogram)
from .eeg_map import (TopoMap_Zi,display_topo_RGB)
from .eeg_map import (s1020_get_epos2d,s1010_get_epos2d,s1005_get_epos2d)
#__all__ = ['ICA', 'SVD','pylfsr', 'ml', 'example','data','load_data','cwt','utils']
