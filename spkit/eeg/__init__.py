from __future__ import absolute_import, division, print_function

name = "Signal Processing toolkit | EEG Processing"
__version__ = '0.0.9.5'
__author__ = 'Nikesh Bajaj'

import sys, os

sys.path.append(os.path.dirname(__file__))

from .artifact_correction import (ICA_filtering, CBIeye)
from .atar_algorithm import (ATAR,ATAR_1Ch,ATAR_mCh)
from .eeg_map import (GridInter,GridInterpolation,showTOPO)
from .eeg_map import (TopoMap_Zi,display_topo_RGB)
from .eeg_map import (s1020_get_epos2d,s1010_get_epos2d,s1005_get_epos2d,presets)

from .eeg_processing import (rhythmic_powers,rhythmic_powers_win)
from .eeg_map import (topomap,presets,gen_ssfi)

##########TO BE REMOVED###############

from .atar_algorithm import (ATAR_mCh_noParallel)
from .eeg_map import (ch_names,pos,s1020_get_epos2d_,TopoMap,Gen_SSFI)
from .eeg_processing import (RhythmicDecomposition, Periodogram)

#__all__ = ['ICA', 'SVD','pylfsr', 'ml', 'example','data','load_data','cwt','utils']
