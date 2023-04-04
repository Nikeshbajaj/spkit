"""
GEOMETRY + GEOMAPPING = GEOMAGIC
----------------------------------
Author @ Nikesh Bajaj
updated on Date: 16 March 2023
Version : 0.0.3
Github :  https://github.com/Nikeshbajaj/spkit
Contact: n.bajaj@qmul.ac.uk | n.bajaj@imperial.ac.uk | nikesh.bajaj@qmul.ac.uk

"""

from __future__ import absolute_import, division, print_function

name = "Signal Processing toolkit | GEOMETRY | Experimental Stage"
__version__ = '0.0.3'
__author__ = 'Nikesh Bajaj'

import sys, os

sys.path.append(os.path.dirname(__file__))

# from .mea_processing import (get_stim_loc, find_bad_channels_idx_v0, find_bad_channels_idx, ch_label2idx, ch_idx2label)
# from .mea_processing import (align_cycles, activation_time_loc, activation_repol_time_loc, extract_egm, egm_features)
# from .mea_processing import (plot_mea_grid,mea_feature_map, mat_list_show, mat_1_show)
# from .mea_processing import (arrange_mea_grid, unarrange_mea_grid, channel_mask, feature_mat)
# from .mea_processing import (compute_cv, analyse_mea_file)

#from .geomagic import *
from .basic_geo import *
from .geomagic import *
