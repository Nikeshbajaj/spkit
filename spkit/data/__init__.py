name = "dataset"
__version__ = '0.0.3'
__author__ = 'Nikesh Bajaj'
import sys, os
sys.path.append(os.path.dirname(__file__))



from .dataGen import (mclassGaus, spiral, mclass_gauss, sinusoidal, moons, gaussian, linear, create_dataset)
from .load_data import (eeg_sample_14ch, eeg_sample_artifact, eeg_sample_1ch,ecg_sample_12leads, optical_sample)
from .load_data import (ppg_sample, gsr_sample, egm_sample, ecg_sample, eda_sample,primitive_polynomials)

# To Be Removed
from .load_data import (eegSample, eegSample_artifact, eegSample_1ch, primitivePolynomials)
from .dataGen import (linear_data)
