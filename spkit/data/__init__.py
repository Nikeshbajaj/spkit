name = "dataset"
__version__ = '0.0.3'
__author__ = 'Nikesh Bajaj'
import sys, os
sys.path.append(os.path.dirname(__file__))



from .dataGen import (mclassGaus, spiral, sinusoidal, moons, gaussian, linear_data, create_dataset)
#from .load_data import (eegSample, eegSample_artifact, eegSample_1ch, primitivePolynomials)
