name = "Signal Processing tool kit"

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.devN' where N is an integer.
#

__version__ = '0.0.2'
__author__ = 'Nikesh Bajaj'
import sys, os

sys.path.append(os.path.dirname(__file__))

#from .DeepNet import DeepNet
from .LogisticRegression import LR
from .Probabilistic import NaiveBayes
from .Trees import ClassificationTree, RegressionTree

__all__ = [ 'LR', 'NaiveBayes',
          'ClassificationTree', 'RegressionTree']
