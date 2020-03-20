from __future__ import absolute_import, division, print_function

name = "Signal Processing toolkit | ML"

__version__ = '0.0.9'
__author__ = 'Nikesh Bajaj'
import sys, os
sys.path.append(os.path.dirname(__file__))

#from .DeepNet import DeepNet
from .LogisticRegression import LR, LogisticRegression
from .Probabilistic import NaiveBayes
from .Trees import ClassificationTree, RegressionTree

__all__ = [ 'LR','LogisticRegression','NaiveBayes',
          'ClassificationTree', 'RegressionTree']
