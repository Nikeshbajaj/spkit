Complex Wavelets
================

Notebook
--------

`View in Jupyter-Notebook <https://nbviewer.jupyter.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/ScalogramCWT_v0.0.9.2.ipynb>`_
~~~~~~~~~~~~~~~~~~~~~~


A quick example to compare different wavelets
----------------------------

::
  
  import numpy as np
  import matplotlib.pyplot as plt

  import spkit
  print('spkit-version ', spkit.__version__)
  import spkit as sp
  from spkit.cwt import ScalogramCWT
  from spkit.cwt import compare_cwt_example
  
  x,fs = sp.load_data.eegSample_1ch()
  t = np.arange(len(x))/fs
  
  compare_cwt_example(x,t,fs=fs)
  

.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/cwt_2.png



Gauss wavelet
-------------


Morlet wavelet
-------------


Gabor wavelet
-------------



Complex Maxican wavelet
-------------


Complex Shannon wavelet
-------------



