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
#TODO

Plot wavelet functions
~~~~~~~~~~~~~~~~~~~~~~~~

::

  import numpy as np
  import matplotlib.pyplot as plt

  import spkit
  print('spkit-version ', spkit.__version__)
  import spkit as sp
  from spkit.cwt import ScalogramCWT

  fs = 128
  tx = np.linspace(-5,5,fs*10+1)
  fx = np.linspace(-fs//2,fs//2,2*len(tx))
  
  # with Q = 2.5, small bandwidth
  f01 = 2#np.linspace(0.1,5,2)[:,None]
  Q1  = 2.5#np.linspace(0.1,5,10)[:,None]
  wt1,wf1 = spkit.cwt.GaussWave(tx,f=fx,f0=f01,Q=Q1)
  
  plt.figure(figsize=(15,4))
  plt.subplot(121)
  plt.plot(tx,wt1.T.real,label='real')
  plt.plot(tx,wt1.T.imag,'--',label='image')
  plt.xlim(tx[0],tx[-1])
  plt.xlabel('time')
  plt.ylabel('Q=2.5')
  plt.legend()
  plt.subplot(122)
  plt.plot(fx,abs(wf1.T), alpha=0.9)
  plt.xlim(fx[0],fx[-1])
  plt.xlim(-5,5)
  plt.xlabel('Frequency')
  plt.show()

  # with Q = 0.5, large bandwidth
  f02 = 2#np.linspace(0.1,5,2)[:,None]
  Q2  = 0.5#np.linspace(0.1,5,10)[:,None]
  wt2,wf2 = spkit.cwt.GaussWave(tx,f=fx,f0=f02,Q=Q2)
  
  plt.figure(figsize=(15,4))
  plt.subplot(121)
  plt.plot(tx,wt2.T.real,label='real')
  plt.plot(tx,wt2.T.imag,'--',label='image')
  plt.xlim(tx[0],tx[-1])
  plt.xlabel('time')
  plt.ylabel('Q=0.5')
  plt.legend()
  plt.subplot(122)
  plt.plot(fx,abs(wf2.T), alpha=0.9)
  plt.xlim(fx[0],fx[-1])
  plt.xlim(-5,5)
  plt.xlabel('Frequency')
  plt.show()
 

Signal Analysis
~~~~~~~~~~~~~~~~~~~~~~~~





Morlet wavelet
-------------
#TODO

Gabor wavelet
-------------
#TODO

Poisson wavelet
-------------
#TODO

Maxican wavelet 
-------------
#TODO

Shannon wavelet
-------------
#TODO


