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
#TODO Equations
   
.. math::
  
  \psi(t) &= e^{-a(t-t_0)^{2}} \cdot e^{-2\pi jf_0(t-t_0)}\\
  \psi(f) &= \sqrt{\pi/a}\left( e^{-2\pi jft_0}\cdot e^{-\pi^{2}((f-f_0)^{2})/a}\right)
  

**where** :math:`a = (\frac{f_0}{Q})^{2}`

**where**

.. math::
   a = \left(f_0/Q \right)^{2}

.. math::
   
   a &= ( f_0/Q )^{2}

.. math::
   
   a &= (\frac{f_0}{Q})^{2} 
   

.. math::
   
   a &= \left( \frac{f_0}{Q} \right)^{2} 
   
   
:math: a = \left( \\frac{f_0}{Q} \right)^{2}
   
.. math::
   a = \left(f_0/Q \right)^{2}
   
   
:math:`\frac{ \sum_{t=0}^{N}f(t,k) }{N}`

:math:`\\frac{f_0}{Q}`

::
  
  import numpy as np
  import matplotlib.pyplot as plt

  import spkit
  print('spkit-version ', spkit.__version__)
  import spkit as sp
  from spkit.cwt import ScalogramCWT

**Parameters for a Gauss wavelet**:

  - **f0 - center frequency**
  - **Q  - associated with spread of bandwidth, as a = (f0/Q)^2**

Plot wavelet functions
~~~~~~~~~~~~~~~~~~~~~~~~

::
  
  fs = 128                                   #sampling frequency 
  tx = np.linspace(-5,5,fs*10+1)             #time 
  fx = np.linspace(-fs//2,fs//2,2*len(tx))   #frequency range
  
  f01 = 2     #np.linspace(0.1,5,2)[:,None]   
  Q1  = 2.5   #np.linspace(0.1,5,10)[:,None]
  wt1,wf1 = sp.cwt.GaussWave(tx,f=fx,f0=f01,Q=Q1)


  f02 = 2    #np.linspace(0.1,5,2)[:,None]
  Q2  = 0.5  #np.linspace(0.1,5,10)[:,None]
  wt2,wf2 = sp.cwt.GaussWave(tx,f=fx,f0=f02,Q=Q2)

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
 
 
.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/wavelets/gauss_1.png
.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/wavelets/gauss_2.png



With a range of scale parameters
~~~~~~~~~~~~~~~~~~~~~~~~

::
  
  f0 = np.linspace(0.5,10,10)[:,None]
  Q  = np.linspace(1,5,10)[:,None]
  #Q  = 1

  wt,wf = sp.cwt.GaussWave(tx,f=fx,f0=f0,Q=Q)

  plt.figure(figsize=(15,4))
  plt.subplot(121)
  plt.plot(tx,wt.T.real, alpha=0.8)
  plt.plot(tx,wt.T.imag,'--', alpha=0.6)
  plt.xlim(tx[0],tx[-1])
  plt.xlabel('time')
  plt.subplot(122)
  plt.plot(fx,abs(wf.T), alpha=0.6)
  plt.xlim(fx[0],fx[-1])
  plt.xlim(-20,20)
  plt.xlabel('Frequency')
  plt.show()


.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/wavelets/gauss_3_range.png



Signal Analysis - EEG
~~~~~~~~~~~~~~~~~~~~~

::
  
  
  x,fs = sp.load_data.eegSample_1ch()
  t = np.arange(len(x))/fs

  print('shape ',x.shape, t.shape)

  plt.figure(figsize=(15,3))
  plt.plot(t,x)
  plt.xlabel('time')
  plt.ylabel('amplitude')
  plt.xlim(t[0],t[-1])
  plt.grid()
  plt.show()
  
  
.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/wavelets/signal_1.png



Scalogram with default parameters
~~~~~~~~~~~~~~~~~~~~~

## With default setting of f0 and Q
# f0 = np.linspace(0.1,10,100)
# Q = 0.5

::
  
  XW,S = ScalogramCWT(x,t,fs=fs,wType='Gauss',PlotPSD=True)
  
  
.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/wavelets/gauss_psd_1.png


With a range of frequency and Q
~~~~~~~~~~~~~~~~~~~~~

# from 0.1 to 10 Hz of analysis range and 100 points

::
  
  f0 = np.linspace(0.1,10,100)
  Q  = np.linspace(0.1,5,100)
  XW,S = ScalogramCWT(x,t,fs=fs,wType='Gauss',PlotPSD=True,f0=f0,Q=Q)
  
  
.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/wavelets/gauss_psd_2.png

# from 5 to 10 Hz of analysis range and 100 points

::
  
  
  f0 = np.linspace(5,10,100)
  Q  = np.linspace(1,4,100)
  XW,S = ScalogramCWT(x,t,fs=fs,wType='Gauss',PlotPSD=True,f0=f0,Q=Q)
  
.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/wavelets/gauss_psd_3.png


# With constant Q

::
   
  f0 = np.linspace(5,10,100)
  Q  = 2
  XW,S = ScalogramCWT(x,t,fs=fs,wType='Gauss',PlotPSD=True,f0=f0,Q=Q)
  
  
.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/wavelets/gauss_psd_4.png


# From 12 to 24 Hz 

::
  
  f0 = np.linspace(12,24,100)
  Q  = 4
  XW,S = ScalogramCWT(x,t,fs=fs,wType='Gauss',PlotPSD=True,f0=f0,Q=Q)
  
.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/wavelets/gauss_psd_5.png


With a plot of analysis wavelets
~~~~~~~~~~~~~~~~~~~~~
::
  
  f0 = np.linspace(12,24,100)
  Q  = 4
  XW,S = ScalogramCWT(x,t,fs=fs,wType='Gauss',PlotPSD=True,PlotW=True, f0=f0,Q=Q)

.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/wavelets/gauss_psd_6_1.png
.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/wavelets/gauss_psd_6_2.png



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


