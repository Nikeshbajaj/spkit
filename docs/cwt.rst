Complex Wavelets
================

Notebook
--------

`View in Jupyter-Notebook <https://nbviewer.jupyter.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/ScalogramCWT_v0.0.9.2.ipynb>`_
~~~~~~~~~~~~~~~~~~~~~~


.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/nav_logo.svg
   :width: 200
   :align: right
   :target: https://nbviewer.jupyter.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/ScalogramCWT_v0.0.9.2.ipynb


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


.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/nav_logo.svg
   :width: 100
   :align: right
   :target: https://nbviewer.jupyter.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/ScalogramCWT_v0.0.9.2.ipynb

-------------

Gauss wavelet
-------------
..
  #TODO Equations clean

The Gauss Wavelet function in time and frequency domain are defined as :math:`\psi(t)` and :math:`\psi(f)` as below;
   
.. math::
  
  \psi(t) &= e^{-a(t-t_0)^{2}} \cdot e^{-2\pi jf_0(t-t_0)}\\
  \psi(f) &= \sqrt{\pi/a}\left( e^{-2\pi jft_0}\cdot e^{-\pi^{2}((f-f_0)^{2})/a}\right)
  
**where**

.. math::
   a = \left( \frac{f_0}{Q} \right)^{2} 


**Parameters for a Gauss wavelet**:

  - **f0 - center frequency**
  - **Q  - associated with spread of bandwidth, as a = (f0/Q)^2**


..
  **where** :math:`a = \left(\frac{f_0}{Q} \right)^{2}`
  
  **where**
  
  .. math::
     a = \left(f_0/Q \right)^{2}

  .. math::
     a = ( f_0/Q )^{2}

  .. math::
     a = ( \frac{f_0}{Q} )^{2} 


  .. math::
     a = \left( \frac{f_0}{Q} \right)^{2} 


  :math: `a = \left( \frac{f_0}{Q} \right)^{2}`

  .. math::
     a = \left(f_0/Q \right)^{2}


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


#TODO Speech/Audio Signal

Speech
~~~~~~~~~~~~~~~~~~~~~
#TODO

Audio
~~~~~~~~~~~~~~~~~~~~~
#TODO


Morlet wavelet
-------------
#TODO

The Morlet Wavelet function in time and frequency domain are defined as :math:`\psi(t)` and :math:`\psi(f)` as below;
   
.. math::
  
  \psi(t) &= C_{\sigma}\pi^{-0.25}  e^{-0.5t^2} \left(e^{j\sigma t}-K_{\sigma} \right)\\
  \psi(w) &= C_{\sigma}\pi^{-0.25} \left(e^{-0.5(\sigma-w)^2} -K_{\sigma}e^{-0.5w^2} \right)
  
**where**

.. math::
   C_{\sigma} &= \left(1+ e^{-\sigma^{2}} - 2e^{-\frac{3}{4}\sigma^{2}}   \right)^{-0.5}\\
   K_{\sigma} &=e^{-0.5\sigma^{2}}\\
   w &= 2\pi f


::
  
  XW,S = ScalogramCWT(x,t,fs=fs,wType='Morlet',PlotPSD=True)


Gabor wavelet
-------------
#TODO

The Gabor Wavelet function (technically same as Gaussian) in time and frequency domain are defined as :math:`\psi(t)` and :math:`\psi(f)` as below;
   
.. math::
   \psi(t) &= e^{-(t-t_0)^2/a^2}e^{-jf_0(t-t_0)}\\
   \psi(f) &= e^{-((f-f_0)a)^2}e^{-jt_0(f-f_0)}
 
**where**
:math:`a` is oscilation rate and :math:`f_0`  is center frequency

::
  
  XW,S = ScalogramCWT(x,t,fs=fs,wType='Gabor',PlotPSD=True)


Poisson wavelet
-------------
Poisson wavelet is defined by positive integers ($n$), unlike other, and associated with Poisson probability distribution 

The Poisson Wavelet function in time and frequency domain are defined as :math:`\psi(t)` and :math:`\psi(f)` as below;

#Type 1 (n)
~~~~~~

.. math::
   \psi(t) &= \left(\frac{t-n}{n!}\right)t^{n-1} e^{-t}\\
   \psi(w) &= \frac{-jw}{(1+jw)^{n+1}}
 
**where**

Admiddibility const :math:`C_{\psi} =\frac{1}{n}` and  :math:`w = 2\pi f`

::
  
  XW,S = ScalogramCWT(x,t,fs=fs,wType='Poisson',method = 1,PlotPSD=True)


#Type 2
~~~~~~

.. math::
   \psi(t) &= \frac{1}{\pi} \frac{1-t^2}{(1+t^2)^2}\\
   \psi(t) &= p(t) + \frac{d}{dt}p(t)\\
   \psi(w) &= |w|e^{-|w|}
 
 
**where**
   
.. math::
   p(t) &=\frac{1}{\pi}\frac{1}{1+t^2}\\
   w &= 2\pi f
   
::
  
  XW,S = ScalogramCWT(x,t,fs=fs,,wType='Poisson',method = 2,PlotPSD=True)
  
  
#Type 3 (n)
~~~~~~
 
.. math::
   \psi(t) &= \frac{1}{2\pi}(1-jt)^{-(n+1)}\\
   \psi(w) &= \frac{1}{\Gamma{n+1}}w^{n}e^{-w}u(w)
 

**where**

.. math::
   \text{unit step function }\quad u(w) &=1 \quad \text{ if  $w>=0$ }\quad \text{else  } 0\\
   w &= 2\pi f

::
  
  XW,S = ScalogramCWT(x,t,fs=fs,wType='Poisson',method = 3,PlotPSD=True)
  
  
#TODO   

Maxican wavelet 
-------------
Complex Mexican hat wavelet is derived from the conventional Mexican hat wavelet. It is a low-oscillation wavelet which is modulated by a complex exponential function with frequency :math:`f_0` `Ref <https://en.wikipedia.org/wiki/Complex_Mexican_hat_wavelet>`_..

The Maxican Wavelet function in time and frequency domain are defined as :math:`\psi(t)` and :math:`\psi(f)` as below;
   
.. math::
   \psi(t) &= \frac{2}{\sqrt{3}} \pi^{-\frac{1}{4}}\left(\sqrt{\pi}(1-t^2)e^{-\frac{1}{2}t^2} - \left(\sqrt{2}jt + \sqrt{\pi}erf\left[\frac{j}{\sqrt{2}}t \right] (1-t^2)e^{-\frac{1}{2}t^2}\right)\right)e^{-2\pi jf_0 t}\\\\
   \psi(w) &= 2\sqrt{\frac{2}{3}}\pi^{-1/4}(w-w_0)^2e^{-\frac{1}{2} (w-w_0)^2}  \quad \text{ if  $w\ge 0$,}\quad \text{ 0  else}
 
 
**where**  :math:`w = 2\pi f` and :math:`w_0 = 2\pi f_0`

::
  
  XW,S = ScalogramCWT(x,t,fs=fs,wType='cMaxican',PlotPSD=True)


#TODO

Shannon wavelet
-------------
Complex Shannon wavelet is the most simplified wavelet function, exploiting Sinc function by modulating with sinusoidal, which results in an ideal bandpass filter. Real Shannon wavelet is modulated by only a cos function `Ref <https://en.wikipedia.org/wiki/Shannon_wavelet>`_.

The Shannon Wavelet function in time and frequency domain are defined as :math:`\psi(t)` and :math:`\psi(f)` as below;
   
.. math::
   \psi(t) &= Sinc(t/2) \cdot e^{-2j\pi f_0t}\\
   \psi(w) &= \prod \left( \frac{w-w_0}{\pi} \right)
 

**where**

where :math:`\prod (x) = 1` if :math:`x \leq 0.5`, 0 else and :math:`w = 2\pi f` and :math:`w_0 = 2\pi f_0`

::
  
  XW,S = ScalogramCWT(x,t,fs=fs,wType='cShannon',PlotPSD=True)
  
  
#TODO
