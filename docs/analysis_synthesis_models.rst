Analysis and Synthesis Models
=============================

DFT Analysis and Synthesis
-------------------------


::
  
  import numpy as np
  import matplotlib.pyplot as plt
  import spkit as sp
  
  
  X, ch_names = sp.data.load_data.eegSample()
  fs=128
  x = X[:,1]
  t = np.arange(len(x))/fs
  print(x.shape)
  
  
  mX, pX, N = sp.dft_analysis(x, window='boxcar')
  y = sp.dft_synthesis(mX, pX, M=N, window='boxcar')
  print(y.shape)
  

Plot figures

::
  
  plt.figure(figsize=(13,8))
  plt.subplot(311)
  plt.plot(t,x)
  plt.xlim([t[0],t[-1]])
  plt.grid()
  plt.xlabel('time (s)')
  plt.title('Original signal')
  plt.ylabel('amplitude (μV)')
  plt.subplot(323)
  fr = (fs/2)*np.arange(len(mX))/(len(mX)-1)
  plt.plot(fr,mX)
  plt.xlim([fr[0],fr[-1]])
  plt.grid()
  plt.ylabel('|X| (dB)')
  plt.title('Magnitude spectrum')
  plt.subplot(324)
  plt.plot(fr,pX)
  plt.xlim([fr[0],fr[-1]])
  plt.grid()
  plt.ylabel('<|X|')
  plt.title('Phase spectrum')

  plt.subplot(313)
  plt.plot(t,y)
  plt.xlim([t[0],t[-1]])
  plt.grid()
  plt.title('Reconstructed signal')
  plt.xlabel('time (s)')
  plt.ylabel('amplitude (μV)')
  plt.tight_layout()
  plt.show()
  

.. image:: https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/dft_analysis_synthesis_1.png
  
  
Effect of windowing

::
  
  mX, pX, N = sp.dft_analysis(x, window='hamm',plot=2, fs=fs)
  


.. image:: https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/dft_analysis_synthesis_ham_3.png
  
  

No windowing

::
  
  mX, pX, N = sp.dft_analysis(x, window='boxcar',plot=2, fs=fs)


.. image:: https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/dft_analysis_synthesis_2.png
  
::
  
  #check for more details
  help(sp.dft_analysis)
  help(sp.dft_synthesis)
  
  
  
STFT Analysis and Synthesis
---------------------------


::
  
  import numpy as np
  import matplotlib.pyplot as plt
  import spkit as sp
  
  X,ch_names = sp.data.load_data.eegSample()
  fs=128
  x = X[:,1]
  t = np.arange(len(x))/fs
  
  # STFT Analysis
  mXt,pXt = sp.stft_analysis(x, winlen=128, overlap=32,window='blackmanharris',nfft=None)
  print(mXt.shape, pXt.shape)
  
  # STFT Synthesis - reconstruct back from STFT
  y = sp.stft_synthesis(mXt, pXt, winlen=128, overlap=32)
  
  print(y.shape)
  # Reconstructed signal might have a longer length, if original signal size was not multiple of overlap size
  # extra samples can be simply discarded
  
  
Plot figures:  
::
  
  plt.figure(figsize=(13,8))
  plt.subplot(311)
  plt.plot(t,x)
  plt.xlim([t[0],t[-1]])
  plt.grid()
  plt.title('Original signal')
  plt.ylabel('amplitude (μV)')

  plt.subplot(312)
  plt.imshow(mXt.T,aspect='auto',origin='lower',cmap='jet',extent=[t[0],t[-1],0,fs/2])
  plt.title('STFT: Spectrogram')
  plt.ylabel('frequency (Hz)')

  plt.subplot(313)
  plt.plot(t,y[:len(t)])
  plt.xlim([t[0],t[-1]])
  plt.grid()
  plt.title('Reconstructed signal')
  plt.xlabel('time (s)')
  plt.ylabel('amplitude (μV)')
  plt.tight_layout()
  plt.show()
  
  
  
.. image:: https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/stft_analysis_synthesis_1.png
  
  
  
::  
  
  #check for details
  help(sp.stft_analysis)
  help(sp.stft_synthesis)
  


Fractional Fourier Transform:FRFT :: Analysis and Synthesis
---------------------------

::
  
  X,names = sp.data.load_data.eegSample()
  fs=128
  x = X[:,1]
  t = np.arange(len(x))/fs
  print(x.shape)

  # Analysis
  Xa = sp.frft(x.copy(),alpha=0.2)
  print(Xa.shape)

  # Synthesis
  y = sp.frft(Xa.copy(),alpha=-0.2)
  print(y.shape)



plots

::
  
  plt.figure(figsize=(13,6))
  plt.subplot(311)
  plt.plot(t,x)
  plt.xlim([t[0],t[-1]])
  plt.grid()
  plt.title('x(t)')
  #plt.xlabel('time (s)')
  plt.ylabel('amplitude (μV)')

  plt.subplot(312)
  plt.plot(t,Xa.real,label='real')
  plt.plot(t,Xa.imag,label='imag')
  plt.xlim([t[0],t[-1]])
  plt.grid()
  plt.title(r'FRFT(x(t)), $\alpha=0.2$')
  #plt.xlabel('time (s)')
  plt.ylabel('amplitude (μV)')
  plt.legend()


  plt.subplot(313)
  plt.plot(t,y.real)
  plt.xlim([t[0],t[-1]])
  plt.grid()
  plt.title('Reconstructed signal: x(t)')
  #plt.xlabel('time (s)')
  plt.ylabel('amplitude (μV)')
  plt.tight_layout()
  plt.show()

  
.. image:: https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/frft_analysis_synthesis_1.png
  
  
  
  
Sinasodal Model for Analysis and Synthesis
---------------------------


::
  
  import requests
  from scipy.io import wavfile
  import IPython
  
  
  
  
  
  help(sp.sineModel_analysis)
  
  help(sp.sineModel_synthesis)
  
  
  
.. image:: https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/sinasodal_model_analysis_synthesis_1.png
  
  
   
  
 
