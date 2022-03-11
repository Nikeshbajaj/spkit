Analysis and Synthesis Models
=============================


DFT: Analysis and Synthesis
===========================

::
  
  import numpy as np
  import matplotlib.pyplot as plt
  import spkit as sp
  
  
  X, ch_names = sp.data.load_data.eegSample()
  fs=128
  x = X[:,1]
  t = np.arange(len(x))/fs
  print(x.shape)
  
Analysis and Synthesis
~~~~~~~~~~~~~~~~~~~~~~


::
  
  # Analysis
  mX, pX, N = sp.dft_analysis(x, window='boxcar')
  
  # Synthesis
  y = sp.dft_synthesis(mX, pX, M=N, window='boxcar')
  print(y.shape)
  


Plot figures
~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~

::
  
  mX, pX, N = sp.dft_analysis(x, window='hamm',plot=2, fs=fs)
  


.. image:: https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/dft_analysis_synthesis_ham_3.png
  
  

No windowing
~~~~~~~~~~~~~~~~~~~~~~

::
  
  mX, pX, N = sp.dft_analysis(x, window='boxcar',plot=2, fs=fs)


.. image:: https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/dft_analysis_synthesis_2.png
  
::
  
  #check for more details
  help(sp.dft_analysis)
  help(sp.dft_synthesis)
  

`View in Jupyter-Notebook <https://github.com/Nikeshbajaj/Notebooks/blob/master/spkit/SP/Analysis_Sythesis_Models.ipynb>`_
~~~~~~~~~~~~~~~~~~~~~~ 
  
STFT: Analysis and Synthesis
===========================


::
  
  import numpy as np
  import matplotlib.pyplot as plt
  import spkit as sp
  
  X,ch_names = sp.data.load_data.eegSample()
  fs=128
  x = X[:,1]
  t = np.arange(len(x))/fs


Analysis and Synthesis
~~~~~~~~~~~~~~~~~~~~~~
::
  
  # STFT Analysis
  mXt,pXt = sp.stft_analysis(x, winlen=128, overlap=32,window='blackmanharris',nfft=None)
  print(mXt.shape, pXt.shape)
  
  # STFT Synthesis - reconstruct back from STFT
  y = sp.stft_synthesis(mXt, pXt, winlen=128, overlap=32)
  
  print(y.shape)
  # Reconstructed signal might have a longer length, if original signal size was not multiple of overlap size
  # extra samples can be simply discarded
  
  
Plot figures:
~~~~~~~~~~~~~~~~~~~~~~

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
  

`View in Jupyter-Notebook <https://github.com/Nikeshbajaj/Notebooks/blob/master/spkit/SP/Analysis_Sythesis_Models.ipynb>`_
~~~~~~~~~~~~~~~~~~~~~~ 

FRFT: Fractional Fourier Transform
===========================

::
  
  X,names = sp.data.load_data.eegSample()
  fs=128
  x = X[:,1]
  t = np.arange(len(x))/fs
  print(x.shape)


Analysis and Synthesis
~~~~~~~~~~~~~~~~~~~~~~

::
  
  # Analysis
  Xa = sp.frft(x.copy(),alpha=0.2)
  print(Xa.shape)

  # Synthesis
  y = sp.frft(Xa.copy(),alpha=-0.2)
  print(y.shape)



Plot figures
~~~~~~~~~~~~~~~~~~~~~~
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
  

`View in Jupyter-Notebook-1 <https://github.com/Nikeshbajaj/Notebooks/blob/master/spkit/SP/Analysis_Sythesis_Models.ipynb>`_
~~~~~~~~~~~~~~~~~~~~~~ 
`View in Jupyter-Notebook-2 <https://nbviewer.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/FRFT_demo_sine.ipynb>`_
~~~~~~~~~~~~~~~~~~~~~~ 
  
  
Sinasodal Model: Analysis and Synthesis
=============================


::
  
  import requests
  from scipy.io import wavfile
  import IPython
  
  path2 = 'https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/spkit/data/singing-female.wav'
  print(path2)
  
  
  req = requests.get(path2)
  with open('myfile.wav', 'wb') as f:
          f.write(req.content)

  fs, x = wavfile.read('myfile.wav')
  t = np.arange(len(x))/fs

  x=x.astype(float)

  print(x.shape, fs)
  

Analysis and Synthesis
~~~~~~~~~~~~~~~~~~~~~~ 

::
    
  # Analysis
  N=20
  fXst, mXst, pXst = sp.sineModel_analysis(x,fs,winlen=3001,overlap=750,
                            window='blackmanharris', nfft=None, thr=-10, 
                            maxn_sines=N,minDur=0.01, freq_devOffset=10,freq_devSlope=0.1)

  print(fXst.shape, mXst.shape, pXst.shape)
  
  # Synthesis
  
  Xr = sp.sineModel_synthesis(fXst, mXst, pXst,fs,overlap=750,crop_end=False)
  print(Xr.shape)
  
  # Residual
  
  Xd = x - Xr[:len(x)]
  
  
Plots
~~~~~~~~~~~~~~~~~~~~~~ 

::
  
  plt.figure(figsize=(13,15))
  plt.subplot(511)
  plt.plot(t,x)
  plt.xlim([t[0],t[-1]])
  plt.grid()
  plt.title('Original Auido: x(t)')
  #plt.xlabel('time (s)')
  plt.ylabel('amplitude (μV)')



  mXt,pXt = sp.stft_analysis(x, winlen=441, overlap=220,window='blackmanharris',nfft=None)

  plt.subplot(512)
  plt.imshow(mXt.T,aspect='auto',origin='lower',cmap='jet',extent=[t[0],t[-1],0,fs/2])
  plt.title('Spectrogram of x(t)')
  #plt.xlabel('time (s)')
  plt.ylabel('frequency (Hz)')



  fXt1 = (fXst.copy())*(mXst>0)
  fXt1[fXt1==0]=np.nan


  plt.subplot(513)
  tx = t[-1]*np.arange(fXt1.shape[0])/fXt1.shape[0]

  plt.plot(tx,fXt1,'-k',alpha=0.5)
  #plt.ylim([0,fs/2])
  plt.xlim([0,tx[-1]])

  plt.title(f'Sinasodals Tracks: n={N}')
  plt.xlabel('time (s)')
  plt.ylabel('frequency (Hz)')
  plt.grid(alpha=0.3)
  
  plt.subplot(514)
  plt.plot(t,Xr[:len(t)])
  plt.xlim([t[0],t[-1]])
  plt.grid()
  plt.title(f'Reconstructed Audio from {N} Sinasodals: $x_r(t)$')
  #plt.xlabel('time (s)')
  plt.ylabel('amplitude')


  mXrt,pXrt = sp.stft_analysis(Xr, winlen=441, overlap=220,window='blackmanharris',nfft=None)

  plt.subplot(515)
  plt.imshow(mXrt.T,aspect='auto',origin='lower',cmap='jet',extent=[t[0],t[-1],0,fs/2])
  plt.title(r'Spectrogram of $x_r(t)$')
  #plt.xlabel('time (s)')
  plt.ylabel('frequency (Hz)')
  plt.tight_layout()
  plt.show()

  print('Original Audio: $x(t)$')
  display(IPython.display.Audio(x,rate=fs))

  print(f'Reconstructed Audio: $x_r(t)$')
  display(IPython.display.Audio(Xr,rate=fs))
  
  
.. image:: https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/sinasodal_model_analysis_synthesis_1.png
  

Residual
~~~~~~~~~~~~~~~~~~~~~~ 

::
  
  mXdt,pXdt = sp.stft_analysis(Xd, winlen=441, overlap=220,window='blackmanharris',nfft=None)
  
  plt.figure(figsize=(13,6))
  plt.subplot(211)
  plt.plot(t,Xd)
  plt.xlim([t[0],t[-1]])
  plt.grid()
  plt.title(r'Residual: Discarded part of Audio: $x_d(t) = x(t)-x_r(t)$')
  #plt.xlabel('time (s)')
  plt.ylabel('amplitude (μV)')

  plt.subplot(212)
  plt.imshow(mXdt.T,aspect='auto',origin='lower',cmap='jet',extent=[t[0],t[-1],0,fs/2])
  plt.title(r'Spectrogram of $x_d(t)$')
  #plt.xlabel('time (s)')
  plt.ylabel('frequency (Hz)')

  plt.tight_layout()
  plt.show()
  IPython.display.Audio(Xd,rate=fs)
  
   
.. image:: https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/sinasodal_model_analysis_synthesis_residual_1.png
  
  
Audio output
~~~~~~~~~~~~~~~~~~~~~~ 

Original Audio
---------------------------

.. raw:: html

    <audio controls="controls">
      <source src="https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/spkit/data/singing-female.wav" type="audio/wav"> 
    </audio>
    
https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/spkit/data/singing-female.wav

`Original Audio <https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/spkit/data/singing-female.wav>`_


Reconstructed Audio
---------------------------

.. raw:: html

    <audio controls="controls">
      <source src="https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/spkit/data/singing_female_recons.wav" type="audio/wav">
    </audio>
    

https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/spkit/data/singing_female_recons.wav

`Reconstructed Audio <https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/spkit/data/singing_female_recons.wav>`_
  

Residual Audio - hissing sound
---------------------------
.. raw:: html

    <audio controls="controls">
      <source src="https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/spkit/data/singing_female_residual.wav" type="audio/wav">
    </audio>
    
`Residual Audio <https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/spkit/data/singing_female_residual.wav>`_


::
  
  # check for help
  help(sp.sineModel_analysis)
  help(sp.sineModel_synthesis)
  
  
`View in Jupyter-Notebook-1 <https://github.com/Nikeshbajaj/Notebooks/blob/master/spkit/SP/Analysis_Sythesis_Models.ipynb>`_
~~~~~~~~~~~~~~~~~~~~~~ 
`View in Jupyter-Notebook-2 <https://github.com/Nikeshbajaj/Notebooks/blob/master/spkit/SP/Sinasodal_Model_AnalysisSynthesis.ipynb>`_
~~~~~~~~~~~~~~~~~~~~~~   
