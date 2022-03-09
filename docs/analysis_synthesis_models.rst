Analysis and Synthesis Models
=============================

DFT Analysis and Synthesis
-------------------------

::
  
  import numpy as np
  import matplotlib.pyplot as plt
  import scipy.linalg as LA
  import spkit as sp
  
  help(sp.dft_analysis)
  
  help(sp.dft_synthesis)
  
  
  
STFT Analysis and Synthesis
---------------------------


::
  
  import numpy as np
  import matplotlib.pyplot as plt
  import spkit as sp
  
  X,names = sp.data.load_data.eegSample()
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
  
  
  
Sinasodal Model for Analysis and Synthesis
---------------------------


::
  
  help(sp.sineModel_analysis)
  
  help(sp.sineModel_synthesis)
  
  
 
