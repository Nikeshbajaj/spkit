Signal Filtering
=========





Removing DC component (removing drift) - using IIR
-------------------------------------------------

::
  
  import numpy as np
  import matplotlib.pyplot as plt
  
  import spkit as sp
  
  xf = sp.filterDC(x,alpha=256,return_background=False)
  
  
  
Removing DC component (removing drift) - using Savitzky-Golay filter
-------------------------------------------------

::
  
  import numpy as np
  import matplotlib.pyplot as plt
  
  import spkit as sp
  
  xf = sp.filterDC_sGolay(x,window_length=127, polyorder=3)
  
  
  
Filtering frequency components - using IIR (butterworth) filter
-------------------------------------------

::
  
  import numpy as np
  import matplotlib.pyplot as plt
  
  import spkit as sp
  
  #highpass
  Xf = sp.filter_X(X,band =[0.5],btype='highpass',order=5,fs=128.0,ftype='filtfilt')   
  
  #bandpass
  Xf = sp.filter_X(X,band =[1, 4],btype='bandpass',order=5,fs=128.0,ftype='filtfilt')
  
  #lowpass
  Xf = sp.filter_X(X,band =[40],btype='lowpass',order=5,fs=128.0,ftype='filtfilt')
  


Wavelet Filtering
-----------------


::
  
  import spkit as sp
  
  
  xf = sp.wavelet_filtering(x,wv='db3',threshold='optimal')
  
  #check help(sp.wavelet_filtering)
  
  
Wavelet Filtering - on smaller windows
-----------------


::
  
  import spkit as sp
  
  
  xf = sp.wavelet_filtering_win(x,wv='db3',threshold='optimal',winsize=128)
  
  #check help(sp.wavelet_filtering)
  
  
  
#TODO - figures- details
