ATAR - Artifact Removal Algorithm for EEG
================

Notebook
--------

`View in Jupyter-Notebook <https://nbviewer.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/ATAR_Algorithm_EEG_Artifact_Removal.ipynb>`_
~~~~~~~~~~~~~~~~~~~~~~


.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/nav_logo.svg
   :width: 200
   :align: right
   :target: https://nbviewer.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/ATAR_Algorithm_EEG_Artifact_Removal.ipynb
   
   
**ATAR Algorithm -  Automatic and Tunable Artifact Removal Algorithm for EEG Signal.** 

The algorithm is based on wavelet packet decomposion (WPD), the full description of algorithm can be found here [Automatic and Tunable Artifact Removal Algorithm for EEG](https://doi.org/10.1016/j.bspc.2019.101624) from the article. 
The algorithm is applied on the given multichannel signal X (n,nch), window wise and reconstructed with overall add method. The defualt window size is set to 1 sec (128 samples). For each window, the threshold $\theta_\alpha$ is computed and applied to filter the wavelet coefficients.
There is manily one parameter that can be tuned $\beta$ with different operating modes and other settings.
Here is the list of parameters and there simplified meaning given:
Parameters:
* $\beta$: This is a main parameter to tune, highher the value, more aggressive the algorithm to remove the artifacts. By default it is set to 0.1. $\beta$ is postive float value.
* ***OptMode***: This sets the mode of operation, which decides hoe to remove the artifact. By default it is set to 'soft', which means Soft Thresholding, in this mode, rather than removing the pressumed artifact, it is suppressed to the threshold, softly. OptMode='linAtten', suppresses the pressumed artifact depending on how far it is from threshold. Finally, the most common mode - Elimination (OptMode='elim'), which remove the pressumed artifact.
    * Soft Thresholding and Linear Attenuation require addition parameters to set the associated thresholds which are by default set to bf=2, gf=0.8. 
* ***wv=db3***: Wavelet funtion, by default set to db3, could be any of ['db3'.....'db38', 'sym2.....sym20', 'coif1.....coif17', 'bior1.1....bior6.8', 'rbio1.1...rbio6.8', 'dmey']
* $k_1$, $k_2$: Lower and upper bounds on threshold $\theta_\alpha$.
* ***IPR=[25,75]***: interpercentile range, range used to compute threshold

**APIs**

There are three functions in **spkit.eeg** for **ATAR algorithm**

* spkit.eeg.ATAR_1Ch(...)
* spkit.eeg.ATAR_mCh(...)
* spkit.eeg.ATAR_mCh_noParallel(...)

***sp.eeg.ATAR_1Ch*** is for single channel input signal x of shape (n,), where as, ***sp.eeg.ATAR_mCh*** is for multichannel signal X with shape (n,ch), which uses joblib for parallel processing of multi channels. For some OS, joblib raise an error of ***BrokenProcessPool***, in that case use  ***sp.eeg.ATAR_mCh_noParallel***, which is same as ***sp.eeg.ATAR_mCh***, except parallel processing. Alternatively, use ***sp.eeg.ATAR_1Ch*** with for loop for each channel.


.. image:: https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/atar_beta_tune.gif


A quick example
---------------

::
  
   import numpy as np
   import matplotlib.pyplot as plt

   import spkit as sp
   from spkit.data import load_data

   print(sp.__version__)

   X,ch_names = load_data.eegSample()
   fs = 128

   # high=pass filtering
   Xf = sp.filter_X(X,band=[0.5], btype='highpass',fs=fs,verbose=0).T
   Xf.shape

   # ATAR Algorithm
   XR = sp.eeg.ATAR_mCh_noParallel(Xf.copy(),verbose=0)

   #plots
   t = np.arange(Xf.shape[0])/fs
   plt.figure(figsize=(15,8))
   plt.subplot(221)
   plt.plot(t,Xf+np.arange(-7,7)*200)
   plt.xlim([t[0],t[-1]])
   #plt.xlabel('time (sec)')
   plt.yticks(np.arange(-7,7)*200,ch_names)
   plt.grid()
   plt.title('Xf: 14 channel - EEG Signal (filtered)')
   plt.subplot(223)
   plt.plot(t,XR+np.arange(-7,7)*200)
   plt.xlim([t[0],t[-1]])
   plt.xlabel('time (sec)')
   plt.yticks(np.arange(-7,7)*200,ch_names)
   plt.grid()
   plt.title('XR: Corrected Signal')
   plt.subplot(224)
   plt.plot(t,(Xf-XR)+np.arange(-7,7)*200)
   plt.xlim([t[0],t[-1]])
   plt.xlabel('time (sec)')
   plt.yticks(np.arange(-7,7)*200,ch_names)
   plt.grid()
   plt.title('Xf - XR: Difference (removed signal)')
   plt.subplots_adjust(wspace=0.1,hspace=0.3)
   plt.show()
  

.. image:: https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/atar_exp1.png

