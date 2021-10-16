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
