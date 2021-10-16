Blind Source Seperation - ICA Based Artifact Removal
================

Notebook
--------

`View in Jupyter-Notebook <https://nbviewer.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/ICA_based_Artifact_Removal.ipynb>`_
~~~~~~~~~~~~~~~~~~~~~~


.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/nav_logo.svg
   :width: 200
   :align: right
   :target: https://nbviewer.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/ICA_based_Artifact_Removal.ipynb
   

::
  
  ICA_filtering(X, winsize=128, ICA_method='extended-infomax', kur_thr=2,
              corr_thr=0.8, AF_ch_index=[0, 13], F_ch_index=[1, 2, 11, 12],
              verbose=True, hopesize=None)


This algorithm includes following three approaches to removal artifact in EEG

1. Kurtosis based artifacts - mostly for motion artifacts
  kur_thr: (default 2) threshold on kurtosis of IC commponents to remove, higher it is, more peaky component is selected
       : +ve int value
2. Correlation Based Index (CBI) for eye movement artifacts
   For applying CBI method, index of prefrontal (AF - First Layer of electrodes towards frontal lobe) and frontal lobe (F - second layer of electrodes) channels
    needs to be provided.
   For case of 14-channels Emotiv Epoc
   ch_names = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
   PreProntal Channels =['AF3','AF4'], Fronatal Channels = ['F7','F3','F4','F8']
   AF_ch_index =[0,13] :  (AF - First Layer of electrodes towards frontal lobe)
   F_ch_index =[1,2,11,12] : (F - second layer of electrodes)
   if AF_ch_index or F_ch_index is None, CBI is not applied

3. Correlation of any independent component with many EEG channels
   If any indepentdent component is correlated fo corr_thr% (80%) of elecctrodes, is considered to be artifactual
   -- Similar like CBI, except, not comparing fronatal and prefrontal but all
   corr_thr: (deafult 0.8) threshold to consider correlation, higher the value less IC are removed and vise-versa
        : float [0-1],
        : if None, this  is not applied
   
   
