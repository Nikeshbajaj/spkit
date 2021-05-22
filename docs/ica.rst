Independent Component Analysis - ICA
----------

`View in Jupyter-Notebook <https://nbviewer.jupyter.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/ICA_EEG_example.ipynb>`_
~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/nav_logo.svg
   :width: 200
   :align: right
   :target: https://nbviewer.jupyter.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/ICA_EEG_example.ipynb

-----------   

::
  
  import numpy as np
  import matplotlib.pyplot as plt
  from spkit import ICA
  from spkit.data import load_data
  X,ch_names = load_data.eegSample()
  nC =  len(ch_names)

  x = X[128*10:128*12,:]
  t = np.arange(x.shape[0])/128.0

  ica = ICA(n_components=nC,method='fastica')
  ica.fit(x.T)
  s1 = ica.transform(x.T)

  ica = ICA(n_components=nC,method='infomax')
  ica.fit(x.T)
  s2 = ica.transform(x.T)

  ica = ICA(n_components=nC,method='picard')
  ica.fit(x.T)
  s3 = ica.transform(x.T)

  ica = ICA(n_components=nC,method='extended-infomax')
  ica.fit(x.T)
  s4 = ica.transform(x.T)


  methods = ('fastica', 'infomax', 'picard', 'extended-infomax')
  icap = ['ICA'+str(i) for i in range(1,15)]

  plt.figure(figsize=(15,15))
  plt.subplot(321)
  plt.plot(t,x+np.arange(nC)*200)
  plt.xlim([t[0],t[-1]])
  plt.yticks(np.arange(nC)*200,ch_names)
  plt.grid(alpha=0.3)
  plt.title('X : EEG Data')

  S = [s1,s2,s3,s4]
  for i in range(4):
      plt.subplot(3,2,i+2)
      plt.plot(t,S[i].T+np.arange(nC)*700)
      plt.xlim([t[0],t[-1]])
      plt.yticks(np.arange(nC)*700,icap)
      plt.grid(alpha=0.3)
      plt.title(methods[i])

  plt.show()
  
 
.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/eeg_ica_4.png


  .. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/nav_logo.svg
   :width: 100
   :align: right
   :target: https://nbviewer.jupyter.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/ICA_EEG_example.ipynb

-----------   
 
  
  
