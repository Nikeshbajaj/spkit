Dispersion Entropy
==================
#TODO

.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/nav_logo.svg
   :width: 200
   :align: right
   :target: https://nbviewer.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/Dispersion_Entropy_1_demo_EEG.ipynb
-----------------------------------------------------------------------------------------------------------------



Backgorund
----------
#TODO


Dispersion entropy
--------
::
  
  import numpy as np
  import matplotlib.pyplot as plt
  import spkit as sp
  
  X,ch_names = sp.load_data.eegSample()
  fs=128
  
  #filtering 
  Xf = sp.filter_X(X,band=[1,20],btype='bandpass',verbose=0)
  
  Xi = Xf[:,0].copy() # only one channel
  
  de,prob,patterns_dict,_,_= sp.dispersion_entropy(Xi,classes=10, scale=1, emb_dim=2, delay=1,return_all=True)
  print(de)

2.271749287746759

Probability of all the patterns found

::
  
  plt.stem(prob)
  plt.xlabel('pattern #')
  plt.ylabel('probability')
  plt.show()

.. image:: https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/DE_pat_1.png
  

Pattern dictionary

::
  
  patterns_dict
  
{(1, 1): 18,
 (1, 2): 2,
 (1, 4): 1,
 (2, 1): 2,
 (2, 2): 23,
 (2, 3): 2,
 (2, 5): 1,
 (3, 1): 1,
 (3, 2): 2,  
  

top 10 patters

::
  
  PP = np.array([list(k)+[patterns_dict[k]] for k in patterns_dict])
  idx = np.argsort(PP[:,-1])[::-1]
  PP[idx[:10],:-1]

array([[ 5,  5],
       [ 6,  6],
       [ 4,  4],
       [ 7,  7],
       [ 6,  5],
       [ 5,  6],
       [10, 10],
       [ 4,  5],
       [ 5,  4],
       [ 8,  8]], dtype=int64)
       

Dispersion Entropy with sliding window
--------     

::
  
  de_temporal = []
  win = np.arange(128)
  while win[-1]<Xi.shape[0]:
      de,_ = sp.dispersion_entropy(Xi[win],classes=10, scale=1, emb_dim=2, delay=1,return_all=False)
      win+=16
      de_temporal.append(de)x
      
   
  plt.figure(figsize=(10,3))
  plt.plot(de_temporal)
  plt.xlim([0,len(de_temporal)])
  plt.xlabel('window')
  plt.ylabel('Dispersion Entropy')
  plt.show()
  
  
.. image:: https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/DE_temp_1.png
  


Dispersion Entropy multiscale
--------

::
  
  
  for scl in [1,2,3,5,10,20,30]:
    de,_ = sp.dispersion_entropy(Xi,classes=10, scale=scl, emb_dim=2, delay=1,return_all=False)
    print(f'Sacle: {scl}, \t: DE: {de}')
    
 
Sacle: 1, 	: DE: 2.271749287746759
Sacle: 2, 	: DE: 2.5456280627759336
Sacle: 3, 	: DE: 2.6984938704051236
Sacle: 5, 	: DE: 2.682837351130069
Sacle: 10, 	: DE: 2.5585556625642476
Sacle: 20, 	: DE: 2.7480275694000103
Sacle: 30, 	: DE: 2.4767472897625806


  help(sp.dispersion_entropy)
  
  
Mltiscale-refined Dispersion Entropy
--------

::
  
  de,_ = sp.dispersion_entropy_multiscale_refined(Xi,classes=10, scales=[1, 2, 3, 4, 5], emb_dim=2, delay=1)
  print(de)
 
2.543855087400606


::
  
  help(sp.dispersion_entropy_multiscale_refined)


`View in Jupyter-Notebook for details <https://nbviewer.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/Dispersion_Entropy_1_demo_EEG.ipynb>`_
----------------


.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/nav_logo.svg
   :width: 100
   :align: right
   :target: https://nbviewer.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/Dispersion_Entropy_1_demo_EEG.ipynb

-----------   
