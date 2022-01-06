Ramanjum Filter Banks  
=====================

Period estimation using RFB - (spkit version 0.0.9.4) 
----------------------------------------------------

Example: 1
----------

::
  
  import numpy as np
  import matplotlib.pyplot as plt
  import scipy.linalg as LA
  import spkit as sp
  
  
  seed = 10
  np.random.seed(seed)
  period = 10
  SNR = 0

  x1 = np.zeros(30)
  x2 = np.random.randn(period)
  x2 = np.tile(x2,10)
  x3 = np.zeros(30)
  x  = np.r_[x1,x2,x3]
  x /= LA.norm(x,2)

  noise  = np.random.randn(len(x))
  noise /= LA.norm(noise,2)

  noise_power = 10**(-1*SNR/20)

  noise *= noise_power
  x_noise = x + noise

  plt.figure(figsize=(15,3))
  plt.plot(x,label='signal: x')
  plt.plot(x_noise, label='signal+noise: x_noise')
  plt.xlabel('sample (n)')
  plt.legend()
  plt.show()


  Pmax = 40  #Largest expected period in the input
  Rcq  = 10   # Number of repeats in each Ramanujan filter
  Rav  = 2    #Number of repeats in each averaging filter
  Th   = 0.2   #Outputs of the RFB are thresholded to zero for all values less than Th*max(output)

  y = sp.RFB(x_noise,Pmax, Rcq, Rav, Th)

  plt.figure(figsize=(15,5))
  im = plt.imshow(y.T,aspect='auto',cmap='jet',extent=[1,len(x_noise),Pmax,1])
  plt.colorbar(im)
  plt.xlabel('sample (n)')
  plt.ylabel('period (in samples)')
  plt.show()

  plt.stem(np.arange(1,y.shape[1]+1),np.sum(y,0))
  plt.xlabel('period (in samples)')
  plt.ylabel('strength')
  plt.show()

  print('top 10 periods: ',np.argsort(np.sum(y,0))[::-1][:10]+1)
  
  
.. image:: https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/RFB_ex1.1.png
   :width: 400
.. image:: https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/RFB_ex1.2.png
   :width: 400
   
.. image:: https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/RFB_ex1.3.png
   :width: 400

 
 top 10 periods:  [10  5 11 18 17 16 15 14 13 12]
  
