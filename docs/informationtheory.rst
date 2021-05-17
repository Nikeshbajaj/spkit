Information Theory for Real-Valued signals
==========================================

Entropy of signal with finit set of values is easy to compute, since frequency for each value can be computed, however, for real-valued signal
it is a little different, because of infinite set of amplitude values. For which spkit comes handy. 

and (other such functions)

Entropy of real-valued signal
-----------

::
  
  import numpy as np
  import matplotlib.pyplot as plt
  import spkit as sp
  
  x = np.random.rand(10000)
  y = np.random.randn(10000)
  
  plt.figure(figsize=(12,5))
  plt.subplot(121)
  sp.HistPlot(x,show=False)

  plt.subplot(122)
  sp.HistPlot(y,show=False)
  plt.show()
  
 
.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/entropy_12.jpg


Shannan entropy
~~~~~~~~~~~~~~~

::
  
  #Shannan entropy
  H_x = sp.entropy(x,alpha=1)
  H_y = sp.entropy(y,alpha=1)
  print('Shannan entropy')
  print('Entropy of x: H(x) = ',H_x)
  print('Entropy of y: H(y) = ',H_y)
  
  
::
  
  Shannan entropy
  Entropy of x: H(x) =  4.4581180171280685
  Entropy of y: H(y) =  5.04102391756942

Rényi entropy
~~~~~~~~~~~~~~~

::
  
  #Rényi entropy
  Hr_x= sp.entropy(x,alpha=2)
  Hr_y= sp.entropy(y,alpha=2)
  print('Rényi entropy')
  print('Entropy of x: H(x) = ',Hr_x)
  print('Entropy of y: H(y) = ',Hr_y)
  
::
  
  Rényi entropy
  Entropy of x: H(x) =  4.456806796146617
  Entropy of y: H(y) =  4.828391418226062


Mutual Information & Joint Entropy
-----------

::
  
  I_xy = sp.mutual_Info(x,y)
  print('Mutual Information I(x,y) = ',I_xy)
  
  H_xy= sp.entropy_joint(x,y)
  print('Joint Entropy H(x,y) = ',H_xy)
  
::

  Joint Entropy H(x,y) =  9.439792556949234
  Mutual Information I(x,y) =  0.05934937774825322

Conditional entropy
-----------

::
  
  H_x1y= sp.entropy_cond(x,y)
  H_y1x= sp.entropy_cond(y,x)
  print('Conditional Entropy of : H(x|y) = ',H_x1y)
  print('Conditional Entropy of : H(y|x) = ',H_y1x)
  
::
  
  Conditional Entropy of : H(x|y) =  4.398768639379814
  Conditional Entropy of : H(y|x) =  4.9816745398211655

Cross entropy & Kullback–Leibler divergence
-----------  

::
  
  H_xy_cross= sp.entropy_cross(x,y)
  D_xy= sp.entropy_kld(x,y)
  print('Cross Entropy of : H(x,y) = :',H_xy_cross)
  print('Kullback–Leibler divergence : Dkl(x,y) = :',D_xy)

::
  
  Cross Entropy of : H(x,y) = : 11.591688735915701
  Kullback–Leibler divergence : Dkl(x,y) = : 4.203058010473213
