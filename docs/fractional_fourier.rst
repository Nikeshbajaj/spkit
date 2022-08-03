Fractional Fourier Transform
============================

#TODO

`View in Jupyter-Notebook <https://nbviewer.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/FRFT_demo_sine.ipynb>`_
~~~~~~~~~~~~~~~~~~~~~~


.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/nav_logo.svg
   :width: 200
   :align: right
   :target: https://nbviewer.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/FRFT_demo_sine.ipynb
   
   

.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/frft_sin_3.gif
   :width: 800



Fractional Fourier Transform
---------

::
  
  import numpy as np
  import matplotlib.pyplot as plt
  import spkit as sp
  
  f0 = 10
  t = np.arange(0,5,0.01)
  x = np.sin(2*np.pi*t*f0)*np.exp(-0.1*t) + np.sin(2*np.pi*t**2*f0)

  print(x.shape)
  plt.figure(figsize=(15,3))
  plt.plot(t,x)
  
  #Fractional Fourier Transform
  y = sp.frft(x,alpha=0.5)
  
  plt.figure(figsize=(15,3))
  plt.plot(t,x)
  plt.plot(t,y.real)
  plt.plot(t,y.imag)
  plt.show()
  
  
  #Inverse fractional Fourier Transform
  xi = sp.frft(y,alpha=-0.5)
  
  plt.figure(figsize=(15,3))
  plt.plot(t,x)
  plt.plot(t,xi.real)
  plt.plot(t,xi.imag)
  plt.show()
  
  help(sp.frft)
  
 
Fast Fractional Fourier Transform
---------
  

::
  
  import numpy as np
  import matplotlib.pyplot as plt
  import spkit as sp
  
  f0 = 10
  t = np.arange(0,5,0.01)
  x = np.sin(2*np.pi*t*f0)*np.exp(-0.1*t) + np.sin(2*np.pi*t**2*f0)

  print(x.shape)
  plt.figure(figsize=(15,3))
  plt.plot(t,x)
  
  #Fractional Fourier Transform
  y = sp.ffrft(x,alpha=0.5)
  
  plt.figure(figsize=(15,3))
  plt.plot(t,x)
  plt.plot(t,y.real)
  plt.plot(t,y.imag)
  plt.show()
  
  
  #Inverse fractional Fourier Transform
  xi = sp.ffrft(y,alpha=-0.5)
  
  plt.figure(figsize=(15,3))
  plt.plot(t,x)
  plt.plot(t,xi.real)
  plt.plot(t,xi.imag)
  plt.show()
  
  help(sp.ffrft)
