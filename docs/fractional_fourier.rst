Fractional Fourier Transform
============================



::
  
  import numpy as np
  import matplotlib.pyplot as plt
  import scipy.linalg as LA
  import spkit as sp
  
  #Fractional Fourier Transform
  y = sp.frft(x,alpha=0.5)
  
  #Fast fractional Fourier Transform
  y = sp.ffrft(x,alpha=0.5)
  
  
  help(sp.frft)
  
  help(sp.ffrft)
  
