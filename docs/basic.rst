
Periodogram
-------------
  
  
::
  
  import numpy as np
  import matplotlib.pyplot as plt
  import spkit as sp
  
  Px = sp.Periodogram(x,fs=128,method ='welch')
  Px = sp.Periodogram(x,fs=128,method ='periodogram')
  
  
A quick stats of an array
-------------
  
  
::
  
  import spkit as sp
  
  x_stats, names = sp.getStats(x, detail_level=1, return_names=True)
  
  detail_level=3
  # ['mean','sd','median','min','max','n','q25','q75','iqr','kur','skw','gmean','entropy']
  detail_level=2
  # ['mean','sd','median','min','max','n','q25','q75','iqr','kur','skw']
  detail_level=1
  # ['mean','sd','median','min','max','n']
  
  
  
Compute statistical outliers
----------------------------
  
  
::
  
  import spkit as sp
  
  idx, idx_bin = sp.OutLiers(x, method='iqr',k=1.5)
  
  
  
  
    
  
