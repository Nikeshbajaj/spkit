Wavelet Filtering
=================

.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/nav_logo.svg
   :width: 200
   :align: right
   :target: https://nbviewer.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/Wavelet_Filtering_1_demo.ipynb
-----------------------------------------------------------------------------------------------------------------


**Background**
----------------
Other than classical frequency filtering, Wavelet filtering is one of common techniques used in signal processing. It allows to filter out short-time duration patterns captured by used wavelet. The patterns to be filtered out depends on the wavelet family (e.g. *db3*) used and number of level of decomposition. 

Algorithmically, it is very straightforward. Decompose a signal :math:`x(n)`, into wavelet coefficients :math:`X(k)`, where each coefficient represents the strength of wavelet pattern at particular time. With some threshold, remove the coefficients by zeroing out and reconstruct the signal back.

The machanism to choose a threshold on the strength of wavelet coefficient depends on the application and objective. To remove the noise and compress the signal, a widely used approach is to filter out all the wavelet coefficients with smaller strength.

Literature [1] suggest the **optimal threshold** on the wavelet coeffiecient is



.. math::
  
  \theta = \tilde{\sigma} \sqrt{2log(N)}
  
where :math:`\tilde{\sigma}` is estimation of noise variance and :math:`N` length of signal


.. math::
  
  \tilde{\sigma} = median(|X(k)|)/0.6745

and :math:`X(k)` are wavelet coeffients of :math:`x(n)`

There are other methods to choose threshold too. One can choose a :math:`\theta =1.5\times SD(X(k))` or :math:`\theta =IQR(X(k))` as to select the outliers, by standard deviation and interquartile range, respectively.

According to the theory, the **optimal threshold** should be applied by zeroing out the coefficients below with magnitude lower than threshold $|X(k)|<\theta$, and for later two methods of thresholds,standard deviation and interquartile range, the coefficients outside of the threshold should be zeroing out, since they reprepresent the outliers. However, some of the (weired) articles use these thresholds in other-way round.

A simple block-diagram shown below is the procedure of wavelet filtering.


.. image:: https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/wavelet_filtering_block_dia_1.png


**References:**
* [1] D.L. Donoho, J.M. Johnstone, **Ideal spatial adaptation by wavelet shrinkage** Biometrika, 81 (1994), pp. 425-455


API
--
* **spkit.wavelet_filtering(...)**
* **spkit.wavelet_filtering_win(...)**


In ***spkit***, we have implemented all three methods for threshold computing, can be chosen by *threshold = 'optimal', 'sd' or 'iqr'* or can be passed as a float value for a fixed threshold, e.g. *threshold = 0.5*. It also support to choose, if you want to zero out coefficient below the threshold or above by setting *filter_out_below* True or False. However, default setting is *threshold='optimal'* and *filter_out_below=True*.

There are a few more options to tune threshold and mode of filtering. For more details see doc section or run ```help(sp.wavelet_filtering)```

Example
----------------
::
  
  import numpy as np
  import matplotlib.pyplot as plt
  import spkit as sp
  print('spkit version',sp.__version__)
  
  x,fs = sp.load_data.eegSample_1ch()
  x = x[fs*2:fs*8]
  t = np.arange(len(x))/fs
  
  plt.figure(figsize=(15,3))
  plt.plot(t,x)
  plt.xlim([t[0],t[-1]])
  plt.grid()
  plt.xlabel('time (s)')
  plt.show()
  
'spkit version 0.0.9.4'
  
.. image:: https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/signal_1.png
  
  
::
  
  xf = sp.wavelet_filtering(x.copy(),wv='db3',threshold='optimal',verbose=1,WPD=False,show=True,fs=fs)

WPD: False  wv: db3  threshold: optimal  k: 1.5  mode: elim  filter_out_below?: True



.. image:: https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/Wavelet_filtering_3.png



`View in Jupyter-Notebook for details and more examples <https://nbviewer.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/Wavelet_Filtering_1_demo.ipynb>`_
----------------
  
  
.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/nav_logo.svg
   :width: 100
   :align: right
   :target: https://nbviewer.org/github/Nikeshbajaj/Notebooks/blob/master/spkit/SP/Wavelet_Filtering_1_demo.ipynb

-----------   
