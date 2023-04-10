Multi-Electrode Arrays Processing
================================

.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/docs_fig/mea_proce_2.png
   :width: 800
   :align: center
   
 

Backgorund
----------

#TODO

Multi-Electrode Arrays System utilies an array of electrods mounted on a small plate as a grid electrods (e.g. 60) evenly spaced (700mm apart).
It is used to analyse the eletrophysiology of cells/tissues under different clinical conditions by stimulating with certain voltage on a regular intervals. As shown in figure below, a plate of MEA system of 60 electrodes (source: https://www.multichannelsystems.com/products/meas-60-electrodes). One of the commonly used research field is the cardiac electrophysiology.

.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/docs_fig/mea_plate_source.png
   :width: 400
   :align: center


This python library analyse the recorded signal file, by extracting the electrograms (EGMs) from signal recoding of each eletrodes, and extracting the features of each EGM.

#TODO

.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/docs_fig/mea_proce_3.png
   :width: 800
   :align: center






Complete Analysis of a recording
--------------------------------

#TODO

One of the simple function to provide complete analysis of recorded file is to use ```spkit.mea.analyse_mea_file``` function.
This uses the default settings of all the paramters for extracting electrograms, identifying bad eletrodes, extracting features and plotting figures.

```spkit.mea.analyse_mea_file``` needs two essential inputs, ```files_name``` :  a full path of recoding file in '.h5' format and ```stim_fhz``` frequency of stimulus in Hz.


::
  
  import spkit as sp
  sp.mea.analyse_mea_file(files_name,stim_fhz=1)




.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/docs_fig/mea_proce_3.png
   :width: 800
   :align: center










Extracting EGM
--------------

.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/docs_fig/mea_grid_egm_1.png
   :width: 800
   :align: center


EGM Processing & Feature Extractions
------------------------------------

.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/docs_fig/egm_processing_1.png
   :width: 800
   :align: center



Conduction and Activation Map
------------------------------------

.. image:: https://raw.githubusercontent.com/spkit/spkit.github.io/master/assets/images/docs_fig/mea_act_cv_map_2.png
   :width: 800
   :align: center






#TODO

