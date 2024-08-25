from __future__ import absolute_import, division, print_function

import sys, os
if sys.version_info[:2] < (3, 3):
    old_print = print
    def print(*args, **kwargs):
        flush = kwargs.pop('flush', False)
        old_print(*args, **kwargs)
        if flush:
            file = kwargs.get('file', sys.stdout)
            # Why might file=None? IDK, but it works for print(i, file=None)
            file.flush() if file is not None else sys.stdout.flush()

import numpy as np
import pickle, os
from ..utils import deprecated

@deprecated("due to naming convension, please use 'eeg_sample_14ch' for updated/improved functionality [spkit-0.0.9.7]")
def eegSample(fname = 'EEG16SecData.pkl',return_info=False):
	fpath = os.path.join(os.path.dirname(__file__),'files', fname)
	with open(fpath, "rb") as f:
		X,ch_names = pickle.load(f)
	
	if return_info:
		info = dict(fs =128)
		return X,ch_names, info
	return X,ch_names

@deprecated("due to naming convension, please use 'eeg_sample_artifact' for updated/improved functionality [spkit-0.0.9.7]")
def eegSample_artifact(fname = 'EEG16sec_artifact.pkl'):
	fpath = os.path.join(os.path.dirname(__file__),'files', fname)
	with open(fpath, "rb") as f:
		data = pickle.load(f)

	return data

@deprecated("due to naming convension, please use 'eeg_sample_1ch' for updated/improved functionality [spkit-0.0.9.7]")
def eegSample_1ch(ch=1,xtype='X_atar_elim'):
	fname = 'EEG16sec_artifact.pkl'
	fpath = os.path.join(os.path.dirname(__file__),'files', fname)

	#data = pickle.load(open(fpath, "rb"))
	with open(fpath, "rb") as f:
		data = pickle.load(f)

	x = data[xtype][:,ch]
	fs = data['fs']
	return x,fs

@deprecated("due to naming convension, please use 'primitive_polynomials' for updated/improved functionality [spkit-0.0.9.7]")
def primitivePolynomials(fname = 'primitive_polynomials_GF2_dict.txt'):
	'''List of Primitive Polynomials as dictionary

	Key of dictionary is the order of polynomials, from 2, till 31

	Returns
	-------
	plist: List of all Primitive Polynomials as dictionary for each polynomial order

	Examples
	--------
	import spkit as sp
	plist = sp.data.primitivePolynomials()
	pplist = plist[3]
	print(pplist)
	'''
	fpath = os.path.join(os.path.dirname(__file__),'files', fname)
	f = open(fpath, "rb")
	lines = f.readlines()
	f.close()
	plist = eval(lines[0].decode())
	return plist


# --- new version -------
def eeg_sample_14ch(fname = 'EEG16SecData.pkl'):
	r"""Load 14 channel EEG sample recording of 16 second duration.

	Recorded sample is a part of PhyAAt Dataset[1]

	Returns
	-------
	  X : 2d-array
	  fs :  sampling frequency
	  ch_names : list of channel names, in same order as signal

	References
	----------
	* [1] PhyAAt Dataset - https://phyaat.github.io/

	See Also
	--------
	eda_sample: Electrodermal activity (EDA)
	gsr_sample: Galvanic Skin Response (GSR)
	eeg_sample_1ch: Electroencephalography (EEG) - 1-channel
	eeg_sample_artifact: Electroencephalography (EEG) processed
	ecg_sample_12leads: Electrocardiogram (ECG) - 12-leads
	ecg_sample:  Electrocardiogram (ECG) - 1-lead
	optical_sample: Optical Mapped Signal
	ppg_sample: Photoplethysmogram (PPG)
	egm_sample: Electrogram (EGM)

	Examples
	--------
	import numpy as np
	import matplotlib.pyplot as plt
	import spkit as sp
	X, fs, ch_names = sp.data.eeg_sample_14ch()
	t = np.arange(X.shape[0])/fs
	sep = 300 
	plt.figure(figsize=(10,6))
	plt.plot(t,X + np.arange(X.shape[1])*sep)
	plt.xlim([t[0],t[-1]])
	plt.xlabel('time (s)')
	plt.yticks(np.arange(X.shape[1])*sep,ch_names)
	plt.grid()
	plt.show()
	"""

	fpath = os.path.join(os.path.dirname(__file__),'files' , fname)
	with open(fpath, "rb") as f:
		X,ch_names = pickle.load(f)
	fs = 128
	return X,fs,ch_names

def eeg_sample_artifact(fname = 'EEG16sec_artifact.pkl'):
	r"""Load 14 channel EEG sample recording of 16 second duration with artifacts and processed.

	Recorded sample is a part of PhyAAt Dataset[1]

	Returns
	--------
	data : dictionary contants signals with following keys
	     'X_raw' :  Raw EEG signal
		 'X_fil' :  Raw EEG Filtered with high pass filter (0.5Hz)
		 'X_ica' :  Aftifact removed using ICA
		 'X_atar_soft' : artifact removed using ATAR soft thresholding mode
		 'X_atar_elim' : artifact removed using ATAR elimination mode

		 'fs' :  sampling frequency
		 'info' : information
		 'ch_names' : channel names

	References
	----------
	* [1] PhyAAt Dataset - https://phyaat.github.io/

	See Also
	--------
	eda_sample: Electrodermal activity (EDA)
	gsr_sample: Galvanic Skin Response (GSR)
	eeg_sample_14ch: Electroencephalography (EEG) - 14-channel
	eeg_sample_1ch: Electroencephalography (EEG) - 1-channel
	ecg_sample_12leads: Electrocardiogram (ECG) - 12-leads
	ecg_sample:  Electrocardiogram (ECG) - 1-lead
	optical_sample: Optical Mapped Signal
	ppg_sample: Photoplethysmogram (PPG)
	egm_sample: Electrogram (EGM)

	Examples
	--------
	import numpy as np
	import matplotlib.pyplot as plt
	import spkit as sp
	data = sp.data.eeg_sample_artifact()
	X = data['X_raw']
	fs = data['fs']
	ch_names = data['ch_names']
	t = np.arange(X.shape[0])/fs
	sep = 50 
	plt.figure(figsize=(10,6))
	plt.plot(t,X + np.arange(X.shape[1])*sep)
	plt.xlim([t[0],t[-1]])
	plt.xlabel('time (s)')
	plt.yticks(np.arange(X.shape[1])*sep,ch_names)
	plt.grid()
	plt.show()
	"""
	fpath = os.path.join(os.path.dirname(__file__),'files', fname)
	with open(fpath, "rb") as f:
		data = pickle.load(f)
	return data

def eeg_sample_1ch(ch=1,xtype='X_raw'):
	r"""Load a single channel EEG sample recording of 16 second duration.

	Recorded sample is a part of PhyAAt Dataset[1]

	Parameters
	----------
	ch: int, 
	  -  channel number, 0, 13
	
	Returns
	-------
	x: 1d-array
	fs: int, 
	- sampling frequency
	
	References
	----------
	* [1] PhyAAt Dataset - https://phyaat.github.io/

	See Also
	--------
	eda_sample: Electrodermal activity (EDA)
	gsr_sample: Galvanic Skin Response (GSR)
	eeg_sample_14ch: Electroencephalography (EEG) - 14-channel
	eeg_sample_artifact: Electroencephalography (EEG) processed
	ecg_sample_12leads: Electrocardiogram (ECG) - 12-leads
	ecg_sample:  Electrocardiogram (ECG) - 1-lead
	optical_sample: Optical Mapped Signal
	ppg_sample: Photoplethysmogram (PPG)
	egm_sample: Electrogram (EGM)

	Examples
	--------
	import numpy as np
	import matplotlib.pyplot as plt
	import spkit as sp
	x,fs = sp.data.eeg_sample_1ch(ch=0)
	t = np.arange(len(x))/fs
	plt.figure(figsize=(12,3))
	plt.plot(t,x)
	plt.xlim([t[0],t[-1]])
	plt.xlabel('time (s)')
	plt.grid()
	plt.show()
	"""
	fname = 'EEG16sec_artifact.pkl'
	fpath = os.path.join(os.path.dirname(__file__),'files', fname)
	with open(fpath, "rb") as f:
		data = pickle.load(f)
	x = data[xtype][:,ch]
	fs = data['fs']
	return x,fs

def ecg_sample_12leads(sample=1):
	r"""Load 12 lead ECG sample.

	Parameters
	----------
	sample : int, {1,2,3,-1}
	 - sample number
	 - if -1, return all three samples

	Returns
	-------
	X: 2d-array, (n,ch)
	fs: int, sampling frequency
	lead_names: list of names for leads

	See Also
	--------
	eda_sample: Electrodermal activity (EDA)
	gsr_sample: Galvanic Skin Response (GSR)
	eeg_sample_14ch: Electroencephalography (EEG) - 14-channel
	eeg_sample_1ch: Electroencephalography (EEG) - 1-channel
	eeg_sample_artifact: Electroencephalography (EEG) processed
	ecg_sample:  Electrocardiogram (ECG) - 1-lead
	optical_sample: Optical Mapped Signal
	ppg_sample: Photoplethysmogram (PPG)
	egm_sample: Electrogram (EGM)

	Examples
	--------
	import numpy as np
	import matplotlib.pyplot as plt
	import spkit as sp
	X,fs,lead_names = sp.data.ecg_sample_12leads(sample=3)
	t = np.arange(X.shape[0])/fs
	plt.figure(figsize=(6,8))
	plt.plot(t,X + np.arange(X.shape[1])*5)
	plt.xlim([t[0],t[-1]])
	plt.grid()
	plt.yticks(np.arange(X.shape[1])*5, lead_names)
	plt.xlabel('time (s)')
	plt.ylabel('ECG Leads')
	plt.show()
	"""
	assert sample in [1,2,3,-1]

	fname = 'ecg_3samples_12leads.pkl'
	fpath = os.path.join(os.path.dirname(__file__),'files', fname)
	with open(fpath, "rb") as f:
		data = pickle.load(f)
	
	X = data['X'] if sample==-1 else data['X'][sample-1]
	fs = data['fs']
	lead_names = data['lead_names']
	return X,fs,lead_names

def cell_mea_sample_8x8(sample=1):
	r"""Load Cell EP Data.

	From a sample of MEA 8x8 Grid, 60Channels

	    .. code-block::

                             MEA 8x8 GRID

                   | 21 | 31 | 41 | 51 | 61  | 71 |
               |12 | 22 | 32 | 42 | 52 | 62  | 72 | 82 |
               |13 | 23 | 33 | 43 | 53 | 63  | 73 | 83 |
               |14 | 24 | 34 | 44 | 54 | 64  | 74 | 84 |
               |15 | 25 | 35 | 45 | 55 | 65  | 75 | 85 |
               |16 | 26 | 36 | 46 | 56 | 66  | 76 | 86 |
               |17 | 27 | 37 | 47 | 57 | 67  | 77 | 87 |
                   | 28 | 38 | 48 | 58 | 68  | 78 |


	Ref : https://www.multichannelsystems.com/products/microelectrode-arrays

	Parameters
	----------
	sample : int, {1}
	 - sample number
	 - if -1, return all three samples

	Returns
	-------
	X: 2d-array, (n,ch)
	fs: int, sampling frequency
	ch_names: list of channel names

	See Also
	--------
	eda_sample: Electrodermal activity (EDA)
	gsr_sample: Galvanic Skin Response (GSR)
	eeg_sample_14ch: Electroencephalography (EEG) - 14-channel
	eeg_sample_1ch: Electroencephalography (EEG) - 1-channel
	eeg_sample_artifact: Electroencephalography (EEG) processed
	ecg_sample:  Electrocardiogram (ECG) - 1-lead
	optical_sample: Optical Mapped Signal
	ppg_sample: Photoplethysmogram (PPG)
	egm_sample: Electrogram (EGM)

	Examples
	--------
	import numpy as np
	import matplotlib.pyplot as plt
	import spkit as sp
	X,fs,lead_names = sp.data.ecg_sample_12leads(sample=3)
	t = np.arange(X.shape[0])/fs
	plt.figure(figsize=(6,8))
	plt.plot(t,X + np.arange(X.shape[1])*5)
	plt.xlim([t[0],t[-1]])
	plt.grid()
	plt.yticks(np.arange(X.shape[1])*5, lead_names)
	plt.xlabel('time (s)')
	plt.ylabel('ECG Leads')
	plt.show()
	"""
	assert sample in [1,2,3,-1]

	fname = 'ecg_3samples_12leads.pkl'
	fpath = os.path.join(os.path.dirname(__file__),'files', fname)
	with open(fpath, "rb") as f:
		data = pickle.load(f)
	
	X = data['X'] if sample==-1 else data['X'][sample-1]
	fs = data['fs']
	lead_names = data['lead_names']
	return X,fs,lead_names

def optical_sample(sample=1,species=None):
	r"""Load sample(s) of optical mapping signals

	Parameters
	----------
	sample : int, {1,2,3,-1}
	  - sample number
	  - if -1, return all three samples 


	species: str, default=None, {None, 'rabbit'}
	  - data of species
	  - for 'rabbit', data of two cameras
	  

	Returns
	-------
	X : array,
	  - shape (n,) or (n,3) if species= None
	  - shape (n,2) or shape (n,6) for species='rabbit',
	  
	fs : int

	See Also
	--------
	eda_sample: Electrodermal activity (EDA)
	gsr_sample: Galvanic Skin Response (GSR)
	eeg_sample_14ch: Electroencephalography (EEG) - 14-channel
	eeg_sample_1ch: Electroencephalography (EEG) - 1-channel
	eeg_sample_artifact: Electroencephalography (EEG) processed
	ecg_sample_12leads: Electrocardiogram (ECG) - 12-leads
	ecg_sample:  Electrocardiogram (ECG) - 1-lead
	ppg_sample: Photoplethysmogram (PPG)
	egm_sample: Electrogram (EGM)

	Examples
	--------
	import numpy as np
	import matplotlib.pyplot as plt
	import spkit as sp
	x,fs = sp.data.optical_sample(sample=1)
	t = np.arange(len(x))/fs
	plt.figure(figsize=(12,3))
	plt.plot(t,x)
	plt.xlim([t[0],t[-1]])
	plt.xlabel('time (s)')
	plt.grid()
	plt.show()

	#sp.data.optical_sample
	import numpy as np
	import matplotlib.pyplot as plt
	import spkit as sp
	X,fs = sp.data.optical_sample(sample=1,species='rabbit')
	t = np.arange(X.shape[0])/fs
	plt.figure(figsize=(10,4))
	plt.subplot(211)
	plt.plot(t,X[:,0],color='C0',label='cam1')
	plt.xlim([t[0],t[-1]])
	plt.grid()
	plt.legend(loc=1)
	plt.subplot(212)
	plt.plot(t,X[:,1],color='C1',label='cam2')
	plt.xlim([t[0],t[-1]])
	plt.grid()
	plt.legend(loc=1)
	plt.xlabel('time (s)')
	plt.subplots_adjust(hspace=0)
	plt.suptitle('Optical: Rabbit')
	plt.show()
	"""
	
	assert sample in [1,2,3,-1]

	if species is None:
		fname = 'optical_bio_samples.pkl'
		fpath = os.path.join(os.path.dirname(__file__),'files', fname)
		with open(fpath, "rb") as f:
			data = pickle.load(f)
		X = data['X'] if sample==-1 else data['X'][:,sample-1]
		fs = data['fs']

	elif species=='rabbit':
		fname = 'optical_rabbit_samples.pkl'
		fpath = os.path.join(os.path.dirname(__file__),'files', fname)
		with open(fpath, "rb") as f:
			data = pickle.load(f)
		X1 = data['X1'] if sample==-1 else data['X1'][:,sample-1]
		X2 = data['X2'] if sample==-1 else data['X2'][:,sample-1]
		X = np.c_[X1,X2]
		fs = data['fs']
	return X,fs

def ppg_sample(sample=1):
	r"""Load sample(s) of PPG signals
	   
	**Photoplethysmogram Signal**

	- An optically obtained signal, recorded by Pulse Sensor at lower sampling rate (128)
	- Sample is part of PhyAAt Dataset [2]

	Parameters
	----------
	sample : int, {1,2,-1}
	 - sample number
	 - if -1, return all two samples

	See Also
	--------
	eda_sample: Electrodermal activity (EDA)
	gsr_sample: Galvanic Skin Response (GSR)
	eeg_sample_14ch: Electroencephalography (EEG) - 14-channel
	eeg_sample_1ch: Electroencephalography (EEG) - 1-channel
	eeg_sample_artifact: Electroencephalography (EEG) processed
	ecg_sample_12leads: Electrocardiogram (ECG) - 12-leads
	ecg_sample:  Electrocardiogram (ECG) - 1-lead
	optical_sample: Optical Mapped Signal
	egm_sample: Electrogram (EGM)


	Returns
	-------
	X : array, (n,) or list of two
	fs : int

	References
	----------
	* [1] wikipedia - https://en.wikipedia.org/wiki/Photoplethysmogram
	* [2] PhyAAt Dataset - https://phyaat.github.io/

	Examples
	--------
	#sp.data.ppg_sample
	#Example 1
	import numpy as np
	import matplotlib.pyplot as plt
	import spkit as sp
	x,fs = sp.data.ppg_sample(sample=1)
	x = x[:int(fs*20)]
	t = np.arange(len(x))/fs
	plt.figure(figsize=(12,3))
	plt.plot(t,x)
	plt.xlim([t[0],t[-1]])
	plt.xlabel('time (s)')
	plt.ylabel('PPG Signal')
	plt.grid()
	plt.show()


	################################
	#Example 2
	#sp.data.ppg_sample
	import numpy as np
	import matplotlib.pyplot as plt
	import spkit as sp
	x,fs = sp.data.ppg_sample(sample=2)
	x = x[:int(fs*20)]
	t = np.arange(len(x))/fs
	plt.figure(figsize=(12,3))
	plt.plot(t,x)
	plt.xlim([t[0],t[-1]])
	plt.xlabel('time (s)')
	plt.ylabel('PPG Signal')
	plt.grid()
	plt.show()
	"""
	assert sample in [1,2,-1]

	fname = 'ppg_samples.pkl'
	fpath = os.path.join(os.path.dirname(__file__),'files', fname)
	with open(fpath, "rb") as f:
		data = pickle.load(f)
	
	fs = data['fs']
	X = [data['x1'],data['x2']]
	if sample==1:
		return X[0],fs
	if sample==2:
		return X[1],fs
	return X,fs

def eda_sample(sample=1):
	r"""Load sample(s) of Electrodermal activity (EDA)
	 
	 Widely known as Electrodermal activity (EDA) 

	- Sample is recorded using copper plats at lower sampling rate (128)
	- Sample is part of PhyAAt Dataset [2]

	Parameters
	----------
	sample : int, {1,2,-1}
	 - sample number
	 - if -1, return all two samples

	Returns
	-------
	X : array, (n,) or (n,2)
	fs : int

	References
	----------
	* [1] wikipedia - https://en.wikipedia.org/wiki/Electrodermal_activity
	* [2] PhyAAt Dataset - https://phyaat.github.io/

	See Also
	--------
	gsr_sample: Galvanic Skin Response (GSR)
	eeg_sample_14ch: Electroencephalography (EEG) - 14-channel
	eeg_sample_1ch: Electroencephalography (EEG) - 1-channel
	eeg_sample_artifact: Electroencephalography (EEG) processed
	ecg_sample_12leads: Electrocardiogram (ECG) - 12-leads
	ecg_sample:  Electrocardiogram (ECG) - 1-lead
	optical_sample: Optical Mapped Signal
	ppg_sample: Photoplethysmogram (PPG)
	egm_sample: Electrogram (EGM)

	Examples
	--------
	#sp.data.eda_sample
	#Example 1
	import numpy as np
	import matplotlib.pyplot as plt
	import spkit as sp
	x,fs = sp.data.eda_sample(sample=1)
	x = x[:int(fs*60)]
	t = np.arange(len(x))/fs
	plt.figure(figsize=(12,3))
	plt.plot(t,x)
	plt.xlim([t[0],t[-1]])
	plt.xlabel('time (s)')
	plt.ylabel('GSR Signal')
	plt.grid()
	plt.show()


	############################
	#sp.data.gsr_sample
	#Example 2
	import numpy as np
	import matplotlib.pyplot as plt
	import spkit as sp
	x,fs = sp.data.eda_sample(sample=2)
	x = x[:int(fs*60)]
	t = np.arange(len(x))/fs
	plt.figure(figsize=(12,3))
	plt.plot(t,x)
	plt.xlim([t[0],t[-1]])
	plt.xlabel('time (s)')
	plt.ylabel('GSR Signal')
	plt.grid()
	plt.show()
	"""

	assert sample in [1,2,-1]

	fname = 'gsr_samples.pkl'
	fpath = os.path.join(os.path.dirname(__file__),'files', fname)
	with open(fpath, "rb") as f:
		data = pickle.load(f)

	fs = data['fs']
	X = [data['x1'],data['x2']]
	if sample==1:
		return X[0],fs
	if sample==2:
		return X[1],fs
	return X,fs

def gsr_sample(sample=1):
	r"""Load sample(s) of Galvanic Skin Response (GSR) or 
	 
	 Widely known as Electrodermal activity (EDA) 

	- Sample is recorded using copper plats at lower sampling rate (128)
	- Sample is part of PhyAAt Dataset [2]

	Parameters
	----------
	sample : int, {1,2,-1}
	 - sample number
	 - if -1, return all two samples

	Returns
	-------
	X : array, (n,) or (n,2)
	fs : int

	References
	----------
	* [1] wikipedia - https://en.wikipedia.org/wiki/Electrodermal_activity
	* [2] PhyAAt Dataset - https://phyaat.github.io/

	See Also
	--------
	eda_sample: Electrodermal activity (EDA)
	eeg_sample_14ch: Electroencephalography (EEG) - 14-channel
	eeg_sample_1ch: Electroencephalography (EEG) - 1-channel
	eeg_sample_artifact: Electroencephalography (EEG) processed
	ecg_sample_12leads: Electrocardiogram (ECG) - 12-leads
	ecg_sample:  Electrocardiogram (ECG) - 1-lead
	optical_sample: Optical Mapped Signal
	ppg_sample: Photoplethysmogram (PPG)
	egm_sample: Electrogram (EGM)

	Examples
	--------
	#sp.data.gsr_sample
	#Example 1
	import numpy as np
	import matplotlib.pyplot as plt
	import spkit as sp
	x,fs = sp.data.gsr_sample(sample=1)
	x = x[:int(fs*60)]
	t = np.arange(len(x))/fs
	plt.figure(figsize=(12,3))
	plt.plot(t,x)
	plt.xlim([t[0],t[-1]])
	plt.xlabel('time (s)')
	plt.ylabel('GSR Signal')
	plt.grid()
	plt.show()


	############################
	#sp.data.gsr_sample
	#Example 2
	import numpy as np
	import matplotlib.pyplot as plt
	import spkit as sp
	x,fs = sp.data.gsr_sample(sample=2)
	x = x[:int(fs*60)]
	t = np.arange(len(x))/fs
	plt.figure(figsize=(12,3))
	plt.plot(t,x)
	plt.xlim([t[0],t[-1]])
	plt.xlabel('time (s)')
	plt.ylabel('GSR Signal')
	plt.grid()
	plt.show()
	"""

	return eda_sample(sample=sample)

def egm_sample(sample=1):
	"""Load sample(s) of Electrograms and ECG

     - EGM: Contact Electrical Activity (Physiological Signal)
	 - Sample of 5 signals
	 
	Parameters
	----------
	sample : int, {1,2,3,4,5,-1}
	 - sample number
	 - if -1, return all 5 samples

	Returns
	-------
	X : array, (n,) or (n,5)
	fs : int

	References
	----------
	* [1] wikipedia - 

	See Also
	--------
	eda_sample: Electrodermal activity (EDA)
	gsr_sample: Galvanic Skin Response (GSR)
	eeg_sample_14ch: Electroencephalography (EEG) - 14-channel
	eeg_sample_1ch: Electroencephalography (EEG) - 1-channel
	eeg_sample_artifact: Electroencephalography (EEG) processed
	ecg_sample_12leads: Electrocardiogram (ECG) - 12-leads
	ecg_sample:  Electrocardiogram (ECG) - 1-lead
	optical_sample: Optical Mapped Signal
	ppg_sample: Photoplethysmogram (PPG)

	Examples
	--------
	#sp.data.ecg_sample
	#Example 1
	import numpy as np
	import matplotlib.pyplot as plt
	import spkit as sp
	x,fs = sp.data.egm_sample(sample=1)
	x = x[:int(fs*0.2)]
	t = np.arange(len(x))/fs
	plt.figure(figsize=(12,3))
	plt.plot(t,x)
	plt.xlim([t[0],t[-1]])
	plt.xlabel('time (s)')
	plt.ylabel('EGM Signal')
	plt.grid()
	plt.show()


	############################
	#sp.data.ecg_sample
	#Example 2
	import numpy as np
	import matplotlib.pyplot as plt
	import spkit as sp
	x,fs = sp.data.egm_sample(sample=2)
	x = x[:int(fs*0.2)]
	t = np.arange(len(x))/fs
	plt.figure(figsize=(12,3))
	plt.plot(t,x)
	plt.xlim([t[0],t[-1]])
	plt.xlabel('time (s)')
	plt.ylabel('EGM Signal')
	plt.grid()
	plt.show()
	"""

	assert sample in [1,2,3,4,5,-1]

	fname = 'ecg_egm_smaple_28s.pkl'
	fpath = os.path.join(os.path.dirname(__file__),'files', fname)
	with open(fpath, "rb") as f:
		data = pickle.load(f)

	fs = data['fs']
	X =  data['X'][:,4:]
	if sample==-1:
		return X,fs
	return X[:,sample-1],fs

def ecg_sample(sample=1):
	"""Load sample(s) of ECG sampled at high rate

     - ECG :  Contact Electrocardiogram, recorded at high rate
	 
	Parameters
	----------
	sample : int, {1,2,3,4,-1}
	 - sample number
	 - if -1, return all 4 samples

	Returns
	-------
	X : array, (n,) or (n,4)
	fs : int

	References
	----------
	* [1] wikipedia - 

	See Also
	--------
	eda_sample: Electrodermal activity (EDA)
	gsr_sample: Galvanic Skin Response (GSR)
	eeg_sample_14ch: Electroencephalography (EEG) - 14-channel
	eeg_sample_1ch: Electroencephalography (EEG) - 1-channel
	eeg_sample_artifact: Electroencephalography (EEG) processed
	ecg_sample_12leads: Electrocardiogram (ECG) - 12-leads
	optical_sample: Optical Mapped Signal
	ppg_sample: Photoplethysmogram (PPG)
	egm_sample: Electrogram (EGM)

	Examples
	--------
	#sp.data.ecg_sample
	#Example 1
	import numpy as np
	import matplotlib.pyplot as plt
	import spkit as sp
	x,fs = sp.data.ecg_sample(sample=1)
	x = x[:int(fs*3)]
	t = np.arange(len(x))/fs
	plt.figure(figsize=(12,3))
	plt.plot(t,x)
	plt.xlim([t[0],t[-1]])
	plt.xlabel('time (s)')
	plt.ylabel('ECG Signal')
	plt.grid()
	plt.show()


	############################
	#sp.data.ecg_sample
	#Example 2
	import numpy as np
	import matplotlib.pyplot as plt
	import spkit as sp
	x,fs = sp.data.ecg_sample(sample=2)
	x = x[:int(fs*10)]
	t = np.arange(len(x))/fs
	plt.figure(figsize=(12,3))
	plt.plot(t,x)
	plt.xlim([t[0],t[-1]])
	plt.xlabel('time (s)')
	plt.ylabel('ECG Signal')
	plt.grid()
	plt.show()
	"""

	assert sample in [1,2,3,4,-1]

	fname = 'ecg_egm_smaple_28s.pkl'
	fpath = os.path.join(os.path.dirname(__file__),'files', fname)
	with open(fpath, "rb") as f:
		data = pickle.load(f)

	fs = data['fs']
	X =  data['X'][:,:4]
	if sample==-1:
		return X,fs
	return X[:,sample-1],fs

def primitive_polynomials(fname = 'primitive_polynomials_GF2_dict.txt'):
	r"""List of Primitive Polynomials as dictionary

	Key of dictionary is the order of polynomials, from 2, till 31

	Parameters
	----------

	Returns
	-------
	plist: List of all Primitive Polynomials as dictionary for each polynomial order

	Examples
	--------
	>>> #sp.data.primitive_polynomials
	>>> import spkit as sp
	>>> plist = sp.data.primitive_polynomials()
	>>> pplist = plist[5]
	>>> print(pplist)
	[[5, 2], [5, 4, 2, 1], [5, 4, 3, 2]]
	"""

	fpath = os.path.join(os.path.dirname(__file__),'files', fname)
	f = open(fpath, "rb")
	lines = f.readlines()
	f.close()
	plist = eval(lines[0].decode())
	return plist
