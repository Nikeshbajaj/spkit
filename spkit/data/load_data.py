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

def eegSample(fname = 'EEG16SecData.pkl',return_info=False):
	fpath = os.path.join(os.path.dirname(__file__), fname)
	with open(fpath, "rb") as f:
		X,ch_names = pickle.load(f)
	
	if return_info:
		info = dict(fs =128)
		return X,ch_names, info
	return X,ch_names

def eegSample_artifact(fname = 'EEG16sec_artifact.pkl'):
	fpath = os.path.join(os.path.dirname(__file__), fname)
	with open(fpath, "rb") as f:
		data = pickle.load(f)

	return data

def eegSample_1ch(ch=1,xtype='X_atar_elim'):
	fname = 'EEG16sec_artifact.pkl'
	fpath = os.path.join(os.path.dirname(__file__), fname)
	data = pickle.load(open(fpath, "rb"))
	x = data[xtype][:,ch]
	fs = data['fs']
	return x,fs

# --- new version -------
def eeg_sample_14ch(fname = 'EEG16SecData.pkl'):
	'''Load 14 channel EEG sample recording of 16 second duration.


	returns
	  -  X : 2d-array
	  -  ch_names : list of channel names, in same order as signal
	  -  fs :  sampling frequency
	'''

	fpath = os.path.join(os.path.dirname(__file__), fname)
	with open(fpath, "rb") as f:
		X,ch_names = pickle.load(f)
	fs = 128
	return X,ch_names, fs

def eeg_sample_artifact(fname = 'EEG16sec_artifact.pkl'):
	'''Load 14 channel EEG sample recording of 16 second duration with artifacts and processed.

	returns
	  -  data : dictionary contants signals with following keys
	     'X_raw' :  Raw EEG signal
		 'X_fil' :  Raw EEG Filtered with high pass filter (0.5Hz)
		 'X_ica' :  Aftifact removed using ICA
		 'X_atar_soft' : artifact removed using ATAR soft thresholding mode
		 'X_atar_elim' : artifact removed using ATAR elimination mode

		 'fs' :  sampling frequency
		 'info' : information
		 'ch_names' : channel names
	      
	'''
	fpath = os.path.join(os.path.dirname(__file__), fname)
	with open(fpath, "rb") as f:
		data = pickle.load(f)
		
	fs=128
	return data, fs

def eeg_sample_1ch(ch=1,xtype='X_raw'):
	'''Load single channel EEG sample recording of 16 second duration.

	 returns
	   x - 1d-array
	   fs - sampling frequency
	'''
	fname = 'EEG16sec_artifact.pkl'
	fpath = os.path.join(os.path.dirname(__file__), fname)
	data = pickle.load(open(fpath, "rb"))
	x = data[xtype][:,ch]
	fs = data['fs']
	return x,fs



def primitivePolynomials(fname = 'primitive_polynomials_GF2_dict.txt'):
	'''
	List of Primitive Polynomials

	'''
	fpath = os.path.join(os.path.dirname(__file__), fname)
	f = open(fpath, "rb")
	lines = f.readlines()
	f.close()
	return eval(lines[0].decode())
