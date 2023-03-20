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

def eegSample(fname = 'EEG16SecData.pkl'):
	fpath = os.path.join(os.path.dirname(__file__), fname)
	with open(fpath, "rb") as f:
		X,ch_names = pickle.load(f)
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

def primitivePolynomials(fname = 'primitive_polynomials_GF2_dict.txt'):
	fpath = os.path.join(os.path.dirname(__file__), fname)
	f = open(fpath, "rb")
	lines = f.readlines()
	f.close()
	return eval(lines[0].decode())
