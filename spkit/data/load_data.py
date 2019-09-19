import numpy as np
import pickle, os

def eegSample(fname = 'EEG16SecData.pkl'):
	fpath = os.path.join(os.path.dirname(__file__), fname)
	with open(fpath, "rb") as f:
		X,ch_names = pickle.load(f)
	return X,ch_names

def primitivePolynomials(fname = 'primitive_polynomials_GF2_dict.txt'):
	fpath = os.path.join(os.path.dirname(__file__), fname)
	f = open(fpath, "rb")
	lines = f.readlines()
	f.close()
	return eval(lines[0].decode())
