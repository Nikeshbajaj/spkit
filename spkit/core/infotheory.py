from __future__ import absolute_import, division, print_function

import sys

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
import matplotlib.pyplot as plt
from scipy.signal import periodogram, welch

'''
Author @ Nikesh Bajaj
Date: 18 Apr 2019
Version : 0.0.3
Github :  https://github.com/Nikeshbajaj/spkit
Contact: n.bajaj@qmul.ac.uk
'''

# Probability distribuation is computed using histogram
# and optimal bin size of histogram is computed using Freedman–Diaconis rule

def entropy(x,alpha=1,ignoreZero=False,base=2,normalize=False):
    '''
    Rényi entropy of order α
    alpha:[0,inf]
         :0: Max-entropy
             H(x) = log(N)
             where N = number of bins
         :1: Shannan entropy
             H(x) = -\sum{Px*log(Px)}
         :2: Collision entropy
             H(x) = 1/(1-α)*log{\sum{Px^α}}
         :inf:Min-entropy:
             H(x) = -log(max(Px))
    base: base of log:
        : if 2, entropy is in bits, e-nats, 10 -bans
    ignoreZero: if true, probabilities with zero value will be omited, before computations
          : It doesn't make much of difference
    '''
    if base=='e': base =np.exp(1)
    frq,_ = np.histogram(x,bins='fd')
    N = len(frq)
    if alpha==0:
        H = np.log(N)
    else:
        Pr = frq/np.sum(frq)
        Pr = Pr[Pr>0] if ignoreZero else Pr+1e-10
        if alpha==1:
            H  = -np.sum(Pr*np.log(Pr))
        elif alpha==np.inf or alpha=='inf':
            H  = -np.log(np.max(Pr))
        else:
            H = (1.0/(1.0-alpha))*np.log(np.sum(Pr**alpha))
    H /=np.log(base)
    if normalize:
        H /= (np.log(N)/np.log(base))
    return H

def entropy_joint(x,y,ignoreZero=False,base=2):
    '''
	H(X,Y) = \sum {P(x,y)*np.log(P(x,y))}

    Computing joint probability using histogram2d from numpy
    '''
    # computing the optimal bin size using Freedman–Diaconis rule-
    _,bins = np.histogram(x,bins='fd')
    binx = np.ceil((np.max(x)-np.min(x))/(bins[1]-bins[0])).astype(int)

    _,bins = np.histogram(y,bins='fd')
    biny = np.ceil((np.max(y)-np.min(y))/(bins[1]-bins[0])).astype(int)

    #Computing joint probability
    frq = np.histogram2d(x,y,bins=[binx,biny])[0]
    Prxy = frq/np.sum(frq)
    Prxy = Prxy[Prxy>0] if ignoreZero else Prxy + 1e-10
    Hxy = - np.sum(Prxy*np.log(Prxy))
    if base!='e': Hxy /= np.log(base)
    return Hxy

def entropy_cond(x,y,ignoreZero=False,base=2):
    '''H(X|Y) = H(X,Y) - H(Y)'''
    Hxy = entropy_joint(x,y,ignoreZero=ignoreZero,base=base)
    Hy  = entropy(y,ignoreZero=ignoreZero,base=base)
    Hx1y = Hxy-Hy
    return Hx1y

def mutual_Info(x,y,ignoreZero=False,base=2):
    '''I(X;Y) = H(X)+H(Y)-H(X,Y)'''
    '''I(X;Y) = H(X) - H(X|Y)'''
    I = entropy(x,ignoreZero=ignoreZero,base=base)+\
        entropy(y,ignoreZero=ignoreZero,base=base)-\
        entropy_joint(x,y,ignoreZero=ignoreZero,base=base)
    return I

def entropy_kld(x,y,base=2):
    '''
    H_xy =  \sum{Px*log(Px/Py)}
    Cross entropy - Kullback–Leibler divergence
    '''
    _,bins = np.histogram(x,bins='fd')
    binx = bins[1]-bins[0]

    _,bins = np.histogram(y,bins='fd')
    biny = bins[1]-bins[0]

    binxy = np.min([binx,biny])
    xy = np.r_[x,y]

    nbins = np.ceil((max(xy)-min(xy))/binxy).astype(int)

    frq,_ = np.histogram(x,bins=nbins)
    PrX = frq/np.sum(frq)

    frq,_ = np.histogram(y,bins=nbins)
    PrY = frq/np.sum(frq)

    #ignoring to be divided by 0
    PrX += 1e-10
    PrY += 1e-10

    H  = np.sum(PrX*np.log(PrX/PrY))
    if base !='e': H = H/np.log(base)
    return H

def entropy_cross(x,y,base=2):
    '''
    Cross entropy
    H_xy = - \sum{Px*log(Py)}
    '''

    _,bins = np.histogram(x,bins='fd')
    binx = bins[1]-bins[0]

    _,bins = np.histogram(y,bins='fd')
    biny = bins[1]-bins[0]

    binxy = np.min([binx,biny])
    xy = np.r_[x,y]

    nbins = np.ceil((max(xy)-min(xy))/binxy).astype(int)

    frq,_ = np.histogram(x,bins=nbins)
    PrX = frq/np.sum(frq)

    frq,_ = np.histogram(y,bins=nbins)
    PrY = frq/np.sum(frq)

    #ignoring to be divided by 0
    PrX += 1e-10
    PrY += 1e-10

    H  = -np.sum(PrX*np.log(PrY))
    if base !='e': H = H/np.log(base)
    return H

def entropy_spectral(x,fs,method='fft',alpha=1,base=2,normalize=True,axis=-1,nperseg=None,bining=True):
    '''
    Spectral Entropy
    Measure of the uncertainity of frequency components in a Signal
    For Uniform distributed signal and Gaussian distrobutated signal, their entropy is quite different, but in spectral domain, both have same entropy
    x: 1d array
    fs = sampling frequency
    method: 'fft' use periodogram and 'welch' uses wekch method
    base: base of log, 2 for bit, 10 for nats
    '''
    x = np.asarray(x)
    # Compute and normalize power spectrum
    if method == 'fft':
        _, psd = periodogram(x,fs, axis=axis)
    elif method == 'welch':
        _, psd = welch(x, fs, nperseg=nperseg, axis=axis)

    psd_norm = psd / psd.sum(axis=axis, keepdims=True)
    if bining:
        se = entropy(psd_norm,alpha=alpha,base=base,normalize=normalize)
    else:

        se = -(psd_norm * np.log(psd_norm+1e-5)).sum(axis=axis)
        se /=np.log(base)
        if normalize:
            se /= (np.log(psd_norm.shape[axis]+1e-5)/np.log(base))
    return se

def entropy_sample(X, m, r):
    """
    Sample Entropy
    Ref: https://en.wikipedia.org/wiki/Sample_entropy
    """

    N = len(X)
    B = 0.0
    A = 0.0
    # Split time series and save all templates of length m
    xmi = np.array([X[i : i + m] for i in range(N - m)])
    xmj = np.array([X[i : i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([X[i : i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])
    return -np.log(A / B)


def entropy_approx(X, m, r) -> float:
    """
    Approximate Entropy
    Ref: https://en.wikipedia.org/wiki/Approximate_entropy
    """
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[X[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))
    N = len(X)
    return _phi(m) - _phi(m + 1)

def TD_Embed(x, order=3, delay=1):
    """
    Time delay Embedding Matrix.
    input
    ----------
    x : 1d-array, time series of shape (n,)
    order : int, Embedding dimension (order).
    delay : int,  Delay.

    output
    -------
    X: Embedded Matrix: ndarray
        Embedded time-series, of shape (n - (order - 1) * delay, order)
    """
    N = len(x)
    assert order * delay<=N # order * delay < N
    assert delay >=1 #delay>=1
    assert order>=2 #order>=2
    X = np.zeros((order, N - (order - 1) * delay))
    for i in range(order):
        X[i] = x[(i * delay):(i * delay + X.shape[1])]
    return X.T

def entropy_svd(x, order=3, delay=1,base=2, normalize=False):
    '''
    Singular Value Decomposition entropy.
    '''
    if base=='e': base = np.e
    x = np.array(x)
    M = TD_Embed(x, order=order, delay=delay)
    W = np.linalg.svd(M, compute_uv=False)
    W /= sum(W)
    H_svd = -np.multiply(W, np.log(W)).sum()
    H_svd /=np.log(base)
    if normalize:
        H_svd /= (np.log(order)/np.log(base))
    return H_svd

def entropy_permutation(x, order=3, delay=1, base=2,normalize=False):
    """
    Permutation Entropy.
    """
    if base=='e': base = np.e
    x = np.array(x)
    ran_order = range(order)
    Hmult = np.power(order, ran_order)
    sorted_idx = TD_Embed(x, order=order, delay=delay).argsort(kind='quicksort')
    Hval = (np.multiply(sorted_idx, Hmult)).sum(1)
    _, freq = np.unique(Hval, return_counts=True)
    Pr = freq/freq.sum()
    H_perm = -np.multiply(Pr, np.log(Pr)).sum()
    H_perm /=np.log(base)
    if normalize:
        H_perm /= (np.log(factorial(order))/np.log(base))
    return H_perm




def HistPlot(x,show=False,norm=False):
    frq,bins = np.histogram(x,bins='fd')
    if norm: frq = frq/np.sum(frq)
    plt.bar(bins[:-1],frq,width=0.8*(bins[1]-bins[0]),alpha=0.5)
    #plt.plot(bins[:-1],frq,'--k',lw=0.7)
    if show: plt.show()

def binSize_FD(x):
    IQR = stats.iqr(x)
    bw =  2.0*IQR/(len(x)**(1/3))
    return bw
