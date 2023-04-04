'''
Information Theory techniques
--------------------------------
Author @ Nikesh Bajaj
updated on Date: 27 March 2023. Version : 0.0.5
updated on Date: 1 Jan 2022, Version : 0.0.3
updated on Date: 18 Apr 2019, Version : 0.0.1
Github :  https://github.com/Nikeshbajaj/spkit
Contact: n.bajaj@qmul.ac.uk
'''

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
from scipy import stats as scipystats
from .infomation_theory_advance import cdf_mapping

# Probability distribuation is computed using histogram
# and optimal bin size of histogram is computed using Freedman–Diaconis rule

def entropy_(x,alpha=1,ignoreZero=False,base=2,normalize=False,return_n_bins=False,is_discrete =False):
    '''
    Rényi entropy of order α
    alpha:[0,inf]
         :0: Max-entropy
             H(x) = log(N)
             where N = number of bins
         :1: Shannan entropy
             H(x) = -\sum{Px*log(Px)}
         :2: Collision entropy or Rényi entropy
             H(x) = 1/(1-α)*log{\sum{Px^α}}
         :inf:Min-entropy:
             H(x) = -log(max(Px))
    base: base of log:
        : if 2, entropy is in bits, e-nats, 10 -bans
    ignoreZero: if true, probabilities with zero value will be omited, before computations
          : It doesn't make much of difference
    '''
    if base=='e': base =np.exp(1)
    if is_discrete:
        _,frq = np.unique(x,return_counts=True)
    else:
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
    if return_n_bins:
        return H,N
    return H

def entropy_joint_(x,y,ignoreZero=False,base=2):
    '''
	H(X,Y) = \sum {P(x,y)*np.log(P(x,y))}

    Computing joint probability using histogram2d from numpy

    max{H(x),H(y)} <= H(X,Y) <= H(x) + H(y)

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

def entropy_cond_(x,y,ignoreZero=False,base=2):
    '''H(X|Y) = H(X,Y) - H(Y)

    0 <= H(X|Y) <= H(x)

    '''
    Hxy = entropy_joint(x,y,ignoreZero=ignoreZero,base=base)
    Hy  = entropy(y,ignoreZero=ignoreZero,base=base)
    Hx1y = Hxy-Hy
    return Hx1y

def mutual_Info_(x,y,ignoreZero=False,base=2):
    '''I(X;Y) = H(X)+H(Y)-H(X,Y)'''
    '''I(X;Y) = H(X) - H(X|Y)

    0 <= I(X;Y) <= min{ H(x), H(y) }
    '''
    I = entropy(x,ignoreZero=ignoreZero,base=base)+\
        entropy(y,ignoreZero=ignoreZero,base=base)-\
        entropy_joint(x,y,ignoreZero=ignoreZero,base=base)
    return I

def entropy_kld_(x,y,base=2):
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

def entropy_cross_(x,y,base=2):
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

# IMPROVED PRECISION OF COMPUTATIONS


def entropy(x,alpha=1,base=2,normalize=False,is_discrete=False,bins='fd',return_n_bins=False,ignoreZero=False):
    '''
    Rényi entropy of order α
    alpha:[0,inf]
         :0: Max-entropy
             H(x) = log(N)
             where N = number of bins
         :1: Shannan entropy
             H(x) = -\sum{Px*log(Px)}
         :2: Collision entropy or Rényi entropy
             H(x) = 1/(1-α)*log{\sum{Px^α}}
         :inf:Min-entropy:
             H(x) = -log(max(Px))
    base: base of log:
        : if 2, entropy is in bits, e-nats, 10 -bans
    ignoreZero: if true, probabilities with zero value will be omited, before computations
          : It doesn't make much of difference
    '''
    if base=='e': base =np.exp(1)
    if is_discrete:
        _,frq = np.unique(x,return_counts=True)
    else:
        frq,_ = np.histogram(x,bins=bins)
    
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
    if return_n_bins:
        return H,N
    return H

def entropy_joint(x,y,base=2,is_discrete=False,bins='fd',return_n_bins=False,ignoreZero=False):
    '''
	H(X,Y) = \sum {P(x,y)*np.log(P(x,y))}

    Computing joint probability using histogram2d from numpy

    max{H(x),H(y)} <= H(X,Y) <= H(x) + H(y)

    '''
    if is_discrete:
        Nx = len(np.unique(x))
        Ny = len(np.unique(y))
    else:
        # computing the optimal bin size using Freedman–Diaconis rule, or defined by bins
        if isinstance(bins,str):
            _,bin_ws = np.histogram(x,bins=bins)
            Nx = np.ceil((np.max(x)-np.min(x))/(bin_ws[1]-bin_ws[0])).astype(int)

            _,bin_ws = np.histogram(y,bins=bins)
            Ny = np.ceil((np.max(y)-np.min(y))/(bin_ws[1]-bin_ws[0])).astype(int)
        elif isinstance(bins,int):
            Nx,Ny = bins,bins
        elif isinstance(bins,list):
            assert len(bins)==2
            Nx,Ny = bins[0], bins[1]
        else:
            raise ValueError("Undefied way of 'bins', it should be 'str, int or list of arrays")
    #Computing joint probability
    frq = np.histogram2d(x,y,bins=[Nx,Ny])[0]
    Prxy = frq/np.sum(frq)
    Prxy = Prxy[Prxy>0] if ignoreZero else Prxy + 1e-10
    Hxy = - np.sum(Prxy*np.log(Prxy))
    if base!='e': Hxy /= np.log(base)
    if return_n_bins:
        return Hxy, (Nx,Ny)
    return Hxy

def entropy_cond(x,y,base=2,is_discrete=False,bins='fd',return_n_bins=False,verbose=False,ignoreZero=False):
    '''H(X|Y) = H(X,Y) - H(Y)

    0 <= H(X|Y) <= H(x)

    '''
    Hxy,(Nx,Ny) = entropy_joint(x,y,ignoreZero=ignoreZero,base=base,is_discrete=is_discrete,bins=bins,return_n_bins=True)
    Hy , Ny_i   = entropy(y,ignoreZero=ignoreZero,base=base,is_discrete=is_discrete,bins=Ny,return_n_bins=True)
    if verbose: print(Nx,Ny,Ny_i)
    Hx1y = Hxy-Hy
    if return_n_bins:
        return Hx1y, (Nx,Ny)
    return Hx1y

def mutual_Info(x,y,base=2,is_discrete=False,bins='fd',return_n_bins=False,verbose=False,ignoreZero=False):
    '''
       I(X;Y) = H(X)+H(Y)-H(X,Y)
       
       I(X;Y) = H(X) - H(X|Y)

    0 <= I(X;Y) <= min{ H(x), H(y) }
    '''
    
    
    Hx,Nx = entropy(x,ignoreZero=ignoreZero,base=base,is_discrete=is_discrete,bins=bins,return_n_bins=True)
    Hy,Ny = entropy(y,ignoreZero=ignoreZero,base=base,is_discrete=is_discrete,bins=bins,return_n_bins=True)
    
    Hxy,(Nx_i,Ny_i) = entropy_joint(x,y,ignoreZero=ignoreZero,base=base,is_discrete=is_discrete,bins=[Nx,Ny],return_n_bins=True)
    
    if verbose: print(Nx,Ny,Nx_i,Ny_i)
    
    I = Hx + Hy - Hxy
    
    #I = entropy(x,ignoreZero=ignoreZero,base=base)+\
    #    entropy(y,ignoreZero=ignoreZero,base=base)-\
    #    entropy_joint(x,y,ignoreZero=ignoreZero,base=base)
    if return_n_bins:
        return I,(Nx,Ny)
    return I

def entropy_kld(x,y,base=2,is_discrete=False,bins='fd',verbose=False,pre_version=False,return_n_bins=False):
    '''
    H_xy =  \sum{Px*log(Px/Py)}
    Cross entropy - Kullback–Leibler divergence
    '''
    
    if pre_version:
        _,bins = np.histogram(x,bins='fd')
        binx = bins[1]-bins[0]

        _,bins = np.histogram(y,bins='fd')
        biny = bins[1]-bins[0]

        binxy = np.min([binx,biny])
        xy = np.r_[x,y]

        N = np.ceil((max(xy)-min(xy))/binxy).astype(int)
        
        frq,_ = np.histogram(x,bins=N)
        PrX = frq/np.sum(frq)

        frq,_ = np.histogram(y,bins=N)
        PrY = frq/np.sum(frq)

        #ignoring to be divided by 0
        PrX += 1e-10
        PrY += 1e-10

        H_kl  = np.sum(PrX*np.log(PrX/PrY))
        
    else:        
        if is_discrete:
            Nx = len(np.unique(x))
            Ny = len(np.unique(y))
        else:
            # computing the optimal bin size using Freedman–Diaconis rule, or defined by bins
            if isinstance(bins,str):
                _,bin_ws = np.histogram(x,bins=bins)
                Nx = np.ceil((np.max(x)-np.min(x))/(bin_ws[1]-bin_ws[0])).astype(int)
                
                _,bin_ws = np.histogram(y,bins=bins)
                Ny = np.ceil((np.max(y)-np.min(y))/(bin_ws[1]-bin_ws[0])).astype(int)
            
            elif isinstance(bins,int):
                Nx,Ny  = bins, bins
            
            elif isinstance(bins,list):
                assert len(bins)==2
                Nx,Ny = bins[0], bins[1]
            
            else:
                raise ValueError("Undefied way of 'bins', it should be 'str, int or list of arrays")
        
        if verbose: print(Nx,Ny, isinstance(Nx,int), isinstance(Ny,int),type(Nx),type(Ny))
        
        if isinstance(Nx,(int,np.integer)) and isinstance(Ny,(int,np.integer)):
            N = np.max([Nx,Ny]).astype(int)
            frq = np.histogram2d(x,y,bins=[N,N])[0]
        else:
            #"Undefied way of 'bins', it should be 'str, int or list of arrays of same size"
            assert (isinstance(Nx,np.ndarray) and isinstance(Ny,np.ndarray)) or (isinstance(Nx,list) and isinstance(Ny,list))
            assert len(Nx)==len(Ny)
            frq = np.histogram2d(x,y,bins=[Nx,Ny])[0]
            N = len(Nx)
            
        
        frx = frq.sum(1)
        PrX = frx/frx.sum()
        
        fry = frq.sum(0)
        PrY = fry/fry.sum()

        #ignoring to be divided by 0
        PrX += 1e-10
        PrY += 1e-10

        H_kl  = np.sum(PrX*np.log(PrX/PrY))
    
    if base !='e': H_kl = H_kl/np.log(base)
        
    if return_n_bins:
        return H_kl, (N,N)
    return H_kl

def entropy_cross(x,y,base=2,is_discrete=False,bins='fd',verbose=False,pre_version=False,return_n_bins=False,):
    '''
    Cross entropy
    H_xy = - \sum{Px*log(Py)}
    '''
    
    if pre_version:
        _,bins = np.histogram(x,bins='fd')
        binx = bins[1]-bins[0]

        _,bins = np.histogram(y,bins='fd')
        biny = bins[1]-bins[0]

        binxy = np.min([binx,biny])
        xy = np.r_[x,y]

        N = np.ceil((max(xy)-min(xy))/binxy).astype(int)

        frq,_ = np.histogram(x,bins=N)
        PrX = frq/np.sum(frq)

        frq,_ = np.histogram(y,bins=N)
        PrY = frq/np.sum(frq)

        #ignoring to be divided by 0
        PrX += 1e-10
        PrY += 1e-10

        H_cross  = -np.sum(PrX*np.log(PrY))
    else:
        if is_discrete:
            Nx = len(np.unique(x))
            Ny = len(np.unique(y))
        else:
            # computing the optimal bin size using Freedman–Diaconis rule, or defined by bins
            if isinstance(bins,str):
                _,bin_ws = np.histogram(x,bins=bins)
                Nx = np.ceil((np.max(x)-np.min(x))/(bin_ws[1]-bin_ws[0])).astype(int)
                
                _,bin_ws = np.histogram(y,bins=bins)
                Ny = np.ceil((np.max(y)-np.min(y))/(bin_ws[1]-bin_ws[0])).astype(int)
            
            elif isinstance(bins,int):
                Nx,Ny  = bins, bins
            
            elif isinstance(bins,list):
                assert len(bins)==2
                Nx,Ny = bins[0], bins[1]
            
            else:
                raise ValueError("Undefied way of 'bins', it should be 'str, int or list of arrays")
        
        if verbose: print(Nx,Ny, isinstance(Nx,int), isinstance(Ny,int),type(Nx),type(Ny))
        
        if isinstance(Nx,(int,np.integer)) and isinstance(Ny,(int,np.integer)):
            N = np.max([Nx,Ny]).astype(int)
            frq = np.histogram2d(x,y,bins=[N,N])[0]
        else:
            #"Undefied way of 'bins', it should be 'str, int or list of arrays of same size"
            assert (isinstance(Nx,np.ndarray) and isinstance(Ny,np.ndarray)) or (isinstance(Nx,list) and isinstance(Ny,list))
            assert len(Nx)==len(Ny)
            frq = np.histogram2d(x,y,bins=[Nx,Ny])[0]
            N = len(Nx)
            
        #frq = np.histogram2d(x,y,bins=[N,N])[0]
        frx = frq.sum(1)
        PrX = frx/frx.sum()
        
        fry = frq.sum(0)
        PrY = fry/fry.sum()

        #ignoring to be divided by 0
        PrX += 1e-10
        PrY += 1e-10

        H_cross = -np.sum(PrX*np.log(PrY))
        
    if base !='e': H_cross = H_cross/np.log(base)
    
    if return_n_bins:
        return H_cross, (N,N)
    return H_cross
    '''
    Cross entropy
    H_xy = - \sum{Px*log(Py)}
    '''
    
    if pre_version:
        _,bins = np.histogram(x,bins='fd')
        binx = bins[1]-bins[0]

        _,bins = np.histogram(y,bins='fd')
        biny = bins[1]-bins[0]

        binxy = np.min([binx,biny])
        xy = np.r_[x,y]

        N = np.ceil((max(xy)-min(xy))/binxy).astype(int)

        frq,_ = np.histogram(x,bins=N)
        PrX = frq/np.sum(frq)

        frq,_ = np.histogram(y,bins=N)
        PrY = frq/np.sum(frq)

        #ignoring to be divided by 0
        PrX += 1e-10
        PrY += 1e-10

        H_cross  = -np.sum(PrX*np.log(PrY))
    else:
        if is_discrete:
            Nx = len(np.unique(x))
            Ny = len(np.unique(y))
        else:
            # computing the optimal bin size using Freedman–Diaconis rule, or defined by bins
            if isinstance(bins,str):
                _,bin_ws = np.histogram(x,bins=bins)
                Nx = np.ceil((np.max(x)-np.min(x))/(bin_ws[1]-bin_ws[0])).astype(int)
                
                _,bin_ws = np.histogram(y,bins=bins)
                Ny = np.ceil((np.max(y)-np.min(y))/(bin_ws[1]-bin_ws[0])).astype(int)
            
            elif isinstance(bins,int):
                Nx,Ny  = bins, bins
            
            elif isinstance(bins,list):
                assert len(bins)==2
                Nx,Ny = bins[0], bins[1]
            
            else:
                raise ValueError("Undefied way of 'bins', it should be 'str, int or list of arrays")
        
        if verbose: print(Nx,Ny, isinstance(Nx,int), isinstance(Ny,int),type(Nx),type(Ny))
        
        if isinstance(Nx,(int,np.integer)) and isinstance(Ny,(int,np.integer)):
            N = np.max([Nx,Ny]).astype(int)
            frq = np.histogram2d(x,y,bins=[N,N])[0]
        else:
            #"Undefied way of 'bins', it should be 'str, int or list of arrays of same size"
            assert (isinstance(Nx,np.ndarray) and isinstance(Ny,np.ndarray)) or (isinstance(Nx,list) and isinstance(Ny,list))
            assert len(Nx)==len(Ny)
            frq = np.histogram2d(x,y,bins=[Nx,Ny])[0]
            N = len(Nx)
            
        #frq = np.histogram2d(x,y,bins=[N,N])[0]
        frx = frq.sum(1)
        PrX = frx/frx.sum()
        
        fry = frq.sum(0)
        PrY = fry/fry.sum()

        #ignoring to be divided by 0
        PrX += 1e-10
        PrY += 1e-10

        H_cross = -np.sum(PrX*np.log(PrY))
        
    if base !='e': H_cross = H_cross/np.log(base)
    
    if return_n_bins:
        return H_cross, (N,N)
    return H_cross



def entropy_spectral(x,fs,method='fft',alpha=1,base=2,normalize=True,axis=-1,nperseg=None,bining=False):
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
        H_perm /= (np.log(np.math.factorial(order))/np.log(base))
    return H_perm

def HistPlot(x,show=False,norm=False):
    frq,bins = np.histogram(x,bins='fd')
    if norm: frq = frq/np.sum(frq)
    plt.bar(bins[:-1],frq,width=0.8*(bins[1]-bins[0]),alpha=0.5)
    #plt.plot(bins[:-1],frq,'--k',lw=0.7)
    if show: plt.show()

def Mu_law(x,Mu=255,companding=True):
    '''
    μ-Law for companding and expanding
    -----------------------------------
    - A way to map gaussian, or laplacian distribuated signal to uniformly distributed one
    - for companding - smaller amplitude values are stretched (since they have high-probabilty)
      and large amplitutde values are compressed (since they have low probabiltity)
    - for expending - it reverts the mapping

    for -1 ≤ x ≤ 1
    Companding:
        y(n) = sign(x)*log(1 + μ|x|)/log(1 + μ)

    expanding:
        y(n) = sign(x)*((1 + μ)**|x| - 1)/μ

    - An alternative to A-Law
    - The μ-law algorithm provides a slightly larger dynamic range than the A-law
      at the cost of worse proportional distortions for small signals.

    Ref: https://en.wikipedia.org/wiki/M-law_algorithm
    -----------------------------------

    input
    -----
    x  : 1d (or nd) array of signal, must be  -1 ≤ x ≤ 1
    Mu : scalar (0≤Mu) - factor for companding-expanding mapping
       : ~0: identity function, higher value of Mu will stretch and compress more
       : use 1e-5 for Mu=0 to avoid 'zero-division' error
    companding: (bool) -  if True, companding is applied, else expanding

    output
    ------
    y: mapped output of same shape as x
    '''
    assert np.max(np.abs(x))<=1
    assert Mu>0 #to avoid zero division error use Mu=1e-5

    if companding:
        #Companding ~ compression ~ encoding
        y = np.sign(x)*np.log(1 + Mu*np.abs(x))/np.log(1+Mu)

    else:
        #Expanding ~ uncompression/expension ~ decoding
        y = np.sign(x)*((1 + Mu)**np.abs(x) - 1)/Mu

    return y

def A_law(x,A=255,companding=True):
    '''
    A-Law for companding and expanding
    ----------------------------------
    - A way to map gaussian, or laplacian distribuated signal to uniformly distributed one
    - for companding - smaller amplitude values are stretched (since they have high-probabilty)
      and large amplitutde values are compressed (since they have low probabiltity)
    - for expending - it reverts the mapping

    for -1 ≤ x ≤ 1 and 1 < A
    Companding:
        y(n) = sign(x)* (A|x|/(1 + log(A)))               if   |x|<1/A
             = sign(x)* ((1 + log(A|x|)) /(1 + log(A)))   else

    expanding:
        y(n) = sign(x)* (|x|(1+log(A)))/A                 if   |x|<1/(1+log(A))
             = sign(x)* (np.exp(-1 + |x|(1+log(A)))) /A   else

    - An alternative to μ-Law
    - The μ-law algorithm provides a slightly larger dynamic range than the A-law
      at the cost of worse proportional distortions for small signals.

    Ref: https://en.wikipedia.org/wiki/A-law_algorithm
    ------------------------------------------

    input
    -----
    x  : 1d (or nd) array of signal, must be  -1 ≤ x ≤ 1
    A  : scalar (1≤A) - factor for companding-expanding mapping
       : 1: identity function, higher value of A will stretch and compress more
    companding: (bool) -  if True, companding is applied, else expanding

    output
    ------
    y: mapped output of same shape as x
    '''
    assert np.max(np.abs(x))<=1
    assert A>=1

    y = np.zeros_like(x)

    if companding:
        #Companding ~ compression ~ encoding
        idx = np.abs(x)<1/A
        y[idx]  = A*np.abs(x[idx])
        y[~idx] = 1 + np.log(A*np.abs(x[~idx]))
        y /= (1 + np.log(A))
    else:
        #Expanding ~ uncompression/expension ~ decoding
        idx = np.abs(x)<(1/(1+np.log(A)))
        y[idx]   = np.abs(x[idx])*(1+np.log(A))
        y[~idx]  = np.exp(-1 + np.abs(x[~idx])*(1+np.log(A)))
        y /= A

    y *= np.sign(x)
    return y

def bin_width(x,method='fd'):
    '''
    Compute bin width using different methods
    -----------------------------------------
    'fd' (Freedman Diaconis Estimator)
      - Robust (resilient to outliers) estimator that takes into account data variability and data size.

    'doane'
      - An improved version of Sturges’ estimator that works better with non-normal datasets.

    'scott'
      - Less robust estimator that that takes into account data variability and data size.

    'stone'
      - Estimator based on leave-one-out cross-validation estimate of the integrated squared error.
         Can be regarded as a generalization of Scott’s rule.

    'rice'
      - Estimator does not take variability into account, only data size.
         Commonly overestimates number of bins required.

    'sturges'
      -  Only accounts for data size. Only optimal for gaussian data and underestimates number of bins
         for large non-gaussian datasets.

    'sqrt'
      - Square root (of data size) estimator, used by Excel and other programs for its speed and simplicity.


    input
    -----
    x :  1d-array or (n-d array)
    method : method to compute bin width and number of bins

    output
    ------
    bw : bin width
    k  : number of bins

    '''
    if np.ndim(x)>1: x = x.reshape(-1)
    x = x[~np.isnan(x)]
    n = len(x)
    if method=='fd':
        #Freedman–Diaconis' choice
        IQR = scipystats.iqr(x)
        bw  = 2.0*IQR/(n**(1/3))
        k = np.ceil((np.max(x)-np.min(x))/bw).astype(int)
    elif method=='sqrt':
        #Square-root choice
        k = np.ceil(np.sqrt(n)).astype(int)
        bw  = (np.max(x)-np.min(x))/k
    elif method=='sturges':
        #Sturges' formula
        k = np.ceil(1 + np.log2(n)).astype(int)
        bw  = (np.max(x)-np.min(x))/k
    elif method=='rice':
        #Rice Rule
        k = np.ceil(2*(n**(1/3))).astype(int)
        bw  = (np.max(x)-np.min(x))/k
    elif method=='doane':
        #Doane's formula
        sk_g = np.abs(scipystats.skew(x))
        sg_g = np.sqrt(6*(n-2)/((n+1)*(n+3)))
        gnn  = 1 + sk_g/sg_g
        k    = 1 + np.log2(n) + np.log2(gnn)
        k = np.ceil(k).astype(int)
        bw  = (np.max(x)-np.min(x))/k
    elif method=='scott':
        #Scott's normal reference rule
        sig = np.std(x)
        bw  = 3.49*sig/(n**(1/3))
        k = np.ceil((np.max(x)-np.min(x))/bw).astype(int)
    else:
        raise NameError("Undefined Method: should be one of {'fd','sqrt','sturges','rice','doane','scott'}")
    return bw,k

def binSize_FD(x):
    '''
    Compute bin width using Freedman–Diaconis rule
    ----------------------------------------------
    bin_width = 2*IQR(x)/(n**(1/3))

    input
    -----
    x: 1d-array

    output
    ------
    bw : bin width
    '''
    IQR = scipystats.iqr(x)
    bw =  2.0*IQR/(len(x)**(1/3))
    return bw

def quantize_FD(x,scale=1,min_bins=2,keep_amp_scale=True):
    '''
    Discretize (quantize) input signal x using optimal bin-width (FD)

    input
    -----

    output
    ------

    '''
    assert scale>0
    x0   = x.copy()
    xmin = np.min(x0)
    xmax = np.max(x0)
    bw = binSize_FD(x0)*scale
    nbins = np.clip(np.ceil((xmax-xmin)/bw).astype(int),min_bins,None)
    x0 = (x0 - xmin)/(xmax-xmin)
    y = np.round(x0*(nbins-1))
    y /=(nbins-1)
    y = np.clip(y,0,1)
    if keep_amp_scale: y = y*(xmax-xmin) + xmin
    return y, nbins

def Quantize(x,scale=1,min_bins=2,A=None, Mu=None,cdf_map=False,nbins=None):
    '''
    input
    -----

    output
    ------
    '''
    x0 = x.copy()
    assert (1*(A is not None) + 1*(Mu is not None) + 1*(cdf_map))<=1
    # both A-Law and Mu-Law can (should) not be applied
    # choose only one
    if A is not None:
        x0 = A_law(x0,A=A,encoding=True)
    if Mu is not None:
        x0 = Mu_law(x0,Mu=Mu,encoding=True)
    if cdf_map:
        x0 = cdf_mapping(x0)
    xmin = np.min(x0)
    xmax = np.max(x0)
    if nbins is None:
        bw = binSize_FD(x0)*scale
        nbins = np.clip(np.ceil((xmax-xmin)/bw).astype(int),min_bins,None)
    x0 = (x0 - xmin)/(xmax-xmin)
    y = np.round(x0*(nbins-1))
    y /=(nbins-1)
    y = np.clip(y,0,1)
    y = y*(xmax-xmin) + xmin
    return y

def quantize_signal(x,n_levels=None,A=None,Mu=None,cdf_map=False,keep_range=True,bin_method='fd',bin_scale=1,min_levels=2):
    '''
    Quantize signal into discreet levels
    ------------------------------------

    Before quantization, signal is equalise using either of three methods
    (1) A-Law, (2) Mu-Law, or (3) Cummulative Distribution Function


    input
    -----
    x: 1d-array, input signal

    n_levels: int, >1, default=None
              if None, number of levels are computed based on optimum bin of signal distribuation
              according to given method (default='fd': Freedman–Diaconis rule).
              While computing, n_levels, following two parameters are effective; bw_scale, min_levels

        if n_levels=None:
            bin_method: str, default='fd', {'fd','sqrt','sturges','rice','doane','scott'}
                        Method to compute bin width 'fd' (Freedman Diaconis Estimator)


            bin_scale: scaler, +ve, default=1
                     It is used to scale the bin width computed by Freedman–Diaconis rule. Higher it is less number of levels are estimated

            min_levels, int, +ve, default=2, while computing n_levels, ensuring the minimum number of levels

    Distribution Equalization: Only one of following can be applied
     A: int,>0, default=None
       if not None, A-Law compading is applied to equalise the distribution
       A>0, A=1, means identity funtion

     Mu: int,>=0, default=None
       if not None, Mu-Law compading is applied to equalise the distribution
       Mu>=0, Mu=0 mean identity funtion

     cdf_map: bool, default=False
            If true, CDF mapping is applied to equalise the distribution


    keep_range: bool, default=True
              If True, return quantized signal is rescaled to its original range, else returned signal is in range of 0 to 1


    output
    ------
    y:  Quantized signal, same size as x
        if keep_range=True, y has same range as x, else 0 to 1

    y_int: Quantized signal as integer level, ranges from 0 to n_levels-1
           Useful to map signal in post-processing


    '''
    # both A-Law and Mu-Law can (should) not be applied
    # choose only one
    assert (1*(A is not None) + 1*(Mu is not None) + 1*(cdf_map))<=1
    # if chosen, number of levels shoudl be at least 2
    if n_levels is not None: assert n_levels>1
    x0 = x.copy()
    x_min = np.nanmin(x0)
    x_max = np.nanmax(x0)
    #print(x_min, x_max)

    if A is not None or Mu is not None:
        x0 = x0/np.abs(x0).max()

    if A is not None:
        x0 = A_law(x0,A=A,companding=True)
    if Mu is not None:
        x0 = Mu_law(x0,Mu=Mu,companding=True)
    if A is not None or Mu is not None:
        x0 = (x0+1)/2

    if cdf_map:
        x0 = cdf_mapping(x0)
        #x0 = 2*(x0-0.5)

    if A is None and Mu is None and (cdf_map==False):
        #x0_min, x0_max = np.min(x0), np.min(x0)
        x0 = x0 - np.nanmin(x0)
        x0 = x0/np.nanmax(x0)

    #print(np.min(x0), np.max(x0), 'from 0 to 1')

    # SINGAL IS NOW BEWEEN 0 to 1
    if n_levels is None:
        xmin,xmax = np.min(x0), np.max(x0)
        bw, _ = bin_width(x0,method=bin_method)
        #bw = binSize_FD(x0)*bin_scale
        bw *= bin_scale
        n_levels = np.clip(np.ceil((xmax-xmin)/bw).astype(int),min_levels,None)
        #print(n_levels)

    #x0 = (x0 - xmin)/(xmax-xmin)
    y_int = np.round(x0*(n_levels-1))

    y = y_int/(n_levels-1)
    y = np.clip(y,0,1)

    if keep_range:
        #y = 2*(y-0.5)
        y = y*(x_max-x_min) + x_min
    return y, y_int
#
def plotJointEntropyXY(x,y,Venn=True, DistPlot=True,JointPlot=True,printVals=True):
    '''
    Analyse Entropy-Venn Diagrams of X and Y
    ----------------------------------------
    input
    -----
    x: 1d-array
    y: 1d-array of same size as x
    Venn: if True, plot Venn diagram
    DistPlot: if True, plot distribuation of x and y
    JointPlot: if True, plot joint distribuation
    printVals: if True, print values
    output
    -------
    Nothing is returned, just plots and prints
    '''
    assert x.shape==y.shape
    import seaborn as sns
    hx = entropy(x)
    hy = entropy(y)
    mi = mutual_Info(x,y)
    mi_norm = mi/np.min([hx,hy])

    nplots = Venn+JointPlot+DistPlot
    #print(nplots)
    if nplots:
        #print(5+6*(nplots>1),5+2*(nplots>2))
        fig = plt.figure(figsize=(6+6*(nplots>1),4+3*(nplots>2)))
        fign=1
        if Venn:
            #print(1+1*(nplots>2),1+1*(nplots>1),fign)
            ax = plt.subplot(1+1*(nplots>2),1+1*(nplots>1),fign)
            crx = (hx+hy)*(1-mi_norm)
            circle_x = plt.Circle((0,  0), hx, color='C0',lw=2,fill=False,clip_on=False,label='H (x)')
            circle_y = plt.Circle((crx, 0), hy, color='C1',lw=2,fill=False,clip_on=False,label='H (y)')
            xmin = 0-max(hx,hy)
            xmx = 0+2*hx+2*hy
            ax.add_patch(circle_x)
            ax.add_patch(circle_y)
            ax.set_xlim([xmin*1.5,xmx])
            ax.set_ylim([xmin*1.5,max(hx,hy)*1.5])
            ax.plot(0, 0,'.C0')
            ax.plot(crx, 0,'.C1')
            ax.legend()
            ax.set_title('Venn Diagram')
            ax.axis('off')
            fign+=1
        if DistPlot:
            #print(1+1*(nplots>2),1+1*(nplots>1),fign)
            ax = plt.subplot(1+1*(nplots>2),1+1*(nplots>1),fign)
            ax = sns.kdeplot(x,label='x',fill=True)
            ax = sns.kdeplot(y,label='y',ax=ax,fill=True)
            ax.legend()
            ax.set_title('distribuation')
            fign+=1
        if JointPlot:
            #print(1+1*(nplots>2),1+1*(nplots>1),fign)
            ax = plt.subplot(1+1*(nplots>2),1+1*(nplots>1),fign)
            ax = sns.histplot(x=x, y=y,ax=ax,common_norm=True,kde=True,cbar=True,cbar_kws=dict(shrink=.75))
            ax = sns.kdeplot(x =x,y=y,ax=ax,color='k',alpha=0.5)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('joint distribuation')
            #plt.show()
            #fign+=1

        plt.show()
    if printVals:
        print('-----Unnormalized-------')
        print('H(x)  \t',hx)
        print('H(y)  \t',hy)
        print('I(x,y)\t',mi)
        #print('MI',hx-sp.entropy_cond(xx,yy), hy-sp.entropy_cond(yy,xx))
        print('H(x|y)\t',entropy_cond(x,y))
        print('H(y|x)\t',entropy_cond(y,x))
        print('H(x,y)\t',entropy_joint(x,y))
        print('')
        print('-----Normalized-------')
        print('H(x)/log(N)          \t\t',entropy(x,normalize=True))
        print('H(y)/log(N)          \t\t',entropy(y,normalize=True))
        print('I(x,y)/min{H(x),H(y)}\t\t',mi_norm)
        print('H(x|y)/H(x)          \t\t',entropy_cond(x,y)/hx)
        print('H(y|x)/H(y)          \t\t',entropy_cond(y,x)/hy)
        print('H(x,y)/(H(x)+H(y))   \t\t',entropy_joint(x,y)/(hx+hy))
        #print('H(x,y)\t/(H(x)+H(y))\t',(sp.entropy_joint(x,y)-np.max([hx,hy]))/(hx+hy-np.max([hx,hy])))
        print('(H(x,y)-max{H(x),H(y)})/min{H(x),H(y)}\t\t',(entropy_joint(x,y)-np.max([hx,hy]))/(np.min([hx,hy])))
