"""
Information Theory techniques
-----------------------------
Author @ Nikesh Bajaj
updated on Date: 27 March 2023. Version : 0.0.5
updated on Date: 1 Jan 2022, Version : 0.0.3
updated on Date: 18 Apr 2019, Version : 0.0.1
Github :  https://github.com/Nikeshbajaj/spkit
Contact: n.bajaj@qmul.ac.uk
"""

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
from .information_theory_advance import cdf_mapping
from ..utils import deprecated
import warnings
warnings.filterwarnings('once')

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
def entropy(x,alpha=1,base=2,normalize=False,is_discrete=False,bins='fd',return_n_bins=False,ignoreZero=False,esp=1e-10):
    r"""Entropy :math:`H(X)`

    Given a sequence or signal x (1d array), compute entropy H(x).

    If 'is_discrete' is true, given x is considered as discreet sequence and to compute entropy,
    frequency of all the unique values of x are computed and used to estimate H(x), which is straightforward.

    However for real-valued sequence (is_discrete=False), first a density of x is computed using histogram method,
    and using optimal bin-width (Freedman Diaconis Estimator), as it is set to bins='fd'.

    **Rényi entropy of order α (generalised form) alpha:[0,inf]**
            * alpha = 0: Max-entropy
                :math:`H(x) = log(N)`
                where N = number of bins
            * alpha= 1: Shannan entropy
                :math:`H(x) = -\sum{P(x)*log(P(x))}`
            * alpha = 2 or .. : Collision entropy or Rényi entropy
                :math:`H(x) = 1/(1-α)*log{\sum{P(x)^α}}`
            * alpha = inf:Min-entropy:
                :math:`H(x) = -log(max(P(x)))`

    

    Parameters
    ----------

    x : 1d array
      - input sequence or signal

    is_discrete: bool, default=False.
      - If True, frequency of unique values are used to estimate H(x)

    alpha : float between 0 to infinity [0,inf], (default=1)
         * alpha = 1 (default), shannan entropy is computed - :math:`H(x) = -\sum{Px*log(Px)}`
         * alpha = 0, maximum entropy: :math:`H(x) = log(N)`, where N = number of bins
         * alpha = 'inf' or np.inf, minimum entropy, :math:`H(x) = -log(max(Px))`
         * for any other value of alpha, Collision entropy or Rényi entropy, :math:`H(x) = 1/(1-α)*log{\sum{Px^α}}`

    base: base of log, (default=2)
      - decides the unit of entropy
      - if base=2 (default) unit of entropy is in bits, base=e, nats, base=10, bans

    .. versionadded:: 0.0.9.5

    normalize: bool, default = False
        - if true, normalised entropy is returned, :math:`H(x)/max{H(x)} = H(x)/log(N)`, which has range 0 to 1.
        - It is useful, while comparing two different sources to enforce the range of entropy between 0 to 1.

    bins: {str, int}, bins='fd' (default)
        * str decides the method of compute bin-width, bins='fd' (default) is considered as optimal bin-width 
          of a real-values signal/sequence. check help(spkit.bin_width) for more Methods
        * if bins is integer, then fixed number of bins are computed. It is useful, while comparing 
          two different sources by enforcing the same number of bins.

    return_n_bins: bool, (default=False)
        - if True, number of bins are also returned.

    ignoreZero: bool, default =False
        - if true, probabilities with zero value will be omited, before computations
        - It doesn't make much of difference


    Returns
    -------
    H : Entropy value
    N : number of bins, only if return_n_bins=True

    Notes
    -----
    * wikipedia - https://en.wikipedia.org/wiki/Entropy_(information_theory)


    See Also
    --------
    entropy_sample : Sample Entropy
    entropy_approx : Approximate Entropy
    dispersion_entropy : Dispersion Entropy
    entropy_spectral : Spectral Entropy
    entropy_svd :  SVD Entropy
    entropy_permutation :  Permutation Entropy
    entropy_differential : Differential Entropy


    Examples
    --------
    >>> import numpy as np
    >>> import spkit as sp
    >>> np.random.seed(1)
    >>> x = np.random.rand(10000)
    >>> y = np.random.randn(10000)
    >>> #Shannan entropy
    >>> H_x = sp.entropy(x,alpha=1)
    >>> H_y = sp.entropy(y,alpha=1)
    >>> print('Shannan entropy')
    >>> print('Entropy of x: H(x) = ',H_x)
    >>> print('Entropy of y: H(y) = ',H_y)
    >>> print('')
    >>> Hn_x = sp.entropy(x,alpha=1, normalize=True)
    >>> Hn_y = sp.entropy(y,alpha=1, normalize=True)
    >>> print('Normalised Shannan entropy')
    >>> print('Entropy of x: H(x) = ',Hn_x)
    >>> print('Entropy of y: H(y) = ',Hn_y)
    >>> np.random.seed(None)
        Shannan entropy
        Entropy of x: H(x) =  4.458019387223165
        Entropy of y: H(y) =  5.043357283463282
        Normalised Shannan entropy
        Entropy of x: H(x) =  0.9996833158270148
        Entropy of y: H(y) =  0.8503760993630085
    """

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
        Pr = Pr[Pr>0] if ignoreZero else Pr+esp
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

def entropy_joint(x,y,base=2,is_discrete=False,bins='fd',return_n_bins=False,ignoreZero=False,esp=1e-10):
    r"""Joint Entropy :math:`H(X,Y)`

	.. math:: H(X,Y) = \sum {P(x,y)*log(P(x,y))}

    Computing joint probability using histogram2d from numpy

    .. math::  max\{H(x),H(y)\} <= H(X,Y) <= H(x) + H(y)

    Parameters
    ----------
    x,y :  1d-arrays

    is_discrete: bool, default=False.
      - If True, frequency of unique values are used to estimate H(x,y)

    base: base of log, default=2
      - decides the unit of entropy
      - if base=2 unit of entropy is in bits and base=e for nats, base=10 for bans

    bins: {str, int, [int, int]}. default='fd'
      - str decides the method of compute bin-width, bins='fd' is considered as optimal bin-width of a real-values signal/sequence.
      - check help(spkit.bin_width) for more Methods
      - if bins is an integer, then fixed number of bins are computed for both x, and y.
      - if bins is a list of 2 integer ([Nx, Ny]),then Nx and Ny are number of bins for x, and y respectively.

    return_n_bins: bool, (default=False),
      - if True, number of bins are also returned.

    ignoreZero: bool, default=True 
       - if True, probabilities with zero value will be omited, before computations
       - It doesn't make much of difference


    Returns
    -------
    Hxy : Joint Entropy H(x,y)
    (Nx, Ny) : tuple,
      - number of bins for x and y, respectively (only if return_n_bins=True)

    References
    ----------
    * wikipedia 

    See Also
    --------
    entropy_cond : Conditional Entropy
    mutual_info : Mutual Information
    entropy_kld : KL-diversion Entropy
    entropy_cross : Cross Entropy

    Examples
    --------
    >>> #sp.entropy_joint
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import spkit as sp
    >>> X, fs, ch_names = sp.data.eeg_sample_14ch()
    >>> x,y1 = X[:,0],X[:,5]
    >>> H_xy1= sp.entropy_joint(x,y1)
    >>> print('Joint Entropy')
    >>> print(f'- H(x,y1) = {H_xy1}')
    Joint Entropy
    - H(x,y1) = 8.52651374518646
    """
    if is_discrete:
        Nx = len(np.unique(x))
        Ny = len(np.unique(y))
    else:
        # computing the optimal bin size using Freedman–Diaconis rule, or defined by bins
        if isinstance(bins,str):
            #_,bin_ws = np.histogram(x,bins=bins)
            #Nx = np.ceil((np.max(x)-np.min(x))/(bin_ws[1]-bin_ws[0])).astype(int)

            #_,bin_ws = np.histogram(y,bins=bins)
            #Ny = np.ceil((np.max(y)-np.min(y))/(bin_ws[1]-bin_ws[0])).astype(int)
            Nx = bin_width(x,method=bins)[1]
            Ny = bin_width(y,method=bins)[1]

        elif isinstance(bins,int):
            Nx,Ny = bins,bins
        elif isinstance(bins,list):
            assert len(bins)==2
            Nx,Ny = bins[0], bins[1]
        else:
            raise ValueError("Undefied way of 'bins', it should be 'str, int or list of 2 int")
    #Computing joint probability
    frq = np.histogram2d(x,y,bins=[Nx,Ny])[0]
    Prxy = frq/np.sum(frq)
    Prxy = Prxy[Prxy>0] if ignoreZero else Prxy + esp
    Hxy = - np.sum(Prxy*np.log(Prxy))
    if base!='e': Hxy /= np.log(base)
    if return_n_bins:
        return Hxy, (Nx,Ny)
    return Hxy

def entropy_cond(x,y,base=2,is_discrete=False,bins='fd',return_n_bins=False,verbose=False,ignoreZero=False):
    r"""Conditional Entropy :math:`H(X|Y)`

    .. math::  H(X|Y) = H(X,Y) - H(Y)

    .. math::  0 <= H(X|Y) <= H(X)


    Parameters
    ----------
    x,y :  1d-arrays

    is_discrete: bool, default=False.
      - If True, frequency of unique values are used to estimate H(x|y)

    base: base of log, default=2
      - decides the unit of entropy
      - if base=2 unit of entropy is in bits, base=e for nats, base=10 for bans

    bins: {str, int, [int, int]}. default='fd'
       - str decides the method of compute bin-width, bins='fd' is considered as optimal bin-width of a real-values signal/sequence.
       - check help(spkit.bin_width) for more Methods
       - if bins is an integer, then fixed number of bins are computed for both x, and y.
       - if bins is a list of 2 integer ([Nx, Ny]),then Nx and Ny are number of bins for x, and y respectively.

    return_n_bins: bool, (default=False)
      - if True, number of bins are also returned.

    ignoreZero: bool, default=False,
       - if true, probabilities with zero value will be omited, before computations
       - It doesn't make much of difference


    Returns
    -------
    Hx1y : Conditional Entropy H(x,y)
    (Nx, Ny) : tuple
      - number of bins for x and y, respectively (only if return_n_bins=True)
    
    See Also
    --------
    entropy_joint : Joint Entropy
    mutual_info : Mutual Information
    entropy_kld : KL-diversion Entropy
    entropy_cross : Cross Entropy

    Examples
    --------
    >>> #sp.entropy_cond
    >>> import numpy as np
    >>> import spkit as sp
    >>> X, fs, ch_names = sp.data.eeg_sample_14ch()
    >>> X = X - X.mean(1)[:, None]
    >>> x,y1 = X[:,0],X[:,5]
    >>> y2 = sp.add_noise(y1,snr_db=0)
    >>> H_x = sp.entropy(x)
    >>> H_x1y1= sp.entropy_cond(x,y1)
    >>> H_x1y2= sp.entropy_cond(x,y2)
    >>> print('Conditional Entropy')
    >>> print(f'- H(x|y1) = {H_x1y1}')
    >>> print(f'- H(x|y2) = {H_x1y2}')
    >>> print(f'- H(x) = {H_x}')
    Conditional Entropy
    - H(x|y1) = 4.096371831484375
    - H(x|y2) = 4.260323284620403
    - H(x) = 4.648381759654535
    """
    Hxy,(Nx,Ny) = entropy_joint(x,y,ignoreZero=ignoreZero,base=base,is_discrete=is_discrete,bins=bins,return_n_bins=True)
    Hy , Ny_i   = entropy(y,ignoreZero=ignoreZero,base=base,is_discrete=is_discrete,bins=Ny,return_n_bins=True)
    if verbose: print(Nx,Ny,Ny_i)
    Hx1y = Hxy-Hy
    if return_n_bins:
        return Hx1y, (Nx,Ny)
    return Hx1y

@deprecated("due to naming consistency, please use 'mutual_info' for updated/improved functionality. [spkit-0.0.9.7]")
def mutual_Info(x,y,base=2,is_discrete=False,bins='fd',return_n_bins=False,verbose=False,ignoreZero=False):
    '''Mututal Information :math:`I(X;Y)`

       .. math::  I(X;Y) = H(X)+H(Y)-H(X,Y)

       .. math::  I(X;Y) = H(X) - H(X|Y)

       .. math::  0 <= I(X;Y) <= min{ H(x), H(y) }

    Parameters
    ----------
    x,y :  1d-arrays

    is_discrete: bool, default=False. If True, frequency of unique values are used to estimate I(x;y)

    base: base of log: decides the unit of entropy
        if base=2 (default) unit of entropy is in bits, base=e, nats, base=10, bans

    bins: {str, int, [int, int]}. str decides the method of compute bin-width, bins='fd' (default) is considered as optimal bin-width of a real-values signal/sequence.
            check help(spkit.bin_width) for more Methods
            if bins is an integer, then fixed number of bins are computed for both x, and y.
            if bins is a list of 2 integer ([Nx, Ny]),then Nx and Ny are number of bins for x, and y respectively.

    return_n_bins: bool, (default=False), if True, number of bins are also returned.

    ignoreZero: if true, probabilities with zero value will be omited, before computations
                  : It doesn't make much of difference


    Returns
    -------
    I : Conditional Entropy I(x;y)
    (Nx, Ny) : number of bins for x and y, respectively (only if return_n_bins=True)

    See Also
    --------
    entropy_joint : Joint Entropy
    entropy_cond : Conditional Entropy
    mutual_info : Mutual Information
    entropy_kld : KL-diversion Entropy
    entropy_cross : Cross Entropy

    Examples
    --------
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

def mutual_info(x,y,base=2,is_discrete=False,bins='fd',return_n_bins=False,verbose=False,ignoreZero=False):
    r"""Mututal Information :math:`I(X;Y)`

       .. math::  I(X;Y) = H(X)+H(Y)-H(X,Y)

       .. math::  I(X;Y) = H(X) - H(X|Y)

       .. math::  0 <= I(X;Y) <= min\{ H(x), H(y)\}

    Parameters
    ----------
    x,y :  1d-arrays

    is_discrete: bool, default=False.
      - If True, frequency of unique values are used to estimate I(x,y)

    base: base of log,
      - decides the unit of entropy
      - if base=2 (default) unit of entropy is in bits, base=e, nats, base=10, bans

    bins: {str, int, [int, int]}.
        - str decides the method of compute bin-width, bins='fd' (default) is considered as optimal bin-width of a real-values signal/sequence.
        - check :func:`bin_width` for more Methods
        - if bins is an integer, then fixed number of bins are computed for both x, and y.
        - if bins is a list of 2 integer ([Nx, Ny]),then Nx and Ny are number of bins for x, and y respectively.

    return_n_bins: bool, (default=False)
        - if True, number of bins are also returned.

    ignoreZero: bool, default=False
        - if true, probabilities with zero value will be omited, before computations
        - It doesn't make much of difference


    Returns
    -------
    I : Mutual Information I(x,y)
    (Nx, Ny) : tuple of 2
     - number of bins for x and y, respectively (only if return_n_bins=True)

    See Also
    --------
    entropy_joint : Joint Entropy
    entropy_cond : Conditional Entropy
    entropy_kld : KL-diversion Entropy
    entropy_cross : Cross Entropy


    Examples
    --------
    >>> #sp.mutual_info
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import spkit as sp
    >>> np.random.seed(1)
    >>> x = np.random.randn(1000)
    >>> y1 = 0.1*x + 0.9*np.random.randn(1000)
    >>> y2 = 0.9*x + 0.1*np.random.randn(1000)
    >>> I_xy1 = sp.mutual_info(x,y1)
    >>> I_xy2 = sp.mutual_info(x,y2)
    >>> print(r'I(x,y1) = ',I_xy1, '\t| y1 /= e x')
    >>> print(r'I(x,y2) = ',I_xy2, '\t| y2 ~ x')
    >>> np.random.seed(None)
    I(x,y1) =  0.29196123466326007 	| y1 /= e x
    I(x,y2) =  2.6874431530714116 	| y2 ~ x
    """


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

def entropy_kld(x,y,base=2,is_discrete=False,bins='fd',verbose=False,pre_version=False,return_n_bins=False,esp=1e-10):
    r"""Cross Entropy Kullback–Leibler divergence :math:`H_{kl}(X,Y)`

    .. math::   H_{kl} =  \sum{Px*log(Px/Py)}
    
    Cross Entropy - Kullback–Leibler divergence

    Parameters
    ----------
    x,y :  1d-arrays

    is_discrete: bool, default=False.
      - If True, frequency of unique values are used to estimate H_{kl}(x,y)

    base: base of log, default=2
      - decides the unit of entropy
      - if base=2 unit of entropy is in bits, base=e for nats, base=10 for bans

    bins: {str, int, [int, int]}. default='fd'
        - str decides the method of compute bin-width, bins='fd' is considered as optimal bin-width of a real-values signal/sequence.
        - check :func:`bin_width` for more Methods
        - if bins is an integer, then fixed number of bins are computed for both x, and y.
        - if bins is a list of 2 integer ([Nx, Ny]),then Nx and Ny are number of bins for x, and y respectively.

    return_n_bins: bool, default=False
       - if True, number of bins are also returned.

    ignoreZero: bool, default=False
        - if true, probabilities with zero value will be omited, before computations
        - It doesn't make much of difference

    Returns
    -------
    H_xy : scaler, H_kl(x,y)
      - Cross entropy Kullback–Leibler divergence 
    (N, N) : tuple, 
       - number of bins for x and y, enforce to maximum of both (only if return_n_bins=True)

    See Also
    --------
    entropy_joint : Joint Entropy
    entropy_cond : Conditional Entropy
    mutual_info : Mutual Information
    entropy_kld : KL-diversion Entropy
    entropy_cross : Cross Entropy

    Examples
    --------
    >>> #sp.entropy_kld
    >>> import numpy as np
    >>> import spkit as sp
    >>> np.random.seed(1)
    >>> X, fs, ch_names = sp.data.eeg_sample_14ch()
    >>> X = X - X.mean(1)[:, None]
    >>> x,y1 = X[:,0],X[:,5]
    >>> y2 = sp.add_noise(y1,snr_db=0)
    >>> H_x = sp.entropy(x)
    >>> H_xy1= sp.entropy_kld(x,y1)
    >>> H_xy2= sp.entropy_kld(x,y2)
    >>> print('Cross Entropy - KL')
    >>> print(f'- H_kl(x,y1) = {H_xy1}')
    >>> print(f'- H_kl(x,y2) = {H_xy2}')
    >>> print(f'- H(x) = {H_x}')
    >>> np.random.seed(None)
    Cross Entropy - KL
    - H_kl(x,y1) = 0.37227231154384194
    - H_kl(x,y2) = 1.8806537173845745
    - H(x) = 4.648381759654535
    """

    if pre_version:
        # TOBE deprecated in FUTURE VERSION
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
        PrX += esp
        PrY += esp

        H_kl  = np.sum(PrX*np.log(PrX/PrY))

    else:
        if is_discrete:
            Nx = len(np.unique(x))
            Ny = len(np.unique(y))
        else:
            # computing the optimal bin size using Freedman–Diaconis rule, or defined by bins
            if isinstance(bins,str):
                #_,bin_ws = np.histogram(x,bins=bins)
                #Nx = np.ceil((np.max(x)-np.min(x))/(bin_ws[1]-bin_ws[0])).astype(int)

                #_,bin_ws = np.histogram(y,bins=bins)
                #Ny = np.ceil((np.max(y)-np.min(y))/(bin_ws[1]-bin_ws[0])).astype(int)
                Nx = bin_width(x,method=bins)[1]
                Ny = bin_width(y,method=bins)[1]

            elif isinstance(bins,int):
                Nx,Ny = bins,bins
            elif isinstance(bins,list):
                assert len(bins)==2
                Nx,Ny = bins[0], bins[1]
            else:
                raise ValueError("Undefied way of 'bins', it should be 'str, int or list of 2 int")

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
        PrX += esp
        PrY += esp

        H_kl  = np.sum(PrX*np.log(PrX/PrY))

    if base !='e': H_kl = H_kl/np.log(base)

    if return_n_bins:
        return H_kl, (N,N)
    return H_kl

def entropy_cross(x,y,base=2,is_discrete=False,bins='fd',verbose=False,pre_version=False,return_n_bins=False,esp=1e-10):
    r"""Cross Entropy :math:`H_{xy}(X,Y)`

    .. math::   H_{xy} = - \sum{Px*log(Py)}

    Parameters
    ----------
    x,y :  1d-arrays

    is_discrete: bool, default=False. 
      - If True, frequency of unique values are used to estimate :math:`H_{xy}(X,Y)`

    bins: {str, int, [int, int]}. default='fd'
        - str decides the method of compute bin-width, bins='fd' is considered as optimal bin-width of a real-values signal/sequence.
        - check :func:`bin_width` for more Methods
        - if bins is an integer, then fixed number of bins are computed for both x, and y.
        - if bins is a list of 2 integer ([Nx, Ny]),then Nx and Ny are number of bins for x, and y respectively.

    return_n_bins: bool, default=False
       - if True, number of bins are also returned.

    ignoreZero: bool, default=False
        - if true, probabilities with zero value will be omited, before computations
        - It doesn't make much of difference

    Returns
    -------
    H_xy : Cross Entropy H_(x,y)
    (N, N) : number of bins for x and y, enforce to maximum of both (only if return_n_bins=True)

    Notes
    -----
    spkit -todo

    See Also
    --------
    entropy_joint : Joint Entropy
    entropy_cond : Conditional Entropy
    mutual_info : Mutual Information
    entropy_kld : KL-diversion Entropy

    Examples
    --------
    >>> #sp.entropy_cross
    >>> import numpy as np
    >>> import spkit as sp
    >>> np.random.seed(1)
    >>> X, fs, ch_names = sp.data.eeg_sample_14ch()
    >>> X = X - X.mean(1)[:, None]
    >>> x,y1 = X[:,0],X[:,5]
    >>> y2 = sp.add_noise(y1,snr_db=0)
    >>> H_x = sp.entropy(x)
    >>> H_xy1= sp.entropy_cross(x,y1)
    >>> H_xy2= sp.entropy_cross(x,y2)
    >>> print('Cross Entropy')
    >>> print(f'- H_(x,y1) = {H_xy1}')
    >>> print(f'- H_(x,y2) = {H_xy2}')
    >>> print(f'- H(x) = {H_x}')
    >>> np.random.seed(None)
    Cross Entropy
    - H_(x,y1) = 5.020654071198377
    - H_(x,y2) = 6.529035477039111
    - H(x) = 4.648381759654535
    """

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
        PrX += esp
        PrY += esp

        H_cross  = -np.sum(PrX*np.log(PrY))
    else:
        if is_discrete:
            Nx = len(np.unique(x))
            Ny = len(np.unique(y))
        else:
            # computing the optimal bin size using Freedman–Diaconis rule, or defined by bins
            if isinstance(bins,str):
                Nx = bin_width(x,method=bins)[1]
                Ny = bin_width(y,method=bins)[1]

            elif isinstance(bins,int):
                Nx,Ny = bins,bins
            elif isinstance(bins,list):
                assert len(bins)==2
                Nx,Ny = bins[0], bins[1]
            else:
                raise ValueError("Undefied way of 'bins', it should be 'str, int or list of 2 int")

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
        PrX += esp
        PrY += esp

        H_cross = -np.sum(PrX*np.log(PrY))

    if base !='e': H_cross = H_cross/np.log(base)

    if return_n_bins:
        return H_cross, (N,N)
    return H_cross

def entropy_spectral(x,fs,method='fft',alpha=1,base=2,normalize=True,axis=-1,nperseg=None,esp=1e-10):
    r"""Spectral Entropy :math:`H_f(X)`

    Measure of the uncertainity of frequency components in a Signal
    For Uniform distributed signal and Gaussian distrobutated signal, their entropy is quite different,
    but in spectral domain, both have same entropy
    
    .. math::
      H_f(x) = H(F(x))
      
    :math:`F(x)` - FFT of x

    Parameters
    ----------
    x: 1d array
    fs = sampling frequency
    method: 'fft' use periodogram and 'welch' uses wekch method
    base: base of log, 2 for bit, 10 for nats

    Returns
    -------
    H_fx: scalar
     - Spectral entropy

    See Also
    --------
    entropy : Entropy
    entropy_sample : Sample Entropy
    entropy_approx : Approximate Entropy
    dispersion_entropy : Dispersion Entropy
    entropy_svd :  SVD Entropy
    entropy_permutation :  Permutation Entropy
    entropy_differential : Differential Entropy

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import spkit as sp
    >>> np.random.seed(1)
    >>> fs = 1000
    >>> t = np.arange(1000)/fs
    >>> 1 = np.random.randn(len(t))
    >>> x2 = np.cos(2*np.pi*10*t)+np.cos(2*np.pi*30*t)+np.cos(2*np.pi*20*t)
    >>> Hx1 = sp.entropy(x1)
    >>> Hx2 = sp.entropy(x2)
    >>> Hx1_se = sp.entropy_spectral(x1,fs=fs,method='welch',normalize=False)
    >>> Hx2_se = sp.entropy_spectral(x2,fs=fs,method='welch',normalize=False)
    >>> print('Spectral Entropy:')
    >>> print(r' - H_f(x1) = ',Hx1_se,)
    >>> print(r' - H_f(x1) = ',Hx2_se)
    >>> print('Shannon Entropy:')
    >>> print(r' - H_f(x1) = ',Hx1)
    >>> print(r' - H_f(x1) = ',Hx2)
    >>> print('-')
    >>> Hx1_n = sp.entropy(x1,normalize=True)
    >>> Hx2_n = sp.entropy(x2,normalize=True)
    >>> Hx1_se_n = sp.entropy_spectral(x1,fs=fs,method='welch',normalize=True)
    >>> Hx2_se_n = sp.entropy_spectral(x2,fs=fs,method='welch',normalize=True)
    >>> print('Spectral Entropy (Normalised)')
    >>> print(r' - H_f(x1) = ',Hx1_se_n,)
    >>> print(r' - H_f(x1) = ',Hx2_se_n,)
    >>> print('Shannon Entropy (Normalised)')
    >>> print(r' - H_f(x1) = ',Hx1_n)
    >>> print(r' - H_f(x1) = ',Hx2_n)
    >>> np.random.seed(None)
    >>> plt.figure(figsize=(11,4))
    >>> plt.subplot(121)
    >>> plt.plot(t,x1,label='x1: Gaussian Noise',alpha=0.8)
    >>> plt.plot(t,x2,label='x2: Sinusoidal',alpha=0.8)
    >>> plt.xlim([t[0],t[-1]])
    >>> plt.xlabel('time (s)')
    >>> plt.ylabel('x1')
    >>> plt.legend(bbox_to_anchor=(1, 1.2),ncol=2,loc='upper right')
    >>> plt.subplot(122)
    >>> label1 = f'x1: Gaussian Noise \n H(x): {Hx1.round(2)}, H_f(x): {Hx1_se.round(2)}'
    >>> label2 = f'x2: Sinusoidal \n H(x): {Hx2.round(2)}, H_f(x): {Hx2_se.round(2)}'
    >>> P1x,f1q = sp.periodogram(x1,fs=fs,show_plot=True,label=label1)
    >>> P2x,f2q = sp.periodogram(x2,fs=fs,show_plot=True,label=label2)
    >>> plt.legend(bbox_to_anchor=(0.4, 0.4))
    >>> plt.grid()
    >>> plt.tight_layout()
    >>> plt.show()
    Spectral Entropy:
    - H_f(x1) =  6.889476342103717
    - H_f(x1) =  2.7850662938305786
    Shannon Entropy:
    - H_f(x1) =  3.950686888274901
    - H_f(x1) =  3.8204484006660255
    -
    Spectral Entropy (Normalised)
    - H_f(x1) =  0.9826348642134711
    - H_f(x1) =  0.3972294995395931
    Shannon Entropy (Normalised)
    - H_f(x1) =  0.8308686349524237
    - H_f(x1) =  0.8445663920995877
    """
    x = np.asarray(x)
    # Compute and normalize power spectrum
    if method == 'fft':
        _, psd = periodogram(x,fs,axis=axis)
    elif method == 'welch':
        _, psd = welch(x, fs, nperseg=nperseg,axis=axis)

    psd_norm = psd / psd.sum(axis=axis, keepdims=True)
    se = -(psd_norm * np.log(psd_norm+esp)).sum(axis=axis)
    se /=np.log(base)
    if normalize:
        se /= (np.log(psd_norm.shape[axis]+esp)/np.log(base))
    return se

def entropy_sample(x, m, r):
    r"""Sample Entropy :math:`SampEn(X)` or :math:`H_{se}(X)`

    Sample entropy is more suited for temporal source, (non-IID), such as physiological signals, or signals in general.
    Sample entropy like Approximate Entropy ( :func:`entropy_approx`) measures the complexity of a signal by extracting the pattern of m-symbols.
    `m` is also called as embedding dimension. `r` is the tolarance here, which determines the two patterns to be same if 
    their maximum absolute difference is than `r`.

    Sample Entropy avoide the self-similarity between patterns as it is considered in Approximate Entropy

    Parameters
    ----------
    X : 1d-array
       as signal
    m : int
       embedding dimension, usual value is m=3, however, it depends on the signal. 
        If signal has high sampling rate, which means a very smooth signal, then any consequitive 3 samples will be quit similar.
    r : tolarance
       usual value is r = 0.2*std(x)

    Returns
    -------
    SampEn :  float -  sample entropy value

    References
    ----------
    * wikipedia https://en.wikipedia.org/wiki/Sample_entropy

    Notes
    -----
    Computationally and otherwise, Sample Entropy is considered better than Approximate Entropy

    See Also
    --------
    entropy_approx : Approximate Entropy
    dispersion_entropy : Dispersion Entropy
    entropy_spectral : Spectral Entropy
    entropy_svd :  SVD Entropy
    entropy_permutation :  Permutation Entropy
    entropy_differential : Differential Entropy
    entropy : Entropy

    Examples
    --------
    >>> #sp.entropy_sample
    >>> import numpy as np
    >>> import spkit as sp
    >>> t = np.linspace(0,2,200)
    >>> x1 = np.sin(2*np.pi*1*t) + 0.1*np.random.randn(len(t))  # less noisy
    >>> x2 = np.sin(2*np.pi*1*t) + 0.5*np.random.randn(len(t))  # very noisy
    >>> #Sample Entropy
    >>> H_x1 = sp.entropy_sample(x1,m=3,r=0.2*np.std(x1))
    >>> H_x2 = sp.entropy_sample(x2,m=3,r=0.2*np.std(x2))
    >>> print('Sample entropy')
    >>> print('Entropy of x1: SampEn(x1)= ',H_x1)
    >>> print('Entropy of x2: SampEn(x2)= ',H_x2)
    Sample entropy
    Entropy of x1: SampEn(x1)=  0.6757312057041359
    Entropy of x2: SampEn(x2)=  1.6700625342505353
    """

    N = len(x)
    B = 0.0
    A = 0.0
    # Split time series and save all templates of length m
    xmi = np.array([x[i : i + m] for i in range(N - m)])
    xmj = np.array([x[i : i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([x[i : i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])
    return -np.log(A / B)

def entropy_approx(x, m, r):
    """Approximate Entropy :math:`ApproxEn(X)` or :math:`H_{ae}(X)`

    Approximate entropy is more suited for temporal source, (non-IID), such as physiological signals, or signals in general.
    Approximate entropy like Sample Entropy ( :func:`entropy_sample`) measures the complexity of a signal by extracting the pattern of m-symbols.
    `m` is also called as embedding dimension. `r` is the tolarance here, which determines the two patterns to be same if 
    their maximum absolute difference is than `r`.

    Parameters
    ----------
    X : 1d-array
       as signal
    m : int
       embedding dimension, usual value is m=3, however, it depends on the signal. 
        If signal has high sampling rate, which means a very smooth signal, then any consequitive 3 samples will be quit similar.
    r : tolarance
       usual value is r = 0.2*std(x)

    Returns
    -------
    ApproxEn :  float
      - approximate entropy value

    References
    ----------
    * wikipedia -  https://en.wikipedia.org/wiki/Approximate_entropy

    See Also
    --------
    entropy : Entropy
    entropy_sample : Sample Entropy
    dispersion_entropy : Dispersion Entropy
    entropy_spectral : Spectral Entropy
    entropy_svd :  SVD Entropy
    entropy_permutation :  Permutation Entropy
    entropy_differential : Differential Entropy


    Examples
    --------
    >>> import numpy as np
    >>> import spkit as sp
    >>> t = np.linspace(0,2,200)
    >>> x1 = np.sin(2*np.pi*1*t) + 0.1*np.random.randn(len(t))  # less noisy
    >>> x2 = np.sin(2*np.pi*1*t) + 0.5*np.random.randn(len(t))  # very noisy
    >>> #Approximate Entropy
    >>> H_x1 = sp.entropy_approx(x1,m=3,r=0.2*np.std(x1))
    >>> H_x2 = sp.entropy_approx(x2,m=3,r=0.2*np.std(x2))
    >>> print('Approximate entropy')
    >>> print('Entropy of x1: ApproxEn(x1)= ',H_x1)
    >>> print('Entropy of x2: ApproxEn(x2)= ',H_x2)
    Approximate entropy
    Entropy of x1: ApproxEn(x1)=  0.5661144425748899
    Entropy of x2: ApproxEn(x2)=  0.20696275451476875
    """

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        xm = [[x[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in xm if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in xm]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))
    
    N = len(x)
    return _phi(m) - _phi(m + 1)

def TD_Embed(x, order=3, delay=1):
    """Time delay Embedding Matrix

    Extracting Embeddings

    Parameters
    ----------
    x : 1d-array
      - time series of shape (n,)
    order : int, default=3
      - Embedding dimension (order).
    delay : int, default=1
       - Delay.

    Returns
    -------
    X: Embedded Matrix: ndarray
     - Embedded time-series, of shape (n - (order - 1) * delay, order)
    
    See Also
    --------
    signal_embeddings: Signal Embeddings
    create_multidim_space_signal: Creating Multi-Dimensional Signal Space

    Examples
    --------
    #sp.TD_Embed
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    X, fs, ch_names = sp.data.eeg_sample_14ch()
    x1 = X[500:700,0]
    t = np.arange(len(x1))/fs
    Xe = sp.TD_Embed(x1,order=3,delay=2)

    plt.figure(figsize=(10,4))
    plt.subplot(211)
    plt.plot(t,x1)
    plt.xlim([t[0],t[-1]])
    plt.ylabel('x')
    plt.xticks([])
    plt.subplot(212)
    plt.plot(t[:Xe.shape[0]],Xe)
    plt.xlim([t[0],t[-1]])
    plt.ylabel('Embeddings')
    plt.xlabel('time (s)')
    idx = 47
    plt.axvline(t[idx],color='k',lw=1,alpha=0.5)
    plt.plot([t[idx],t[idx],t[idx]],Xe[idx],'o',ms=3)
    plt.plot(t[[idx,idx+5,idx+10]],Xe[idx]+50,ls='-',marker='.',color='k')
    plt.text(t[idx+10],80,' embedding')
    plt.tight_layout()
    plt.show()

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
    r"""Singular Value Decomposition Entropy :math:`H_{\Sigma}(X)`

    Singular Value Decomposition Entropy

    Parameters
    ----------
    x : 1d-array, shape (n,)
      - input signal
    order : int, default=3
      - Embedding dimension (order).
    delay : int, default=1
      - Delay.
    base: scalar>0, deafult=2
      - base of log, 2, 10, 'e'
    normalize: bool, default=False
      - if True, Hx is normalised
    
    Returns
    -------
    H_svd: scalar,
      -  Singular Value Decomposition Entropy

    References
    ----------
    * wikipedia 

    See Also
    --------
    entropy : Entropy
    entropy_sample : Sample Entropy
    entropy_approx : Approximate Entropy
    dispersion_entropy : Dispersion Entropy
    entropy_spectral : Spectral Entropy
    entropy_permutation :  Permutation Entropy
    entropy_differential : Differential Entropy

    Examples
    --------
    >>> import numpy as np
    >>> import spkit as sp
    >>> t = np.linspace(0,2,200)
    >>> x1 = np.sin(2*np.pi*1*t) + 0.01*np.random.randn(len(t))  # less noisy
    >>> x2 = np.sin(2*np.pi*1*t) + 0.5*np.random.randn(len(t))  # very noisy
    >>> #Entropy SVD
    >>> H_x1 = sp.entropy_svd(x1,order=3, delay=1)
    >>> H_x2 = sp.entropy_svd(x2,order=3, delay=1)
    >>> print('Entropy SVD')
    >>> print(r'Entropy of x1: H_s(x1) = ',H_x1)
    >>> print(r'Entropy of x2: H_s(x2) = ',H_x2)
    Entropy SVD
    Entropy of x1: H_s(x1) =  0.34210072132884417
    Entropy of x2: H_s(x2) =  1.394331263550738
    """

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
    r"""Permutation Entropy :math:`H_{\pi}(X)`
    
    Permutation Entropy extracts the patterns as order of embeddings, and compute the entropy of the distribuation
    of the patterns. 
    
    The order of embeddings is the sorting order. For example, pattern of embedding e1 = [1,2,-2], is 
    same as pattern of embedding e2 = [1,20,-5].

    Parameters
    ----------
    x : 1d-array, shape (n,)
      - input signal
    order : int, default=3
      - Embedding dimension (order).
    delay : int, default=1
      - Delay.
    base: scalar>0, deafult=2
      - base of log, 2, 10, 'e'
    normalize: bool, default=False
      - if True, Hx is normalised
    
    Returns
    -------
    H_pi: scalar,
      -  Permutation Entropy

    References
    ----------
    * Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a natural complexity measure 
      for time series." Physical review letters 88.17 (2002): 174102.


    See Also
    --------
    entropy : Entropy
    entropy_sample : Sample Entropy
    entropy_approx : Approximate Entropy
    dispersion_entropy : Dispersion Entropy
    entropy_spectral : Spectral Entropy
    entropy_svd :  SVD Entropy
    entropy_differential : Differential Entropy

    Examples
    --------
    >>> import numpy as np
    >>> import spkit as sp
    >>> t = np.linspace(0,2,200)
    >>> x1 = np.sin(2*np.pi*1*t) + 0.01*np.random.randn(len(t))  # less noisy
    >>> x2 = np.sin(2*np.pi*1*t) + 0.5*np.random.randn(len(t))  # very noisy
    >>> H_x1 = sp.entropy_permutation(x1,order=3, delay=1)
    >>> H_x2 = sp.entropy_permutation(x2,order=3, delay=1)
    >>> print('Permutation Entropy ')
    >>> print('Entropy of x1: H_p(x1) = ',H_x1)
    >>> print('Entropy of x2: H_p(x2) = ',H_x2)
    Permutation Entropy 
    Entropy of x1: H_p(x1) =  1.5156504111997058
    Entropy of x2: H_p(x2) =  2.556358399367036
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

@deprecated("due to naming consistency, please use 'hist_plot' for updated/improved functionality. [spkit-0.0.9.7]")
def HistPlot(x,norm=False,show=False,):
    '''plot histogram with optimal number of bins (fd).

    Parameters
    ----------
        norm:  bool - if norm = True, plot is probability disctribution, else frequency
    '''
    frq,bins = np.histogram(x,bins='fd')
    if norm: frq = frq/np.sum(frq)
    plt.bar(bins[:-1],frq,width=0.8*(bins[1]-bins[0]),alpha=0.5)
    #plt.plot(bins[:-1],frq,'--k',lw=0.7)
    if show: plt.show()

def hist_plot(x,norm=False,bins='fd',show=False,):
    r"""Plot histogram with optimal number of bins (FD).

    Parameters
    ----------
        norm:  bool - if norm = True, plot is probability disctribution, else frequency
        bins:  str, int. if str method to compute number of bins, if int, number of bins

    Returns
    -------
    display: histogram plot

    Examples
    --------
    #sp.hist_plot
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    X, fs, ch_names = sp.data.eeg_sample_14ch()
    x1 = X[:500,0]
    sp.hist_plot(x1)
    x2 = X[800:1200,0]
    sp.hist_plot(x2)
    plt.xlabel('x')
    plt.ylabel('count')
    plt.show()
    """

    frq,bins = np.histogram(x,bins=bins)
    if norm: frq = frq/np.sum(frq)
    plt.bar(bins[:-1],frq,width=0.8*(bins[1]-bins[0]),alpha=0.5)
    if show: plt.show()

def Mu_law(x,Mu=255,companding=True):
    r"""μ-Law for companding and expanding (Non-linear mapping)
    
    μ-Law is one of the ways to map gaussian, or laplacian distribuated signal to uniformly distributed one
    - for companding - smaller amplitude values are stretched (since they have high-probabilty)
      and large amplitutde values are compressed (since they have low probabiltity)
    - for expending - it reverts the mapping

    for -1 ≤ x ≤ 1
    
    Companding:

        .. math::  y(n) = sign(x)* \frac{log(1 + μ|x|)}{log(1 + μ)}

    expanding:

        .. math::  y(n) = sign(x)* \frac{((1 + μ)**|x| - 1)}{μ}

    - An alternative to A-Law
    - The μ-law algorithm provides a slightly larger dynamic range than the A-law
      at the cost of worse proportional distortions for small signals.

    
    Parameters
    ----------
    x  : 1d (or nd) array 
      - array of signal, must be  -1 ≤ x ≤ 1
    Mu : scalar (0≤Mu)
      - factor for companding-expanding mapping
      - ~0: identity function, higher value of Mu will stretch and compress more
      - use 1e-5 for Mu=0 to avoid 'zero-division' error
    
    companding: bool, default=True
      - if True, companding is applied, else expanding

    Returns
    -------
    y: mapped output of same shape as x

    References
    ----------
    * wikipedia -  https://en.wikipedia.org/wiki/M-law_algorithm

    See Also
    --------
    A_law : A-Law for companding and expanding


    Examples
    --------
    #sp.Mu_law
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs,_ = sp.data.ecg_sample_12leads(sample=3)
    x = x[1000:3000,0]
    x = x/(np.abs(x).max()*1.01)
    x = x - np.mean(x)
    x = x/(np.abs(x).max()*1.01)
    t = np.arange(len(x))/fs
    y1 = sp.Mu_law(x.copy(),Mu=5,companding=True)
    y2 = sp.Mu_law(x.copy(),Mu=255,companding=True)
    x2 = sp.Mu_law(y2.copy(),Mu=255,companding=False)
    plt.figure(figsize=(12,7))
    plt.subplot(211)
    plt.plot(t,x,label=f'x')
    plt.plot(t,y1,label=f'y1')
    plt.plot(t,y2,label=f'y2')
    plt.xlim([t[0],t[-1]])
    plt.xlabel('time (s)')
    plt.legend()
    plt.title('$\mu$-Law')
    plt.grid()
    plt.subplot(223)
    sp.hist_plot(x)
    sp.hist_plot(y1)
    sp.hist_plot(y2)
    plt.title('Histogram')
    plt.subplot(224)
    idx = np.argsort(x)
    plt.plot(x[idx],y1[idx],color='C1',label='x-to-y1 ($\mu=5$)')
    plt.plot(x[idx],y2[idx],color='C2',label='x-to-y2 ($\mu=255$)')
    plt.plot(y2[idx],x[idx],color='C3',label='y2-to-x2 ($\mu=255$)')
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.xlabel('x, y2')
    plt.ylabel('y1, y2, x2')
    plt.title(r'$\mu$-Law')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
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
    r"""A-Law for companding and expanding (Non-linear mapping)


    A-Law for companding and expanding
    - A-Law is one of the ways to map gaussian, or laplacian distribuated signal to uniformly distributed one
    - for companding - smaller amplitude values are stretched (since they have high-probabilty)
      and large amplitutde values are compressed (since they have low probabiltity)
    - for expending - it reverts the mapping

    for -1 ≤ x ≤ 1 and 1 < A

    Companding:
        .. math::

            y(n) &= sign(x)* \frac{A|x|}{(1 + log(A))}               if   |x|<1/A
            
                 &= sign(x)* \frac{((1 + log(A|x|))} {(1 + log(A)))}   else

    expanding:

        .. math::
           
           y(n) = sign(x)* \frac{|x|(1+log(A))} {A}                 if   |x|<1/(1+log(A))

            = sign(x)*  \frac{(exp(-1 + |x|(1+log(A))))} {A}       else


            

    - An alternative to μ-Law
    - The μ-law algorithm provides a slightly larger dynamic range than the A-law
      at the cost of worse proportional distortions for small signals.


    Parameters
    ----------
    x  - 1d (or nd) array of signal, must be  -1 ≤ x ≤ 1
    A  - scalar (1≤A) - factor for companding-expanding mapping
       - 1: identity function, higher value of A will stretch and compress more
    companding: (bool) -  if True, companding is applied, else expanding

    Returns
    -------
    y - mapped output of same shape as x

    References
    ----------
    * wikipedia -  https://en.wikipedia.org/wiki/A-law_algorithm

    See Also
    --------
    A_law : A-Law for companding and expanding


    Examples
    --------
    #sp.A_law
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs,_ = sp.data.ecg_sample_12leads(sample=3)
    x = x[1000:3000,0]
    x = x/(np.abs(x).max()*1.01)
    x = x - np.mean(x)
    x = x/(np.abs(x).max()*1.01)
    t = np.arange(len(x))/fs
    y1 = sp.A_law(x.copy(),A=5,companding=True)
    y2 = sp.A_law(x.copy(),A=255,companding=True)
    x2 = sp.A_law(y2.copy(),A=255,companding=False)
    plt.figure(figsize=(12,7))
    plt.subplot(211)
    plt.plot(t,x,label=f'x')
    plt.plot(t,y1,label=f'y1')
    plt.plot(t,y2,label=f'y2')
    plt.xlim([t[0],t[-1]])
    plt.xlabel('time (s)')
    plt.legend()
    plt.title('$A$-Law')
    plt.grid()
    plt.subplot(223)
    sp.hist_plot(x)
    sp.hist_plot(y1)
    sp.hist_plot(y2)
    plt.title('Histogram')
    plt.subplot(224)
    idx = np.argsort(x)
    plt.plot(x[idx],y1[idx],color='C1',label='x-to-y1 (A=5)')
    plt.plot(x[idx],y2[idx],color='C2',label='x-to-y2 (A=255)')
    plt.plot(y2[idx],x[idx],color='C3',label='y2-to-x2 (A=255)')
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.xlabel('x, y2')
    plt.ylabel('y1, y2, x2')
    plt.title(r'$A$-Law')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
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
    r"""Compute bin width for histogram, using different methods


    Compute bin width using different methods

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


    Parameters
    ----------
    x :  1d-array or (n-d array)
    method : method to compute bin width and number of bins

    Returns
    -------
    bw : bin width
    k  : number of bins

    References
    ----------
    * wikipedia
    
    
    Notes
    -----

    See Also
    --------
    hist_plot: # Histogram plot with optimal number of bins

    Examples
    --------
    >>> import numpy as np
    >>> import spkit as sp
    >>> np.random.seed(1)
    >>> t = np.linspace(0,2,200)
    >>> x1 = np.cos(2*np.pi*1*t) + 0.01*np.random.randn(len(t))  # less noisy
    >>> x2 = np.cos(2*np.pi*1*t) + 0.9*np.random.randn(len(t))  # very noisy
    >>> bw1, k1 = sp.bin_width(x1, method='fd')
    >>> bw2, k2 = sp.bin_width(x2, method='fd')
    >>> print(r'Optimal: bin-width of x1 = ',bw1,'\t Number of bins = ',k1)
    >>> print(r'Optimal: bin-width of x2 = ',bw2,'\t Number of bins = ',k2)
    """
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

def quantize_FD(x,scale=1,min_bins=2,keep_amp_scale=True,bin_method='fd'):
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
    #--------------------------
    #bw = binSize_FD(x0)*scale
    bw, _ = bin_width(x0,method=bin_method)  # replacing binSize_FD
    bw *= scale
    #--------------------------
    nbins = np.clip(np.ceil((xmax-xmin)/bw).astype(int),min_bins,None)
    x0 = (x0 - xmin)/(xmax-xmin)
    y = np.round(x0*(nbins-1))
    y /=(nbins-1)
    y = np.clip(y,0,1)
    if keep_amp_scale: y = y*(xmax-xmin) + xmin
    return y, nbins

@deprecated("due to naming consistency, please use 'quantize_signal' for updated/improved functionality. [spkit-0.0.9.7]")
def Quantize(x,A=None, Mu=None,cdf_map=False,nbins=None,min_bins=2,scale=1,bin_method='fd'):
    '''Quantize given signal x


    Inputs
    -----
    - x: 1d-signal
    - non-linear mappings:
        A - for A-Law mapping
        Mu - for Mu-Law mapping
        cdf_map - for CDF (cummulative distribution function) mapping
    - nbins :  number of bins, if None, FD is used to compute number of bins
    - scale : float, scaling factor for the bin-width. deafult 1, means no scalling, 2 means, 2 the size of bin width computated.
    - min_bins: int, minimum number of bins

    Outputs
    ------
    - y: 1d-singal, qualtized.

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
        #bw = binSize_FD(x0)*scale
        bw, _ = bin_width(x0,method=bin_method)
        bw *= scale
        nbins = np.clip(np.ceil((xmax-xmin)/bw).astype(int),min_bins,None)
    x0 = (x0 - xmin)/(xmax-xmin)
    y = np.round(x0*(nbins-1))
    y /=(nbins-1)
    y = np.clip(y,0,1)
    y = y*(xmax-xmin) + xmin
    return y

def quantize_signal(x,n_levels=None,A=None,Mu=None,cdf_map=False,keep_range=True,bin_method='fd',bin_scale=1,min_levels=2):
    r"""Quantize signal into discreet levels

    Quantize signal into discreet levels

    Before quantization, signal is equalise using either of three methods
    (1) A-Law, (2) Mu-Law, or (3) Cummulative Distribution Function

    Parameters
    ----------
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
     - if not None, A-Law compading is applied to equalise the distribution
     - A>0, A=1, means identity funtion

    Mu: int,>=0, default=None
     - if not None, Mu-Law compading is applied to equalise the distribution
     - Mu>=0, Mu=0 mean identity funtion

    cdf_map: bool, default=False
     - If true, CDF mapping is applied to equalise the distribution


    keep_range: bool, default=True
     - If True, return quantized signal is rescaled to its original range, else returned signal is in range of 0 to 1


    Returns
    -------
    y: Quantized signal, same size as x
      - if keep_range=True, y has same range as x, else 0 to 1

    y_int: Quantized signal as integer level, ranges from 0 to n_levels-1
      - Useful to map signal in post-processing

    References
    ----------
    * wikipedia
    
    
    Notes
    -----

    See Also
    --------
    cdf_mapping: CDF Mapping
    A_law: A - Law, Nonlinear mapping
    Mu_law: Mu - Law, Nonlinear mapping

    Examples
    --------
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs,_ = sp.data.ecg_sample_12leads(sample=3)
    x = x[:1000,0]
    t = np.arange(len(x))/fs
    y1, yint = sp.quantize_signal(x.copy(),n_levels=5)
    y2, yint = sp.quantize_signal(x.copy(),n_levels=11)
    y3, yint = sp.quantize_signal(x.copy(),n_levels=31)
    m1 = np.mean((x-y1)**2)
    m2 = np.mean((x-y2)**2)
    m3 = np.mean((x-y3)**2)
    plt.figure(figsize=(12,3))
    plt.plot(t,x,alpha=0.8 ,label='x')
    plt.plot(t,y1,alpha=0.8,label=f'y1: L=5,  MSE={m1:,.4f}')
    plt.plot(t,y2,alpha=0.8,label=f'y2: L=11, MSE={m2:,.4f}')
    plt.plot(t,y3,alpha=0.8,label=f'y3: L=31, MSE={m3:,.4f}')
    plt.xlim([t[0],t[-1]])
    plt.xlabel('time (s)')
    plt.ylabel('x')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    """
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
        #bw = binSize_FD(x0)*bin_scale
        bw, _ = bin_width(x0,method=bin_method)
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
    ''' Analyse Entropy-Venn Diagrams of X and Y

    Analyse Entropy-Venn Diagrams of X and Y

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
    mi = mutual_info(x,y)
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



# def binSize_FD(x):
#     ''' Compute bin width using Freedman–Diaconis rule
#
#
#     Compute bin width using Freedman–Diaconis rule
#     ----------------------------------------------
#     bin_width = 2*IQR(x)/(n**(1/3))
#
#     input
#     -----
#     x: 1d-array
#
#     output
#     ------
#     bw : bin width
#     '''
#     IQR = scipystats.iqr(x)
#     bw =  2.0*IQR/(len(x)**(1/3))
#     return bw
