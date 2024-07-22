'''
Advance Information Theory techniques
--------------------------------
Author @ Nikesh Bajaj
updated on Date: 27 March 2023, Version : 0.0.5
updated on Date: 1 Jan 2022, Version : 0.0.1
Github :  https://github.com/Nikeshbajaj/spkit
Contact: n.bajaj@qmul.ac.uk | n.bajaj@imperial.ac.uk
'''

from __future__ import absolute_import, division, print_function
name = "Signal Processing toolkit | Advance - Information Theory"
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
from .infotheory import *

def low_resolution(x, scale):
    """
    Reducing the time-resolution of signal : x
    ------------------------------------------

    input
    -----
    x : 1d-array of shape=(n,)
    scale: (int) downsampled by factor of scale and averaged around removed samples

    output
    ------
    x_low: shape=(n/scale,) low resultioned signal
    """
    N = int(np.fix(len(x) / scale))
    x_low = np.array([np.mean(x[i * scale:(i + 1) * scale]) for i in range(N)])
    return x_low

def cdf_mapping(x):
    """
    Map the signal x to y from into CDF of x, y will be uniformly
    disctribuated anf ranges from 0 to 1.
    CDF: Cumulative Distribution Function

            y(n) = P(X<=x(n))
    input
    -----
    x : 1d-array of shape=(n,)

    output
    ------
    x_cdf: cumulative distribution function of x

    """
    N   = len(x)
    mu  = np.mean(x)
    sig = np.std(x) if np.std(x) != 0 else 0.001
    ncdf  = scipystats.norm(loc=mu, scale=sig)
    x_cdf = ncdf.cdf(x)
    return x_cdf


def signal_embeddings(x,is_discrete=False,emb_dim=2,delay=1,n_levels=10,scale=1,mapping_type='cdf',A=100,Mu=100,encode=True):
    r"""

    Extract Embeddings from given signal
    ------------------------------------

    Parameters
    ----------
    x :  1d-array, signal
    is_discrete: bool, default=False,
                if False, input signal is discretized by given setting, (n_levels,scale,mapping_type,A,Mu)
                else considered it as discrete and create dictionary of embeddigns

    emb_dim:, int>=1, default 2
    delay:, int>=1 default=1,

    Return
    ------


    embeddings_count: dict, map of each unique pattern/embeding to number of times it occures
                   {p1:f1, p2:f2 ...}, pattern p1, occures f1 times


    x_discrete: 1d-array, discrete signal, same size as input x
               if is_discrete=True, x_discrete is same as input x


    x_emb: input signal ecoded as embeddings
         if encode=True,
         x_emb is array of int encoding the embeddings as defined in 'embeddings_dict'


    embeddings_dict: dict, two dictionaries, 'emb2int', 'int2emb'
                     mapping of embeddings to integer and vice-versa

    """

    if not(is_discrete):

        assert scale>=1 # scaling cannot be less than 1 - downsampling rate
        assert delay>=1 # dalay cannot be less than 1
        if scale>1:
            x = low_resolution(x, scale=scale)
        if mapping_type =='cdf':
            x_mapped = cdf_mapping(x)
        elif mapping_type =='mu-law':
            x_mapped = np.clip(x/np.max(np.abs(x)),-1,1)
            x_mapped = 0.5*(Mu_law(x_mapped,Mu=Mu,companding=True)+1)
        elif mapping_type =='a-law':
            x_mapped = np.clip(x/np.max(np.abs(x)),-1,1)
            x_mapped = 0.5*(A_law(x_mapped,A=A,companding=True)+1)
        elif mapping_type =='fd':
            #x_mapped = np.clip(x/np.max(np.abs(x)),-1,1)
            x_mapped,nbins = quantize_FD(x,scale=1,min_bins=2,keep_amp_scale=False)
            classes = nbins
        else:
            #min-max
            x_mapped = x-np.min(x)
            x_mapped /= np.max(x_mapped)

        x_discrete  = np.round(n_levels * x_mapped + 0.5)
        x_discrete = np.clip(x_discrete,1,n_levels)
    else:
        x_discrete = x.copy()


    N = len(x_discrete)

    x_emb = [list(x_discrete[i:i + emb_dim * delay:delay].astype(int)) for i in range(N - (emb_dim - 1) * delay)]


    patterns, counts = np.unique(x_emb,axis=0,return_counts=True)

    embeddings_count = {tuple(patterns[i]):counts[i] for i in range(len(patterns))}



    k=0
    emb2int = {}
    int2emb = {}
    for p in embeddings_count:
        if p not in emb2int:
            emb2int[p] = k
            int2emb[k] = p
            k+=1

    embeddings_dict = {'emb2int':emb2int, 'int2emb':int2emb}

    if encode:
        x_emb_coded = []
        for p in x_emb:
            k = emb2int[tuple(p)]
            x_emb_coded.append(k)
        x_emb_coded = np.array(x_emb_coded)

        x_emb = x_emb_coded

    return embeddings_count, x_discrete, x_emb, embeddings_dict


def dispersion_entropy(x,classes=10,scale=1,emb_dim=2,delay=1,mapping_type='cdf',de_normalize=False, A=100,Mu=100,return_all=False,warns=True):
    """
    Calculate dispersion entropy of signal x (multiscale)
    ----------------------------------------
    input:
    -----
    x      : input signal x - 1d-array of shape=(n,)
    classes: number of classes - (levels of quantization of amplitude) (default=10)
    emb_dim: embedding dimension,
    delay  : time delay (default=1)
    scale  : downsampled signal with low resolution  (default=1)  - for multipscale dispersion entropy
    mapping_type: mapping method to discretizing signal (default='cdf')
           : options = {'cdf','a-law','mu-law','fd'}
    A  : factor for A-Law- if mapping_type = 'a-law'
    Mu : factor for μ-Law- if mapping_type = 'mu-law'

    de_normalize: (bool) if to normalize the entropy, to make it comparable with different signal with different
                 number of classes and embeding dimensions. default=0 (False) - no normalizations

    if de_normalize=1:
       - dispersion entropy is normalized by log(Npp); Npp=total possible patterns. This is classical
         way to normalize entropy since   max{H(x)}<=np.log(N) for possible outcomes. However, in case of
         limited length of signal (sequence), it would be not be possible to get all the possible patterns
         and might be incorrect to normalize by log(Npp), when len(x)<Npp or len(x)<classes**emb_dim.
         For example, given signal x with discretized length of 2048 samples, if classes=10 and emb_dim=4,
         the number of possible patterns Npp = 10000, which can never be found in sequence length < 10000+4.
         To fix this, the alternative way to nomalize is recommended as follow.
       - select this when classes**emb_dim < (N-(emb_dim-1)*delay)

      de_normalize=2: (recommended for classes**emb_dim > len(x)/scale)
       - dispersion entropy is normalized by log(Npf); Npf [= (len(x)-(emb_dim - 1) * delay)]
         the total  number of patterns founds in given sequence. This is much better normalizing factor.
         In worst case (lack of better word) - for a very random signal, all Npf patterns could be different
         and unique, achieving the maximum entropy and for a constant signal, all Npf will be same achieving to
         zero entropy
       - select this when classes**emb_dim > (N-(emb_dim-1)*delay)

      de_normalize=3:
       - dispersion entropy is normalized by log(Nup); number of total unique patterns (NOT RECOMMENDED)
         -  it does not make sense (not to me, at least)

      de_normalize=4:
       - auto select normalizing factor
       - if classes**emb_dim > (N-(emb_dim-1)*delay), then de_normalize=2
       - if classes**emb_dim > (N-(emb_dim-1)*delay), then de_normalize=2

    output
    ------
    disp_entr  : dispersion entropy of the signal
    prob       : probability distribution of patterns

    if return_all True - also returns

    patterns_dict: disctionary of patterns and respective frequencies
    x_discrete   : discretized signal x
    (Npf,Npp,Nup): Npf - total_patterns_found, Npp - total_patterns_possible) and Nup - total unique patterns found
                 : Npf number of total patterns in discretized signal (not total unique patterns)

    """
    assert scale>=1 # scaling cannot be less than 1 - downsampling rate
    assert delay>=1 # dalay cannot be less than 1
    if scale>1:
        x = low_resolution(x, scale=scale)
    if mapping_type =='cdf':
        x_mapped = cdf_mapping(x)
    elif mapping_type =='mu-law':
        x_mapped = np.clip(x/np.max(np.abs(x)),-1,1)
        x_mapped = 0.5*(Mu_law(x_mapped,Mu=Mu,companding=True)+1)
    elif mapping_type =='a-law':
        x_mapped = np.clip(x/np.max(np.abs(x)),-1,1)
        x_mapped = 0.5*(A_law(x_mapped,A=A,companding=True)+1)
    elif mapping_type =='fd':
        #x_mapped = np.clip(x/np.max(np.abs(x)),-1,1)
        x_mapped,nbins = quantize_FD(x,scale=1,min_bins=2,keep_amp_scale=False)
        classes = nbins
    else:
        #min-max
        x_mapped = x-np.min(x)
        x_mapped /= np.max(x_mapped)

    x_discrete  = np.round(classes * x_mapped + 0.5)
    x_discrete = np.clip(x_discrete,1,classes)

    N = len(x_discrete)

    pttrn_all = [list(x_discrete[i:i + emb_dim * delay:delay].astype(int)) for i in range(N - (emb_dim - 1) * delay)]

    patterns, dispersions = np.unique(pttrn_all,axis=0,return_counts=True)

    prob = dispersions/np.sum(dispersions)

    disp_entr = -np.sum(prob*np.log(prob))

    Npp = classes**emb_dim    #total_patterns_possible
    Npf = np.sum(dispersions) #total_patterns_found
    Nup = len(prob)           #total unique patterns found

    if de_normalize==4: #auto select
        if Npp>(N-(emb_dim-1)*delay):    #same as classes**emb_dim>=(N-(emb_dim-1)*delay)
            de_normalize = 2
        elif Npf>=Npp:                    #same as (classes**emb_dim)<(N-(emb_dim-1)*delay)
            de_normalize = 1
        else:
            de_normalize = 3              # never happens

    if de_normalize==1:
        disp_entr /= np.log(Npp)
        if warns and Npp>(N-(emb_dim-1)*delay):
            print(f'WARNING: Total possible patterns {Npp} can not be in sequence with length {N}')
            print(f'for given emb_dim and delay. Either increase the length of sequence (decrease scale,)')
            print('or choose "de_normalize=2". Set warns=False, to avoid this message')
    elif de_normalize==2:
        disp_entr /= np.log(Npf)
        if warns and Npf>Npp:
            print(f'WARNING: Total possible patterns {Npp} is smaller than total found pattern {Npf}.')
            print(f'Recommonded normizing factor is log({Npp}) to correct choose "de_normalize=1"')
            print('set warns=False, to avoid this message')
    elif de_normalize==3:
        disp_entr /= np.log(Nup)
        if warns:
            print(f'WARNING: using "de_normalize=3 is not recommended. Check documentation at')
            print(f'https://spkit.readthedocs.io/en/latest/#information-theory')
            print('set warns=False, to avoid this message')
    if return_all:
        patterns_dict = {tuple(patterns[i]):dispersions[i] for i in range(len(patterns))}
        return disp_entr, prob, patterns_dict, x_discrete,(Npf,Npp,Nup)
    return disp_entr, prob


def dispersion_entropy_multiscale_refined(x,classes=10,scales=[1,2,3,4,5],emb_dim=2,delay=1,mapping_type='cdf',
                   de_normalize=False, A=100,Mu=100,return_all=False,warns=True):

    """
    Calculate multiscale refined dispersion entropy of signal x
    -----------------------------------------------------------

    compute dispersion entropy at different scales (defined by argument - 'scales') and combining the patterns
    found at different scales to compute final dispersion entropy

    input:
    -----
    x       : input signal x - 1d-array of shape=(n,)
    classes : number of classes - (levels of quantization of amplitude) (default=10)
    emb_dim : embedding dimension,
    delay   : time delay (default=1)
    scales  : list or 1d array of scales to be considered to refine the dispersion entropy

    mapping_type: mapping method to discretizing signal (default='cdf')
           : options = {'cdf','a-law','mu-law','fd'}
    A  : factor for A-Law- if mapping_type = 'a-law'
    Mu : factor for μ-Law- if mapping_type = 'mu-law'

    de_normalize: (bool) if to normalize the entropy, to make it comparable with different signal with different
                 number of classes and embeding dimensions. default=0 (False) - no normalizations

    if de_normalize=1:
       - dispersion entropy is normalized by log(Npp); Npp=total possible patterns. This is classical
         way to normalize entropy since   max{H(x)}<=np.log(N) for possible outcomes. However, in case of
         limited length of signal (sequence), it would be not be possible to get all the possible patterns
         and might be incorrect to normalize by log(Npp), when len(x)<Npp or len(x)<classes**emb_dim.
         For example, given signal x with discretized length of 2048 samples, if classes=10 and emb_dim=4,
         the number of possible patterns Npp = 10000, which can never be found in sequence length < 10000+4.
         To fix this, the alternative way to nomalize is recommended as follow.

      de_normalize=2: (recommended for classes**emb_dim > len(x)/scale)
       - dispersion entropy is normalized by log(Npf); Npf [= (len(x)-(emb_dim - 1) * delay)]
         the total  number of patterns founds in given sequence. This is much better normalizing factor.
         In worst case (lack of better word) - for a very random signal, all Npf patterns could be different
         and unique, achieving the maximum entropy and for a constant signal, all Npf will be same achieving to
         zero entropy
      de_normalize=3:
       - dispersion entropy is normalized by log(Nup); number of total unique patterns (NOT RECOMMENDED)
         -  it does not make sense (not to me, at least)

    output
    ------
    disp_entr  : dispersion entropy of the signal
    prob       : probability distribution of patterns

    if return_all True - also returns

    patterns_dict: disctionary of patterns and respective frequencies
    x_discrete   : discretized signal x
    (Npf,Npp,Nup): Npf - total_patterns_found, Npp - total_patterns_possible) and Nup - total unique patterns found
                 : Npf number of total patterns in discretized signal (not total unique patterns)

    """

    patterns_dict    = {}
    for scale in scales:
        _,_,patterns_dicti,x_discrete,_  = dispersion_entropy(x.copy(),classes=classes,scale=scale,
                    emb_dim=emb_dim,delay=delay,mapping_type=mapping_type,de_normalize=False,
                       A=A,Mu=Mu,return_all=True,warns=False)

        for pattern in patterns_dicti:
            if pattern in patterns_dict:
                patterns_dict[pattern] += patterns_dicti[pattern]
            else:
                patterns_dict[pattern] =  patterns_dicti[pattern]

    dispersion = np.array(list(patterns_dict.values()))

    prob = dispersion/dispersion.sum()

    disp_entr = -np.sum(prob*np.log(prob))

    N   = x.shape[0]
    Npp = classes**emb_dim    #total_patterns_possible
    Npf = np.sum(dispersion) #total_patterns_found
    Nup = len(prob)           #total unique patterns found

    if de_normalize==4: #auto select
        if Npp>(N-(emb_dim-1)*delay):    #same as classes**emb_dim>=(N-(emb_dim-1)*delay)
            de_normalize = 2
        elif Npf>=Npp:                    #same as (classes**emb_dim)<(N-(emb_dim-1)*delay)
            de_normalize = 1
        else:
            de_normalize = 3              # never happens

    if de_normalize==1:
        disp_entr /= np.log(Npp)
        if warns and Npp>(N-(emb_dim-1)*delay)*len(scales):
            print(f'WARNING: Total possible patterns {Npp} can not be in sequence with length {N}')
            print(f'for given emb_dim and delay. Either increase the length of sequence (decrease scale,)')
            print('or choose "de_normalize=2". Set warns=False, to avoid this message')
    elif de_normalize==2:
        disp_entr /= np.log(Npf)
        if warns and Npf>Npp:
            print(f'WARNING: Total possible patterns {Npp} is smaller than total found pattern {Npf}.')
            print(f'Recommonded normizing factor is log({Npp}) to correct choose "de_normalize=1"')
            print('set warns=False, to avoid this message')
    elif de_normalize==3:
        disp_entr /= np.log(Nup)
        if warns:
            print(f'WARNING: using "de_normalize=3 is not recommended. Check documentation at')
            print(f'https://spkit.readthedocs.io/en/latest/#information-theory')
            print('set warns=False, to avoid this message')
    if return_all:
        return disp_entr, prob, patterns_dict, x_discrete,(Npf,Npp,Nup)
    return disp_entr, prob



def signal_delayed_space(x,emb_dim=2,delay=1):
    r"""
    Create Signal Space of emb_dim-dimensions
    -----------------------------------------

    Create multi-dimensional signals corresponding delay

    Parameters
    ----------
    x: 1d-array, input
    emb_dim: int>1 default=2,
    delay: int>=1, delay, lag, between smaples,

    Return
    ------

    X: 2d-array of shape (N - (emb_dim - 1) * delay, emb_dim)

    """
    assert emb_dim>=1 and delay>0

    N = len(x)
    X = []
    nT = N - (emb_dim - 1) * delay
    #print(nT)
    for  k in range(emb_dim):
        xi = x[k*delay:k*delay+nT]
        X.append(xi)
    return np.array(X).T

def create_multidim_space_signal(x,emb_dim=2,delay=1):
    r"""
    Create Signal Space of emb_dim-dimensions
    -----------------------------------------

    Create multi-dimensional signals corresponding delay

    Parameters
    ----------
    x: 1d-array, input
    emb_dim: int>1 default=2,
    delay: int>=1, delay, lag, between smaples,

    Return
    ------

    X: 2d-array of shape (N - (emb_dim - 1) * delay, emb_dim)

    """
    assert emb_dim>=1 and delay>0

    N = len(x)
    X = []
    nT = N - (emb_dim - 1) * delay
    #print(nT)
    for  k in range(emb_dim):
        xi = x[k*delay:k*delay+nT]
        X.append(xi)
    return np.array(X).T

def entropy_differential(x, is_multidim=True, emb_dim=1, delay=1, exclude_constants=True, return_cov=False,raise_error=False):
    r"""
    Differential Entropy of Normally distribuated Multivariant X
    -----------------------------------------------------------

    for            x ∼ N(μ,Σ)

    entropy in nats

        H(x) =  (1/2)ln|Σ| + (1/2)n + (n/2)ln(2π)


    (1/2)n + (n/2)ln(2π) => are constant values for fixed dimension

    Parameters
    ----------
    x: 1d-array, if is_multidim = False, else 2d array
       with (m, N) for N-dimensional space

    if is_multidim=False:

        emb_dim: int>=1, default=1

        delay: int>=1, default=1,

        are used to create multidimensional array


    exclude_constants: if True, then (1/2)n + (n/2)ln(2π) are excluded



    Reference:
    [0] https://statproofbook.github.io/P/mvn-dent.html
    [1] https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    [2] https://en.wikipedia.org/wiki/Differential_entropy
    [3] Gokhale, DV; Ahmed, NA; Res, BC; Piscataway, NJ (May 1989).
     "Entropy Expressions and Their Estimators for Multivariate Distributions".
     IEEE Transactions on Information Theory. 35 (3): 688–692. doi:10.1109/18.30996.

    """


    X = x.copy()

    if not(is_multidim):
        X = create_multidim_space_signal(x,emb_dim=emb_dim, delay=delay)


    assert X.ndim>1

    # Covariance
    M, N = X.shape

    # None of dimension can be zero
    #assert M>=1 and N>=1
    if M==0 or N==0:
        raise ValueError("Neither dimension or number of samples can be zeos. Check shape of input X")

    X = X - X.mean(axis=0, keepdims=True)
    CovX =  np.dot(X.T, X)
    CovX /= (M - 1)

    #print(CovX)

    # compute entropy using the slogdet in numpy rather than np.linalg.det
    # nb: the entropy is the logdet

    (sign, H_x) = np.linalg.slogdet(CovX)

    #print(sign)
    #print(Hx)

    if not sign > 0:
        if raise_error:
            raise ValueError(f"Estimate of differential entropy for the input X {X.shape} couldn't be calculated. "
                             f"Due to determinant of covariace matrix ({CovX.shape}), Det : {np.linalg.det(CovX)}, logDet {H_x} with Sign {sign}")
        else:
            print(f"Estimate of differential entropy for the input X {X.shape} couldn't be calculated. Due to determinant of covariace matrix ({CovX.shape}), Det : {np.linalg.det(CovX)}, logDet {H_x} with Sign {sign}")

    if not(exclude_constants):
        # H(x) =  (1/2)ln|Σ| + (1/2)n + (n/2)ln(2π)
        #print(Hx)
        #print('c', (1/2)*N, (N/2)*np.log(2*np.pi))
        H_x = (1/2)*H_x + (1/2)*N + (N/2)*np.log(2*np.pi)
        #print(Hx)
    if return_cov:
        return H_x, CovX
    return H_x

def entropy_diff_cond_self(X, present_first=True):
    r"""

    Self-Conditional Entropy
    ------------------------
    Information of X(i+1) given X(i)


    H(X_i+1|X_i) = H(X_i+1, X_i) - H(X_i)

    using::
    H(X|Y) =  H(X, Y) - H(Y)


    """
    #det_xi1 = entr(x)
    #det_xi = entr(x[1:, :])
    #hxcx = det_xi1 - det_xi

    assert X.ndim>1

    if X.shape[1]<2:
        raise ValueError("To compute entropy conditioned to past, dimension of X should be atleast 2 ")


    H_xi1_xi  = entropy_differential(X, is_multidim=True)
    if present_first:
        H_xi  = entropy_differential(X[:,1:], is_multidim=True)
    else:
        H_xi  = entropy_differential(X[:,:-1], is_multidim=True)

    H_x1x = H_xi1_xi - H_xi

    return H_x1x

def entropy_diff_cond(X,Y,present_first=True):
    r"""

    Conditional Entropy
    ------------------
    Information of X(i+1) given X(i) and Y(i)


    H(X_i+1|X_i,Y_i) = H(X_i+1,X_i,Y_i) - H(X_i,Y_i)


    """
    # H(X_i+1|X_i,Y_i) = H(X_i+1,X_i,Y_i) - H(X_i,Y_i)

    assert X.shape[0]==Y.shape[0]

    if present_first:
        xi1_xi_yi = np.concatenate((X, Y[:, 1:]), axis=1)
        xi_yi = np.concatenate((X[:, 1:],Y[:, 1:]), axis=1)
    else:
        xi1_xi_yi = np.concatenate((X, Y[:, :-1]), axis=1)
        xi_yi = np.concatenate((X[:, :-1], Y[:, :-1]), axis=1)


    H_xi1_xi_yi  = entropy_differential(xi1_xi_yi, is_multidim=True)
    H_xi_yi      = entropy_differential(xi_yi, is_multidim=True)

    H_x1xy = H_xi1_xi_yi - H_xi_yi

    return H_x1xy

def entropy_diff_joint(x,y):
    r"""

    Joint Entropy
    ------------

    H(X,Y)
    """
    assert x.shape[0]==y.shape[0]

    x_y = np.concatenate((x, y), axis=1)

    H_xy = entropy_differential(x_y, is_multidim=True)

    return H_xy

def entropy_diff_joint_cond(X,Y,present_first=True):
    r"""

    Joint-Conditional Entropy
    ------------------------

    H(X_i+1,Y_i+1|X_i,Y_i) = H(X_i+1,Y_i+1,X_i,Y_i) - H(X_i,Y_i)

    """

    # H(X_i+1,Y_i+1|X_i,Y_i) = H(X_i+1,Y_i+1,X_i,Y_i) - H(X_i,Y_i)

    if present_first:
        xi_yi = np.concatenate((X[:, 1:], Y[:, 1:]), axis=1)
    else:
        xi_yi = np.concatenate((X[:, :-1], Y[:, :-1]), axis=1)


    H_xiyi    = entropy_differential(xi_yi, is_multidim=True)

    H_xy = entropy_diff_joint(X,Y)

    H_xy1xy = H_xy - H_xiyi

    return H_xy1xy

def mutual_info_diff_self(X,present_first=True):
    r"""

    Self Mutual Information
    ------------------
    Predictibility of X(i+1) given X(i)


    I(X_i+1; X_i) = H(X_i+1) - H(X_i+1 | X_i)


    """

    if present_first:
        x0 = X[:,:1].copy()
    else:
        x0 = X[:,-1:].copy()

    H_x = entropy_differential(x0, is_multidim=True)

    H_x1x = entropy_diff_cond_self(X, present_first=present_first)

    I_xx = H_x-H_x1x

    return I_xx

def mutual_info_diff(X,Y,present_first=True):
    r"""

    Mutual Information
    ------------------
    Predictibility of X(i+1) given X(i) and Y(i)

    I(X_i+1; X_i, Y_i) = H(X_i+1) - H(X_i+1 | X_i, Y_i)

    H(X_i+1|X_i,Y_i) = H(X_i+1,X_i,Y_i) - H(X_i,Y_i)

    """

    if present_first:
        x0 = X[:,:1].copy()
    else:
        x0 = X[:,-1:].copy()

    H_x = entropy_differential(x0, is_multidim=True)


    H_x1xy = entropy_diff_cond(X,Y,present_first=present_first)

    I_xy = H_x - H_x1xy

    return I_xy

def transfer_entropy(X,Y,present_first=True):
    r"""

    Transfer Entropy
    ----------------

    TE_{X-->Y}  = I(Y_i+1, X_i | Y_i)

    TE_{X-->Y}  = H(Y_i+1 | Y_i) - H(Y_i+1 | X_i, Y_i)           [Eq1]



    TE_{X-->Y}  = H(Y_i+1, Y_i) - H(Y_i) - H(Y_i+1,X_i,Y_i) +  H(X_i,Y_i)

    TE_{X-->Y}  = H(X_i,Y_i)  +  H(Y_i+1, Y_i) - H(Y_i+1,X_i,Y_i) - H(Y_i)  [Eq2]


    Using:
    H(X_i+1|X_i)     = H(X_i+1, X_i) - H(X_i)          |  entropy_diff_cond_self(X)
    H(X_i+1|X_i,Y_i) = H(X_i+1,X_i,Y_i) - H(X_i,Y_i)   |  entropy_diff_cond(X,Y)

    Ref:

    """

    H_y1y  = entropy_diff_cond_self(Y,present_first=present_first)
    H_y1xy = entropy_diff_cond(Y,X,present_first=present_first)

    TE_x2y = H_y1y - H_y1xy

    return TE_x2y

def transfer_entropy_cond(X,Y,Z,present_first=True):
    r"""

    Conditional Transfer Entopry Or Partial Transfer Entropy
    --------------------------------------------------------

    TE_{X-->Y | Z}   =  I(Y_i+1, X_i | Y_i, Z_i)


    TE_{X-->Y | Z}  =  H(X_i,Y_i, Z_i)  +  H(Y_i+1, Y_i, Z_i) - H(Y_i+1,X_i,Y_i, Z_i) - H(Y_i, Z_i)

    """

    if present_first:
        xi_yi_zi = np.concatenate((X[:, 1:], Y[:, 1:], Z[:, 1:]), axis=1)
        yi1_yi_zi = np.concatenate((Y, Z[:, 1:]), axis=1)
        yi1_yi_xi_zi = np.concatenate((Y, X[:, 1:], Z[:, 1:]), axis=1)
        yi_zi = np.concatenate((Y[:, 1:], Z[:, 1:]), axis=1)
    else:
        xi_yi_zi = np.concatenate((X[:, :-1], Y[:, :-1], Z[:, :-1]), axis=1)
        yi1_yi_zi = np.concatenate((Y, Z[:, :-1]), axis=1)
        yi1_yi_xi_zi = np.concatenate((Y, X[:, :-1], Z[:, :-1]), axis=1)
        yi_zi = np.concatenate((Y[:, :-1], Z[:, :-1]), axis=1)



    H_xi_yi_zi     = entropy_differential(xi_yi_zi, is_multidim=True)
    H_yi1_yi_zi    = entropy_differential(yi1_yi_zi, is_multidim=True)
    H_yi1_yi_xi_zi = entropy_differential(yi1_yi_xi_zi, is_multidim=True)
    H_yi_zi        = entropy_differential(yi_zi, is_multidim=True)


    TE_x2y1z = H_xi_yi_zi + H_yi1_yi_zi - H_yi1_yi_xi_zi - H_yi_zi

    return TE_x2y1z

def partial_transfer_entropy_(X,Y,Z,present_first=True):
    r"""

    Partial Transfer Entropy Or Conditional Transfer Entopry
    --------------------------------------------------------

    TE_{X-->Y | Z}   =  I(Y_i+1, X_i | Y_i, Z_i)


    TE_{X-->Y | Z}  =  H(X_i,Y_i, Z_i)  +  H(Y_i+1, Y_i, Z_i) - H(Y_i+1,X_i,Y_i, Z_i) - H(Y_i, Z_i)

    """

    if present_first:
        xi_yi_zi = np.concatenate((X[:, 1:], Y[:, 1:], Z[:, 1:]), axis=1)
        yi1_yi_zi = np.concatenate((Y, Z[:, 1:]), axis=1)
        yi1_yi_xi_zi = np.concatenate((Y, X[:, 1:], Z[:, 1:]), axis=1)
        yi_zi = np.concatenate((Y[:, 1:], Z[:, 1:]), axis=1)
    else:
        xi_yi_zi = np.concatenate((X[:, :-1], Y[:, :-1], Z[:, :-1]), axis=1)
        yi1_yi_zi = np.concatenate((Y, Z[:, :-1]), axis=1)
        yi1_yi_xi_zi = np.concatenate((Y, X[:, :-1], Z[:, :-1]), axis=1)
        yi_zi = np.concatenate((Y[:, :-1], Z[:, :-1]), axis=1)



    H_xi_yi_zi     = entropy_differential(xi_yi_zi, is_multidim=True)
    H_yi1_yi_zi    = entropy_differential(yi1_yi_zi, is_multidim=True)
    H_yi1_yi_xi_zi = entropy_differential(yi1_yi_xi_zi, is_multidim=True)
    H_yi_zi        = entropy_differential(yi_zi, is_multidim=True)


    TE_x2y1z = H_xi_yi_zi + H_yi1_yi_zi - H_yi1_yi_xi_zi - H_yi_zi

    return TE_x2y1z

def partial_transfer_entropy(X,Y,Z,present_first=True,verbose=False):
    r"""

    Partial Transfer Entropy Or Conditional Transfer Entopry
    --------------------------------------------------------

    TE_{X-->Y | Z}   =  I(Y_i+1, X_i | Y_i, Z_i)


    TE_{X-->Y | Z}  =  H(X_i,Y_i, Z_i)  +  H(Y_i+1, Y_i, Z_i) - H(Y_i+1,X_i,Y_i, Z_i) - H(Y_i, Z_i)

    """

    if isinstance(Z, list):
        if verbose: print('#Z :',len(Z), ' Z0',Z[0].shape)
        if present_first:
            Zi = np.concatenate([Zj[:, 1:] for Zj in Z],axis=1)
        else:
            Zi = np.concatenate([Zj[:, :-1] for Zj in Z],axis=1)

        if verbose: print('Zi shape:',Zi.shape)

    elif Z.ndim>2:
        if verbose: print('Z shape:',Z.shape)
        if present_first:
            Zi = np.concatenate(Z[:,:,1:],axis=1)
        else:
            Zi = np.concatenate(Z[:,:,:-1],axis=1)



    elif Z.ndim==2:
        if verbose: print('Z shape:',Z.shape)
        if present_first:
            Zi = Z[:, 1:]
        else:
            Zi = Z[:, :-1]

    if verbose: print('Zi shape:',Zi.shape)

    if present_first:
        xi_yi_zi = np.concatenate((X[:, 1:], Y[:, 1:], Zi), axis=1)
        yi1_yi_zi = np.concatenate((Y, Zi), axis=1)
        yi1_yi_xi_zi = np.concatenate((Y, X[:, 1:], Zi), axis=1)
        yi_zi = np.concatenate((Y[:, 1:], Zi), axis=1)
    else:
        xi_yi_zi = np.concatenate((X[:, :-1], Y[:, :-1], Zi), axis=1)
        yi1_yi_zi = np.concatenate((Y, Zi), axis=1)
        yi1_yi_xi_zi = np.concatenate((Y, X[:, :-1], Zi), axis=1)
        yi_zi = np.concatenate((Y[:, :-1], Zi), axis=1)

    H_xi_yi_zi     = entropy_differential(xi_yi_zi, is_multidim=True)
    H_yi1_yi_zi    = entropy_differential(yi1_yi_zi, is_multidim=True)
    H_yi1_yi_xi_zi = entropy_differential(yi1_yi_xi_zi, is_multidim=True)
    H_yi_zi        = entropy_differential(yi_zi, is_multidim=True)


    TE_x2y1z = H_xi_yi_zi + H_yi1_yi_zi - H_yi1_yi_xi_zi - H_yi_zi

    return TE_x2y1z

def entropy_granger_causality(X,Y,present_first=True, normalize=False):
    r"""

    Granger Causality based on Differential Entropy
    ----------------------------------------------

    (1) GC_XY (X-->Y)  : H(Y_i+1|Y_i) - H(Y_i+1|X_i,Y_i)
    (2) GC_YX (Y-->X)  : H(X_i+1|X_i) - H(X_i+1|X_i,Y_i)
    (3) GC_XdY (X.Y)   : H(Y_i+1|X_i,Y_i) +  H(X_i+1|X_i,Y_i) -    H(X_i+1,Y_i+1|X_i,Y_i)


    if normalize True
        GC_XY = GC_XY/(I(Y_i+1; Y_i) + GC_XY )
        GC_YX = GC_YX/(I(X_i+1; X_i) + GC_YX )


    Using::
    H(Y_i+1|Y_i) = H(Y_i+1, Y_i) - H(Y_i)
    H(X_i+1|X_i) = H(X_i+1, X_i) - H(X_i)
    H(Y_i+1|X_i,Y_i) = H(Y_i+1,X_i,Y_i) - H(X_i,Y_i)
    H(X_i+1|X_i,Y_i) = H(X_i+1,X_i,Y_i) - H(X_i,Y_i)
    H(X_i+1,Y_i+1|X_i,Y_i) = H(X_i+1,Y_i+1,X_i,Y_i) - H(X_i,Y_i)

    I(X_i+1; X_i) = H(X_i+1) - H(X_i+1 | X_i)
    I(Y_i+1; Y_i) = H(Y_i+1) - H(Y_i+1 | Y_i)

    """

    H_x1x  = entropy_diff_cond_self(X, present_first=present_first)
    H_y1y  = entropy_diff_cond_self(Y, present_first=present_first)

    H_y1xy = entropy_diff_cond(Y,X,present_first=present_first)
    H_x1xy = entropy_diff_cond(X,Y,present_first=present_first)


    H_xy1xy = entropy_diff_joint_cond(X,Y,present_first=present_first)


    gc_xy  = H_y1y  - H_y1xy
    gc_yx  = H_x1x  - H_x1xy
    gc_xdy = H_y1xy + H_x1xy - H_xy1xy


    if normalize:
        I_xx = mutual_info_diff_self(X,present_first=present_first)
        I_yy = mutual_info_diff_self(Y,present_first=present_first)

        gc_xy = gc_xy/(gc_xy + I_yy)
        gc_yx = gc_yx/(gc_yx + I_xx)


    return (gc_xy, gc_yx,gc_xdy)

def show_farmulas():
    r"""
    |
    |
    |     Usuful Formulas
    |
    |

    Differential Entropy of Normally distribuated Multivariant X
    -----------------------------------------------------------

    for            x ∼ N(μ,Σ)

    entropy in nats

        H(x) =  (1/2)ln|Σ| + (1/2)n + (n/2)ln(2π)


    (1/2)n + (n/2)ln(2π) => are constant values for fixed dimension


    code::

        H_x = entropy_differential(x,is_multidim=True, emb_dim=1, delay=1,)



    Self-Conditional Entropy
    ------------------------
    Information of X(i+1) given X(i)

    H(X_i+1|X_i) = H(X_i+1, X_i) - H(X_i)

    using::
    H(X|Y) =  H(X, Y) - H(Y)

    code::

        H_x1x = entropy_diff_cond_self(X, present_first=True)



    Conditional Entropy
    ------------------
    Information of X(i+1) given X(i) and Y(i)

    H(X_i+1|X_i,Y_i) = H(X_i+1,X_i,Y_i) - H(X_i,Y_i)


    code::
        H_x1xy = entropy_diff_cond(X,Y,present_first=True)




    Joint Entropy
    ------------

    H(X,Y)

    code::

        H_xy = entropy_diff_joint(x,y)


    Joint-Conditional Entropy
    ------------------------

    H(X_i+1,Y_i+1|X_i,Y_i) = H(X_i+1,Y_i+1,X_i,Y_i) - H(X_i,Y_i)


    code::

        H_xy1xy = entropy_diff_joint_cond(X,Y,present_first=True)


    Self Mutual Information
    ------------------
    Predictibility of X(i+1) given X(i)


    I(X_i+1; X_i) = H(X_i+1) - H(X_i+1 | X_i)

    code::

        I_xx = mutual_info_diff_self(X,present_first=True)


    Mutual Information
    ------------------
    Predictibility of X(i+1) given X(i) and Y(i)

    I(X_i+1; X_i, Y_i) = H(X_i+1) - H(X_i+1 | X_i, Y_i)

    H(X_i+1|X_i,Y_i) = H(X_i+1,X_i,Y_i) - H(X_i,Y_i)


    code::

        I_xy = mutual_info_diff(X,Y,present_first=True)


    Transfer Entropy
    ----------------

    TE_{X-->Y}  = I(Y_i+1, X_i | Y_i)

    TE_{X-->Y}  = H(Y_i+1 | Y_i) - H(Y_i+1 | X_i, Y_i)           [Eq1]



    TE_{X-->Y}  = H(Y_i+1, Y_i) - H(Y_i) - H(Y_i+1,X_i,Y_i) +  H(X_i,Y_i)

    TE_{X-->Y}  = H(X_i,Y_i)  +  H(Y_i+1, Y_i) - H(Y_i+1,X_i,Y_i) - H(Y_i)  [Eq2]


    Using:
    H(X_i+1|X_i)     = H(X_i+1, X_i) - H(X_i)          |  entropy_diff_cond_self(X)
    H(X_i+1|X_i,Y_i) = H(X_i+1,X_i,Y_i) - H(X_i,Y_i)   |  entropy_diff_cond(X,Y)

    code::

        TE_x2y = transfer_entropy(X,Y,present_first=True)


    Partial Transfer Entropy Or Conditional Transfer Entopry
    --------------------------------------------------------

    TE_{X-->Y | Z}   =  I(Y_i+1, X_i | Y_i, Z_i)


    TE_{X-->Y | Z}  =  H(X_i,Y_i, Z_i)  +  H(Y_i+1, Y_i, Z_i) - H(Y_i+1,X_i,Y_i, Z_i) - H(Y_i, Z_i)

    code::

        TE_x2y1z = partial_transfer_entropy(X,Y,Z,present_first=True,verbose=False)

    Granger Causality based on Differential Entropy
    ----------------------------------------------

    (1) GC_XY (X-->Y)  : H(Y_i+1|Y_i) - H(Y_i+1|X_i,Y_i)
    (2) GC_YX (Y-->X)  : H(X_i+1|X_i) - H(X_i+1|X_i,Y_i)
    (3) GC_XdY (X.Y)   : H(Y_i+1|X_i,Y_i) +  H(X_i+1|X_i,Y_i) -    H(X_i+1,Y_i+1|X_i,Y_i)


    if normalize True
        GC_XY = GC_XY/(I(Y_i+1; Y_i) + GC_XY )
        GC_YX = GC_YX/(I(X_i+1; X_i) + GC_YX )


    Using::
    H(Y_i+1|Y_i) = H(Y_i+1, Y_i) - H(Y_i)
    H(X_i+1|X_i) = H(X_i+1, X_i) - H(X_i)
    H(Y_i+1|X_i,Y_i) = H(Y_i+1,X_i,Y_i) - H(X_i,Y_i)
    H(X_i+1|X_i,Y_i) = H(X_i+1,X_i,Y_i) - H(X_i,Y_i)
    H(X_i+1,Y_i+1|X_i,Y_i) = H(X_i+1,Y_i+1,X_i,Y_i) - H(X_i,Y_i)

    I(X_i+1; X_i) = H(X_i+1) - H(X_i+1 | X_i)
    I(Y_i+1; Y_i) = H(Y_i+1) - H(Y_i+1 | Y_i)


    code::

        gc_xy, gc_yx,gc_xdy = entropy_granger_causality(X,Y,present_first=True, normalize=False)

    """

    print(help(show_farmulas))
