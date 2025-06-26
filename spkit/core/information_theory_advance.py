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
from .information_theory import *
from ..utils import deprecated

def low_resolution(x, scale):
    r"""Reducing the time-resolution of signal :math:`x`

    Reducing the time-resolution of signal : x. It is similar as downsampling (or decimation) of a signal,
    except it is averaged around removed samples.


    Parameters
    ----------
    x : 1d-array of shape=(n,)
    scale: (int) downsampled by factor of scale and averaged around removed samples

    Returns
    -------
    x_low: shape=(n/scale,) low resolution signal
    
    Notes
    -----

    See Also
    --------
    cdf_mapping: # CDF Mapping

    Examples
    --------
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs,_ = sp.data.ecg_sample_12leads(sample=3)
    x = x[1000:2000,0]
    x = x + 0.1*np.random.randn(len(x))
    t = np.arange(len(x))/fs
    N = 5
    y1 = x[::N] # decimation by factor of N
    y2 = sp.low_resolution(x.copy(),scale=N) 
    t1 = t[::N]
    print(y1.shape, y2.shape, x.shape)
    plt.figure(figsize=(10,3))
    plt.plot(t,x,label=f'raw (n={len(x)})')
    plt.plot(t1,y1 - 3,label=f'decimated (n={len(y1)})')
    plt.plot(t1,y2 - 6,label=f'low-resoltion (n={len(y2)})')
    plt.xlim([t[0],t[-1]])
    plt.xlabel('time (s)')
    plt.ylabel('x')
    plt.yticks([0,-3,-6],['raw','y1','y2'])
    plt.legend(bbox_to_anchor=(1,1))
    plt.grid()
    plt.tight_layout()
    plt.show()

    """
    N = int(np.fix(len(x) / scale))
    x_low = np.array([np.mean(x[i * scale:(i + 1) * scale]) for i in range(N)])
    return x_low

def cdf_mapping(x):
    r"""CDF: Cumulative Distribution Function Mapping (Non-linear mapping)

    Map the signal x to y from into CDF of x, y will be uniformly
    disctribuated anf ranges from 0 to 1.
    CDF: Cumulative Distribution Function

            .. math:: y(n) = P(X<=x(n))
    
    Parameters
    ----------
    x : 1d-array of shape=(n,)

    Returns
    -------
    x_cdf: cumulative distribution function of x


    References
    ----------
    * wikipedia - https://en.wikipedia.org/wiki/Cumulative_distribution_function

    Notes
    -----
    * CDF Mapping shift the mean of the signal and the range.

    See Also
    --------
    low_resolution: reducing the resolution of signal

    Examples
    --------
    #sp.cdf_mapping
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs,_ = sp.data.ecg_sample_12leads(sample=3)
    x = x[1000:3000,0]
    x = x - np.mean(x)
    t = np.arange(len(x))/fs
    y = sp.cdf_mapping(x)
    y = y - np.mean(y)
    plt.figure(figsize=(13,6))
    plt.subplot(211)
    plt.plot(t,x,label=f'x')
    plt.plot(t,y,label=f'y')
    plt.xlabel('time (s)')
    plt.legend()
    plt.title('CDF Mapping')
    plt.grid()
    plt.subplot(223)
    sp.hist_plot(x)
    sp.hist_plot(y)
    plt.title('Histogram')
    plt.subplot(224)
    plt.plot(x,y,'.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()
    """
    N   = len(x)
    mu  = np.mean(x)
    sig = np.std(x) if np.std(x) != 0 else 0.001
    ncdf  = scipystats.norm(loc=mu, scale=sig)
    x_cdf = ncdf.cdf(x)
    return x_cdf

def signal_embeddings(x,is_discrete=False,emb_dim=2,delay=1,n_levels=10,scale=1,mapping_type='cdf',A=100,Mu=100,encode=True):
    r"""Extract Embeddings from given signal

    Extract Embeddings from given signal, after discreetization of signal.

    Parameters
    ----------
    x :  1d-array
      - input signal

    is_discrete: bool, default=False,
      - if False, input signal is discretized by given setting, (n_levels,scale,mapping_type,A,Mu)
      - else considered it as discrete and create dictionary of embeddigns

    emb_dim: int>=1, default 2
      - Embedding Dimensions

    delay: int>=1 default=1
      - delay factor, as skipping factor

    n_levels: int, default=10
      - Number of level used for discreetization.

    scale: int>0, default=1
      - if scale>1, applied :func:`low_resolution` function to reduce the resolution

    mapping_type: str, default='cdf' {'cdf','mu-law','a-law','fd'}
      - mapping type to discreetize the signal before extracting embeddings.

    A: int, default=100
     - used if mapping_type='a-law'
    Mu: int, default=100
     - used if mapping_type='mu-law'
    encode: bool, default = True
      -  If True, 

    Returns
    -------

    embeddings_count: dict, 
       - map of each unique pattern/embeding to number of times it occures
       - {p1:f1, p2:f2 ...}, pattern p1, occures f1 times


    x_discrete: 1d-array,
       - discrete signal, same size as input x
       - if is_discrete=True, x_discrete is same as input x


    x_emb: input signal ecoded as embeddings
        - if encode=True, x_emb is array of int encoding the embeddings as defined in 'embeddings_dict'

    embeddings_dict: dict, 
         - two dictionaries with keys: 'emb2int', 'int2emb'
         - mapping of embeddings to integer and vice-versa

    Notes
    -----
    * CDF Mapping shift the mean of the signal and the range.

    See Also
    --------
    low_resolution: reducing the resolution of signal
    
    Examples
    --------
    #sp.signal_embeddings
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x, fs = sp.data.optical_sample(sample=1)
    x = x[1000:2000]
    t = np.arange(len(x))/fs
    embeddings_count, x_discrete, x_emb, embeddings_dict = sp.signal_embeddings(x,n_levels=5,emb_dim=3, delay=10,encode=False)
    plt.figure(figsize=(10,5))
    plt.subplot(311)
    plt.plot(t,x)
    plt.xlim([t[0],t[-1]])
    plt.ylabel('x: signal')
    plt.grid()
    plt.subplot(312)
    plt.plot(t,x_discrete)
    plt.xlim([t[0],t[-1]])
    plt.ylabel('Discreet signal')
    plt.grid()
    plt.subplot(313)
    plt.plot(t[:x_emb.shape[0]],x_emb)
    plt.ylabel('Embeddings')
    plt.xlim([t[0],t[-1]])
    plt.xlabel('time (s)')
    plt.grid()
    plt.tight_layout()
    plt.show()
    print('Count of Embeddings')
    print(embeddings_count)
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
    
    x_emb = np.array(x_emb)
    return embeddings_count, x_discrete, x_emb, embeddings_dict

def dispersion_entropy(x,classes=10,scale=1,emb_dim=2,delay=1,mapping_type='cdf',de_normalize=False, A=100,Mu=100,return_all=False,warns=True):
    r"""Dispersion Entropy of signal :math:`H_{de}(X)`

    Calculate dispersion entropy of signal x (multiscale)

    Dispersion Entropy

    Parameters
    ----------
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

    
    Returns
    -------
    disp_entr  : float
            dispersion entropy of the signal
    prob       : array
             probability distribution of patterns

    if return_all =True:
        patterns_dict: dict
            dictionary of patterns and respective frequencies
        x_discrete   : array
            discretized signal x
        (Npf,Npp,Nup): tuple of 3
            * Npf - total_patterns_found, Npp - total_patterns_possible
            * Nup - total unique patterns found
            * Npf number of total patterns in discretized signal (not total unique patterns)

    References
    ----------
    * https://ieeexplore.ieee.org/document/7434608


    See Also
    --------
    dispersion_entropy_multiscale_refined : Dispersion Entropy multi-scale
    entropy : Entropy
    entropy_sample : Sample Entropy
    entropy_approx : Approximate Entropy
    entropy_spectral : Spectral Entropy
    entropy_svd :  SVD Entropy
    entropy_permutation :  Permutation Entropy
    entropy_differential : Differential Entropy

    Examples
    --------
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    np.random.seed(1)
    x1, fs = sp.data.optical_sample(sample=1)
    t = np.arange(len(x1))/fs
    H_de1, prob1  = sp.dispersion_entropy(x1,classes=10,scale=1)
    print('DE of x1 = ',H_de1)
    plt.figure(figsize=(10,2))
    plt.plot(t,x1, label=f'H_d(x1) = {H_de1:,.2f}')
    plt.xlim([t[0],t[-1]])
    plt.ylabel('x1')
    plt.xlabel('time (s)')
    plt.legend(loc='upper right')
    plt.show()
    
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
    r"""Multiscale refined Dispersion Entropy of signal :math:`H_{de}(X)`

    Calculate multiscale refined dispersion entropy of signal x

    compute dispersion entropy at different scales (defined by argument - 'scales') and combining the patterns
    found at different scales to compute final dispersion entropy

    Parameters
    ----------
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

    Returns
    -------
    disp_entr  : dispersion entropy of the signal
    prob       : probability distribution of patterns

    if return_all True - also returns

    patterns_dict: disctionary of patterns and respective frequencies
    x_discrete   : discretized signal x
    (Npf,Npp,Nup): Npf - total_patterns_found, Npp - total_patterns_possible) and Nup - total unique patterns found
                 : Npf number of total patterns in discretized signal (not total unique patterns)


    See Also
    --------
    dispersion_entropy: Dispersion Entropy
    entropy : Entropy
    entropy_sample : Sample Entropy
    entropy_approx : Approximate Entropy
    entropy_spectral : Spectral Entropy
    entropy_svd :  SVD Entropy
    entropy_permutation :  Permutation Entropy
    entropy_differential : Differential Entropy

    Examples
    --------
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    np.random.seed(1)
    x1, fs = sp.data.optical_sample(sample=1)
    t = np.arange(len(x1))/fs
    H_de1, prob1  = sp.dispersion_entropy_multiscale_refined(x1,classes=10,scales=[1,2,3,4,5,6,7,8])
    print('DE of x1 = ',H_de1)
    plt.figure(figsize=(10,2))
    plt.plot(t,x1, label=f'H_d(x1) = {H_de1:,.2f}')
    plt.xlim([t[0],t[-1]])
    plt.ylabel('x1')
    plt.xlabel('time (s)')
    plt.legend(loc='upper right')
    plt.show()
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
    r"""Create N-dimensional signal space

    Create Signal Space of emb_dim-dimensions

    Create multi-dimensional signals corresponding delay

    Parameters
    ----------
    x: 1d-array, input
    emb_dim: int>1 default=2,
    delay: int>=1, delay, lag, between smaples,

    Return
    ------
    X: 2d-array of shape (N - (emb_dim - 1) * delay, emb_dim)

    References
    ----------
    * wiki

    See Also
    --------
    signal_embeddings: Extract Signal Embeddings

    Examples
    --------
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x, fs = sp.data.optical_sample(sample=3)
    x = x[:500]
    t = np.arange(len(x))/fs
    X = sp.signal_delayed_space(x,emb_dim=5,delay=3)
    plt.figure(figsize=(10,4))
    plt.subplot(211)
    plt.plot(t,x)
    plt.xlim([t[0],t[-1]])
    plt.ylabel('x')
    plt.subplot(212)
    plt.plot(t[:X.shape[0]],X)
    plt.xlim([t[0],t[-1]])
    plt.ylabel('Embeddings')
    plt.xlabel('time (s)')
    plt.show()

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
    r"""Create N-dimensional signal space

    Create Signal Space of emb_dim-dimensions

    Create multi-dimensional signals corresponding delay

    Parameters
    ----------
    x: 1d-array, input
    emb_dim: int>1 default=2,
    delay: int>=1, delay, lag, between smaples,

    Returns
    -------
    X: 2d-array of shape (N - (emb_dim - 1) * delay, emb_dim)

    See Also
    --------
    signal_embeddings: Extract Signal Embeddings

    Examples
    --------
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x, fs = sp.data.optical_sample(sample=3)
    x = x[:500]
    t = np.arange(len(x))/fs
    X = sp.create_multidim_space_signal(x,emb_dim=5,delay=3)
    plt.figure(figsize=(10,4))
    plt.subplot(211)
    plt.plot(t,x)
    plt.xlim([t[0],t[-1]])
    plt.ylabel('x')
    plt.subplot(212)
    plt.plot(t[:X.shape[0]],X)
    plt.xlim([t[0],t[-1]])
    plt.ylabel('Embeddings')
    plt.xlabel('time (s)')
    plt.show()

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
    r"""Differential Entropy :math:`H_{\partial}(X)`

    Differential Entropy of Normally distribuated Multivariant X

    for 

    .. math::
       
       x ∼ N(μ,Σ)

    entropy in nats

        .. math:: H_{\partial}(x) =  (1/2)ln|Σ| + (1/2)n + (n/2)ln(2π)

    :math:`H_{\partial}(X)`

    :math:`(1/2)n + (n/2)ln(2π)` => are constant values for fixed dimension


    Parameters
    ----------
    x: 1d-array, if is_multidim = False, else 2d array
       with (m, N) for N-dimensional space

    if is_multidim=False:
        - emb_dim: int>=1, default=1
        - delay: int>=1, default=1,
        are used to create multidimensional array

    exclude_constants: if True, then (1/2)n + (n/2)ln(2π) are excluded

    Returns
    -------
    H_x: scaler
     - Differential Entropy

    References
    ----------
    * [1] https://statproofbook.github.io/P/mvn-dent.html
    * [2] https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    * [3] https://en.wikipedia.org/wiki/Differential_entropy
    * [4] Gokhale, DV; Ahmed, NA; Res, BC; Piscataway, NJ (May 1989). "Entropy Expressions and Their Estimators for Multivariate Distributions".
     IEEE Transactions on Information Theory. 35 (3): 688–692. doi:10.1109/18.30996.


    See Also
    --------
    dispersion_entropy: Dispersion Entropy
    dispersion_entropy_multiscale_refined: Dispersion Entropy Multiscale
    entropy : Entropy
    entropy_sample : Sample Entropy
    entropy_approx : Approximate Entropy
    entropy_spectral : Spectral Entropy
    entropy_svd :  SVD Entropy
    entropy_permutation :  Permutation Entropy

    Examples
    --------
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x0, fs = sp.data.optical_sample(sample=3)
    t = np.arange(len(x0))/fs
    x1 = x0 + 0.5*np.random.randn(len(t))
    # create multi-dimensional space signals
    X0 = sp.signal_delayed_space(x0,emb_dim=3,delay=2)
    X1 = sp.signal_delayed_space(x1,emb_dim=3,delay=2)
    # Compute Differential Entropy
    Hd0 = sp.entropy_differential(X0)
    Hd1 = sp.entropy_differential(X1)
    print('Differential Entropy')
    print(' - x0: ',Hd0)
    print(' - x1: ',Hd1)
    # Or Compute directly - Compute Differential Entropy
    Hd0 = sp.entropy_differential(x0,is_multidim=False,emb_dim=3, delay=2)
    Hd1 = sp.entropy_differential(x1,is_multidim=False,emb_dim=3, delay=2)
    print('Differential Entropy')
    print(' - x0: ',Hd0)
    print(' - x1: ',Hd1)
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
    r"""Self-Conditional Entropy :math:`H_{\partial}(X_{i+1}|X_i)`

    Self-Conditional Entropy

    Information of :math:`X(i+1)` given :math:`X(i)`


    .. math:: H_{\partial}(X_{i+1}|X_i) = H_{\partial}(X_{i+1}, X_i) - H_{\partial}(X_i)

    using::
    .. math:: H(X|Y) =  H(X, Y) - H(Y)

    Parameters
    ----------
    X: 2d-array,
     -  multi-dimentional signal space, where each column (axis=1) are the delayed signals 
    present_first: bool, default=True
     - if True, X[:,0] is present, and X[:,1:] is past, in incresing order
     - if True, X[:,-1] is present, and X[:,:-1] is past

    Returns
    -------
    H_x1x: scaler
     - Self-Conditional Entropy

    References
    ----------
    * wikipedia - 

    See Also
    --------
    entropy_diff_cond: Conditional Entropy

    Examples
    --------
    >>> import numpy as np
    >>> import spkit as sp
    >>> x, fs = sp.data.optical_sample(sample=3)
    >>> #x = sp.add_noise(x,snr_db=20)
    >>> X = sp.signal_delayed_space(x,emb_dim=5,delay=2)
    >>> H_x1x = sp.entropy_diff_cond_self(X, present_first=True)
    >>> print('Self-Conditional Entropy')
    >>> print(f'  H(X(i+1)|X(i)) = {H_x1x}')
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
    r"""Conditional Entropy :math:`H_{\partial}(X_{i+1}|X_i,Y_i)`

    Conditional Entropy

    Information of :math:`X(i+1)` given :math:`X(i)` and :math:`Y(i)`


    .. math:: H_{\partial}(X_{i+1}|X_i,Y_i) = H(X_{i+1},X_i,Y_i) - H(X_i,Y_i)
    
    Parameters
    ----------
    X: 2d-array,
     -  multi-dimentional signal space, where each column (axis=1) are the delayed signals 
    
    Y: 2d-array,
     -  multi-dimentional signal space, where each column (axis=1) are the delayed signals 
    
    present_first: bool, default=True
     - if True, X[:,0] is present, and X[:,1:] is past, in incresing order
     - if True, X[:,-1] is present, and X[:,:-1] is past

    Returns
    -------
    H_x1y: scaler
     - Conditional Entropy

    References
    ----------

    See Also
    --------
    entropy_diff_cond_self: Self-Conditional Entropy

    Examples
    --------
    #sp.entropy_diff_cond
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    X, fs, ch_names = sp.data.eeg_sample_14ch()
    X = X - X.mean(1)[:, None]
    X1 = sp.signal_delayed_space(X[:,0].copy(),emb_dim=5,delay=2)
    Y1 = sp.signal_delayed_space(X[:,2].copy(),emb_dim=5,delay=2)
    Y2 = sp.add_noise(Y1,snr_db=0)
    H_xy1 = sp.entropy_diff_cond(X1,Y1,present_first=True)
    H_xy2 = sp.entropy_diff_cond(X1,Y2,present_first=True)
    H_y1x = sp.entropy_diff_cond(Y1,X1,present_first=True)
    H_y2x = sp.entropy_diff_cond(Y2,X1,present_first=True)
    print('Conditional Entropy')
    print(f'- H(X1|Y1) = {H_xy1}')
    print(f'- H(X1|Y2) = {H_xy2}')
    print(f'- H(Y1|X1) = {H_y1x}')
    print(f'- H(Y2|X1) = {H_y2x}')

    """
    # H(X_{i+1}|X_i,Y_i) = H(X_{i+1},X_i,Y_i) - H(X_i,Y_i)

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

def entropy_diff_joint(X,Y):
    r"""Differential Joint Entropy :math:`H_{\partial}(X,Y)`

    Differential Joint Entropy with 

    .. math:: H_{\partial}(X,Y)
    
    Parameters
    ----------
    X: 2d-array, 1d-array
     -  multi-dimentional signal space, where each column (axis=1) are the delayed signals
     -  or 1d-dimensional signal 
    
    Y: 2d-array, 1d-array
     -  multi-dimentional signal space, where each column (axis=1) are the delayed signals
     -  or 1d-dimensional signal 

    Returns
    -------
    H_xy: scaler
     - Differential Joint Entropy

    References
    ----------
    * wiki

    See Also
    --------
    entropy_diff_cond_self: Self-Conditional Entropy
    entropy_diff_cond: Conditional Entropy

    Examples
    --------
    # Example 1
    #sp.entropy_diff_joint
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    X, fs, ch_names = sp.data.eeg_sample_14ch()
    X = X - X.mean(1)[:, None]
    X1 = sp.signal_delayed_space(X[:,0].copy(),emb_dim=5,delay=2)
    Y1 = sp.signal_delayed_space(X[:,2].copy(),emb_dim=5,delay=2)
    Y2 = sp.add_noise(Y1,snr_db=0)
    H_xy1 = sp.entropy_diff_joint(X1,Y1)
    H_xy2 = sp.entropy_diff_joint(X1,Y2)
    print('Conditional Entropy')
    print(f'- H(X1,Y1) = {H_xy1}')
    print(f'- H(X1,Y2) = {H_xy2}')

    #############################
    # Example 2
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    X, fs, ch_names = sp.data.eeg_sample_14ch()
    X = X - X.mean(1)[:, None]
    x1,y1 = X[:,0].copy(),X[:,2].copy()
    #x1 = x1[:,None]
    #y1 = y1[:,None]
    y2 = sp.add_noise(y1,snr_db=0)
    H_xy1 = sp.entropy_diff_joint(x1,y1)
    H_xy2 = sp.entropy_diff_joint(x1,y2)
    print('Conditional Entropy')
    print(f'- H(X1,Y1) = {H_xy1}')
    print(f'- H(X1,Y2) = {H_xy2}')
    """
    assert X.shape[0]==Y.shape[0]

    if X.ndim==1: X = X[:,None]
    if Y.ndim==1: Y = Y[:,None]
    x_y = np.concatenate((X, Y), axis=1)

    H_xy = entropy_differential(x_y, is_multidim=True)

    return H_xy

def entropy_diff_joint_cond(X,Y,present_first=True):
    r"""Joint-Conditional Entropy :math:`H_{\partial}(X_{i+1},Y_{i+1}|X_i,Y_i)`

    Joint-Conditional Entropy

    .. math::  H_{\partial}(X_{i+1},Y_{i+1}|X_i,Y_i) = H(X_{i+1},Y_{i+1},X_i,Y_i) - H(X_i,Y_i)
    
    Parameters
    ----------
    X: 2d-array,
     -  multi-dimentional signal space, where each column (axis=1) are the delayed signals 
    
    Y: 2d-array,
     -  multi-dimentional signal space, where each column (axis=1) are the delayed signals 
    
    present_first: bool, default=True
     - if True, X[:,0] is present, and X[:,1:] is past, in incresing order
     - if True, X[:,-1] is present, and X[:,:-1] is past

    Returns
    -------
    H_x1y: scaler
     - Conditional Joint Entropy

    References
    ----------
    * wiki

    See Also
    --------
    entropy_diff_joint: Joint-Entropy

    Examples
    --------
    #sp.entropy_diff_joint_cond
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    X, fs, ch_names = sp.data.eeg_sample_14ch()
    X = X - X.mean(1)[:, None]
    # Example 1
    X1 = sp.signal_delayed_space(X[:,0].copy(),emb_dim=5,delay=2)
    Y1 = sp.signal_delayed_space(X[:,2].copy(),emb_dim=5,delay=2)
    Y2 = sp.add_noise(Y1,snr_db=0)
    H_xy1 = sp.entropy_diff_joint_cond(X1,Y1)
    H_xy2 = sp.entropy_diff_joint_cond(X1,Y2)
    print('Conditional Entropy')
    print(f'- H(X1(i+1),Y1(i+1) | X1(i),Y1(i)) = {H_xy1}')
    print(f'- H(X1(i+1),Y2(i+1) | X1(i),Y2(i)) = {H_xy2}')
    """

    # H(X_{i+1},Y_{i+1}|X_i,Y_i) = H(X_{i+1},Y_{i+1},X_i,Y_i) - H(X_i,Y_i)

    if present_first:
        xi_yi = np.concatenate((X[:, 1:], Y[:, 1:]), axis=1)
    else:
        xi_yi = np.concatenate((X[:, :-1], Y[:, :-1]), axis=1)


    H_xiyi = entropy_differential(xi_yi, is_multidim=True)

    H_xy = entropy_diff_joint(X,Y)

    H_xy1xy = H_xy - H_xiyi

    return H_xy1xy

def mutual_info_diff_self(X,present_first=True):
    r"""Self Mutual Information :math:`I_{\partial}(X_{i+1}; X_i)`

    **Self Mutual Information**
    
    Predictibility of :math:`X(i+1)` given :math:`X(i)`


    .. math:: I_{\partial}(X_{i+1}; X_i) = H(X_{i+1}) - H(X_{i+1} | X_i)
    
    Parameters
    ----------
    X: 2d-array,
     -  multi-dimentional signal space, where each column (axis=1) are the delayed signals 
    present_first: bool, default=True
     - if True, X[:,0] is present, and X[:,1:] is past, in incresing order
     - if True, X[:,-1] is present, and X[:,:-1] is past

    Returns
    -------
    I_x1x: scaler
     - Self-Mutual Information

    References
    ----------
    * wiki

    See Also
    --------
    entropy_diff_joint: Joint-Entropy
    mutual_info_diff_self: Self-Mutual Information

    Examples
    --------
    #sp.mutual_info_diff_self
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    X, fs, ch_names = sp.data.eeg_sample_14ch()
    X = X - X.mean(1)[:, None]
    # Example 1
    X1 = sp.signal_delayed_space(X[:,0].copy(),emb_dim=5,delay=2)
    Y1 = sp.add_noise(X1,snr_db=0)
    I_x1x = sp.mutual_info_diff_self(X1)
    I_y1y = sp.mutual_info_diff_self(Y1)
    print('Self-Mutual Information')
    print(f'- I(X(i+1)| X(i)) = {I_x1x}')
    print(f'- I(Y(i+1)| Y(i)) = {I_y1y}')

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
    r"""Mutual Information  :math:`I_{\partial}(X_{i+1}; X_i, Y_i)`

    **Mutual Information**
    
    Predictibility of :math:`X(i+1)` given :math:`X(i)` and :math:`Y(i)`

    .. math:: I_{\partial}(X_{i+1}; X_i, Y_i) = H_{\partial}(X_{i+1}) - H_{\partial}(X_{i+1} | X_i, Y_i)

    .. math:: H_{\partial}(X_{i+1}|X_i,Y_i) = H_{\partial}(X_{i+1},X_i,Y_i) - H_{\partial}(X_i,Y_i)

    Parameters
    ----------
    X: 2d-array,
     -  multi-dimentional signal space, where each column (axis=1) are the delayed signals 
    
    Y: 2d-array,
     -  multi-dimentional signal space, where each column (axis=1) are the delayed signals 
    
    present_first: bool, default=True
     - if True, X[:,0] is present, and X[:,1:] is past, in incresing order
     - if True, X[:,-1] is present, and X[:,:-1] is past

    Returns
    -------
    I_x1y: scaler
     - Mutual Information

    References
    ----------
    * wiki

    See Also
    --------
    mutual_info_diff_self: Self-Mutual Information
    entropy_diff_joint: Joint-Entropy

    Examples
    --------
    #sp.mutual_info_diff
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    X, fs, ch_names = sp.data.eeg_sample_14ch()
    X = X - X.mean(1)[:, None]
    # Example 1
    X1 = sp.signal_delayed_space(X[:,0].copy(),emb_dim=5,delay=2)
    Y1 = sp.signal_delayed_space(X[:,2].copy(),emb_dim=5,delay=2)
    Y2 = sp.add_noise(Y1,snr_db=0)
    I_xy1 = sp.mutual_info_diff(X1,Y1)
    I_xy2 = sp.mutual_info_diff(X1,Y2)
    print('Mutual-Information')
    print(f'- I(X1,Y1) = {I_xy1}')
    print(f'- I(X1,Y2) = {I_xy2}')
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
    r""" Transfer Entropy :math:`TE_{X->Y}`

    Transfer Entropy


    :math:`TE_{X->Y}  = I(Y_{i+1}, X_i | Y_i)`

    :math:`TE_{X->Y}  = H(Y_{i+1} | Y_i) - H(Y_{i+1} | X_i, Y_i)`           [Eq1]

    :math:`TE_{X->Y}  = H(Y_{i+1}, Y_i) - H(Y_i) - H(Y_{i+1},X_i,Y_i) +  H(X_i,Y_i)`

    :math:`TE_{X->Y}  = H(X_i,Y_i)  +  H(Y_{i+1}, Y_i) - H(Y_{i+1},X_i,Y_i) - H(Y_i)`  [Eq2]

    Using:
        * :math:`H(X_{i+1}|X_i)     = H(X_{i+1}, X_i) - H(X_i)`  - entropy_diff_cond_self(X)
        * :math:`H(X_{i+1}|X_i,Y_i) = H(X_{i+1},X_i,Y_i) - H(X_i,Y_i)`  - entropy_diff_cond(X,Y)

    Parameters
    ----------
    X: 2d-array,
     -  multi-dimentional signal space, where each column (axis=1) are the delayed signals 
    
    Y: 2d-array,
     -  multi-dimentional signal space, where each column (axis=1) are the delayed signals 
    
    present_first: bool, default=True
     - if True, X[:,0] is present, and X[:,1:] is past, in incresing order
     - if True, X[:,-1] is present, and X[:,:-1] is past

    Returns
    -------
    TE_x2y: scaler
     - Transfer Entropy from x to y

    References
    ----------
    * wikipedia - https://en.wikipedia.org/wiki/Transfer_entropy

    See Also
    --------
    transfer_entropy_cond: Conditional Transfer Entropy
    partial_transfer_entropy: Partial Transfer Entropy
    entropy_granger_causality: Granger Causality based on Differential Entropy

    Examples
    --------
    #sp.transfer_entropy
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    X, fs, ch_names = sp.data.eeg_sample_14ch()
    X = X - X.mean(1)[:, None]
    # Example 1
    X1 = sp.signal_delayed_space(X[:,0].copy(),emb_dim=5,delay=2)
    Y1 = sp.signal_delayed_space(X[:,2].copy(),emb_dim=5,delay=2)
    Y2 = sp.add_noise(Y1,snr_db=0)
    TE_x_y1 = sp.transfer_entropy(X1,Y1)
    TE_x_y2 = sp.transfer_entropy(X1,Y2)
    TE_y1_x = sp.transfer_entropy(Y1,X1)
    TE_y2_x = sp.transfer_entropy(Y2,X1)
    TE_y1_y2 = sp.transfer_entropy(Y1,Y2)
    print('Transfer Entropy')
    print(f'- TE(X1->Y1) = {TE_x_y1}')
    print(f'- TE(X1->Y2) = {TE_x_y2}')
    print(f'- TE(Y1->X1) = {TE_y1_x}')
    print(f'- TE(Y2->X1) = {TE_y2_x}')
    print(f'- TE(Y1->Y2) = {TE_y1_y2}')
    """

    H_y1y  = entropy_diff_cond_self(Y,present_first=present_first)
    H_y1xy = entropy_diff_cond(Y,X,present_first=present_first)

    TE_x2y = H_y1y - H_y1xy

    return TE_x2y

def transfer_entropy_cond(X,Y,Z,present_first=True):
    r"""Conditional Transfer Entopry Or Partial Transfer Entropy :math:`TE_{X->Y | Z}`

    Conditional Transfer Entopry Or Partial Transfer Entropy

    .. math:: TE_{X->Y | Z}   =  I(Y_{i+1}, X_i | Y_i, Z_i)


    .. math:: TE_{X->Y | Z}  =  H(X_i,Y_i, Z_i)  +  H(Y_{i+1}, Y_i, Z_i) - H(Y_{i+1},X_i,Y_i, Z_i) - H(Y_i, Z_i)

    Parameters
    ----------
    X: 2d-array,
     -  multi-dimentional signal space, where each column (axis=1) are the delayed signals 
    
    Y: 2d-array,
     -  multi-dimentional signal space, where each column (axis=1) are the delayed signals 

    Z: 2d-array,
     -  multi-dimentional signal space, where each column (axis=1) are the delayed signals 
    
    present_first: bool, default=True
     - if True, X[:,0] is present, and X[:,1:] is past, in incresing order
     - if True, X[:,-1] is present, and X[:,:-1] is past

    Returns
    -------
    TE_x2y1z: scaler
     - Conditional Transfer Entropy, transfer entropy from x to y, given z

    References
    ----------
    * wikipedia - https://en.wikipedia.org/wiki/Transfer_entropy

    See Also
    --------
    transfer_entropy: Transfer Entropy
    partial_transfer_entropy: Partial Transfer Entropy
    entropy_granger_causality: Granger Causality based on Differential Entropy

    Examples
    --------
    #sp.transfer_entropy_cond
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    X, fs, ch_names = sp.data.eeg_sample_14ch()
    X = X - X.mean(1)[:, None]
    # Example 1
    X1 = sp.signal_delayed_space(X[:,0].copy(),emb_dim=5,delay=2)
    Y1 = sp.signal_delayed_space(X[:,2].copy(),emb_dim=5,delay=2)
    Z1 = sp.signal_delayed_space(X[:,4].copy(),emb_dim=5,delay=2)
    Y2 = sp.add_noise(Y1,snr_db=0)
    TE_x_y1_1z = sp.transfer_entropy_cond(X1,Y1,Z1)
    TE_x_y2_1z = sp.transfer_entropy_cond(X1,Y2,Z1)
    TE_y1_x_1z = sp.transfer_entropy_cond(Y1,X1,Z1)
    TE_y2_x_1z = sp.transfer_entropy_cond(Y2,X1,Z1)
    print('Conditional Transfer Entropy')
    print(f'- TE(X1->Y1 | Z1) = {TE_x_y1_1z}')
    print(f'- TE(X1->Y2 | Z1) = {TE_x_y2_1z}')
    print(f'- TE(Y1->X1 | Z1) = {TE_y1_x_1z}')
    print(f'- TE(Y2->X1 | Z1) = {TE_y2_x_1z}')
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
    r"""Partial Transfer Entropy Or Conditional Transfer Entropy :math:`TE_{X->Y | Z}`

    Partial Transfer Entropy Or Conditional Transfer Entropy

    .. math:: TE_{X->Y | Z}   =  I(Y_{i+1}, X_i | Y_i, Z_i)


    .. math:: TE_{X->Y | Z}  =  H(X_i,Y_i, Z_i)  +  H(Y_{i+1}, Y_i, Z_i) - H(Y_{i+1},X_i,Y_i, Z_i) - H(Y_i, Z_i)

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    See Also
    --------

    Examples
    --------
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
    r"""Partial Transfer Entropy Or Conditional Transfer Entopry :math:`TE_{X->Y | Z}`

    Partial Transfer Entropy Or Conditional Transfer Entopry

    .. math:: TE_{X->Y | Z}   =  I(Y_{i+1}, X_i | Y_i, Z_i)


    .. math:: TE_{X->Y | Z}  =  H(X_i,Y_i, Z_i)  +  H(Y_{i+1}, Y_i, Z_i) - H(Y_{i+1},X_i,Y_i, Z_i) - H(Y_i, Z_i)

    Parameters
    ----------
    X: 2d-array,
     -  multi-dimentional signal space, where each column (axis=1) are the delayed signals 
    
    Y: 2d-array,
     -  multi-dimentional signal space, where each column (axis=1) are the delayed signals 

    Z: 2d-array,
     -  multi-dimentional signal space, where each column (axis=1) are the delayed signals 
    
    present_first: bool, default=True
     - if True, X[:,0] is present, and X[:,1:] is past, in incresing order
     - if True, X[:,-1] is present, and X[:,:-1] is past

    Returns
    -------
    TE_x2y1z: scaler
     - Conditional Transfer Entropy, transfer entropy from x to y, given z

    References
    ----------
    * wikipedia - https://en.wikipedia.org/wiki/Transfer_entropy

    See Also
    --------
    transfer_entropy: Transfer Entropy
    partial_transfer_entropy: Partial Transfer Entropy
    entropy_granger_causality: Granger Causality based on Differential Entropy


    Examples
    --------
    #sp.partial_transfer_entropy
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    X, fs, ch_names = sp.data.eeg_sample_14ch()
    X = X - X.mean(1)[:, None]
    # Example 1
    X1 = sp.signal_delayed_space(X[:,0].copy(),emb_dim=5,delay=2)
    Y1 = sp.signal_delayed_space(X[:,2].copy(),emb_dim=5,delay=2)
    Z1 = sp.signal_delayed_space(X[:,4].copy(),emb_dim=5,delay=2)
    Y2 = sp.add_noise(Y1,snr_db=0)
    TE_x_y1_1z = sp.partial_transfer_entropy(X1,Y1,Z1)
    TE_x_y2_1z = sp.partial_transfer_entropy(X1,Y2,Z1)
    TE_y1_x_1z = sp.partial_transfer_entropy(Y1,X1,Z1)
    TE_y2_x_1z = sp.partial_transfer_entropy(Y2,X1,Z1)
    print('Partial Transfer Entropy')
    print(f'- TE(X1->Y1 | Z1) = {TE_x_y1_1z}')
    print(f'- TE(X1->Y2 | Z1) = {TE_x_y2_1z}')
    print(f'- TE(Y1->X1 | Z1) = {TE_y1_x_1z}')
    print(f'- TE(Y2->X1 | Z1) = {TE_y2_x_1z}')
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
    r"""Granger Causality based on Differential Entropy :math:`GC_{X->Y}, GC_{Y->X}, GC_{X.Y}`

    Granger Causality based on Differential Entropy

    * (1) :math:`GC_{XY} (X->Y)`  : :math:`H(Y_{i+1}|Y_i) - H(Y_{i+1}|X_i,Y_i)`
    * (2) :math:`GC_{YX} (Y->X)`  : :math:`H(X_{i+1}|X_i) - H(X_{i+1}|X_i,Y_i)`
    * (3) :math:`GC_{XdY} (X.Y)`  : :math:`H(Y_{i+1}|X_i,Y_i) +  H(X_{i+1}|X_i,Y_i) -    H(X_{i+1},Y_{i+1}|X_i,Y_i)`


    if normalize True
        * :math:`GC_{XY} = GC_{XY}/(I(Y_{i+1}; Y_i) + GC_{XY})`
        * :math:`GC_{YX} = GC_{YX}/(I(X_{i+1}; X_i) + GC_{YX})`

    Using:
        * :math:`H(Y_{i+1}|Y_i) = H(Y_{i+1}, Y_i) - H(Y_i)`
        * :math:`H(X_{i+1}|X_i) = H(X_{i+1}, X_i) - H(X_i)`
        * :math:`H(Y_{i+1}|X_i,Y_i) = H(Y_{i+1},X_i,Y_i) - H(X_i,Y_i)`
        * :math:`H(X_{i+1}|X_i,Y_i) = H(X_{i+1},X_i,Y_i) - H(X_i,Y_i)`
        * :math:`H(X_{i+1},Y_{i+1}|X_i,Y_i) = H(X_{i+1},Y_{i+1},X_i,Y_i) - H(X_i,Y_i)`
        * :math:`I(X_{i+1}; X_i) = H(X_{i+1}) - H(X_{i+1} | X_i)`
        * :math:`I(Y_{i+1}; Y_i) = H(Y_{i+1}) - H(Y_{i+1} | Y_i)`

    Parameters
    ----------
    X: 2d-array,
     -  multi-dimentional signal space, where each column (axis=1) are the delayed signals 
    
    Y: 2d-array,
     -  multi-dimentional signal space, where each column (axis=1) are the delayed signals 
    
    normalize: bool, default=False
     - if True, GC is normalised

    present_first: bool, default=True
     - if True, X[:,0] is present, and X[:,1:] is past, in incresing order
     - if True, X[:,-1] is present, and X[:,:-1] is past


    Returns
    -------
    gc_xy: scaler
     - Granger Causality from x to y
    gc_yx: scaler
     - Granger Causality from y to x
    gc_xdy: scaler
     - Granger Causality (x  y)

    References
    ----------
    * wikipedia

    See Also
    --------
    transfer_entropy: Transfer Entropy
    transfer_entropy_cond: Conditional Transfer Entropy
    partial_transfer_entropy: Partial Transfer Entropy

    Examples
    --------
    #sp.entropy_granger_causality
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    X, fs, ch_names = sp.data.eeg_sample_14ch()
    X = X - X.mean(1)[:, None]
    # Example 1
    X1 = sp.signal_delayed_space(X[:,0].copy(),emb_dim=5,delay=2)
    Y1 = sp.signal_delayed_space(X[:,2].copy(),emb_dim=5,delay=2)
    Z1 = sp.signal_delayed_space(X[:,4].copy(),emb_dim=5,delay=2)
    Y2 = sp.add_noise(Y1,snr_db=0)
    gc_x1y1, gc_y1x1,gc_x1dy1 = sp.entropy_granger_causality(X1,Y1)
    gc_x1y2, gc_y2x1,gc_x1dy2 = sp.entropy_granger_causality(X1,Y2)
    print('Granger Causality : X1,Y1')
    print(f'- GC(X1->Y1) = {gc_x1y1}')
    print(f'- GC(Y1->X1) = {gc_y1x1}')
    print(f'- GC(X1,Y1)  = {gc_x1dy1}')
    print('-'*10)
    print('Granger Causality : X1,Y2')
    print(f'- GC(X1->Y2) = {gc_x1y2}')
    print(f'- GC(Y2->X1) = {gc_y2x1}')
    print(f'- GC(X1,Y2)  = {gc_x1dy2}')
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
    """Usuful Formulas

    |
    |
    |     Usuful Formulas
    |
    |

    **Differential Entropy of Normally distribuated Multivariant X**
    

    for            .. math:: x ∼ N(μ,Σ)

    entropy in nats

        .. math:: H(x) =  (1/2)ln|Σ| + (1/2)n + (n/2)ln(2π)


    (1/2)n + (n/2)ln(2π) => are constant values for fixed dimension


    code::

        H_x = entropy_differential(x,is_multidim=True, emb_dim=1, delay=1,)



    **Self-Conditional Entropy**
    
    Information of X(i+1) given X(i)

    :math:`H(X_{i+1}|X_i) = H(X_{i+1}, X_i) - H(X_i)`

    using:
    :math:`H(X|Y) =  H(X, Y) - H(Y)`

    code::

        H_x1x = entropy_diff_cond_self(X, present_first=True)



    **Conditional Entropy**

    Information of X(i+1) given X(i) and Y(i)

    :math:`H(X_{i+1}|X_i,Y_i) = H(X_{i+1},X_i,Y_i) - H(X_i,Y_i)`


    code::
        H_x1xy = entropy_diff_cond(X,Y,present_first=True)




    **Joint Entropy**

    :math:`H(X,Y)`

    code::

        H_xy = entropy_diff_joint(X,Y)


    **Joint-Conditional Entropy**

    :math:`H(X_{i+1},Y_{i+1}|X_i,Y_i) = H(X_{i+1},Y_{i+1},X_i,Y_i) - H(X_i,Y_i)`


    code::

        H_xy1xy = entropy_diff_joint_cond(X,Y,present_first=True)


    **Self Mutual Information**

    Predictibility of X(i+1) given X(i)


    :math:`I(X_{i+1}; X_i) = H(X_{i+1}) - H(X_{i+1} | X_i)`

    code::

        I_xx = mutual_info_diff_self(X,present_first=True)


    **Mutual Information**

    Predictibility of X(i+1) given X(i) and Y(i)

    :math:`I(X_{i+1}; X_i, Y_i) = H(X_{i+1}) - H(X_{i+1} | X_i, Y_i)`

    :math:`H(X_{i+1}|X_i,Y_i) = H(X_{i+1},X_i,Y_i) - H(X_i,Y_i)`


    code::

        I_xy = mutual_info_diff(X,Y,present_first=True)


    **Transfer Entropy**

    :math:`TE_{X->Y}  = I(Y_{i+1}, X_i | Y_i)`

    :math:`TE_{X-->Y}  = H(Y_{i+1} | Y_i) - H(Y_{i+1} | X_i, Y_i)`          [Eq1]



    :math:`TE_{X-->Y}  = H(Y_{i+1}, Y_i) - H(Y_i) - H(Y_{i+1},X_i,Y_i) +  H(X_i,Y_i)`

    :math:`TE_{X-->Y}  = H(X_i,Y_i)  +  H(Y_{i+1}, Y_i) - H(Y_{i+1},X_i,Y_i) - H(Y_i)`  [Eq2]


    Using:
    :math:`H(X_{i+1}|X_i)     = H(X_{i+1}, X_i) - H(X_i)`          |  entropy_diff_cond_self(X)
    :math:`H(X_{i+1}|X_i,Y_i) = H(X_{i+1},X_i,Y_i) - H(X_i,Y_i)`   |  entropy_diff_cond(X,Y)

    code::

        TE_x2y = transfer_entropy(X,Y,present_first=True)


    **Partial Transfer Entropy Or Conditional Transfer Entopry**

    :math:`TE_{X-->Y | Z}   =  I(Y_{i+1}, X_i | Y_i, Z_i)`


    :math:`TE_{X-->Y | Z}  =  H(X_i,Y_i, Z_i)  +  H(Y_{i+1}, Y_i, Z_i) - H(Y_{i+1},X_i,Y_i, Z_i) - H(Y_i, Z_i)`

    code::

        TE_x2y1z = partial_transfer_entropy(X,Y,Z,present_first=True,verbose=False)

    **Granger Causality based on Differential Entropy**
    

    (1) GC_XY (X-->Y)  : :math:`H(Y_{i+1}|Y_i) - H(Y_{i+1}|X_i,Y_i)`
    (2) GC_YX (Y-->X)  : :math:`H(X_{i+1}|X_i) - H(X_{i+1}|X_i,Y_i)`
    (3) GC_XdY (X.Y)   : :math:`H(Y_{i+1}|X_i,Y_i) +  H(X_{i+1}|X_i,Y_i) -    H(X_{i+1},Y_{i+1}|X_i,Y_i)`


    if normalize True
        :math:`GC_XY = GC_XY/(I(Y_{i+1}; Y_i) + GC_XY )`
        :math:`GC_YX = GC_YX/(I(X_{i+1}; X_i) + GC_YX )`


    Using::
    :math:`H(Y_{i+1}|Y_i) = H(Y_{i+1}, Y_i) - H(Y_i)`
    :math:`H(X_{i+1}|X_i) = H(X_{i+1}, X_i) - H(X_i)`
    :math:`H(Y_{i+1}|X_i,Y_i) = H(Y_{i+1},X_i,Y_i) - H(X_i,Y_i)`
    :math:`H(X_{i+1}|X_i,Y_i) = H(X_{i+1},X_i,Y_i) - H(X_i,Y_i)`
    :math:`H(X_{i+1},Y_{i+1}|X_i,Y_i) = H(X_{i+1},Y_{i+1},X_i,Y_i) - H(X_i,Y_i)`

    :math:`I(X_{i+1}; X_i) = H(X_{i+1}) - H(X_{i+1} | X_i)
    :math:`I(Y_{i+1}; Y_i) = H(Y_{i+1}) - H(Y_{i+1} | Y_i)`


    code::

        gc_xy, gc_yx,gc_xdy = entropy_granger_causality(X,Y,present_first=True, normalize=False)

    """

    print(help(show_farmulas))
