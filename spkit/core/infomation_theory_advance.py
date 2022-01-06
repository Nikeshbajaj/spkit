'''
Advance Information Theory techniques
--------------------------------
Author @ Nikesh Bajaj
updated on Date: 1 Jan 2022
Version : 0.0.1
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
from scipy import stats
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
    ncdf  = stats.norm(loc=mu, scale=sig)
    x_cdf = ncdf.cdf(x)
    return x_cdf

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
