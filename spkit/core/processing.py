"""
Basic signal processing methods
--------------------------------
Author @ Nikesh Bajaj
updated on Date: 27 March 2023. Version : 0.0.5
updated on Date: 26 Sep 2021, Version : 0.0.4
Github :  https://github.com/Nikeshbajaj/spkit
Contact: n.bajaj@qmul.ac.uk | n.bajaj@imperial.ac.uk
"""

from __future__ import absolute_import, division, print_function
name = "Signal Processing toolkit | Processing"
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
import scipy, copy #spkit
from scipy import signal
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
from scipy import stats as scipystats
from copy import deepcopy
import pywt as wt
from scipy.interpolate import interp1d, interp2d
import seaborn as sns

sys.path.append("..")
#sys.path.append(".")
from .information_theory import entropy
from .advance_techniques import peak_interp, dft_analysis
from ..utils import ProgBar, ProgBar_JL
from ..utils_misc.borrowed import resize
from ..utils import deprecated

@deprecated("due to naming consistency, please use 'filterDC' for updated/improved functionality. [spkit-0.0.9.7]")
def filterDC_(x,alpha=256):
    r"""TO BE DEPRECIATED  - use filterDC instead

    Filter out DC component - Remving drift using Recursive (IIR type) filter

          y[n] = ((alpha-1)/alpha) * ( x[n] - x[n-1] -y[n-1])

          where y[-1] = x[0], x[-1] = x[0]
          resulting y[0] = 0
    
    Parameters
    ----------
    x    : (vector) input signal

    alpha: (scalar) filter coefficient, higher it is, more suppressed dc component (0 frequency component)
         : with alpha=256, dc component is suppressed by 20 dB

    initialize_zero: (bool): If True, running backgrpund b will be initialize it with x[0], resulting y[0] = 0
          if False, b = 0, resulting y[0] ~ x[0], and slowly drifting towards zeros line
          - recommended to set True

    Returns
    ------
        y : output vector

    """
    b = x[0]
    y = np.zeros(len(x))
    for i in range(len(x)):
        b = ((alpha - 1) * b + x[i]) / alpha
        y[i] = x[i]-b
    return y

@deprecated("due to naming consistency, please use 'filterDC' for updated/improved functionality. [spkit-0.0.9.7]")
def filterDC_X(X,alpha=256,return_background=False,initialize_zero=True):
    r"""TO BE DEPRECIATED   - use filterDC instead

    Filter out DC component - Remving drift using Recursive (IIR type) filter

          y[n] = ((alpha-1)/alpha) * ( x[n] - x[n-1] -y[n-1])

          where y[-1] = x[0], x[-1] = x[0]
          resulting y[0] = 0
    
    Parameters
    -----------
    x    : (vecctor) input signal

    alpha: (scalar) filter coefficient, higher it is, more suppressed dc component (0 frequency component)
         : with alpha=256, dc component is suppressed by 20 dB

    initialize_zero: (bool): If True, running backgrpund b will be initialize it with x[0], resulting y[0] = 0
          if False, b = 0, resulting y[0] ~ x[0], and slowly drifting towards zeros line
          - recommended to set True

    Returns
    -------
        y : output vector

    """
    B = X[0] if initialize_zero else 0*X[0]
    if return_background:
        Bg = np.zeros_like(X)
    Y = np.zeros_like(X)
    for i in range(X.shape[0]):
        B = ((alpha - 1) * B + X[i]) / alpha
        Y[i] = X[i]-B
        if return_background: Bg[i]= copy.copy(B)
    if return_background: return Y, Bg
    return Y

@deprecated("due to naming consistency, please use 'filter_X' for updated/improved functionality. [spkit-0.0.9.7]")
def filter_X_(X,fs=128.0,band =[0.5],btype='highpass',order=5,ftype='filtfilt',verbose=1,use_joblib=False):
    r"""Buttorworth filtering -  basic filtering
    

    X : (vecctor) input signal single channel (n,) or multi-channel, channel axis should be 1 shape ~ (n,ch)

    band: cut of frequency, for lowpass and highpass, band is list of one, for bandpass list of two numbers
    btype: filter type
    order: order of filter
    ftype: filtering approach type, 'filtfilt', 'lfilter', 'SOS',
         : lfilter is causal filter, which introduces delay, filtfilt does not introduce any delay, but it is non-causal filtering
          SOS:  Filter a signal using IIR Butterworth SOS method. A forward-backward digital filter using cascaded second-order sections:
    Xf: filtered signal of same size as X
    """
    if verbose: print(X.shape, 'channels axis = 1')

    b,a = signal.butter(order,np.array(band)/(0.5*fs),btype=btype)

    if ftype=='lfilter':
        if np.ndim(X)>1:
            if use_joblib:
                try:
                    Xf  = np.array(Parallel(n_jobs=-1)(delayed(signal.lfilter)(b,a,X[:,i]) for i in range(X.shape[1]))).T
                except:
                    print('joblib paraller failed computing with loops- turn off --> use_joblib=False')
                    Xf  = np.array([signal.lfilter(b,a,X[:,i]) for i in range(X.shape[1])]).T
            else:
                Xf  = np.array([signal.lfilter(b,a,X[:,i]) for i in range(X.shape[1])]).T
        else:
            Xf  = signal.lfilter(b,a,X)
    elif ftype=='filtfilt':
        if np.ndim(X)>1:
            if use_joblib:
                try:
                    Xf  = np.array(Parallel(n_jobs=-1)(delayed(signal.filtfilt)(b,a,X[:,i]) for i in range(X.shape[1]))).T
                except:
                    print('joblib paraller failed computing with loops- turn off --> use_joblib=False')
                    Xf  = np.array([signal.filtfilt(b,a,X[:,i]) for i in range(X.shape[1])]).T
            else:
                Xf  = np.array([signal.filtfilt(b,a,X[:,i]) for i in range(X.shape[1])]).T
        else:
            Xf  = signal.filtfilt(b,a,X)
    else:
        raise NameError('ftype')
    return Xf

@deprecated("due to naming consistency, please use 'periodogram' for updated/improved functionality. [spkit-0.0.9.7]")
def Periodogram(x,fs=128,method ='welch',win='hann',nfft=None,scaling='density',average='mean',detrend='constant',nperseg=None, noverlap=None,show=False,showlog=True):
    r"""Computing Periodogram

    **Computing Periodogram using Welch or Periodogram method**

    Parameters
    ----------
    x: 1d array, (n,)
        - input signal, 
    fs: sampling frequency
    method: {'welch','None'}
        -  if None, then periodogram is computed without welch method
    win: {'hann', 'ham',}
        - window function
    scaling: {'density', 'spectrum'}
        - 'density'--V**2/Hz 'spectrum'--V**2
    average: {'mean', 'median'}
        - averaging method
    detrend: False, 'constant', 'linear'
    nfft:  None, n-point FFT

    Returns
    -------
    Px: |periodogram|

    References
    ----------
    * wikipedia - https://en.wikipedia.org/wiki/Periodogram
    
    
    Notes
    -----
    #

    See Also
    --------
    periodogram: new version

    Examples
    --------
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs = sp.load_data.eeg_sample_1ch()
    Px1 = sp.Periodogram(x,fs=128,method ='welch',nperseg=128)
    Px2 = sp.Periodogram(x,fs=128,method =None,nfft=128)
    frq = (fs/2)*np.arange(len(Px1))/(len(Px1)-1)
    plt.figure(figsize=(5,4))
    plt.plot(frq,np.log(Px1),label='Welch')
    plt.plot(frq,np.log(Px2),label='Periodogram')
    plt.xlim([frq[0],frq[-1]])
    plt.grid()
    plt.ylabel('V**2/Hz')
    plt.xlabel('Frequency (Hz)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    """
    if method ==None:
        frq, Px = scipy.signal.periodogram(x,fs,win,nfft=nfft,scaling=scaling,detrend=detrend)
    elif method =='welch':
        #f, Pxx = scipy.signal.welch(x,fs,win,nperseg=np.clip(len(x),0,256),scaling=scaling,average=average,detrend=detrend)
        frq, Px = scipy.signal.welch(x,fs,win,nperseg=nperseg,noverlap=noverlap,average=average,nfft=nfft,scaling=scaling,detrend=detrend)
    if show:
        if showlog:
            plt.plot(frq,np.log(np.abs(Px)))
        else:
            plt.plot(frq,np.abs(Px))
        plt.xlim([frq[0],frq[-1]])
        plt.xlabel('Frequency (Hz)')
        if scaling=='density':
            plt.ylabel('V**2/Hz')
        elif scaling=='spectrum':
            plt.ylabel('V**2')
        plt.grid()
        plt.show()
    
    return np.abs(Px)

@deprecated("due to naming/module consistency, please use 'sp.stats.get_stats' for updated/improved functionality. [spkit-0.0.9.7]")
def getStats(x,detail_level=1,return_names=False):
    r"""Quick Statistics of a given sequence x, excluding NaN values


    returns stats and names of statistics measures

    Parameters
    ----------

    Returns
    -------

    References
    ----------
    * wikipedia - 
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    spkit: #TODO

    Examples
    --------
    import numpy as np
    import spkit as sp
    #TODO
    """
    stats_names =['mean','sd','median','min','max','n','q25','q75','iqr','kur','skw','gmean','entropy']
    esp=1e-5
    if isinstance(x,int) or isinstance(x,float): x =  [x]
    if isinstance(x,list):x = np.array(x)
    assert len(x.shape)==1
    #logsum = self.get_exp_log_sum(x)
    x = x+esp
    mn = np.nanmean(x)
    sd = np.nanstd(x)
    md = np.nanmedian(x)
    min0 = np.nanmin(x)
    max0 = np.nanmax(x)

    n = len(x) - sum(np.isnan(x))

    if detail_level==1:
        return np.r_[mn,sd,md,min0,max0,n], stats_names[:6]

    q25 = np.nanquantile(x,0.25)
    q75 = np.nanquantile(x,0.75)
    iqr = scipystats.iqr(x[~np.isnan(x)])
    kur = scipystats.kurtosis(x,nan_policy='omit')
    skw = scipystats.skew(x[~np.isnan(x)])
    if detail_level==2:
        return np.r_[mn,sd,md,min0,max0,n,q25,q75,iqr,kur,skw], stats_names[:11]

    gmn = scipystats.gmean(x[~np.isnan(x)])
    entropy = entropy(x[~np.isnan(x)])
    names =['mean','sd','median','min','max','n','q25','q75','iqr','kur','skw','gmean','entropy']
    return np.r_[mn,sd,md,min0,max0,n,q25,q75,iqr,kur,skw,gmn,entropy], stats_names

@deprecated("due to naming/module consistency, please use sp.stats.get_stats' for updated/improved functionality. [spkit-0.0.9.7]")
def getQuickStats(x):
    r"""
    """
    if isinstance(x,int) or isinstance(x,float): x =  [x]
    if isinstance(x,list):x = np.array(x)
    n = len(x)-np.sum(np.isnan(x))
    mn = np.nanmean(x)
    md = np.nanmedian(x)
    sd = np.nanstd(x)
    se = sd/np.sqrt(n-1)
    min0 = np.nanmin(x)
    max0 = np.nanmax(x)
    return [mn,sd,se,md,min0,max0,n]

@deprecated("due to naming consistency, please use 'sp.stats.outliers' for updated/improved functionality. [spkit-0.0.9.7]")
def OutLiers(x, method='iqr',k=1.5, include_lower=True,include_upper=True,return_lim=False):
    r"""Identyfying outliers

    Using
    1. Interquartile Range: below Q1 - k*IQR and above Q3 + k*IQR
    2. Stander Deviation:   below Mean -k*SD(x) above Mean + k*SD(x)

    Parameters
    ----------
    x :  1d array or nd-array

    method = 'iqr' or 'sd'
    k : (default 1.5), factor for range, for SD k=2 is widely used
    include_lower: if False, excluding lower outliers
    include_upper: if False excluding upper outliers
     - At least one of (include_lower, include_upper) should be True
    return_lim: if True, return includes lower and upper limits (lt, ul)

    Returns
    -------
    idx: index of outliers in x
    idx_bin: binary array of same size as x, indicating outliers
    (lt,ut): lower and upper limit for outliers, if  return_lim is True


    References
    ----------
    * wikipedia - 
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    spkit: #TODO

    Examples
    --------
    import numpy as np
    import spkit as sp
    #TODO
    """

    assert (include_upper+include_lower)
    xi = x.copy()
    if method =='iqr':
        q1 = np.nanquantile(xi,0.25)
        q3 = np.nanquantile(xi,0.75)
        ut = q3 + k*(q3-q1)
        lt = q1 - k*(q3-q1)
    elif method =='sd':
        sd = np.nanstd(xi)
        ut = np.nanmean(xi) + k*sd
        lt = np.nanmean(xi) - k*sd
    else:
        print('Define method')
        return None

    if not(include_lower): lt = -np.inf
    idx_bin = (xi>=ut) | (xi<=lt)
    idx = np.where(idx_bin)
    if return_lim:
        return idx, idx_bin, (lt,ut)
    return idx, idx_bin


#New Updated Versions
def filterDC(X,alpha=256,return_background=False,initialize_zero=True):
    r"""Filter out DC component using IIR filter

    Filter out DC component - Remving drift using Recursive (IIR type) filter
    
          .. math::
             y[n] = ((alpha-1)/alpha) * ( x[n] - x[n-1] -y[n-1])

          where :math:`y[-1] = x[0]`, :math:`x[-1] = x[0]`
          resulting :math:`y[0] = 0`

    Implemenatation works for single (1d array) or multi-channel (2d array)

    Parameters
    ----------

    X : array, 1d, 2d, (n,) or (n,ch) 
        - input signal single channel (n,) or multi-channel, channel axis should be 1 shape ~ (n,ch)

    alpha: scalar, default alpha=256
        - filter coefficient, higher it is, more suppressed dc component (0 frequency component)
        - with alpha=256, dc component is suppressed by 20 dB

    initialize_zero: bool, default True
        - If True, running background b will be initialize it with x[0], resulting y[0] = 0
        - if False, b = 0, resulting y[0] ~ x[0], and slowly drifting towards zeros line
        - recommended to set True

    Returns
    -------

    Y : output vector, 
    - shape same as input X (n,) or (n,ch)


    References
    ----------
    *  https://en.wikipedia.org/wiki/DC_bias

    Notes
    -----
    -  :func:`filterDC` employes causal IIR filter to remove drift from signal, which introduces the shift (delay) in the filtered signal.
       It is recommonded to check out :func:`filterDC_sGolay`  

    See Also
    --------
    filterDC_sGolay: filter out DC using Savitzky-Golay filter

    Examples
    --------
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x = sp.create_signal_1d(n=1000, sg_polyorder=1, sg_winlen=1, seed=1)
    x -= x.mean()
    y = sp.filterDC(x.copy(),alpha=32)
    plt.figure(figsize=(12,3))
    plt.plot(x,label='x : raw signal')
    plt.plot(y,label='y : filtered - dc removed')
    plt.plot(x-y,label='x-y : removed component (drift)')
    plt.legend()
    plt.xlim([0,len(x)])
    plt.show()
    """
    
    B = X[0] if initialize_zero else 0*X[0]
    if return_background:
        Bg = np.zeros_like(X)
    Y = np.zeros_like(X)
    for i in range(X.shape[0]):
        B = ((alpha - 1) * B + X[i]) / alpha
        Y[i] = X[i]-B
        if return_background: Bg[i]= copy.copy(B)
    if return_background: return Y, Bg
    return Y

def filterDC_sGolay(X, window_length=127, polyorder=3, deriv=0, delta=1.0, mode='interp', cval=0.0,return_background=False):
    r"""Filter out DC component using Savitzky-Golay filter

    **Filter out DC component - Removing drift using Savitzky-Golay filter**
    

    Savitzky-Golay filter for multi-channels signal: From Scipy library

    Parameters
    ----------

    X : array, 1d, 2d, (n,) or (n,ch)
        - input signal single channel (n,) or multi-channel, channel axis should be 1 shape ~ (n,ch)
    
    window_length: int, odd number, 
            - length of window, larger the value, more flat drift is removed.
            - use smaller size to remove small duration fluctuations.
    polyorder: int, non-negetive
            - order of polynomial to be fitted. Higher the order, more curvey drift is captured and removed.
    
    others: deriv=0, delta=1.0, mode='interp', cval=0.0
            - parameters as same as in scipy.signal.savgol_filter
            :(polyorder=3, deriv=0, delta=1.0, mode='interp', cval=0.0)

    return_background: bool, False,
        - If True, returns the removed drift (residual) = (X - Y)

    Returns
    -------

    Y : corrected signal
    Xm: background removed -  return only if return_background is True



    References
    ----------
    * https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter

    Notes
    -----

    See Also
    --------
    filterDC: filter out DC using IIR Filter

    Examples
    --------
    # Example 1
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x = sp.create_signal_1d(n=1000, sg_polyorder=1, sg_winlen=1, seed=1)
    x -= x.mean()
    y = sp.filterDC_sGolay(x.copy(),window_length=127, polyorder=1)
    plt.figure(figsize=(12,3))
    plt.plot(x,label='x : raw signal')
    plt.plot(y,label='y : filtered - dc removed')
    plt.plot(x-y,label='x-y : removed component (drift)')
    plt.legend()
    plt.xlim([0,len(x)])
    plt.tight_layout()
    plt.show()



    ###################################
    # Example 2
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    X,fs,lead_names = sp.data.ecg_sample_12leads(sample=1)
    x = X[:,0]
    t = np.arange(x.shape[0])/fs
    y = sp.filterDC_sGolay(x.copy(),window_length=fs, polyorder=1)
    plt.figure(figsize=(12,3))
    plt.plot(t,x,label='x : raw signal')
    plt.plot(t,y,label='y : filtered - dc removed')
    plt.plot(t,x-y,label='x-y : removed component (drift)')
    plt.xlim([t[0],t[-1]])
    plt.legend(bbox_to_anchor=(1,1))
    plt.grid()
    plt.xlabel('time (s)')
    plt.tight_layout()
    plt.show()
    """

    if isinstance(window_length, float):
        if window_length==int(window_length):
            window_length = int(window_length)
        else:
            raise ValueError(f"'window_length' should be an odd integer>=0, passed window_length = {window_length}")

    if np.ndim(X)>1:
        Xm = savgol_filter(X, window_length, polyorder,deriv=deriv, delta=delta, axis=0, mode=mode, cval=cval)
    else:
        Xm = savgol_filter(X, window_length, polyorder,deriv=deriv, delta=delta, axis=-1, mode=mode, cval=cval)
    Y = X - Xm
    if return_background: return Y, Xm
    return Y

def filter_X(X,fs=128.0,band =[0.5],btype='highpass',order=5,ftype='filtfilt',verbose=False,use_joblib=False,filtr_keywors=dict()):
    r"""Spectral filtering using Buttorworth

    **Buttorworth filtering -  basic filtering**
    
    Wrapper for spectral filtering.

    Parameters
    ----------

    X : array, 1d, 2d, (n,) or (n,ch),
        - input signal single channel (n,) or multi-channel, channel axis should be 1 shape ~ (n,ch)
    fs: scaler, default=128
        - smapling frequency
    band: list of one or two scaler values.
        -  cut of frequency, for lowpass and highpass, band is list of one, for bandpass list of two numbers
    btype: str, {'highpass','lowpass','bandpass','bandstop'}
        - band of filter
    order: int, positive, default=5,
        - order of filter
    ftype: {'filtfilt', 'lfilter', 'SOS', 'sosfilt','sosfiltfilt'}
        -  filtering approach type, 'SOS' is mapped to 'sosfiltfilt'
        - lfilter is causal filter, which introduces delay, filtfilt does not introduce any delay, but it is non-causal filtering
        SOS:  Filter a signal using IIR Butterworth SOS method. A forward-backward digital filter using cascaded second-order sections.
        NOTE: 'SOS' is Recommended

    Returns
    -------

    Xf: filtered signal of same size as X

    References
    ----------

    * wikipedia - https://en.wikipedia.org/wiki/Butterworth_filter

    Notes
    -----
    * Filtering using Buttorworth

    See Also
    --------
    filter_smooth_sGolay : Smoothing signal using Savitzky-Golay filter
    filter_smooth_gauss : Smoothing signal using Gaussian function
    filter_smooth_mollifier : Smoothing signal using mollifier
    filter_with_kernel : Filtering signal using provided kernel
    filtering_pipeline: Filtering pipeline
    filter_powerline : Filtering out the powerline interference
    wavelet_filtering: Wavelet Filtering

    Examples
    --------
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs = sp.load_data.eeg_sample_1ch()
    t = np.arange(len(x))/fs
    x1 = sp.filter_X(x.copy(),fs=128.0, band=[4], btype='lowpass',ftype='SOS')
    x2 = sp.filter_X(x.copy(),fs=128.0, band=[8,12], btype='bandpass',ftype='SOS')
    x3 = sp.filter_X(x.copy(),fs=128.0, band=[32], btype='highpass',ftype='SOS')
    X = np.c_[x,x1,x2,x3]
    plt.figure(figsize=(12,5))
    plt.plot(t,X + np.arange(X.shape[1])*50)
    plt.xlim([t[0],t[-1]])
    plt.yticks(np.arange(X.shape[1])*50, ['Raw EEG','<4Hz (Lowpass)', '8-12Hz (Bandpass)','32<(Highpass)'])
    plt.xlabel('time (s)')
    plt.show()
    """

    filtfunlist = {'sosfiltfilt':signal.sosfiltfilt, 'sosfilt':signal.sosfilt, 'lfilter':signal.lfilter, 'filtfilt':signal.filtfilt}

    if ftype.lower()=='sos': ftype = 'sosfiltfilt'
    ftype = ftype.lower()
    # Should be One of the 'sosfiltfilt', 'sosfilt', 'lfilter', 'filtfilt', or 'sos'
    assert ftype in filtfunlist.keys()

    if verbose: print(X.shape, 'channels axis = 1', ' filter type ', ftype)

    ifSOS = False
    if ftype=='sosfiltfilt' or ftype=='sosfilt':
        sos = signal.butter(order, np.array(band)/(0.5*fs), btype=btype, output="sos")
        ifSOS = True
    else:
        b,a = signal.butter(order,np.array(band)/(0.5*fs),btype=btype)

    FILTER_  = filtfunlist[ftype]

    if ifSOS:
        if np.ndim(X)>1:
            if use_joblib:
                try:
                    Xf  = np.array(Parallel(n_jobs=-1)(delayed(FILTER_)(sos,X[:,i],**filtr_keywors) for i in range(X.shape[1]))).T
                except:
                    print('joblib paraller failed computing with loops- turn off --> use_joblib=False')
                    Xf  = np.array([FILTER_(sos,X[:,i],**filtr_keywors) for i in range(X.shape[1])]).T
            else:
                Xf  = np.array([FILTER_(sos,X[:,i],**filtr_keywors) for i in range(X.shape[1])]).T
        else:
            Xf  = FILTER_(sos,X,**filtr_keywors)
    else:
        if np.ndim(X)>1:
            if use_joblib:
                try:
                    Xf  = np.array(Parallel(n_jobs=-1)(delayed(FILTER_)(b,a,X[:,i],**filtr_keywors) for i in range(X.shape[1]))).T
                except:
                    print('joblib paraller failed computing with loops- turn off --> use_joblib=False')
                    Xf  = np.array([FILTER_(b,a,X[:,i],**filtr_keywors) for i in range(X.shape[1])]).T
            else:
                Xf  = np.array([FILTER_(b,a,X[:,i],**filtr_keywors) for i in range(X.shape[1])]).T
        else:
            Xf  = FILTER_(b,a,X,**filtr_keywors)
    return Xf

def filter_powerline(X,fs=1000, powerline=50):
    r"""Remove Powerline interefences

    Equivilent to lowpass butterworth filter of order 1

    Parameters
    ----------
    X: array, (n,) or (n,ch)
     - 1d-array signal, or multi-channel signal
    fs: int, default=1000
     - sampling frequency
    powerline: int {50,60}
     - powerline interfence 50Hz, or 60Hz
    
    Returns
    -------
    Xf: same size as input X
     -  filtered signal

    References
    ----------
    * wikipedia - 
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    spkit: #TODO

    Examples
    --------
    >>> import numpy as np
    >>> import spkit as sp
    

    """
    if fs >= 100:
        b = np.ones(int(fs / powerline))
    else:
        b = np.ones(2)
    a = [len(b)]

    if len(X.shape)>1:
        Xf = np.array([signal.filtfilt(b,a, X[:,i], method="pad") for i in range(X.shape[1])]).T
    else:
        Xf = signal.filtfilt(b, a, X, method="pad")
    return Xf

def filter_smooth_sGolay(X, window_length=127, polyorder=3, deriv=0, delta=1.0, mode='interp', cval=0.0):
    r"""Smoothing filter using Savitzky-Golay filter
    
    **Smoothing filter using Savitzky-Golay filter**

    Savitzky-Golay filter for multi-channels signal: From Scipy library

    Parameters
    ----------
    X : array,
     - input signal single channel (n,) or multi-channel, channel axis should be 1 shape ~ (n,ch)
    
    window_length: int, odd number, 
        - length of window, larger the value, more flat drift is removed.
        - use smaller size to retain small duration of fluctuations and curves.
    polyorder: int, non-negetive
        - order of polynomial to be fitted. Higher the order, more curvey smooth is captured.
    
    others: deriv=0, delta=1.0, mode='interp', cval=0.0
        - parameters as same as in scipy.signal.savgol_filter
        - (deriv=0, delta=1.0, mode='interp', cval=0.0)

    Returns
    -------
    Y : same shape as X,
     - corrected signal
    
    References
    ----------


    See Also
    --------
    filter_smooth_gauss: Smoothing signal using Gaussian kernel
    filter_smooth_mollifier: Smoothing signal using Mollifier
    filter_with_kernel: filtering signal using custom kernel
    filter_X: Spectral filtering

    Examples
    --------
    #sp.filter_smooth_sGolay
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs = sp.data.ppg_sample(sample=1)
    x = x[:int(fs*5)]
    x = x - x.mean()
    t = np.arange(len(x))/fs
    xf1 = sp.filter_smooth_sGolay(x.copy(),window_length=31, polyorder=2)
    xf2 = sp.filter_smooth_sGolay(x.copy(),window_length=31, polyorder=5)
    xf3 = sp.filter_smooth_sGolay(x.copy(),window_length=51, polyorder=3)
    plt.figure(figsize=(12,3))
    plt.plot(t,x,label='x: signal')
    plt.plot(t,xf1,label='xf1: (wL=31, order=2)')
    plt.plot(t,xf2,label='xf2: (wL=31, order=5)')
    plt.plot(t,xf3,label='xf3: (wL=51, order=3)')
    plt.xlim([t[0],t[-1]])
    plt.xlabel('time (s)')
    plt.ylabel('PPG Signal')
    plt.grid()
    plt.legend(bbox_to_anchor=(1,1))
    plt.title('Savitzky-Golay Smoothing')
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(12,3))
    plt.plot(t,x-xf1,label='x-xf1')
    plt.plot(t,x-xf2-40,label='x-xf2')
    plt.plot(t,x-xf3-80,label='x-xf3')
    plt.xlim([t[0],t[-1]])
    plt.xlabel('time (s)')
    plt.ylabel('Residual')
    plt.legend(bbox_to_anchor=(1,1))
    plt.grid()
    plt.tight_layout()
    plt.show()

    """

    if np.ndim(X)>1:
        Y = savgol_filter(X, window_length, polyorder,deriv=deriv, delta=delta, axis=0, mode=mode, cval=cval)
    else:
        Y = savgol_filter(X, window_length, polyorder,deriv=deriv, delta=delta, axis=-1, mode=mode, cval=cval)
    return Y

def filter_smooth_gauss(X, window_length=11, sigma_scale=2.7,iterations=1,mode='same'):
    r"""Smoothing filter using Gaussian Kernel and 1d-ConvFB
    
    **Smoothing filter using Gaussian Kernel and 1d-ConvFB**
    
    sigma : sigma for gaussian kernel, if None, sigma=window_length/6

    Parameters
    ----------

    X : array,
     - input signal single channel (n,) or multi-channel, channel axis should be 1 shape ~ (n,ch)
    
    window_length: int >1, length of gaussian kernel
    
    sigma_scale: float, deafult=2.7
        - To control width/spread of gauss

    iterations: int, >=1, default=1
     - repeating gaussian smoothing iterations times

    mode: {'same','valid','full'},
      - convolution mode in , `same` make sense.

    Returns
    -------
    Y : Smoothed signal

    References
    ----------
    * wikipedia
    
    
    Notes
    -----

    See Also
    --------
    filter_smooth_sGolay: Smoothing signal using Savitzky-Golay filter
    filter_smooth_mollifier: Smoothing signal using Mollifier
    filter_with_kernel: filtering signal using custom kernel
    filter_X: Spectral filtering

    Examples
    --------
    #sp.filter_smooth_gauss
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs = sp.data.ppg_sample(sample=1)
    x = x[:int(fs*5)]
    x = x - x.mean()
    t = np.arange(len(x))/fs
    xf1 = sp.filter_smooth_gauss(x.copy(),window_length=31, sigma_scale=2.7)
    xf2 = sp.filter_smooth_gauss(x.copy(),window_length=31, sigma_scale=5.4)
    xf3 = sp.filter_smooth_gauss(x.copy(),window_length=51, sigma_scale=2.7)
    plt.figure(figsize=(12,3))
    plt.plot(t,x,label='x: signal')
    plt.plot(t,xf1,label=r'xf1: (wL=31, $\sigma=2.7$)')
    plt.plot(t,xf2,label=r'xf2: (wL=31, $\sigma=5.4$)')
    plt.plot(t,xf3,label=r'xf3: (wL=51, $\sigma=2.7$)')
    plt.xlim([t[0],t[-1]])
    plt.xlabel('time (s)')
    plt.ylabel('PPG Signal')
    plt.grid()
    plt.legend(bbox_to_anchor=(1,1))
    plt.title('Gaussian Smoothing')
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(12,3))
    plt.plot(t,x-xf1,label='x-xf1')
    plt.plot(t,x-xf2-40,label='x-xf2')
    plt.plot(t,x-xf3-80,label='x-xf3')
    plt.xlim([t[0],t[-1]])
    plt.xlabel('time (s)')
    plt.ylabel('Residual')
    plt.legend(bbox_to_anchor=(1,1))
    plt.grid()
    plt.tight_layout()
    plt.show()
    """

    g_kernel = gaussian_kernel(window_length=window_length, sigma_scale=sigma_scale)

    if np.ndim(X)>1:
        Y = []
        for x in X.T:
            y = conv1d_fb(x, kernel=g_kernel, iterations=iterations,mode=mode)
            Y.append(y)
        Y = np.array(Y).T
    else:
        Y = conv1d_fb(X, kernel=g_kernel, iterations=iterations,mode=mode)
    return Y

def filter_smooth_mollifier(X, window_length=11,s=None,p=None,r=None,iterations=1,mode='same'):
    r"""Smoothing filter using Mollifier kernel and 1d-ConvFB

    **Smoothing filter using Mollifier kernel and 1d-ConvFB**
    
    **Mollifier: Kurt Otto Friedrichs**

    Generalized function

    .. math:: f(x) =  exp(-s/(1-|x|**p))    for |x|<1,   x \in [-r, r]

    Convolving with a mollifier, signals's sharp features are smoothed, while still remaining close
    to the original nonsmooth (generalized) signals.

    Intuitively, given a function which is rather irregular, by convolving it with a mollifier the function gets "mollified".

    This function is infinitely differentiable, non analytic with vanishing derivative for |x| = 1,
    can be therefore used as mollifier as described in [1]. This is a positive and symmetric mollifier.[2]


    Parameters
    ----------

    X : array,
     - input signal single channel (n,) or multi-channel, channel axis should be 1 shape ~ (n,ch)
    
    window_length: int >1, length of kernel

    s: scaler, s>0, default=None, which means; s =1,
       Spread of the middle width, heigher the value of s, narrower the width

    p: scaler, p>0, default=None, which means; p=2,
       Order of flateness of the peak at the top,
       p=2, smoother, p=1, triangulare type
       Higher it is, more flat the peak.

    r: float, 0<r<1, default=None, which means; r=0.999,
        it is used to compute x = [-r, r]
        recommonded to keep it r=0.999

    iterations: int, >=1, repeating gaussian smoothing iterations times
    mode: convolution mode in {'same','valid','full'}, 'same make sense'


    Returns
    -------
    Y : array
     - Mollified signal, of same shape as input X



    References
    ----------
    * [1] https://en.wikipedia.org/wiki/Mollifier
    * [2] https://en.wikipedia.org/wiki/Kurt_Otto_Friedrichs
    
    
    Notes
    -----

    See Also
    --------
    filter_smooth_sGolay: Smoothing signal using Savitzky-Golay filter
    filter_smooth_gauss: Smoothing signal using Gaussian kernel
    filter_with_kernel: filtering signal using custom kernel
    filter_X: Spectral filtering

    Examples
    --------
    #sp.filter_smooth_mollifier
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    seed = 4
    N = 10
    fs = 100
    x   = sp.create_signal_1d(n=2*fs,seed=seed,sg_polyorder=3, sg_winlen=11)
    x_n = sp.create_signal_1d(n=2*fs,seed=seed,sg_polyorder=1, sg_winlen=11)
    idx = np.arange(len(x))
    np.random.seed(seed)
    np.random.shuffle(idx)
    x[idx[:N]] = x_n[idx[:N]]
    #x[idx[:N]+1] = x_n[idx[:N]+1]
    #x[idx[:N]+2] = x_n[idx[:N]+2]
    t = np.arange(len(x))/fs
    y = sp.filter_smooth_mollifier(x.copy(),window_length=11, s=1, p=1)
    plt.figure(figsize=(12,4))
    plt.plot(t,x,label='x: signal')
    plt.plot(t,y,label='y: mollified')
    plt.plot(t,x-y1-2,label='x-y:residual')
    plt.xlim([t[0],t[-1]])
    plt.xlabel('time (s)')
    plt.title('Mollifier')
    plt.yticks([0,-2],['x','x-y'])
    plt.grid()
    plt.legend(bbox_to_anchor=(1,1))
    np.random.seed(None)
    plt.show()

    """

    #g_kernel = gaussian_kernel(window_length=window_length, sigma_scale=sigma_scale)
    ker_mol  = friedrichs_mollifier_kernel(window_length,s=s,p=p,r=r)
    if np.ndim(X)>1:
        Y = []
        for x in X.T:
            y = conv1d_fb(x, kernel=ker_mol, iterations=iterations,mode=mode)
            Y.append(y)
        Y = np.array(Y).T
    else:
        Y = conv1d_fb(X, kernel=ker_mol, iterations=iterations,mode=mode)
    return Y

def filter_with_kernel(X, kernel,iterations=1,mode='same'):
    r"""Smoothing/Sharpening using given kernel and 1d-ConvFB

    **Smoothing/Sharpening using given kernel and 1d-ConvFB**

    Smoothing/Sharpening depends on kernel

    Parameters
    ----------

    X : array,
     - input signal single channel (n,) or multi-channel, channel axis should be 1 shape ~ (n,ch)
    
    kernel :  custamised kernel for

    iterations: int, >=1, repeating gaussian smoothing iterations times
    mode: convolution mode in {'same','valid','full'}, 'same make sense'

    Returns
    --------

    Y : procesed signal

    See Also
    --------
    filter_smooth_sGolay: Smoothing signal using Savitzky-Golay filter
    filter_smooth_gauss: Smoothing signal using Gaussian kernel
    filter_smooth_mollifier: Smoothing signal using Mollifier
    filter_X: Spectral filtering

    Examples
    --------
    #sp.filter_with_kernel
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs = sp.data.ppg_sample(sample=1)
    x = x[:int(fs*5)]
    x = x - x.mean()
    t = np.arange(len(x))/fs
    kernel1 = sp.gaussian_kernel(window_length=101,sigma_scale=10)
    kernel2 = sp.friedrichs_mollifier_kernel(window_size=101,s=1,p=1)
    kernel3 = (kernel1 - kernel2)/2
    y1 = sp.filter_with_kernel(x.copy(),kernel=kernel1)
    y2 = sp.filter_with_kernel(x.copy(),kernel=kernel2)
    y3 = sp.filter_with_kernel(x.copy(),kernel=kernel3)
    plt.figure(figsize=(12,5))
    plt.subplot(212)
    plt.plot(t,x,label='x: signal')
    plt.plot(t,y1,label='y1: kernel1')
    plt.plot(t,y2,label='y2: kernel2')
    plt.plot(t,y3,label='y3: kernel3')
    plt.xlim([t[0],t[-1]])
    plt.xlabel('time (s)')
    plt.ylabel('PPG Signal')
    plt.grid()
    plt.legend(bbox_to_anchor=(1,1))
    plt.title('filtering with kernels')
    plt.subplot(231)
    plt.plot(kernel1,label='kernel1')
    plt.plot(kernel2,label='kernel2')
    plt.plot(kernel3,label='kernel3')
    plt.title('Kernels')
    plt.grid()
    plt.tight_layout()
    plt.show()
    """

    if np.ndim(X)>1:
        Y = []
        for x in X.T:
            y = conv1d_fb(x,kernel=kernel, iterations=iterations,mode=mode)
            Y.append(y)
        Y = np.array(Y).T
    else:
        Y = conv1d_fb(X, kernel=kernel, iterations=iterations,mode=mode)
    return Y

def filtering_pipeline(X,fs,trim=[0,0],iir_alpha=0,sg_window=1001,sg_polyorder=1,sg_itr=1,filter_lpf=None, filter_pwrline=None,last_filter_keywords ={},verbose=1):
    r"""Filtering Pipeline

    Applying sequence of filters

    - Useful to set onces and apply all many samples
    - Any filter.]/operation can be excluded or included using parameters

    **Sequence of Operations:**

        1. Trim/crop the signal
            - if sum(trim)>0: X[int(fs*trim[0]):-int(fs*trim[1])
        2. DC Removal:
            - 2.1) IIR DC Removal :  if iir_alpha>0, apply :func:`filterDC`
            - 2.2) SGolay DC Removal: if sg_window>0, apply :func:`filterDC_sGolay`
        3. Powerline intereference removal:
            - if filter_pwrline is not None, apply :func:`filter_powerline`
        4. Lowpass filter:
            - if filter_lpf is not None, apply :func:`filter_X`
        5. Last custom filter :  
            - if len(last_filter_keywords)>0, apply :func:`filter_X`
    
    Parameters
    ----------
    X : array,
     - input signal single channel (n,) or multi-channel, channel axis should be 1 shape ~ (n,ch)

    fs: int, 
     - sampling frequency
    trim: list, default=[0,0]
     - applied only if sum(trim)>0 
     - duration of signal to be removed (triming the signal) from start and end.
     - trim =[1,1] means, removing 1sec from start and 1sec from end
     - X = X[int(fs*trim[0]):-int(fs*trim[1])]
    iir_alpha: scalar, default=0
     - applied only if iir_alpha>0
     - alpha for IIR DC Removal filter
    (sg_window,sg_polyorder,sg_itr): scalars,
     - parameters for DC removal using sGolay filter, applied only is sg_window>0
     - defaults values sg_window=1001,sg_polyorder=1,sg_itr=1
     - sg_window: window length,sg_polyorder: polynomial order,sg_itr: number of iterations to apply
     filter_lpf: scalar, default=None
        - applied only if it not None
        - cut-off frequency for lowpass filter

     filter_pwrline: {50,60},default=None
        - applied only if it not None
        - powerline interefence type 50Hz of 60Hz,

     last_filter_keywords: dict, deafult ={}
       - applied only if dictionary is not empty
       - parameters to be passed for :func:`filter_X`, other than X, fs, and verbose.
     verbose=1,
                     

    Returns
    -------
    Y : Filtered signal

    References
    ----------
    * wikipedia - 
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    filter_smooth_sGolay: Smoothing signal using Savitzky-Golay filter
    filter_smooth_gauss: Smoothing signal using Gaussian kernel
    filter_with_kernel: filtering signal using custom kernel
    filter_X: Spectral filtering

    Examples
    --------
    #sp.filtering_pipeline
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs,lead_names = sp.data.ecg_sample_12leads(sample=1)
    x = x[:,1]
    t = np.arange(len(x))/fs
    t1,t2 = 0.5, 0.5
    y = sp.filtering_pipeline(x,fs,trim=[t1,t2],sg_window=fs+1,sg_polyorder=1,filter_lpf=150,filter_pwrline=50)
    plt.figure(figsize=(12,5))
    plt.subplot(311)
    plt.plot(t,x)
    plt.grid()
    plt.xticks(np.arange(0,t[-1],0.5))
    plt.xlim([t[0],t[-1]])
    plt.ylabel('x: signal')
    plt.subplot(312)
    plt.plot(t[:len(y)]+t1,y,label='x: signal')
    plt.xticks(np.arange(0,t[-1],0.5))
    plt.xlim([t[0],t[-1]])
    plt.ylabel('y: processed')
    plt.grid()
    plt.subplot(313)
    plt.plot(t[:len(y)]+t1,x[:len(y)]-y)
    plt.xticks(np.arange(0,t[-1],0.5))
    plt.xlim([t[0],t[-1]])
    plt.grid()
    plt.ylabel('x-y: residual')
    plt.tight_layout()
    plt.show()
    """
    #if axis==0: X = X.T
    if verbose: print('input: ',X.shape)
    if sum(trim):
        if trim[1]>0:
            X = X[int(fs*trim[0]):-int(fs*trim[1])]
        else:
            X = X[int(fs*trim[0]):]
        if verbose: print(' - Aftering trimming : ',X.shape)

    if iir_alpha>0:
        X = filterDC(X,alpha=iir_alpha)
        if verbose: print(' - Aftering DC filtering : ',X.shape)

    if sg_window>0:
        dX = filterDC_sGolay(X, window_length=sg_window,polyorder=sg_polyorder, return_background=True)[1]
        if sg_itr>1:
            for _ in range(sg_itr-1):
                sg_window = sg_window//2
                if sg_window%2==0: sg_window+=1
                dX = filterDC_sGolay(dX, window_length=sg_window,polyorder=sg_polyorder, return_background=True)[1]

        X = X-dX
        if verbose: print(' - Aftering sgolay filtering : ',X.shape)

    if filter_pwrline is not None:
        X = filter_powerline(X,fs=fs, powerline=filter_pwrline)
        if verbose: print(f' - Aftering Power line {filter_pwrline}Hz filtering : ',X.shape)

    if filter_lpf is not None:
        X = filter_X(X,fs=fs,band =[filter_lpf],btype='lowpass',order=5,ftype='SOS',verbose=verbose,use_joblib=False)
        if verbose: print(' - Aftering LPF filtering : ',X.shape)
    if len(last_filter_keywords):
        # band =[30],btype='lowpass',order=5,ftype='SOS'
        X = filter_X(X,fs=fs,verbose=verbose,**last_filter_keywords)
        if verbose: print(' - Aftering last filter: {last_filter_keywords}, X:shape',X.shape)

    if verbose: print('output: ',X.shape)
    return X

def filtering_pipeline_v2(X,fs,trim=[0,0],iir_alpha=0,sg_window=1001,sg_polyorder=1,filter_lpf =None,filter_pwrline=None,verbose=1,
                            last_filter_keywords ={}):
    r"""Filtering Pipeline V2

    Applying sequence of filters

    - Useful to set onces and apply all many samples
    - Any filter.]/operation can be excluded or included using parameters

    **Sequence of Operations**

    1) Trim/crop the signal
      - if sum(trim)>0: X[int(fs*trim[0]):-int(fs*trim[1])
    2) DC Removal:
     2.1) IIR DC Removal :  if iir_alpha>0, apply :func:`filterDC`
     2.2) SGolay DC Removal: if sg_window>0, apply :func:`filterDC_sGolay`
    3) Powerline intereference removal:
      - if filter_pwrline is not None, apply :func:`filter_powerline`
    4) Lowpass filter:
      - if filter_lpf is not None, apply :func:`filter_powerline`
    5) Last custom filter :  
      - if len(last_filter_keywords)>0, apply :func:`filter_X`
    
    Parameters
    ----------
    X : array,
     - input signal single channel (n,) or multi-channel, channel axis should be 1 shape ~ (n,ch)

    fs: int, 
     - sampling frequency
    trim: list, default=[0,0]
     - applied only if sum(trim)>0 
     - duration of signal to be removed (triming the signal) from start and end.
     - trim =[1,1] means, removing 1sec from start and 1sec from end
     - X = X[int(fs*trim[0]):-int(fs*trim[1])]
    iir_alpha: scalar, default=0
     - applied only if iir_alpha>0
     - alpha for IIR DC Removal filter
    (sg_window,sg_polyorder,sg_itr): scalars,
     - parameters for DC removal using sGolay filter, applied only is sg_window>0
     - defaults values sg_window=1001,sg_polyorder=1,sg_itr=1
     - sg_window: window length,sg_polyorder: polynomial order,sg_itr: number of iterations to apply
     filter_lpf: scalar, default=None
        - applied only if it not None
        - cut-off frequency for lowpass filter

     filter_pwrline: {50,60},default=None
        - applied only if it not None
        - powerline interefence type 50Hz of 60Hz,

     last_filter_keywords: dict, deafult ={}
       - applied only if dictionary is not empty
       - parameters to be passed for :func:`filter_X`, other than X, fs, and verbose.
     verbose=1,
                     

    Returns
    -------
    Y : Filtered signal

    References
    ----------
    * wikipedia - 
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    filter_smooth_sGolay: Smoothing signal using Savitzky-Golay filter
    filter_smooth_gauss: Smoothing signal using Gaussian kernel
    filter_with_kernel: filtering signal using custom kernel
    filter_X: Spectral filtering

    Examples
    --------
    #sp.filtering_pipeline_v2
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs,lead_names = sp.data.ecg_sample_12leads(sample=1)
    x = x[:,1]
    t = np.arange(len(x))/fs
    t1,t2 = 0.5, 0.5
    y = sp.filtering_pipeline_v2(x,fs,trim=[t1,t2],sg_window=fs+1,sg_polyorder=1,filter_lpf=150,filter_pwrline=50)
    plt.figure(figsize=(12,5))
    plt.subplot(311)
    plt.plot(t,x)
    plt.grid()
    plt.xticks(np.arange(0,t[-1],0.5))
    plt.xlim([t[0],t[-1]])
    plt.ylabel('x: signal')
    plt.subplot(312)
    plt.plot(t[:len(y)]+t1,y,label='x: signal')
    plt.xticks(np.arange(0,t[-1],0.5))
    plt.xlim([t[0],t[-1]])
    plt.ylabel('y: processed')
    plt.grid()
    plt.subplot(313)
    plt.plot(t[:len(y)]+t1,x[:len(y)]-y)
    plt.xticks(np.arange(0,t[-1],0.5))
    plt.xlim([t[0],t[-1]])
    plt.grid()
    plt.ylabel('x-y: residual')
    plt.tight_layout()
    plt.show()
    """

    if verbose: print('input: ',X.shape)
    if sum(trim):
        if trim[1]>0:
            X = X[int(fs*trim[0]):-int(fs*trim[1])]
        else:
            X = X[int(fs*trim[0]):]
        if verbose: print(' - Aftering trimming : ',X.shape)

    if iir_alpha>0:
        #Xf = filterDC_X(X.T,alpha=alpha,returnB=False,verbose=verbose).T
        X = filterDC(X,alpha=iir_alpha)
        if verbose: print(' - Aftering DC filtering : ',X.shape)

    if sg_window>0:
        dX = filterDC_sGolay(X, window_length=sg_window,polyorder=sg_polyorder, return_background=True)[1]
        sg_window = sg_window//2
        if sg_window%2==0: sg_window+=1
        dX = filterDC_sGolay(dX, window_length=sg_window,polyorder=sg_polyorder, return_background=True)[1]
        X = X-dX
        if verbose: print(' - Aftering sgolay filtering : ',X.shape)

    if filter_pwrline is not None:
        X = filter_powerline(X,fs=fs, powerline=filter_pwrline)
        if verbose: print(f' - Aftering Power line {filter_pwrline}Hz filtering : ',X.shape)

    if filter_lpf is not None:
        X = filter_X(X,fs=fs,band =[filter_lpf],btype='lowpass',order=5,ftype='filtfilt',verbose=verbose,use_joblib=False)
        if verbose: print(' - Aftering LPF filtering : ',X.shape)
    if verbose: print('output: ',X.shape)
    return X

def conv1d_nan(x,kernel, denormalise_ker=True, boundary='constant', fillvalue=np.nan ):
    r"""1D Convolution with NaN values

    **1D Convolution with NaN values**
    

    In conventional Convolution funtions, if any of the value in
    input x or in kernel is NaN (np.nan), then NaN values are propogated and corrupt other values too.

    To avoid this, this funtion does the convolution in same fashion as conventional
    except, it allows NaN values to exists in input, without propogating them.

    while computation, it simply ignores the NaN value, as it doen not exist, and adjust the computation
    accordingly.

    If No NaN value exist, result is same as conventional convolution

    Parameters
    ----------
    x : 1d-array,
      - input signal single channel (n,) with NaN values.

    kernel: a 1D kernel
      - to use for convolution
      - IMPORTANT NOTE: kernel passed should **NOT** be normalised. If normalised kernel is used,
        the results will be very different than conventional convolution.
        For example:
            kernel_unnorm = np.ones(9)
            kernel_norm = np.ones(9)/9
        kernel_unnorm should be passed, not kernel_norm.

      - To de-normalise a kernel, used  :func:`denorm_kernel`
      - or set denormalise_ker=True
    
    denormalise_ker: bool, default=True
        - If True, first de-normalise kernel

    boundary : str {'fill','constant' ,'wrap', 'symm','symmetric','reflect',}, optional
        A flag indicating how to handle boundaries:

        ``fill`` or ``constant``
           pad input arrays with fillvalue. (default)
        ``wrap``
           circular boundary conditions.
        ``symm`` or 'symmetric'
           symmetrical boundary conditions.

        ``reflect``
           reflecting boundary conditions.

    fillvalue: scalar, optional
        - Value to fill pad input arrays with. Default is np.nan,


    Returns
    -------
    y: 1D-arrray
      - of same size as input x with no NaN propogation
    
    See Also
    --------
    conv2d_nan: 2D Convolution with NaN
    fill_nans_1d: fill NaN of 1d array 
    fill_nans_2d: fill NaN of 2d array 

    Notes
    -----
    See examples below
    
    Examples
    --------
    >>> #sp.conv1d_nan
    >>> import numpy as np
    >>> import spkit as sp
    >>> from scipy import signal
    >>> N = 10
    >>> np.random.seed(seed=200)
    >>> X  = np.random.randint(0,5,N)
    >>> r = 1*(np.abs(np.random.randn(N))<1.4).astype(float)
    >>> r[r==0]=np.nan
    >>> X_nan = X*r
    >>> kernel_norm = np.ones(9)/9
    >>> cx1 = signal.convolve(X_nan,kernel_norm, method='auto',mode='same')
    >>> cx2 = sp.conv1d_nan(X_nan,kernel_norm, denormalise_ker=True)
    >>> print('Convolution kernel: ')
    >>> print(' - ',kernel_norm.round(3))
    >>> print('input signal with Nans')
    >>> print(' - ',X_nan)
    >>> print('Convolution using scipy')
    >>> print(' - ',cx1.round(3))
    >>> print('Convolution using spkit')
    >>> print(' - ',cx2.round(3))
    
    >>> ####################################################
    >>> #sp.conv1d_nan
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import scipy
    >>> import spkit as sp
    >>> seed = 4
    >>> x = (sp.create_signal_1d(n=100,seed=seed,sg_winlen=5)*10).round(0)
    >>> t = np.arange(len(x))/100
    >>> np.random.seed(seed)
    >>> r = np.random.rand(*x.shape)
    >>> x[r<0.05] = np.nan
    >>> kernel = np.ones(7)/7
    >>> np.random.seed(None)
    >>> x_scipy = scipy.signal.convolve(x.copy(), kernel,mode='same')
    >>> x_spkit = sp.conv1d_nan(x.copy(), kernel)
    >>> plt.figure(figsize=(8,7))
    >>> plt.subplot(311)
    >>> plt.plot(t,x,color='C0')
    >>> plt.vlines(t[np.isnan(x)],ymin=np.nanmin(x),ymax=np.nanmax(x),color='r',alpha=0.2,lw=8,label='missing: NaN')
    >>> plt.xlim([t[0],t[-1]])
    >>> plt.ylim([np.nanmin(x),np.nanmax(x)])
    >>> plt.legend(bbox_to_anchor=(1,1.2))
    >>> plt.grid()
    >>> plt.title('Signal with NaNs')
    >>> plt.subplot(312)
    >>> plt.plot(t,x_scipy)
    >>> plt.vlines(t[np.isnan(x_scipy)],ymin=np.nanmin(x),ymax=np.nanmax(x),color='r',alpha=0.2,lw=5.5)
    >>> plt.xlim([t[0],t[-1]])
    >>> plt.ylim([np.nanmin(x),np.nanmax(x)])
    >>> plt.grid()
    >>> plt.title('Convolution uisng Scipy (scipy.signal.convolution)')
    >>> plt.subplot(313)
    >>> plt.plot(t,x_spkit)
    >>> plt.xlim([t[0],t[-1]])
    >>> plt.ylim([np.nanmin(x),np.nanmax(x)])
    >>> plt.grid()
    >>> plt.title('Convolution uisng Spkit (sp.conv1d_nan)')
    >>> plt.tight_layout()
    >>> plt.show()
    """

    x = np.squeeze(x.astype(float))
    kernel = np.squeeze(kernel.astype(float))

    if denormalise_ker: kernel = denorm_kernel(kernel)

    assert x.ndim==1 and kernel.ndim==1

    r1 = x.shape[0]
    kr = kernel.shape[0]

    pad_len = kr//2

    if boundary in ['constant','fill']:
        xaf = np.pad(x,pad_len,mode='constant',constant_values=fillvalue)
    elif boundary in ['symmetric','wrap','reflect']:
        xaf = np.pad(x,pad_len,mode=boundary,)
    elif boundary in ['symm']:
        xaf = np.pad(x,pad_len,mode='symmetric')
    else:
        raise NameError("not supported 'boundary' condition, should be on of {'constant','fill', 'symmetric','wrap','reflect', 'symm'} ")

    r2 = xaf.shape
    y = np.zeros(r2)
    for i in range(x.shape[0]):
        y[i] = np.nanmean(xaf[i:i+kr]*kernel)
    return y[:r1]

def conv2d_nan(x,kernel, boundary='fill', fillvalue=0, denormalise_ker=True):
    r"""2D Convolution with NaN values
    
    **2D Convolution with NaN values**
    

    In conventional Convolution funtions, if any of the value in
    input x or in kernel is NaN (np.nan), then NaN values are propogated and corrupt other values too.

    To avoid this, this funtion does the convolution in same fashion as conventional
    except, it allows NaN values to exists in input, without propogating their effect. It even replaces them.

    While computating, it simply ignores the NaN value, as if does not exist, and adjust the computation
    accordingly.

    If No NaN value exist, result is same as conventional convolution

    Parameters
    ----------

    x: 2D-array with NaN values.

    kernel: a 2D kernel
      - to use for convolution
      - IMPORTANT NOTE: kernel passed should NOT be normalised. If normalised kernel is used,
        the results will be very different than conventional convolution.

        For example:
          kernel_unnorm = np.ones([3,3])
          kernel_norm = np.ones([3,3])/9

        kernel_unnorm should be passed, not kernel_norm.
        
       - To de-normalise a kernel, used  :func:`denorm_kernel`
       - or set denormalise_ker=True

    denormalise_ker: bool, default=True
       - If True, first de-normalise kernel

    boundary : str {'fill','constant' ,'wrap', 'symm','symmetric','reflect',}, optional
        A flag indicating how to handle boundaries:
        ``fill`` or ``constant``
           pad input arrays with fillvalue. (default)
        ``wrap``
           circular boundary conditions.
        ``symm`` or 'symmetric'
           symmetrical boundary conditions.

        ``reflect``
           reflecting boundary conditions.

    fillvalue: scalar, optional
        Value to fill pad input arrays with. Default is np.nan,


    Returns
    -------

    y: 2D-arrray of same size as input x with no NaN propogation

    See Also
    --------
    conv1d_nan, fill_nans_1d, fill_nans_2d

    Example
    --------
    >>> import numpy as np
    >>> import spkit as sp
    >>> from scipy import signal
    >>> N = 10
    >>> np.random.seed(seed=100)
    >>> X  = np.random.randint(0,5,[N,N])
    >>> r = 1*(np.abs(np.random.randn(N,N))<1.4).astype(float)
    >>> r[r==0]=np.nan
    >>> X_nan = X*r
    >>> X_nan

    code::

        array([[ 0., nan,  3.,  0.,  2.,  4.,  2.,  2.,  2.,  2.],
           [ 1., nan,  0.,  4., nan,  4.,  2.,  0.,  3.,  1.],
           [ 2.,  3.,  4.,  4., nan, nan,  4., nan,  3.,  3.],
           [ 3.,  1.,  1., nan,  0.,  2.,  1., nan,  3.,  2.],
           [nan,  0.,  1., nan, nan,  2.,  0.,  0., nan,  2.],
           [ 1.,  0.,  2., nan,  4.,  2.,  0.,  3.,  3.,  3.],
           [ 0.,  1.,  4.,  2.,  3.,  3.,  4.,  2.,  4.,  3.],
           [ 1.,  0.,  4., nan,  4.,  2., nan,  4.,  2., nan],
           [nan,  2., nan,  0., nan,  2.,  3.,  4., nan,  4.],
           [ 3.,  1.,  0.,  0.,  1.,  2.,  4., nan,  3.,  3.]])
    

    >>> kernel_norm = np.ones([3,3])/9
    >>> kernel_norm

    code::

        array([[0.11111111, 0.11111111, 0.11111111],
               [0.11111111, 0.11111111, 0.11111111],
               [0.11111111, 0.11111111, 0.11111111]])

    >>> kernel = np.ones([3,3])
    >>> kernel

    code::

        array([[1., 1., 1.],
               [1., 1., 1.],
               [1., 1., 1.]])

    # Or use denormaliser
    >>> kernel = denorm_kernel(kernel_norm)
    >>> kernel
    
    code::

        array([[1., 1., 1.],
               [1., 1., 1.],
               [1., 1., 1.]])


    >>> signal.convolve2d(X_nan,kernel_norm,boundary='symm',mode='same').round(1)

    code::

        array([[nan, nan, nan, nan, nan, nan, 2.4, 1.9, 1.8, 1.9],
               [nan, nan, nan, nan, nan, nan, nan, nan, nan, 2.2],
               [nan, nan, nan, nan, nan, nan, nan, nan, nan, 2.3],
               [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
               [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
               [nan, nan, nan, nan, nan, nan, 1.8, nan, nan, nan],
               [0.6, 1.4, nan, nan, nan, nan, nan, nan, nan, nan],
               [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
               [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
               [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]])
    

    >>> sp.conv2d_nan(X_nan,kernel,boundary='symm').round(1)

    code::

        array([[0.3, 1.2, 1.7, 1.8, 2.5, 2.8, 2.4, 1.9, 1.8, 1.9],
            [1.3, 1.9, 2.6, 2.4, 3. , 3. , 2.6, 2.2, 2. , 2.2],
            [2. , 1.9, 2.4, 2.2, 2.8, 2.2, 2.2, 2.3, 2.1, 2.3],
            [2. , 1.9, 2. , 2. , 2. , 1.5, 1.5, 1.8, 2.2, 2.5],
            [1.3, 1.1, 0.8, 1.6, 2. , 1.4, 1.2, 1.4, 2.3, 2.5],
            [0.4, 1.1, 1.4, 2.7, 2.7, 2.2, 1.8, 2. , 2.5, 2.9],
            [0.6, 1.4, 1.9, 3.3, 2.9, 2.8, 2.5, 2.8, 3. , 3. ],
            [0.7, 1.7, 1.9, 2.8, 2.3, 3. , 3. , 3.3, 3.3, 3.3],
            [1.6, 1.6, 1. , 1.5, 1.6, 2.6, 3. , 3.3, 3.3, 3.2],
            [2.3, 1.4, 0.5, 0.3, 1. , 2.4, 3. , 3.5, 3.3, 3.2]])

    ##########################################

    Examples
    --------
    >>> #sp.conv2d_nan
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import seaborn as sns
    >>> import scipy
    >>> import spkit as sp
    >>> seed = 2
    >>> I = (sp.create_signal_2d(n=10,seed=seed,sg_winlen=3)*10).round(0)
    >>> np.random.seed(seed)
    >>> r = np.random.rand(*I.shape)
    >>> np.random.seed(None)
    >>> I[r<0.05] = np.nan
    >>> kernel = np.ones([3,3])/9
    >>> I_scipy = scipy.signal.convolve2d(I.copy(), kernel,mode='same', boundary='fill',fillvalue=0)
    >>> I_spkit = sp.conv2d_nan(I.copy(), kernel, boundary='fill',fillvalue=0)
    >>> plt.figure(figsize=(12,4))
    >>> plt.subplot(131)
    >>> sns.heatmap(I, annot=True,cbar=False,xticklabels='', yticklabels='')
    >>> plt.title('Image with NaNs')
    >>> plt.subplot(132)
    >>> sns.heatmap(I_scipy, annot=True,cbar=False,xticklabels='', yticklabels='')
    >>> plt.title('Convolution uisng Scipy \n (scipy.signal.convolve2d)')
    >>> plt.subplot(133)
    >>> sns.heatmap(I_spkit, annot=True,cbar=False,xticklabels='', yticklabels='')
    >>> plt.title('Convolution uisng Spkit \n (sp.conv2d_nan)')
    >>> plt.tight_layout()
    >>> plt.show()
    """

    if denormalise_ker: kernel = denorm_kernel(kernel)

    x = x.astype(float)
    r1,c1 = x.shape
    kr,kc = kernel.shape
    pad_len = kr//2
    if boundary in ['constant','fill']:
        xaf = np.pad(x,pad_len,mode='constant',constant_values=fillvalue)
    elif boundary in ['symmetric','wrap','reflect']:
        xaf = np.pad(x,pad_len,mode=boundary,)
    elif boundary in ['symm']:
        xaf = np.pad(x,pad_len,mode='symmetric')
    else:
        raise NameError("not supported 'boundary' condition, should be on of {'constant','fill', 'symmetric','wrap','reflect', 'symm'} ")

    r2,c2 = xaf.shape
    y = np.zeros([r2,c2])
    #print(r1,c1,kr,kc,pad_len,r2,c2,y.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            #print(xaf[i:i+kr,j:j+kc], kernel)
            y[i,j] = np.nanmean(xaf[i:i+kr,j:j+kc]*kernel)
    return y[:r1,:c1]

def conv1d_fb(x, kernel, iterations=1,mode='same', return_full=False):
    r"""1D Forward-Backward-Convolution (ConvFB)
    
    **1D Forward-Backward-Convolution (ConvFB)**

    Parameters
    ----------
    x: np.array (n,)
      - signal
    kernel: kernel to be used
    iterations >=1, 
      - applying conv_fb recursively
    return_full: bool,
      - if true, it will return 3 times of length of signal

    Returns
    -------
    y: output signal

    See Also
    --------
    conv1d_nan, fill_nans_1d, fill_nans_2d

    Examples
    --------
    #sp.conv1d_fb
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    fs=100
    seed=1
    x = sp.create_signal_1d(n=200,seed=seed)
    np.random.seed(seed)
    x = sp.add_noise(x,snr_db=0)
    np.random.seed(None)
    t = np.arange(len(x))/fs
    kernel = sp.gaussian_kernel(window_length=11)
    y1 = signal.convolve(x.copy(),kernel, method='auto',mode='same')
    y2 = sp.conv1d_fb(x.copy(),kernel,iterations=2)
    plt.figure(figsize=(12,3))
    plt.plot(t,x,label='x: signal')
    plt.plot(t,y1,label='y1: conv')
    plt.plot(t,y2,label='y2: conv_fb, itr=2')
    plt.xlim([t[0],t[-1]])
    plt.ylabel('signal')
    plt.grid()
    plt.legend(loc = 'upper left')
    plt.tight_layout()
    plt.show()

    """
    # pad
    y = np.hstack((x[::-1], x, x[::-1]))
    for _ in range(iterations):
        x_f = np.convolve(y, kernel, mode=mode)
        x_b = np.convolve(y[::-1], kernel, mode=mode)[::-1]

        w = np.arange(0, len(x_f), 1)
        w = w/np.max(w)
        y = x_f*w + x_b*(1-w)

    if return_full: return y
    return y[len(x):len(x)*2]

def conv1d_fft(x,y):
    r"""1D Convolution using FFT
    
    Parameters
    ----------
    x,y: arrays
     - input signals

    Returns
    -------
    z = convolution of x and y

    References
    ----------
    * wikipedia - https://en.wikipedia.org/wiki/Convolution_theorem
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    conv1d_nan, fill_nans_1d, fill_nans_2d, conv1d_fb

    Examples
    --------
    #sp.conv1d_fft
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    fs=100
    seed=100
    x = sp.create_signal_1d(n=200,seed=seed)
    t = np.arange(len(x))/fs
    kernel = sp.gaussian_kernel(window_length=11)
    y1 = signal.convolve(x.copy(),kernel, method='auto',mode='same')
    y2 = sp.conv1d_fft(x.copy(),kernel)
    y2x = y2[len(kernel)//2:len(kernel)//2+len(x)]
    plt.figure(figsize=(12,3))
    plt.plot(t,x,label='x: signal')
    plt.plot(t,y1,label='y1: conv')
    plt.plot(t,y2x,label='y2: conv FFT')
    plt.xlim([t[0],t[-1]])
    plt.ylabel('signal')
    plt.grid()
    plt.legend(loc = 'upper left')
    plt.tight_layout()
    plt.show()
    """
    N = len(x) + len(y)-1
    P = 2**np.ceil(np.log2(N)).astype(int)
    z = np.fft.fft(x,P)*np.fft.fft(y,P)
    z = np.fft.ifft(z)
    z = z[:N]
    return z

def periodogram(x,fs=128,show_plot=False, method ='welch',win='hann',nfft=None,
                scaling='density',average='mean',detrend='constant',nperseg=None, noverlap=None,
                showlog=True,show=False,label=None,return_frq=True):
    r"""Computing Periodogram

    **Computing Periodogram using Welch or Periodogram method**
    

    Parameters
    -----------
    x: 1d array, (n,)
        - input signal, 
    fs: sampling frequency
    method: {'welch','None'}
        -  if None, then periodogram is computed without welch method
    win: {'hann', 'ham', ..}
        : window function
    scaling: {'density', 'spectrum'}
        - 'density'--V**2/Hz 'spectrum'--V**2
    average: {'mean', 'median'}
        - averaging method
    detrend: False, 'constant', 'linear'
    nfft:  None, n-point FFT

    Returns
    -------
        Px : array, |periodogram|
        Frq : array, frequency

    References
    ----------
    * wikipedia - https://en.wikipedia.org/wiki/Periodogram
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    spkit: #TODO

    Examples
    --------
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs = sp.load_data.eeg_sample_1ch()
    Px1, frq1 = sp.periodogram(x,fs=128,method ='welch',nperseg=128)
    Px2, frq2 = sp.periodogram(x,fs=128,method =None,nfft=128)
    plt.figure(figsize=(5,4))
    plt.plot(frq1,np.log(Px1),label='Welch')
    plt.plot(frq2,np.log(Px2),label='Periodogram')
    plt.xlim([frq[0],frq[-1]])
    plt.grid()
    plt.ylabel('V**2/Hz')
    plt.xlabel('Frequency (Hz)')
    plt.legend()
    plt.show()

    """
    if method ==None:
        frq, Px = scipy.signal.periodogram(x,fs,win,nfft=nfft,scaling=scaling,detrend=detrend)
    elif method =='welch':
        #f, Pxx = scipy.signal.welch(x,fs,win,nperseg=np.clip(len(x),0,256),scaling=scaling,average=average,detrend=detrend)
        frq, Px = scipy.signal.welch(x,fs,win,nfft=nfft,scaling=scaling,detrend=detrend,nperseg=nperseg,noverlap=noverlap,average=average)
    
    if show_plot:
        if showlog:
            plt.plot(frq,np.log(np.abs(Px)),label=label)
        else:
            plt.plot(frq,np.abs(Px))
        plt.xlim([frq[0],frq[-1]])
        plt.xlabel('Frequency (Hz)')
        if scaling=='density':
            plt.ylabel('V**2/Hz')
        elif scaling=='spectrum':
            plt.ylabel('V**2')
        plt.grid()
        if show: plt.show()
    if return_frq:
        return np.abs(Px), frq
    return np.abs(Px)


"""
BASIC WAVELET FILTERING
------------------------
"""

def get_theta(w,N,k=1.5,method='optimal',IPR=[0.25,0.75]):
    r"""Threshold for wavelet filtering
    
    Parameters
    ----------
    w: wavelet coeeficients
    N: length of signal x for noise eastimation
    method:  method to compute threshold
          - 'optimal' - optimal threshold based on noise estimation
          - 'sd'      - mu :math:`\pm` k*sd
          - 'iqr'     - Q1 - k*IQR, Q3 + k*IQR
    k: for outlier computation as above
    IPR   : Inter-percentile range: quartile to be considers for inter-quartile range IPR = [0.25, 0.75]
          : could be [0.3, 0.7] for more aggressive threshold

    Returns
    -------
    theta_l: lower threshold for wavelet coeeficients
    theta_u: upper threshold for wavelet coeeficients


    References
    ----------
    * wikipedia
    
    
    Notes
    -----
    #--

    See Also
    --------
    spkit: #TODO

    Examples
    --------
    import spkit as sp

    """
    
    if method =='optimal':
        sig = np.median(abs(w))/0.6745
        theta_u = sig*np.sqrt(2*np.log(N))
        theta_l = -theta_u
    elif method =='sd':
        theta_u = np.mean(w) + k*np.std(w)
        theta_l = np.mean(w) - k*np.std(w)
    elif method=='iqr':
        r = scipystats.iqr(w)
        q1 = np.quantile(w,IPR[0])
        q3 = np.quantile(w,IPR[1])
        #assert r ==q3-q1
        theta_u = q3 + k*r
        theta_l = q1 - k*r
    return theta_l, theta_u

def wavelet_filtering(x,wv='db3',threshold='optimal',filter_out_below=True,k=1.5,mode='elim',show=False,wpd_mode='symmetric',
        wpd_maxlevel=None,packetwise=False,WPD=False,lvl=[],verbose=False,fs=128.0,sf=1,IPR=[0.25,0.75], figsize=(11,6),plot_abs_coef=False):

    r"""Wavelet Filtering


    **Wavelet Filtering**

    Wavelet filtering is applied on signal by decomposing signal into wavelet domain, filtering out wavelet coefficients
    and reconstructing signal back.  Signal is decompose using DWT with wavelet function specified as wv (e.g. db3), and
    filtering out coefficients using by `threshold`, `mode`, and `filter_out_below` arguments.

    
    1. Wavelet Transform 
        :math:`W(k) = DWT(x(n))`

    2. Filtering (with `elim` mode)
       - :math:`W(k)=0`
          - if :math:`|W(k)|<=threshold`  and if filter_out_below = True
          - if :math:`|W(k)|>threshold`   and if filter_out_below = False
    
    3. Reconstruction
         :math:`x'(n) = IDWT(W(k))`



    Parameters
    ----------

    x: 1d array

    Threshold Computation method:

    threshold: 'str' or float, default - 'optimal'
                - if str, method to compute threshold, example : 'optimal', 'sd', 'iqr'
                - 'optimal': threshold = sig*sqrt(2logN), sig = median(|w|)/0.6745
                - 'sd' : threshold = k*SD(w)
                - 'iqr': threshold = q3+kr, threshold_l =q1-kr, where r = IQR(w)  #Tukey's fences
                - 'ttt': Modified Thompson Tau test (ttt) # TODO
    
    wv : str, 'db3'(default)
        - **Wavelet family**
        - {'db3'.....'db38', 'sym2.....sym20', 'coif1.....coif17', 'bior1.1....bior6.8', 'rbio1.1...rbio6.8', 'dmey'}

    mode: str,  default = 'elim'
        - 'elim' - remove the coeeficient (by zering out), 
        - 'clip' - cliping the coefficient to threshold

    filter_out_below: bool, default True,
        - if true, wavelet coefficient below threshold are eliminated else obove threshold

    wpd_mode = str,  default 'symmetric'
            -  **Wavelet Decomposition modes**
            -  ['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization']
                
    wpd_maxlevel: int
        - level of decomposition,
            if None, max level posible is used
    
    packetwise: bool, if true, 
        - thresholding is applied to each packet/level individually, else globally
    
    WPD: if true, 
        - WPD is applied as wavelet transform
    
    lvl: list
        list of levels/packets apply the thresholding, if empty, applied to all the levels/packets

    show: bool, deafult=False,
        - if to plot figure, it True, following are used
        figsize: default=(11,6), size of figure
        plot_abs_coef: bool, deafult=False,
                    if True,plot abs coefficient value, else signed

    Returns
    -------
    xR:  filtered signal, same size as x


    References
    ----------
    * wikipedia - 
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    wavelet_filtering_win: applying wavelet filtering on smaller windows
    filter_X: spectral filtering

    Examples
    --------
    import spkit as sp
    x,fs = sp.load_data.eeg_sample_1ch()
    xr = sp.wavelet_filtering(x,fs=fs, wv='db3', threshold='optimal',show=True)

    #################
    
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x = sp.create_signal_1d(n=1000,seed=1,sg_polyorder=5, sg_winlen=11)
    x = sp.add_noise(x,snr_db=10)
    xr = sp.wavelet_filtering(x.copy(),wv='db3',threshold='optimal',show=True)
    """

    assert isinstance(threshold,str) or isinstance(threshold, float)
    #'method for computing threshold is not defined. Must be one of optimal,sd,iqr or a float value'

    if filter_out_below: assert mode=='elim'
    if verbose:
        print('WPD:',WPD,' wv:',wv,' threshold:',threshold,' k:',k,' mode:',mode,' filter_out_below?:',filter_out_below)

    N = len(x)
    if WPD: # Wavelet Packet Decomposition
        wp = wt.WaveletPacket(x, wavelet=wv, mode=wpd_mode,maxlevel=wpd_maxlevel)
        wr = [wp[node.path].data for node in wp.get_level(wp.maxlevel, 'natural') ]
        WR = np.hstack(wr)
        nodes = [node for node in wp.get_level(wp.maxlevel, 'natural')]
    else:  # Wavelet Transform
        wr = wt.wavedec(x,wavelet=wv, mode=wpd_mode,level=wpd_maxlevel)
        WR = np.hstack(wr)
        nodes = np.arange(len(wr))

    if verbose>1:
        print(f'signal length: {len(x)},  #coefficients: {len(WR)}, #nodes: {len(nodes)}')

    if not(packetwise):
        if isinstance(threshold,str):
            theta_l, theta_u = get_theta(WR,N,k=k,method=threshold,IPR=IPR)
        else:
            theta_l, theta_u = -threshold, threshold

        theta_l, theta_u = sf*theta_l, sf*theta_u

        if verbose>1: print(f'global thresholds: {threshold}, {theta_l, theta_u}')


    for i in range(len(nodes)):
        #for node in wp.get_level(wp.maxlevel, 'natural'):
        if len(lvl)==0 or (i in lvl):
            if verbose>2: print(f'node #: {i}')
            c = wp[nodes[i].path].data if WPD else wr[i]
            if packetwise:
                if isinstance(threshold,str):
                    theta_l, theta_u = get_theta(c,len(c),k=k,method=threshold,IPR=IPR)
                else:
                    theta_l, theta_u = -threshold, threshold
                theta_l, theta_u = sf*theta_l, sf*theta_u
                if verbose>2: print(f'local thresholds: {threshold}, {theta_l, theta_u}')

            if filter_out_below:
                idx = (c>=theta_l) & (c<=theta_u)
                #mode='elim'
                c[idx] = 0
            else:
                idx = (c<=theta_l) | (c>=theta_u)
                if mode=='elim':
                    c[idx] = 0
                elif mode=='clip':
                    c = np.clip(c,theta_l, theta_u)

            if WPD:
                wp[nodes[i].path].data = c
            else:
                wr[i] = c

    #Reconstruction
    if WPD:
        xR = wp.reconstruct(update=False)
    else:
        xR = wt.waverec(wr, wavelet = wv)

    if show:
        plt.figure(figsize=figsize)
        plt.subplot(211)
        if plot_abs_coef:
            plt.plot(np.abs(WR),alpha=0.8,label='Coef.',color='C0')
            ytiW =[0,np.max(np.abs(WR))]
            plt.ylabel('Wavelete Coefficients \n |W|')
        else:
            plt.plot(WR,alpha=0.8,label='Coef.',color='C0')
            ytiW =[np.min(WR),np.max(WR)]
            plt.ylabel('Wavelete Coefficients \n W')

        #print('maxlevel :',wp.maxlevel)
        if WPD: wr = [wp[node.path].data for node in wp.get_level(wp.maxlevel, 'natural') ]
        WRi = np.hstack(wr)

        if plot_abs_coef:
            plt.plot(np.abs(WRi),color='C3',alpha=0.8,label='Filtered Coff.')
            ytiW = ytiW+[np.max(np.abs(WRi))]
        else:
            plt.plot(WRi,color='C3',alpha=0.8,label='Filtered Coff.')
            ytiW = ytiW+[np.min(WRi),np.max(WRi)]

        if not(packetwise):
            if plot_abs_coef:
                ytiW = ytiW+list(np.abs([theta_l, theta_u]))
                plt.axhline(np.abs(theta_l),color='C2',ls='--',lw=2, alpha=0.7)
                plt.axhline(np.abs(theta_u),color='C2',ls='--',lw=2, alpha=0.7)
            else:
                ytiW = ytiW+[theta_l, theta_u]
                plt.axhline(theta_l,color='C2',ls='--',lw=2, alpha=0.7)
                plt.axhline(theta_u,color='C2',ls='--',lw=2, alpha=0.7)

        ytiW = list(set(ytiW))
        plt.yticks(ytiW)

        ki = 0
        for i in range(len(wr)):
            ki+=len(wr[i])
            plt.axvline(ki,color='k',ls='--',lw=1)


        plt.grid()
        plt.legend(framealpha=0.5)
        plt.xlim([0,len(WRi)])

        plt.subplot(212)
        if WPD:
            t = np.arange(len(wp.data))/fs
            plt.plot(t,wp.data,color='C0',alpha=0.8,label='signal')
        else:
            t = np.arange(len(x))/fs
            plt.plot(t,x,color='C0',alpha=0.8,label='signal')
        plt.plot(t,xR,color='C3',alpha=0.8,label='corrected')
        plt.ylabel('Signal')
        plt.yticks([np.min(xR),np.min(x),0,np.max(xR),np.max(x)])
        plt.xlim([t[0],t[-1]])
        plt.legend(framealpha=0.5)
        plt.grid()
        plt.show()
    return xR

def wavelet_filtering_win(x,winsize=128,wv='db3',threshold='optimal',filter_out_below=True,k=1.5,mode='elim',wpd_mode='symmetric',
                wpd_maxlevel=None,packetwise=False,WPD=True,lvl=[],verbose=False,sf=1,
                hopesize=None, wintype='hamming',windowing_before=False,IPR=[0.25, 0.75]):

    r"""Wavelet Filtering window-wise


    **Wavelet Filtering applied to smaller windows**
    

    Same as wavelet_filtering fumction, applied to smaller overlapping windows and reconstructed by overlap-add method

    for documentation, check help(wavelet_filtering)


    Parameters
    ----------

    x: 1d array

    winsize: int deafult=128
        -  size of window to apply wavelet filtering
    hopesize: int, deafult=None
        -  shift factor, if winsize=128, and hopesize=64, then 50% overlap windowing is used.
        - if None, hopesize = winsize//2

    Threshold Computation method:

    threshold: 'str' or float, default - 'optimal'
                - if str, method to compute threshold, example : 'optimal', 'sd', 'iqr'
                - 'optimal': threshold = sig*sqrt(2logN), sig = median(|w|)/0.6745
                - 'sd' : threshold = k*SD(w)
                - 'iqr': threshold = q3+kr, threshold_l =q1-kr, where r = IQR(w)  #Tukey's fences
                - 'ttt': Modified Thompson Tau test (ttt) # TODO

    mode: str,  default = 'elim'
        - 'elim' - remove the coeeficient (by zering out), 
        - 'clip' - cliping the coefficient to threshold

    filter_out_below: bool, default True,
        - if true, wavelet coefficient below threshold are eliminated else obove threshold

    Wavelet Decomposition modes:
    wpd_mode = ['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization']
                default 'symmetric'

    wpd_maxlevel: level of decomposition, if None, max level posible is used

    Wavelet family:
    wv = ['db3'.....'db38', 'sym2.....sym20', 'coif1.....coif17', 'bior1.1....bior6.8', 'rbio1.1...rbio6.8', 'dmey']
        :'db3'(default)

    packetwise: bool, if true, 
        - thresholding is applied to each packet/level individually, else globally
    
    WPD: if true, 
        - WPD is applied as wavelet transform
    
    lvl: list
        list of levels/packets apply the thresholding, if empty, applied to all the levels/packets

    show: bool, deafult=False,
        - if to plot figure, it True, following are used
        figsize: default=(11,6), size of figure
        plot_abs_coef: bool, deafult=False,
                    if True,plot abs coefficient value, else signed

    Returns
    -------
    xR:  filtered signal, same size as x

    References
    ----------
    * wikipedia - 
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    spkit: #TODO

    Examples
    --------
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x = sp.create_signal_1d(n=1000,seed=1,sg_polyorder=5, sg_winlen=11)
    x = x + 0.1*np.random.randn(len(x))
    t = np.arange(len(x))/100
    xr = sp.wavelet_filtering_win(x.copy(),winsize=128,wv='db3',threshold='optimal')
    plt.figure(figsize=(12,3))
    plt.plot(t,x, label='x: noisy signal')
    plt.plot(t,xr, label='x: filtered signal')
    plt.xlim([t[0],t[-1]])
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    plt.grid()
    plt.show()
    """

    if hopesize is None: hopesize = winsize//2
    M   = winsize
    H   = hopesize
    hM1 = (M+1)//2
    hM2 = M//2

    xt  = np.hstack([np.zeros(hM2),x,np.zeros(hM1)])

    pin  = hM1
    pend = xt.size-hM1
    wh   = signal.get_window(wintype,M)
    if verbose: print('Windowing before apply : ',windowing_before)

    xR   = np.zeros(xt.shape)
    pf=0
    while pin<=pend:
        if verbose:
            if 100*pin/float(pend)>=pf+1:
                pf = 100*pin/float(pend)
                pbar = '|'+'#'*int(pf)+' '*(99-int(pf))+'|'
                print(str(np.round(pf,2))+'%'+pbar,end='\r', flush=True)
        xi = xt[pin-hM1:pin+hM2]
        if windowing_before: xi *=wh

        xr = wavelet_filtering(xi,wv=wv,threshold=threshold,filter_out_below=filter_out_below,k=k,mode=mode,wpd_mode=wpd_mode,wpd_maxlevel=wpd_maxlevel,
               packetwise=packetwise,WPD=WPD,lvl=lvl,verbose=0,sf=sf,IPR=IPR)

        if not(windowing_before): xr *=wh

        xR[pin-hM1:pin+hM2] += H*xr  ## Overlap Add method
        pin += H
    xR = xR[hM2:-hM1]/sum(wh)

    return xR

# TOBE Removed
@deprecated("due to naming consistency, please use 'wpa_coeff' for updated/improved functionality. [spkit-0.0.9.7]")
def WPA_coeff(x,wv='db3',mode='symmetric',maxlevel=None, verticle_stacked=False):
    r"""Wavelet Packet Decomposition

    Wavelet Packet Decomposition

    input
    -----
        x: 1d signal array
        wv :  wavelet type - default 'db3'
        mode='symmetric'
        maxlevel=None -  maximum levels of decomposition will result in 2**maxlevel packets

        verticle_stacked : if True, coeeficients are vertically stacked -  good for temporal alignment

    output
    -----
        WK: Wavelet Packet Coeeficients
        if verticle_stacked True : shape (2**maxlevel, k), 2**maxlevel - packets with k coeeficient in each
        if verticle_stacked False: shape (2**maxlevel * k, )

    Parameters
    ----------

    Returns
    -------

    References
    ----------
    * wikipedia - 
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    spkit: #TODO

    Examples
    --------
    >>> import numpy as np
    >>> import spkit as sp
    >>> #TODO

    """
    wp = wt.WaveletPacket(x, wavelet=wv, mode=mode,maxlevel=maxlevel)
    wr = [wp[node.path].data for node in wp.get_level(wp.maxlevel, 'natural') ]
    WK = np.vstack(wr) if verticle_stacked else np.hstack(wr)
    return WK

@deprecated("due to naming consistency, please use 'wpa_coeff_win' for updated/improved functionality. [spkit-0.0.9.7]")
def WPA_temporal(x,winsize=128,overlap=64,wv='db3',mode='symmetric',maxlevel=None,verticle_stacked=True,pad=True,verbose=0):
    r"""Wavelet Packet Decomposition - window-wise

    Wavelet Packet Decomposition -  for each window and stacked together
    

    input
    -----
        x: 1d signal array
        wv :  wavelet type - default 'db3'
        mode='symmetric'
        maxlevel=None -  maximum levels of decomposition will result in 2**maxlevel packets

        winsize: size of each window, samples at the end will be discarded, if len(x)%overlap is not eqaul to 0
        to avoid, padd with zeros
        overlap: overlap

    output
    -----
        Wtemp

        Parameters
    ----------

    Returns
    -------

    References
    ----------
    * wikipedia - 
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    spkit: #TODO

    Examples
    --------
    >>> import numpy as np
    >>> import spkit as sp
    >>> #TODO

    """
    winsize = int(winsize)
    overlap = int(overlap)
    xi = x.copy()
    if pad:
        if x.shape[0]%overlap!=0:
            if verbose: print('padding', overlap - x.shape[0]%overlap)
            xi = np.r_[x, x[-1]*np.ones(overlap - x.shape[0]%overlap)]

    win =np.arange(winsize)
    W =[]
    while win[-1]<xi.shape[0]:
        Wi = WPA_coeff(xi[win],verticle_stacked=verticle_stacked,wv=wv,mode=mode,maxlevel=maxlevel)
        W.append(Wi)
        win +=overlap
    Wtemp = np.hstack(W) if verticle_stacked else np.vstack(W).T
    return Wtemp

@deprecated("due to naming consistency, please use 'wpa_plot' for updated/improved functionality. [spkit-0.0.9.7]")
def WPA_plot(x,winsize=128,overlap=64,verticle_stacked=True,wv='db3',mode='symmetric',maxlevel=None,inpterp='sinc',
             fs=128,plot=True,pad=True,verbose=0, plottype='abs',figsize=(15,8)):

    r"""WPA Window-wise - Plot

    Wavelet Packet Decomposition -  temporal - Plot
    -------------------------------------

    return Wavelet coeeficients packet vs time

    Parameters
    ----------

    Returns
    -------

    References
    ----------
    * wikipedia - 
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    spkit: #TODO

    Examples
    --------
    >>> import numpy as np
    >>> import spkit as sp
    >>> #TODO

    """
    xi = x.copy()
    if pad:
        if x.shape[0]%overlap!=0:
            if verbose: print('padding', overlap - x.shape[0]%overlap)
            xi = np.r_[x, x[-1]*np.ones(overlap - x.shape[0]%overlap)]

    Wp = WPA_temporal(xi,winsize=winsize,overlap=overlap,wv=wv,mode=mode,maxlevel=maxlevel,
                         verticle_stacked=verticle_stacked,pad=False,verbose=0)

    if fs is None: fs =1
    t = np.arange(len(xi))/fs

    if plottype=='abs':
        Wp = np.abs(Wp)
    elif plottype=='abs_log':
        Wp = np.log(np.abs(Wp))
    elif plottype=='abs_log_p1':
        Wp = np.log(np.abs(Wp)+1)
    elif plottype=='abs_log10':
        Wp = np.log10(np.abs(Wp))
    elif plottype=='abs_log10_p1':
        Wp = np.log10(np.abs(Wp)+1)

    if plot:
        plt.figure(figsize=figsize)
        plt.subplot(211)
        plt.imshow(Wp,aspect='auto',origin='lower',interpolation=inpterp,cmap='jet',extent=[t[0], t[-1], 1, Wp.shape[0]])
        plt.xlabel('time (s)')
        plt.ylabel('packet')
        plt.subplot(212)
        plt.plot(t,xi)
        plt.xlim([t[0], t[-1]])
        plt.grid()
        plt.xlabel('time (s)')
        plt.ylabel('x: amplitude')
        plt.show()
    return Wp

@deprecated("due to naming consistency, please use 'wavelet_decompositions' for updated/improved functionality. [spkit-0.0.9.7]")
def Wavelet_decompositions(x,wv='db3',L = 6,threshold=np.inf,show=True,WPD=False):
    r"""Decomposing signal into wavelet components


    Decomposing signal into different level of wavalet based signals

    Parameters
    ----------

    Returns
    -------

    References
    ----------
    * wikipedia - 
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    spkit: #TODO

    Examples
    --------
    >>> import numpy as np
    >>> import spkit as sp
    >>> #TODO

    """

    XR = []
    N = L+1
    if WPD: N = 2**L
    for l in range(N):
        lvl = list(range(N))
        lvl.remove(l)
        if show:print(lvl)
        xr = wavelet_filtering(x.copy(), wv=wv, threshold=threshold,
                              WPD=WPD, lvl=lvl,wpd_maxlevel=L,verbose=show,show=show,packetwise=True)
        XR.append(xr)

    XR = np.array(XR).T
    XR.shape
    return XR

#updated versions
def wpa_coeff(x,wv='db3',mode='symmetric',maxlevel=None, verticle_stacked=True):
    r"""Wavelet Packet Decomposition

    Wavelet Packet Decomposition

    Parameters
    ----------
    x: 1d signal array
    wv :  wavelet type, default = 'db3'
    mode: {'symmetric', ..}, default='symmetric'
    maxlevel: int, or None, default=None 
      -  maximum levels of decomposition will result in 2**maxlevel packets

    verticle_stacked: bool, default=True
      - if True, coefficients are vertically stacked -  good for temporal alignment

    Returns
    -------
    WK: Wavelet Packet Coeeficients
      - if verticle_stacked True : shape (2**maxlevel, k), 2**maxlevel - packets with k coeeficient in each
      - if verticle_stacked False: shape (2**maxlevel * k, )

    
    References
    ----------
    * wikipedia - https://en.wikipedia.org/wiki/Wavelet_packet_decomposition
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    spkit: #TODO

    Examples
    --------
    #sp.wpa_coeff
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs,lead_names = sp.data.ecg_sample_12leads(sample=2)
    x = x[:int(fs*5),5]
    x = sp.filterDC_sGolay(x, window_length=fs//3+1)
    t = np.arange(len(x))/fs
    WK = sp.wpa_coeff(x,wv='db3',mode='symmetric',maxlevel=3, verticle_stacked=True)
    plt.figure(figsize=(10,5))
    plt.subplot(211)
    plt.plot(t,x)
    plt.xlim([t[0],t[-1]])
    plt.xlabel('time (s)')
    plt.ylabel('ECG Signal')
    plt.grid()
    plt.subplot(212)
    plt.imshow(np.log(np.abs(WK)+0.01),aspect='auto',cmap='jet',)
    plt.ylabel('Wavelet Packets')
    plt.xlabel('time (n)')
    plt.tight_layout()
    plt.show()

    """
    wp = wt.WaveletPacket(x, wavelet=wv, mode=mode,maxlevel=maxlevel)
    wr = [wp[node.path].data for node in wp.get_level(wp.maxlevel, 'natural') ]
    WK = np.vstack(wr) if verticle_stacked else np.hstack(wr)
    return WK

def wpa_coeff_win(x,winsize=128,overlap=None,wv='db3',mode='symmetric',maxlevel=None,verticle_stacked=True,pad=False,verbose=0):
    r"""Wavelet Packet Decomposition - window-wise

    Wavelet Packet Decomposition -  for each window and stacked together
    

    Parameters
    ----------
    x: 1d signal array
    wv :  wavelet type, default = 'db3'
    winsize: int, default=128
     -  window size, samples at the end will be discarded, if len(x)%overlap is not eqaul to 0, to avoid, pad the signal
    overlap: int, default=None,
     - if None, overlap= winsize//2
     - shift of window
    mode: {'symmetric', ..}, default='symmetric'
    maxlevel: int, or None, default=None 
      -  maximum levels of decomposition will result in 2**maxlevel packets

    verticle_stacked: bool, default=True
      - if True, coefficients are vertically stacked -  good for temporal alignment
    
    pad: bool, default=False,
      -  if True, signal will be padded with last value to make len(x)%overlap==0
    
    verbose: bool, deault=False
      - Verbosity mode

    Returns
    -------
    WK_seq: 2D array, (N,K), 
      -  N = 2**maxlevel, number of packets
      -  K, number of coefficients in each packet

    References
    ----------
    * wikipedia - https://en.wikipedia.org/wiki/Wavelet_packet_decomposition
    
    
    Notes
    -----
    :func:`wpa_coeff_win` improves the time resolution of the WPA, compared to :func:`wpa_coeff` for long sequence

    See Also
    --------
    wpa_coeff, wpa_plot, wavelet_decomposed_signals, cwt

    Examples
    --------
    #sp.wpa_coeff_win
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs,lead_names = sp.data.ecg_sample_12leads(sample=2)
    x = x[:int(fs*5),5]
    x = sp.filterDC_sGolay(x, window_length=fs//3+1)
    t = np.arange(len(x))/fs
    WK = sp.wpa_coeff_win(x,wv='db3',winsize=512,overlap=256,mode='symmetric',maxlevel=3, verticle_stacked=True,pad=False)
    plt.figure(figsize=(10,5))
    plt.subplot(211)
    plt.plot(t,x)
    plt.xlim([t[0],t[-1]])
    plt.xlabel('time (s)')
    plt.ylabel('ECG Signal')
    plt.grid()
    plt.subplot(212)
    plt.imshow(np.log(np.abs(WK)+0.01),aspect='auto',cmap='jet',)
    plt.ylabel('Wavelet Packets')
    plt.xlabel('time (n)')
    plt.tight_layout()
    plt.show()

    """
    #wpa_coeff(x,wv='db3',mode='symmetric',maxlevel=None, verticle_stacked=True)
    winsize = int(winsize)
    overlap = int(overlap)
    xi = x.copy()
    if pad:
        if x.shape[0]%overlap!=0:
            if verbose: print('padding', overlap - x.shape[0]%overlap)
            xi = np.r_[x, x[-1]*np.ones(overlap - x.shape[0]%overlap)]

    win =np.arange(winsize)
    W =[]
    while win[-1]<xi.shape[0]:
        Wi = wpa_coeff(xi[win],verticle_stacked=verticle_stacked,wv=wv,mode=mode,maxlevel=maxlevel)
        W.append(Wi)
        win +=overlap
    WK_seq = np.hstack(W) if verticle_stacked else np.vstack(W).T
    return WK_seq

def wpa_plot(x,winsize=128,overlap=64,verticle_stacked=True,wv='db3',mode='symmetric',maxlevel=None,inpterp='sinc',
             fs=128,plot=True,pad=True,verbose=0, plottype='abs',figsize=(15,8)):
    r"""WPA Window-wise Plot

    Wavelet Packet Decomposition -  temporal - Plot
    

    Parameters
    ----------
    x: 1d signal array
    wv :  wavelet type, default = 'db3'
    winsize: int, default=128
     -  window size, samples at the end will be discarded, if len(x)%overlap is not eqaul to 0, to avoid, pad the signal
    overlap: int, default=None,
     - if None, overlap= winsize//2
     - shift of window
    mode: {'symmetric', ..}, default='symmetric'
    maxlevel: int, or None, default=None 
      -  maximum levels of decomposition will result in 2**maxlevel packets

    verticle_stacked: bool, default=True
      - if True, coefficients are vertically stacked -  good for temporal alignment
    
    pad: bool, default=False,
      -  if True, signal will be padded with last value to make len(x)%overlap==0
    
    verbose: bool, deault=False
      - Verbosity mode

    plottype: {'abs','abs_log','abs_log_p1','abs_log10','abs_log10_p1'}
      - used to plot Wavelet coefficients
    figsize: figure size
    inpterp: interpolation type

    Returns
    -------
    WK_seq: 2D array, (N,K), 
      -  N = 2**maxlevel, number of packets
      -  K, number of coefficients in each packet

    References
    ----------
    * wikipedia - https://en.wikipedia.org/wiki/Wavelet_packet_decomposition
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    wpa_coeff, wpa_plot, wavelet_decomposed_signals, cwt

    Examples
    --------
    #sp.wpa_plot
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs,lead_names = sp.data.ecg_sample_12leads(sample=2)
    x = x[:int(fs*5),5]
    x = sp.filterDC_sGolay(x, window_length=fs//3+1)
    t = np.arange(len(x))/fs
    WK = sp.wpa_plot(x,wv='db3',winsize=512,overlap=256,mode='symmetric',maxlevel=3,plottype='abs_log_p1',
                    verticle_stacked=True,pad=False,figsize=(10,6))

    """
    xi = x.copy()
    if pad:
        if x.shape[0]%overlap!=0:
            if verbose: print('padding', overlap - x.shape[0]%overlap)
            xi = np.r_[x, x[-1]*np.ones(overlap - x.shape[0]%overlap)]
    #wpa_coeff_win(x,winsize=128,overlap=64,wv='db3',mode='symmetric',maxlevel=None,verticle_stacked=True,pad=True,verbose=0
    Wp = wpa_coeff_win(xi,winsize=winsize,overlap=overlap,wv=wv,mode=mode,maxlevel=maxlevel,
                         verticle_stacked=verticle_stacked,pad=False,verbose=0)

    if fs is None: fs =1
    t = np.arange(len(xi))/fs

    if plottype=='abs':
        Wp = np.abs(Wp)
    elif plottype=='abs_log':
        Wp = np.log(np.abs(Wp))
    elif plottype=='abs_log_p1':
        Wp = np.log(np.abs(Wp)+1)
    elif plottype=='abs_log10':
        Wp = np.log10(np.abs(Wp))
    elif plottype=='abs_log10_p1':
        Wp = np.log10(np.abs(Wp)+1)

    if plot:
        plt.figure(figsize=figsize)
        plt.subplot(211)
        plt.imshow(Wp,aspect='auto',origin='lower',interpolation=inpterp,cmap='jet',extent=[t[0], t[-1], 1, Wp.shape[0]])
        plt.xlabel('time (s)')
        plt.ylabel('wavelet packets')
        plt.subplot(212)
        plt.plot(t,xi)
        plt.xlim([t[0], t[-1]])
        plt.grid()
        plt.xlabel('time (s)')
        plt.ylabel('x: amplitude')
        plt.tight_layout()
        plt.show()
    return Wp

def wavelet_decomposed_signals(x,wv='db3',L=3,WPD=False,threshold=np.inf,plot_final=True,plot_each=False,fs=100,show=True,figsize=(10,8)):
    r"""Decomposing signal into signals of wavelet components

    .. math::
       x(t) &=  x_{w_0}(t) + x_{w_1}(t) + x_{w_2}(t) + \cdots + x_{w_{K-1}}(t)

       x(t) &= \sum_{k=0}^{K-1} x_{w_k}(t)
       
    Decomposing signal into different levels of 'wavalet based signals'

    'wavalet based signal' is reconstructed signal from wavelet coefficients such that
    some of all the decomposed signal is equal to original signals

    Parameters
    ----------
    x: 1d signal array
    wv:  wavelet type, default = 'db3'
    L : int, default=3
        -  number of level for decomposition
        -  if WPA False, then L+1 decomposed signals are returned
        -  if WPA is True, then 2**L decomposed signals are returned
    threshold: scalar, deafult='np.inf'
        - to apply wavelet filtering
        - if 'np.inf', no filtering is applied
    plot_each: bool, default=False
        - plot of each decomposed part is shown.
        - USE only to analyse, it slow down the computations
    WPD: bool, default False
        - if True, Wavelet Packet Decomposition is applied,resultings 2**L decomposed signals 
        - if False, Wavelet Transform is applied,resultings L+1 decomposed signals 
    fs: int, default=100,
        - sampling frequency, only used for plot labeling
    show: bool, deafult=True
        - if True, plt.show() is executed for plot, else not. Useful to edit plot later
    figsize:tuple defualt =(10,8)
        - figure size   
        
    Returns
    -------
        XR: (n,K) array, 
            - K decomposed signals
            - n, same as length of signal

    References
    ----------
    * wikipedia - https://en.wikipedia.org/wiki/Wavelet_transform
    * wikipedia - https://en.wikipedia.org/wiki/Wavelet_packet_decomposition
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    wpa_coeff, wpa_plot, wavelet_decomposed_signals, cwt

    Examples
    --------
    #sp.wavelet_decomposed_signals
    #Example 1: Wavelet Transform
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs,lead_names = sp.data.ecg_sample_12leads(sample=2)
    x = x[:int(fs*5),5]
    x = sp.filterDC_sGolay(x, window_length=fs//3+1)
    t = np.arange(len(x))/fs
    XR = sp.wavelet_decomposed_signals(x,wv='db3',L=3,plot_each=False,WPD=False)
    plt.show()

    #################################
    #Example 2: Wavelet Packet Decomposition
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs,lead_names = sp.data.ecg_sample_12leads(sample=2)
    x = x[:int(fs*5),5]
    x = sp.filterDC_sGolay(x, window_length=fs//3+1)
    t = np.arange(len(x))/fs
    XR = sp.wavelet_decomposed_signals(x,wv='db3',L=3,plot_each=False,WPD=True)
    plt.show()
    """

    XR = []
    N = L+1
    if WPD: N = 2**L
    for l in range(N):
        lvl = list(range(N))
        lvl.remove(l)
        if plot_each:print(lvl)
        xr = wavelet_filtering(x.copy(), wv=wv, threshold=threshold,
                              WPD=WPD, lvl=lvl,wpd_maxlevel=L,verbose=plot_each,show=plot_each,packetwise=True)
        XR.append(xr)

    XR = np.array(XR).T
    if plot_final:
        n, K = XR.shape
        t = np.arange(n)/fs
        plt.figure(figsize=figsize)
        plt.subplot(K+1,1,1)
        plt.plot(t,x,'C1')
        plt.xlim([t[0],t[-1]])
        plt.ylabel(f'x')
        plt.xticks([])
        plt.yticks([])
        if WPD:
            plt.title(f'Decomposed using Wavelet Packet Decomposition(L={L})')
        else:
            plt.title(f'Decomposed using Wavelet Transform (L={L})')
        for i in range(K):
            plt.subplot(K+1,1,i+2)
            plt.plot(t,XR[:,i])
            plt.xlim([t[0],t[-1]])
            if i<K-1:plt.xticks([])
            plt.yticks([])
            plt.ylabel(f'$Xw_{i}$')
        plt.xlabel('time (s)')
        plt.subplots_adjust(hspace=0)
        #plt.tight_layout()

        if show: plt.show()
    return XR

def sinc_interp(x):
    r"""Sinc interpolation

    Upsampling input signal with a factor of 2
    
    Using FFT approach to smooth the interpolated zeros.

    Parameters
    ----------
    x: 1d-array
      -  input signal, with length of n

    Returns
    -------
    y: 1d-array
      - upsampled by factor of 2, with length of 2*n -1
    

    Notes
    -----
    There are effects on boundary 

    Examples
    --------
    #sp.sinc_interp
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    fs=50
    x = sp.create_signal_1d(n=50,seed=2,sg_winlen=20)
    y = sp.sinc_interp(x.copy())
    t = np.arange(len(x))/fs
    t0 = np.arange(len(y))/(2*fs)
    plt.figure(figsize=(10,2))
    plt.plot(t0,y.real,'.',alpha=0.5, label=f'y (n={len(y)})')
    plt.plot(t,x,'.',alpha=0.5,label=f'x (n={len(x)})')
    plt.xlabel('time (s)')
    plt.legend()
    plt.show()
    """
    N = len(x)
    y = np.zeros(2*N-1) + 1j*0
    y[::2] = x.copy()
    t0 = np.arange(-(2*N-3),(2*N-3)+1)/2
    x_intp = conv1d_fft(y, np.sinc(t0))
    #x_intp = x_intp[2*N-2:-2*N+3]
    x_intp = x_intp[2*N-2-1:-2*N+3]
    return x_intp

"""
NEW UPDATES 16/03/2023 : All TESTES Aug 2024
---------------------------------------------
"""

def graph_filter_dist(X,V,d=0.1,ftype='mean',exclude_self=False,default_value=np.nan,esp=0,sort_with_dist=False,verbose=False):
    r"""Graph Filtering with (Proximity)
    
    Graph Filtering with Distance

    A value of a  vertice (node) is refined by value of neibouring vertices (nodes)

    The refinement could be - mean, media, standard deviation or any custom operation passed to function as `ftype`
    Commonly a filter operation could be permutation invariant, such as np.mean, np.median, np.std, 
    However for custom operation passed as function `ftype`, values can be passed in sorted order by turning `sort_with_dist=True`
    
    Parameters
    ----------
    X: 1d-array (n,)
      - values of vertices iof graph
    
    V: array - shape (n,d)
      - location of vertices in d-dimensional sapce
      - for 3D Geomartry, (n,3)

    d: float, 
      - distance of the vertices to be considered

    ftype: str, or Callable, default='mean'
      - filter operation, {'mean','median','std','min','max'} or any numpy operation as str = 'np.newfun'
      - using 'mean' , value of vertices is refined with average of neibours defined by distance d.
      - if `ftype` is callable, make sure it return single value given neigbours values

    exclude_self: bool, default=False,
      - if True, while refining/filtering the value of the vertice, value of itself if excluded.

    default_value: float, default=np.nan
      - default value to start with. 
      - In case of no neibours and excluding itself, vertices will have this value.

    esp: used for excluding itself

    sort_with_dist:bool, default=False
        - if True, values passed to filter are sorted in order of smallest distance to largest
        - in case of permutation variant filter, this could be used.
        - it adds extra computations, so turn it off, if not useful.

    Returns
    -------
    Y: 1d-array of same size as X
      -  refined/filtered values

    
    Notes
    -----

    See Also
    --------
    graph_filter_adj, spatial_filter_adj, spatial_filter_dist

    Examples
    --------
    #sp.graph_filter_dist
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp

    V = sp.geometry.get_ellipsoid(n1=50, n2=50, rx=1, ry=2, rz=1,)
    V += 0.01*np.random.randn(V.shape[0],V.shape[1])
    X = sp.create_signal_1d(V.shape[0],bipolar=False,sg_winlen=21,sg_polyorder=2,seed=1)
    X += 0.1*np.random.randn(X.shape[0]) 

    Y1 = sp.graph_filter_dist(X,V,d=0.01,ftype='mean',exclude_self=False)
    Y2 = sp.graph_filter_dist(X,V,d=0.2,ftype='mean',exclude_self=False)
    Y3 = sp.graph_filter_dist(X,V,d=0.5,ftype='mean',exclude_self=False)

    fig, ax = plt.subplots(1,4,subplot_kw={"projection": "3d"},figsize=(15,7))

    Ys =[X, Y1, Y2, Y3]
    TITLEs = ['raw', r'$d=0.01$',r'$d=0.1$', r'$d=0.2$']
    for i in range(4):
        ax[i].scatter3D(V[:,0], V[:,1], V[:,2], c=Ys[i], cmap='jet',s=10)
        ax[i].axis('off')
        ax[i].view_init(elev=60, azim=45, roll=15)
        ax[i].set_xlim([-1,1])
        ax[i].set_ylim([-2,2])
        ax[i].set_zlim([-1,1])
        ax[i].set_title(TITLEs[i])

    plt.subplots_adjust(hspace=0,wspace=0)
    plt.show()
    """

    assert X.shape[0]==V.shape[0]

    FILTER_ = None
    if isinstance(ftype, str):
        if ftype in ['mean','median','std','min','max']:
            FILTER_ = eval(f"np.nan{ftype}")
        else:
            try:
                FILTER_ = eval(f"np.{ftype}")
                FILTER_(np.random.randn(5))
            except:
                raise NameError("Undefined ftype, it could be one of {'mean', 'median', or numpy function, such as np.nanstd or a callable function}")
    elif callable(ftype):
        FILTER_ = ftype

    try:
        FILTER_(np.random.randn(5))
    except:
        raise NameError("Undefined ftype, it could be one of {'mean', 'median', or numpy function, such as np.nanstd or a callable function}")

    Y = np.zeros_like(X) + default_value
    for i,p in enumerate(V):
        #dist = np.sqrt(np.sum((V-p)**2,1))
        dist = np.linalg.norm(V-p,axis=1)
        idx = np.where((dist<d))[0]
        
        if sort_with_dist:
            idx = idx[np.argsort(dist[idx])]
        
        if exclude_self:
            idx = list(idx)
            if i in idx: idx.remove(i)
        
        if verbose: print('i:',i, ' idx:',idx)
        xi  = X[idx].copy()
        if len(xi) and len(xi[~np.isnan(xi)]):
            Y[i] = FILTER_(xi)
    return Y

def graph_filter_adj(X,AdjM,V=None,ftype='mean',exclude_self=False,sort_with_dist=False, default_value=np.nan):
    r"""Graph Filtering with Adjacency Matrix
    
    Graph Filtering with  Adjacency Matrix

    A value of a  vertice (node) is refined by value of connect vertices (nodes) defined in  Adjacency Matrix

    The refinement could be - mean, media, standard deviation or any custom operation passed to function as `ftype`
    Commonly a filter operation could be permutation invariant, such as np.mean, np.median, np.std, 
    However for custom operation passed as function `ftype`, values can be passed in sorted order by turning `sort_with_dist=True`  


    Parameters
    ----------
    X: 1d-array (n,)
      - values of vertices iof graph
    
    AdjM: array - shape (n,n), binary [0,1]
      - Adjacency Matrix with 0s and 1s
      - 0 mena not connected, 1 means connected

    V: array - shape (n,d), default=None
      - location of vertices in d-dimensional sapce
      - for 3D Geomartry, (n,3)
      - used if `sort_with_dist` is True

    ftype: str, or Callable, default='mean'
      - filter operation, {'mean','median','std','min','max'} or any numpy operation as str = 'np.newfun'
      - using 'mean' , value of vertices is refined with average of neibours defined by distance d.
      - if `ftype` is callable, make sure it return single value given neigbours values

    exclude_self: bool, default=False,
      - if True, while refining/filtering the value of the vertice, value of itself if excluded.

    default_value: float, default=np.nan
      - default value to start with. 
      - In case of no neibours and excluding itself, vertices will have this value.

    sort_with_dist:bool, default=False
        - if True, values passed to filter are sorted in order of smallest distance to largest
        - in case of permutation variant filter, this could be used.
        - it adds extra computations, so turn it off, if not useful.

    Returns
    -------
    Y: 1d-array of same size as X
      -  refined/filtered values
    
    
    Notes
    -----
    * This function ..

    See Also
    --------
    spkit: Home

    Examples
    --------
    #sp.graph_filter_adj
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp

    V = sp.geometry.get_ellipsoid(n1=50, n2=50, rx=1, ry=2, rz=1,)
    V += 0.01*np.random.randn(V.shape[0],V.shape[1])

    X = sp.create_signal_1d(V.shape[0],bipolar=False,sg_winlen=21,sg_polyorder=2,seed=1)
    X += 0.1*np.random.randn(X.shape[0]) 

    AdjM1, _  = sp.geometry.get_adjacency_matrix_kNN(V, K=5)
    AdjM2, _  = sp.geometry.get_adjacency_matrix_kNN(V, K=21)
    AdjM3, _  = sp.geometry.get_adjacency_matrix_kNN(V, K=31)


    Y1 = sp.graph_filter_adj(X,AdjM1,ftype='mean',exclude_self=False)
    Y2 = sp.graph_filter_adj(X,AdjM2,ftype='mean',exclude_self=False)
    Y3 = sp.graph_filter_adj(X,AdjM3,ftype='mean',exclude_self=False)

    fig, ax = plt.subplots(1,4,subplot_kw={"projection": "3d"},figsize=(15,7))

    Ys =[X, Y1, Y2, Y3]
    TITLEs = ['raw', r'$K=5$',r'$K=21$', r'$K=31$']
    for i in range(4):
        ax[i].scatter3D(V[:,0], V[:,1], V[:,2], c=Ys[i], cmap='jet',s=10)
        ax[i].axis('off')
        ax[i].view_init(elev=60, azim=45, roll=15)
        ax[i].set_xlim([-1,1])
        ax[i].set_ylim([-2,2])
        ax[i].set_zlim([-1,1])
        ax[i].set_title(TITLEs[i])

    plt.subplots_adjust(hspace=0,wspace=0)
    plt.show()

    """
    assert X.shape[0]==AdjM.shape[0]
    if V is not None:
        assert V.shape[0]==X.shape[0]



    FILTER_ = None
    if isinstance(ftype, str):
        if ftype in ['mean','median','std','min','max']:
            FILTER_ = eval(f"np.nan{ftype}")
        else:
            try:
                FILTER_ = eval(f"np.{ftype}")
                FILTER_(np.random.randn(5))
            except:
                raise NameError("Undefined ftype, it could be one of {'mean', 'median', or numpy function, such as np.nanstd or a callable function}")
    elif callable(ftype):
        FILTER_ = ftype

    try:
        FILTER_(np.random.randn(5))
    except:
        raise NameError("Undefined ftype, it could be one of {'mean', 'median', or numpy function, such as np.nanstd or a callable function}")

    Y = np.zeros_like(X) + default_value
    for i in range(AdjM.shape[0]):
        idx = list(np.where(AdjM[i])[0])
        if exclude_self:
            idx = list(set(idx) - set([i]))
        else:
            idx = list(set(idx) | set([i]))
        idx.sort()
        
        if len(idx):
            idx = np.array(idx)
            if sort_with_dist and V is not None:
                dist = np.linalg.norm(V-V[i],axis=1)
                idx = idx[np.argsort(dist[idx])]

            xi  = X[idx].copy()
            if len(xi) and len(xi[~np.isnan(xi)]):
                Y[i] = FILTER_(xi)
    return Y

def spatial_filter_dist(X,V,r=0.1,ftype='mean',exclude_self=False,sort_with_dist=False,default_value=np.nan,esp=0):
    r"""Spatial Filter with Distance (Proximity)
    
    Spatial Filter with Distance

    .. note::

        It is same as Graph Filtering, with special case as 3D


    A value of a  vertice (node) is refined by value of neibouring vertices (nodes)

    The refinement could be - mean, media, standard deviation or any custom operation passed to function as `ftype`
    Commonly a filter operation could be permutation invariant, such as np.mean, np.median, np.std, 
    However for custom operation passed as function `ftype`, values can be passed in sorted order by turning `sort_with_dist=True`
    
    Parameters
    ----------
    X: 1d-array (n,)
      - values of vertices iof graph
    
    V: array - shape (n,d)
      - location of vertices in d-dimensional sapce
      - for 3D Geomartry, (n,3)

    r: float, 
      - radius/distance of the vertices to be considered

    ftype: str, or Callable, default='mean'
      - filter operation, {'mean','median','std','min','max'} or any numpy operation as str = 'np.newfun'
      - using 'mean' , value of vertices is refined with average of neibours defined by distance d.
      - if `ftype` is callable, make sure it return single value given neigbours values

    exclude_self: bool, default=False,
      - if True, while refining/filtering the value of the vertice, value of itself if excluded.

    default_value: float, default=np.nan
      - default value to start with. 
      - In case of no neibours and excluding itself, vertices will have this value.

    esp: used for excluding itself

    sort_with_dist:bool, default=False
        - if True, values passed to filter are sorted in order of smallest distance to largest
        - in case of permutation variant filter, this could be used.
        - it adds extra computations, so turn it off, if not useful.

    Returns
    -------
    Y: 1d-array of same size as X
      -  refined/filtered values
    
    Notes
    -----

    See Also
    --------
    graph_filter_adj, spatial_filter_adj, graph_filter_dist

    Examples
    --------
    #sp.spatial_filter_dist
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp

    V = sp.geometry.get_ellipsoid(n1=50, n2=50, rx=1, ry=2, rz=1,)
    V += 0.01*np.random.randn(V.shape[0],V.shape[1])
    X = sp.create_signal_1d(V.shape[0],bipolar=False,sg_winlen=21,sg_polyorder=2,seed=1)
    X += 0.1*np.random.randn(X.shape[0]) 

    Y1 = sp.spatial_filter_dist(X,V,r=0.01,ftype='mean',exclude_self=False)
    Y2 = sp.spatial_filter_dist(X,V,r=0.2,ftype='mean',exclude_self=False)
    Y3 = sp.spatial_filter_dist(X,V,r=0.5,ftype='mean',exclude_self=False)

    fig, ax = plt.subplots(1,4,subplot_kw={"projection": "3d"},figsize=(15,7))

    Ys =[X, Y1, Y2, Y3]
    TITLEs = ['raw', r'$r=0.01$',r'$r=0.1$', r'$r=0.2$']
    for i in range(4):
        ax[i].scatter3D(V[:,0], V[:,1], V[:,2], c=Ys[i], cmap='jet',s=10)
        ax[i].axis('off')
        ax[i].view_init(elev=60, azim=45, roll=15)
        ax[i].set_xlim([-1,1])
        ax[i].set_ylim([-2,2])
        ax[i].set_zlim([-1,1])
        ax[i].set_title(TITLEs[i])

    plt.subplots_adjust(hspace=0,wspace=0)
    plt.show()
    """

    assert X.shape[0]==V.shape[0]

    FILTER_ = None
    if isinstance(ftype, str):
        if ftype in ['mean','median','std','min','max']:
            FILTER_ = eval(f"np.nan{ftype}")
        else:
            try:
                FILTER_ = eval(f"np.{ftype}")
                FILTER_(np.random.randn(5))
            except:
                raise NameError("Undefined ftype, it could be one of {'mean', 'median', or numpy function, such as np.nanstd or a callable function}")
    elif callable(ftype):
        FILTER_ = ftype

    try:
        FILTER_(np.random.randn(5))
    except:
        raise NameError("Undefined ftype, it could be one of {'mean', 'median', or numpy function, such as np.nanstd or a callable function}")

    Y = np.zeros_like(X) + default_value
    for i,p in enumerate(V):
        dist = np.linalg.norm(V-p,axis=1)
        idx = np.where((dist<r))[0]

        if sort_with_dist:
            idx = idx[np.argsort(dist[idx])]
        
        if exclude_self:
            idx = list(idx)
            if i in idx: idx.remove(i)
        #if verbose: print('i:',i, ' idx:',idx)
        xi  = X[idx].copy()
        if len(xi) and len(xi[~np.isnan(xi)]):
            Y[i] = FILTER_(xi)
    return Y

def spatial_filter_adj(X,AdjM,V=None, ftype='mean',exclude_self=False,sort_with_dist=False,default_value=np.nan):
    r"""Spatial Filter with Adjacency Matrix: AdjM

    **Spatial Filter with Adjacency Matrix (Connection Matrix) : AdjM**

        .. note::

            It is same as Graph Filtering, with special case as 3D

    A value of a  vertice (node) is refined by value of connect vertices (nodes) defined in  Adjacency Matrix

    The refinement could be - mean, media, standard deviation or any custom operation passed to function as `ftype`
    Commonly a filter operation could be permutation invariant, such as np.mean, np.median, np.std, 
    However for custom operation passed as function `ftype`, values can be passed in sorted order by turning `sort_with_dist=True`  


    Parameters
    ----------
    X: 1d-array (n,)
      - values of vertices iof graph
    
    AdjM: array - shape (n,n), binary [0,1]
      - Adjacency Matrix with 0s and 1s
      - 0 mena not connected, 1 means connected

    V: array - shape (n,d), default=None
      - location of vertices in d-dimensional sapce
      - for 3D Geomartry, (n,3)
      - used if `sort_with_dist` is True

    ftype: str, or Callable, default='mean'
      - filter operation, {'mean','median','std','min','max'} or any numpy operation as str = 'np.newfun'
      - using 'mean' , value of vertices is refined with average of neibours defined by distance d.
      - if `ftype` is callable, make sure it return single value given neigbours values

    exclude_self: bool, default=False,
      - if True, while refining/filtering the value of the vertice, value of itself if excluded.

    default_value: float, default=np.nan
      - default value to start with. 
      - In case of no neibours and excluding itself, vertices will have this value.

    sort_with_dist:bool, default=False
        - if True, values passed to filter are sorted in order of smallest distance to largest
        - in case of permutation variant filter, this could be used.
        - it adds extra computations, so turn it off, if not useful.

    Returns
    -------
    Y: 1d-array of same size as X
      -  refined/filtered values
    

    See Also
    --------
    spkit: Home

    Examples
    --------
    #sp.graph_filter_adj
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp

    V = sp.geometry.get_ellipsoid(n1=50, n2=50, rx=1, ry=2, rz=1,)
    V += 0.01*np.random.randn(V.shape[0],V.shape[1])

    X = sp.create_signal_1d(V.shape[0],bipolar=False,sg_winlen=21,sg_polyorder=2,seed=1)
    X += 0.1*np.random.randn(X.shape[0]) 

    AdjM1, _  = sp.geometry.get_adjacency_matrix_kNN(V, K=5)
    AdjM2, _  = sp.geometry.get_adjacency_matrix_kNN(V, K=21)
    AdjM3, _  = sp.geometry.get_adjacency_matrix_kNN(V, K=31)


    Y1 = sp.spatial_filter_adj(X,AdjM1,ftype='mean',exclude_self=False)
    Y2 = sp.spatial_filter_adj(X,AdjM2,ftype='mean',exclude_self=False)
    Y3 = sp.spatial_filter_adj(X,AdjM3,ftype='mean',exclude_self=False)

    fig, ax = plt.subplots(1,4,subplot_kw={"projection": "3d"},figsize=(15,7))

    Ys =[X, Y1, Y2, Y3]
    TITLEs = ['raw', r'$K=5$',r'$K=21$', r'$K=31$']
    for i in range(4):
        ax[i].scatter3D(V[:,0], V[:,1], V[:,2], c=Ys[i], cmap='jet',s=10)
        ax[i].axis('off')
        ax[i].view_init(elev=60, azim=45, roll=15)
        ax[i].set_xlim([-1,1])
        ax[i].set_ylim([-2,2])
        ax[i].set_zlim([-1,1])
        ax[i].set_title(TITLEs[i])

    plt.subplots_adjust(hspace=0,wspace=0)
    plt.show()
    """
    assert X.shape[0]==AdjM.shape[0]

    FILTER_ = None
    if isinstance(ftype, str):
        if ftype=='mean':
            FILTER_ = np.nanmean
        elif ftype=='median':
            FILTER_ = np.nanmedian
        else:
            try:
                FILTER_ = eval(f"np.{ftype}")
                FILTER_(np.random.randn(5))
            except:
                raise NameError("Undefined ftype, it could be one of {'mean', 'median', or numpy function, such as np.nanstd or a callable function}")
    elif callable(ftype):
        FILTER_ = ftype

    try:
        FILTER_(np.random.randn(5))
    except:
        raise NameError("Undefined ftype, it could be one of {'mean', 'median', or numpy function, such as np.nanstd or a callable function}")

    Y = np.zeros_like(X) + default_value
    for i in range(AdjM.shape[0]):
        idx = list(np.where(AdjM[i])[0])
        #print(idx)
        if exclude_self:
            idx = list(set(idx) - set([i]))
        else:
            idx = list(set(idx) | set([i]))
        idx.sort()
        #print(idx)
        xi  = X[idx].copy()
        if len(xi) and len(xi[~np.isnan(xi)]):
            Y[i] = FILTER_(xi)
    return Y

def add_noise(x, snr_db=10,return_noise=False):
    r"""Add Gaussian Noise to Signal

    Add Gaussian Noise to Signal

    SNR =  sig_pw/noise_pw
    SNR_db = sig_pw_db - noise_pw_db

    noise_pw = sig_pw/SNR
    noise_pw = 10**( (sig_pw_db-SNR_db)/10 )

    noise ~ N(0, sqrt(noise_pw))
    
    
    Parameters
    ----------
    x: nd array
     -  input signal, 1d or multi-dimensional

    snr_db: scalar
     - Desired Signal to Noise Ratio (SNR) in Decible (dB)
    
    return_noise: bool, default=False
     - if True, returns the added noise
    
    Returns
    -------
    y: array, same shape as x
     - Noisy signal
    
    
    Notes
    -----
    #TODO

    Examples
    --------
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    np.random.seed(1)
    t = np.linspace(0,2,200)
    x = np.cos(2*np.pi*1*t)
    SNRs = [30,20,10,5,0]
    plt.figure(figsize=(10,5))
    plt.plot(t,x,label=f'x: signal')
    for i,snr in enumerate(SNRs):
        plt.plot(t,sp.add_noise(x,snr)-(i+1)*2,label=f'SNR = {snr} dB')
    
    plt.xlim([t[0],t[-1]])
    plt.legend(bbox_to_anchor=(1,1))
    plt.grid()
    plt.yticks([])
    np.random.seed(None)
    plt.tight_layout()
    plt.show()
    """
    if not(isinstance(x, np.ndarray)):
        x = np.array(x)
    sig_pw_db = 10 * np.log10(np.nanmean(x**2))
    noise_pw_db = sig_pw_db - snr_db
    noise_pw = 10 ** (noise_pw_db / 10)

    # Generate an sample of white noise
    #noise_mean = sig.mean(0)
    noise_mean = 0
    noise = np.random.normal(noise_mean, np.sqrt(noise_pw), x.shape)
    if return_noise:
        return x + noise, noise
    return x+noise

def fill_nans_1d(x, pkind='linear'):
    r"""Fill nan values with interpolation/exterpolation for 1D
    
    **Fill nan values with interpolation/exterpolation for 1D**


    Parameters
    ----------
    x :  1d array,
      -  array with missing values as NaN 
    pkind : kind of interpolation, default='linear
      -  one of {'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'. 'zero'}
            

    Returns
    -------
    y : 1d array resulting array with interpolated values instead of nans
        same shape-size as x

    Examples
    --------
    >>> import numpy as np
    >>> import spkit as sp
    >>> x = np.array([1,1,np.nan, 3,4,9,3, np.nan])
    >>> y = fill_nans_1d(x)
    >>> y

      array([ 1.,  1.,  2.,  3.,  4.,  9.,  3., -3.])


    See Also
    --------
    fill_nans_2d

    """

    if np.sum(~np.isnan(x))>1:
        aindexes = np.arange(x.shape[0])
        agood_indexes, = np.where(np.isfinite(x))
        f = interp1d(agood_indexes
                   , x[agood_indexes]
                   , bounds_error=False
                   , copy=False
                   , fill_value="extrapolate"
                   , kind=pkind)
        return f(aindexes)
    else:
        return x

def fill_nans_2d(X,pkind='linear',filter_size=3,method='conv',clip_range=[None,None], restore_original_values=True,):
    r""" Fill nan values with interpolation/exterpolation for 2D

    **Fill nan values with interpolation/exterpolation for 2D**

    This function applies
      1. Inter/exter-polation to estimate missing values
      2. Smooting with  filter_size, if `filter_size`>0
      3. Restore original values, if `restore_original_values` True
      4. Finally, clipping values if `clip_range` is provided.


    This function uses 'fill_nans_1d' for each column and each row.
    This results two inter/exter-polated values for each missing value

    To fill the missing value, function takes average of both, which
    reduces the variability along both axis.

    Further to remove any artifacts created by new values, smoothing is applied.
    However, original values are restored.

    Finally, if clip_range is passed, values in new matrix are clipped using it.

    Parameters
    ----------
    X: 2d-array
      -  array with missing values, denoted by np.nan

    pkind : kind of interpolation, default='linear
      - one of {'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'. 'zero'}
            

    filter_size: int
      - A 2d-filter size to apply smoothing, a kernel of filter_size x filter_size
      - is created and convoled with matrix


    method: str, {'conv', 'conv_nan'}, default = 'conv'
       -  convolution method 
       - if method='conv', conventional convolution is applied, in this case,
       - 'filter_size' can be of any length>1
       - if  method='conv_nan', a convolution operation, that can handle NaN values is used.
          For this, filter_size should be an odd number, if even number is passed, 1 is added to make it odd
    
    restore_original_values: bool, default =True
       - if True, original values are restored
    
    clip_range: list of two [l0,l1]
       - After (1) inter-exter-polation, (2) applied smotthing,(3) restoring original values
       - matrix values are clipped with clip_range.
       - This is only applied if at least of the clip_range values is not None.


    Returns
    -------
    XI : New Matrix, 
       - where NaN are filled, but original values are left un-changed, except clipping
    Xk : New Matrix,
       - same as XI, except, original values are not restored, or clipped.

    Examples
    --------
    >>> import numpy as np
    >>> import spkit as sp
    >>> np.random.seed(seed=2)
    >>> X  = np.random.randint(0,10,[5,5])
    >>> r = 1*(np.abs(np.random.randn(5,5))<1.4).astype(float)
    >>> r[r==0]=np.nan
    >>> X_nan = X*r
    >>> print(X_nan)

    .. code-block:: bash
        
        array([[ 8.,  8.,  6.,  2., nan],
               [ 7.,  2., nan,  5.,  4.],
               [nan,  5.,  7.,  3.,  6.],
               [ 4.,  3.,  7.,  6.,  1.],
               [nan,  5.,  8.,  4., nan]])


    >>> X_filled, X_smooth = sp.fill_nans_2d(X_nan)
    >>> print(X_filled.round(1))

    .. code-block:: bash
        
        array([[8. , 8. , 6. , 2. , 1.9],
               [7. , 2. , 4.8, 5. , 4. ],
               [4.5, 5. , 7. , 3. , 6. ],
               [4. , 3. , 7. , 6. , 1. ],
               [3.3, 5. , 8. , 4. , 0.9]])


    >>> #####################################
    >>> #sp.fill_nans_2d
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import seaborn as sns
    >>> import scipy
    >>> import spkit as sp
    >>> seed = 2
    >>> I = (sp.create_signal_2d(n=10,seed=seed,sg_winlen=3)*10).round(0)
    >>> np.random.seed(seed)
    >>> r = np.random.rand(*I.shape)
    >>> I[r<0.05] = np.nan
    >>> np.random.seed(None)
    >>> kernel = np.ones([2,2])/4
    >>> I_spkit = sp.conv2d_nan(I.copy(), kernel, boundary='reflect',fillvalue=0)
    >>> I_fill, I_sm  = sp.fill_nans_2d(I.copy())
    >>> plt.figure(figsize=(12,4))
    >>> plt.subplot(131)
    >>> sns.heatmap(I, annot=True,cbar=False,xticklabels='', yticklabels='')
    >>> plt.title('Matrix with NaNs')
    >>> plt.subplot(132)
    >>> sns.heatmap(I_fill, annot=True,cbar=False,xticklabels='', yticklabels='')
    >>> plt.title('Filling NaNs with \n (sp.fill_nans_2d)')
    >>> plt.subplot(133)
    >>> sns.heatmap(I_spkit, annot=True,cbar=False,xticklabels='', yticklabels='')
    >>> plt.title('Smoothing with \n (sp.conv2d_nan)')
    >>> plt.tight_layout()
    >>> plt.show()
    
    See Also
    --------
    conv2d_nan, fill_nans_1d

    """
    Xi = np.array([fill_nans_1d(xi, pkind=pkind) for xi in X])
    Xj = np.array([fill_nans_1d(xi, pkind=pkind) for xi in X.T]).T
    Xk = np.nanmean(np.array([Xi,Xj]),axis=0)
    if filter_size>0:
        if method=='conv_nan':
            if filter_size%2==0: filter_size+=1
            kernel = np.ones([filter_size,filter_size])
            Xk = conv2d_nan(Xk,kernel)
        else:
            kernel = np.ones([filter_size,filter_size])/(filter_size*filter_size)
            Xk = signal.convolve2d(Xk,kernel,boundary='symm',mode='same')
    
    if restore_original_values:
        Xl = Xk*1*(np.isnan(X))
        XI = np.nansum(np.array([X,Xl]),axis=0)
    else:
        XI =  Xk
    if clip_range[0] is not None or clip_range[1] is not None:
        XI = np.clip(XI, clip_range[0],clip_range[1])
    return XI, Xk

def denorm_kernel(kernel,mode=None,keep_scale=False, esp=1e-5):
    r"""De-normalise 1d/2d Kernel
    
    **De-normalise 1d or 2d Kernel**

    Often we use normalised kernels for processing (e.g. convolution, filtering), However, in some cases,
    we like to have un-normalised kernel, such as :func:`conv1d_nan` and :func:`conv2d_nan`.

    De-normalising kernel will scale the weights of the kernel, keeping the relative differences.
     
    Parameters
    ----------
    kernel: 1d-array, 2d-array
       -  kernel, example : kernel = [1/3, 1/3, 1/3]

    Returns
    -------
    nkernel: same size as input
       -  denprmised kernel, nkernel = [1, 1, 1]

    
    Notes
    -----
    #TODO

    See Also
    --------
    conv1d_nan, conv2d_nan

    Examples
    --------
    >>> #sp.denorm_kernel
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import spkit as sp
    >>> kernel = np.ones([3,3])/9
    >>> kernel
    >>> print(kernel)
    
    .. code-block::

         array([[0.11111111, 0.11111111, 0.11111111],
               [0.11111111, 0.11111111, 0.11111111],
               [0.11111111, 0.11111111, 0.11111111]])


    >>> denorm_kernel(kernel)

    .. code-block::

        array([[1., 1., 1.],
               [1., 1., 1.],
               [1., 1., 1.]])


    """
    if not (isinstance(kernel, np.ndarray)):
        kernel = np.array(kernel)

    #assert kernel.ndim ==2 or kernel.ndim==1

    S = np.abs(np.nansum(kernel)) if keep_scale else 1.
    if S==0 or S < esp: S=1
    M = S/np.nanmax(np.abs(kernel))
    
    if kernel.ndim ==2:
        r,c = kernel.shape
    else:
        r = kernel.shape[0]
        c = r
    if mode=='mid':
        if r%2==1 and r==c:
            r1 = r//2
            m_value = kernel[r1, r1] if kernel.ndim ==2 else kernel[r1]
            #m_valu*M = 1
            if np.abs(m_value)>esp: M = S/m_value
        elif r%2==1 and c%2==1:
            r1,c1 = r//2, c//2
            m_value = kernel[r1, c1] if kernel.ndim ==2 else kernel[r1]
            if np.abs(m_value)>esp: M = S/m_value
    return kernel*np.abs(M)

def gaussian_kernel(window_length, sigma_scale=2.7,sigma=None):
    r"""Gaussian Kernel
    
    **Gaussian Kernel**
    

    Generating Gaussian kernel of given window length and sigma.

    sigma = window_length / 6

    Parameters
    ----------
    window_length: int, length of window


    sigma_scale: float, to control the width and spread of gaussian curve

    Returns
    -------

    ker: gaussian kernel of given window

    See Also
    --------
    friedrichs_mollifier_kernel: Kurt Otto Friedrichs kernel

    Examples
    --------
    #sp.gaussian_kernel
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs = sp.data.ppg_sample(sample=1)
    x = x[:int(fs*5)]
    x = x - x.mean()
    t = np.arange(len(x))/fs
    kernel1 = sp.gaussian_kernel(window_length=101,sigma_scale=10)
    kernel2 = sp.friedrichs_mollifier_kernel(window_size=101,s=1,p=1)
    kernel3 = (kernel1 - kernel2)/2
    y1 = sp.filter_with_kernel(x.copy(),kernel=kernel1)
    y2 = sp.filter_with_kernel(x.copy(),kernel=kernel2)
    y3 = sp.filter_with_kernel(x.copy(),kernel=kernel3)
    plt.figure(figsize=(12,5))
    plt.subplot(212)
    plt.plot(t,x,label='x: signal')
    plt.plot(t,y1,label='y1: kernel1')
    plt.plot(t,y2,label='y2: kernel2')
    plt.plot(t,y3,label='y3: kernel3')
    plt.xlim([t[0],t[-1]])
    plt.xlabel('time (s)')
    plt.ylabel('PPG Signal')
    plt.grid()
    plt.legend(bbox_to_anchor=(1,1))
    plt.title('filtering with kernels')
    plt.subplot(231)
    plt.plot(kernel1,label='kernel1')
    plt.plot(kernel2,label='kernel2')
    plt.plot(kernel3,label='kernel3')
    plt.title('Kernels')
    plt.grid()
    plt.tight_layout()
    plt.show()


    """
    if sigma is None: sigma = window_length / 6.
    if sigma_scale is None: sigma_scale=2.7
    t = np.linspace(-sigma_scale*sigma, sigma_scale*sigma, window_length)
    gaussian_func = lambda t, sigma: 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(t**2)/(2*sigma**2))
    ker = gaussian_func(t, sigma)
    return ker / np.sum(ker)

def friedrichs_mollifier_kernel(window_size, s=1, p=2, r=0.999):
    r"""Mollifier: Kurt Otto Friedrichs
    
    **Mollifier: Kurt Otto Friedrichs**

    Generalized function

    .. math:: 
       f(x) =  exp(-s/(1-|x|**p))    for |x|<1,   x \in [-r, r]

    Convolving with a mollifier, signals's sharp features are smoothed, while still remaining close
    to the original nonsmooth (generalized) signals.

    Intuitively, given a function which is rather irregular, by convolving it with a mollifier the function gets "mollified".

    This function is infinitely differentiable, non analytic with vanishing derivative for |x| = 1,
    can be therefore used as mollifier as described in [1]. This is a positive and symmetric mollifier.[15]

    Parameters
    ----------
    window_size: int, size of windows

    s: scaler, s>0, default=1,
     - Spread of the middle width, heigher the value of s, narrower the width

    p: scaler, p>0, default=2,
     - Order of flateness of the peak at the top,
     - p=2, smoother, p=1, triangulare type
     - Higher it is, more flat the peak.

    r: float, 0<r<1, default=0.999,
      - it is used to compute x = [-r, r]
      - recommonded to keep it r=0.999


    Returns
    -------
    ker_mol: mollifier kernel

    References
    ----------
    * [1] https://en.wikipedia.org/wiki/Mollifier
    * [2] https://en.wikipedia.org/wiki/Kurt_Otto_Friedrichs

    See also
    --------
    gaussian_kernel: Gaussian Kernel

    Examples
    --------
    #sp.friedrichs_mollifier_kernel
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs = sp.data.ppg_sample(sample=1)
    x = x[:int(fs*5)]
    x = x - x.mean()
    t = np.arange(len(x))/fs
    kernel1 = sp.gaussian_kernel(window_length=101,sigma_scale=10)
    kernel2 = sp.friedrichs_mollifier_kernel(window_size=101,s=1,p=1)
    kernel3 = (kernel1 - kernel2)/2
    y1 = sp.filter_with_kernel(x.copy(),kernel=kernel1)
    y2 = sp.filter_with_kernel(x.copy(),kernel=kernel2)
    y3 = sp.filter_with_kernel(x.copy(),kernel=kernel3)
    plt.figure(figsize=(12,5))
    plt.subplot(212)
    plt.plot(t,x,label='x: signal')
    plt.plot(t,y1,label='y1: kernel1')
    plt.plot(t,y2,label='y2: kernel2')
    plt.plot(t,y3,label='y3: kernel3')
    plt.xlim([t[0],t[-1]])
    plt.xlabel('time (s)')
    plt.ylabel('PPG Signal')
    plt.grid()
    plt.legend(bbox_to_anchor=(1,1))
    plt.title('filtering with kernels')
    plt.subplot(231)
    plt.plot(kernel1,label='kernel1')
    plt.plot(kernel2,label='kernel2')
    plt.plot(kernel3,label='kernel3')
    plt.title('Kernels')
    plt.grid()
    plt.tight_layout()
    plt.show()

    """
    if r is None: r=0.999
    if s is None: s=1
    if p is None: p=2
    # r should be between (0,1), excluding 0 and 1
    assert r<1 and r>0
    # s should be positive
    assert s>0
    # p should be positive
    assert p>0
    x = np.linspace(-r, r, window_size)
    ker_mol = np.exp(-s/(1-np.abs(x)**p))
    return ker_mol / np.sum(ker_mol)

def get_activation_time(x,fs=1,method='min_dvdt',gradient_method='fdiff',sg_window=11,sg_polyorder=3,gauss_window=0,gauss_itr=1,**kwargs):
    r"""Get Activation Time based on Gradient

    **Get Activation Time based on Gradient**

    Activation Time in cardiac electrograms refered to as time at which depolarisation of cells/tissues/heart occures.

    For biological signals (e.g. cardiac electorgram), an activation time in signal is reflected by maximum negative deflection,
    which is equal to min-dvdt, if signal is a volatge signal and function of time x = v(t)
    However, simply computing derivative of signal is sometime misleads the activation time locatio, due to noise, so derivative of
    a given signal has be computed after pre-processing



    Parameters
    ----------
    x: 1d-array,
       - input signal
    fs: int,
       - sampling frequency, default fs=1, in case only interested in loc

    method : default =  "min_dvdt"
        - Method to compute  **Activation Time**, "min_dvdt" is the used in literature, however, it depends on the kind of signal.
          For Action Potential Signal, "max_dvdt" is considered as Activation Time for Unipolar Electrogram, "min_dvdt" is consodered.
          it can be chosen one of {"max_dvdt", "min_dvdt", "max_abs_dvdt"},
        - Some literation suggests to use max_dvdt instead of min_dvdt, but mostly agree on min_dvdt

    gradient_method: default ="fdiff",
        - {"fdiff", "fgrad","sgdiff","sgdrift_diff","sgsmooth_diff", "gauss_diff"}
        - Method to compute gradient of signal
        - one of {"fdiff", "fgrad", "npdiff","sgdiff","sgdrift_diff","sgsmooth_diff", "gauss_diff"}
        - check :func:`signal_diff` for more details on the method
        - if signal is noisy try "sgsmooth_diff" or "gauss_diff"

    Parameters for gradient_method:
        - used if gradient_method in one of {"sgdiff","sgdrift_diff","sgsmooth_diff", "gauss_diff"}
        - sg_window: sgolay-filter's window length
        - sg_polyorder: sgolay-filter's polynomial order
        - gauss_window: window size of gaussian kernel for smoothing,
        - check :func:`signal_diff` for more details on the method

    Returns
    -------
    at : activation time (ms)
    loc: index
    mag: magnitude of deflection at loc
    dx : derivative of signal x

    References
    ----------
    * https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1458874/
    * https://opencarp.org/documentation/examples/02_ep_tissue/08_lats
    
    
    Notes
    -----
    #TODO


    See Also
    --------
    get_repolarisation_time, mea.activation_time_loc, mea.activation_repol_time_loc


    Examples
    --------
    #sp.get_activation_time
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x, fs = sp.data.ecg_sample(sample=1)
    x = sp.filterDC_sGolay(x,window_length=fs//2+1)
    #x = sp.filter_smooth_gauss(x,window_length=31) 
    x = x[int(0.02*fs):int(0.4*fs)]
    t = 1000*np.arange(len(x))/fs
    # It is a good idea to smooth the signal or use gradient_method = gauss_diff
    at,loc,mag,dx= sp.get_activation_time(x,fs=fs,method='min_dvdt',gradient_method='fdiff')

    plt.figure(figsize=(10,4))
    plt.plot(t,x)
    plt.axvline(at,color='r',label=f'AT = {at:0.2f} ms')
    plt.legend()
    plt.xlabel('time (ms)')
    plt.title('Activation Time (Depolarisation)')
    plt.show()



    """
    dx = signal_diff(x,method=gradient_method,sg_window=sg_window,sg_polyorder=sg_polyorder,gauss_window=gauss_window,gauss_itr=gauss_itr)

    if method=='max_dvdt':
        loc = np.argmax(dx)
    elif method=='min_dvdt':
        loc = np.argmin(dx)
    elif method=='max_abs_dvdt':
        loc = np.argmax(np.abs(dx))
    else:
        raise NameError('Unknown method name, available names of methodes are "max_dvdt", "min_dvdt", "max_abs_dvdt"')
    at = 1000*loc/fs
    mag = dx[loc]
    return at,loc,mag, dx

def get_repolarisation_time(x,fs,at_loc,t_range=[0.5,20],method='min_dvdt',gradient_method='fdiff',sg_window=11,sg_polyorder=3,gauss_window=0,gauss_itr=1,verbose=False,**kwargs):
    r"""Get Repolarisation Time based on Gradient
    
    **Get Repolarisation Time based on Gradient**

    In contrast to 'Activation Time' in cardiac electrograms, Repolarisation Time, also refered as Recovery Time,
    indicates a time at which repolarisation of cells/tissues/heart occures.

    Repolarisation time in signal is again a reflected by maximum deflection (mostly negative), after activation occures.
    That is equal to min-dvdt, after activation time, if signal is a volatge signal and function of time x = v(t)

    However, Repolarisation Time is very hard to detect reliably, due to very small electrogram, which is mostly lost in noise.



    Parameters
    ----------
    x: 1d-array,
       - input signal
    fs: int,
       - sampling frequency

    at_loc: int,  
       - location (index) of activation time, this is used to avoid any deflections before at_loc

    t_range: list of [t0 (ms),t1 (ms)]
        - range of time to restrict the search of repolarisation time.
        -  Search start from (1000*at_loc/fs + t0) to (1000*at_loc/fs + t1) ms of given signal
        
        - during t0 ms to t1 ms
           - if t_range=[None,None], whole input signal after at_loc is considered for search
           - if t_range=[t0,None], excluding signal before 1000*at_loc/fs + t0 ms
           - if t_range=[None,t1], limiting search to 1000*at_loc/fs to t1 ms


        .. note::
            NOTE: It is recommonded to add a small time gap after at_loc,  t_range = [0.5, None] or [1, None]
                As next max deflection is almost always very near (within a few sample) to at_loc, 
                which will leads to rt_loc very close to at_loc

    method : default =  "min_dvdt"
        - Method to compute  **Repolarisation Time**, "min_dvdt" is the used in literature, however, it depends on the kind of signal.
        - Some literation suggests to use max_dvdt instead of min_dvdt, but mostly agree on min_dvdt.
        - There is Wyatt method and Alternative method, suggest diffrent location points as repolarisation time.
        - It can be chosen one of {"max_dvdt", "min_dvdt", "max_abs_dvdt"},


    gradient_method: default ="fdiff", {"fdiff", "fgrad","sgdiff","sgdrift_diff","sgsmooth_diff", "gauss_diff"}
        - Method to compute gradient of signal
        - One of {"fdiff", "fgrad", "npdiff","sgdiff","sgdrift_diff","sgsmooth_diff", "gauss_diff"}
        - Check :func:`signal_diff` for more details on the method
        - If signal is noisy try "sgsmooth_diff" or "gauss_diff"

    Parameters for gradient_method:
        - used if gradient_method in one of ("sgdiff","sgdrift_diff","sgsmooth_diff", "gauss_diff")
        - sg_window: sgolay-filter's window length
        - sg_polyorder: sgolay-filter's polynomial order
        - gauss_window: window size of gaussian kernel for smoothing,
        - check :func:`signal_diff` for more details on the method


    Returns
    -------
    rt : activation time (ms)
    loc: index
    mag: magnitude of deflection
    dx : derivative of signal x

    References
    ----------
    * https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10119409/
    * https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4315451/
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    get_activation_time, mea.activation_time_loc, mea.activation_repol_time_loc


    Examples
    --------
    #sp.get_repolarisation_time
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x, fs = sp.data.ecg_sample(sample=1)
    x = sp.filterDC_sGolay(x,window_length=fs//2+1)
    x = sp.filter_smooth_gauss(x,window_length=31) 
    x = x[int(0.02*fs):int(0.4*fs)]
    t = 1000*np.arange(len(x))/fs
    at,loc,mag,dx= sp.get_activation_time(x,fs=fs,method='min_dvdt',gradient_method='fdiff')
    rt,loc,mag, dx = sp.get_repolarisation_time(x,fs,at_loc=loc,t_range=[50,None],method='min_dvdt',gradient_method='fdiff')

    plt.figure(figsize=(10,4))
    plt.plot(t,x)
    plt.axvline(at,color='r',label=f'AT = {at:0.2f} ms')
    plt.axvline(rt,color='g',label=f'RT = {rt:0.2f} ms')
    plt.legend()
    plt.xlabel('time (ms)')
    plt.title('Activation Time and Repolarisation Time')
    plt.show()
    """


    dx = signal_diff(x,method=gradient_method,sg_window=sg_window,sg_polyorder=sg_polyorder,gauss_window=gauss_window,gauss_itr=gauss_itr)

    t0 = at_loc + int(fs*t_range[0]/1000) if t_range[0] is not None else at_loc
    t1 = at_loc + int(fs*t_range[1]/1000) if t_range[1] is not None else -1

    if verbose: print(t0,t1)
    dxi = dx[t0:t1]

    if method=='max_dvdt':
        loc = np.argmax(dxi)
    elif method=='min_dvdt':
        loc = np.argmin(dxi)
    elif method=='max_abs_dvdt':
        loc = np.argmax(np.abs(dxi))
    else:
        raise NameError('Unknown method name, available names of methodes are "max_dvdt", "min_dvdt", "max_abs_dvdt"')
    loc = loc + t0
    rt = 1000*loc/fs
    mag = dx[loc]
    return rt,loc,mag, dx

def phase_map(X, add_sig=False):
    r"""Phase Mapping of multi channel signals X


    Phase computation

    .. math::
       \theta = tan^{-1}(A_y/A_x)

    # Analytical Signal using Hilbert Transform

    .. math::
        A_x + j*A_y = HT(x)

    if add_sig True

    Parameters
    ----------
    X : (n,ch) shape= (samples, n_channels)
    add_sig: bool, False,
       - if True: :math:`A_x + j*A_y = x + j*HT(x)`
       - else: :math:`A_x + j*A_y = HT(x)`

    Returns
    -------
    PM : (n,ch) shape= (samples, n_channels)

    References
    ----------
    * wikipedia - https://en.wikipedia.org/wiki/Analytic_signal
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    clean_phase, dominent_freq, phase_map_reconstruction, dominent_freq_win, amplitude_equalizer

    Examples
    --------
    >>> #sp.phase_map
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import spkit as sp
    >>> x,fs = sp.data.optical_sample(sample=1)
    >>> x = x[:int(1.5*fs)]
    >>> x = sp.filterDC_sGolay(x,window_length=fs//2+1)
    >>> t = np.arange(len(x))/fs
    >>> xp = sp.phase_map(x,add_sig=False)
    >>> plt.figure(figsize=(10,3))
    >>> plt.subplot(211)
    >>> plt.plot(t,x,label=f'$x(t)$')
    >>> plt.grid()
    >>> plt.legend(bbox_to_anchor=(1,1))
    >>> plt.xlim([t[0],t[-1]])
    >>> plt.xticks(fontsize=0)
    >>> plt.title('Phase mapping of a signal')
    >>> plt.subplot(212)
    >>> plt.plot(t,xp,color='C1',label=r'$\phi (t)$')
    >>> plt.xlim([t[0],t[-1]])
    >>> plt.ylim([-np.pi,np.pi])
    >>> plt.yticks([-np.pi,0,np.pi],[r'$-\pi$','0',r'$\pi$'])
    >>> plt.ylabel('')
    >>> plt.xlabel('time (s)')
    >>> plt.grid()
    >>> plt.legend(bbox_to_anchor=(1,1))
    >>> plt.subplots_adjust(hspace=0)
    >>> plt.tight_layout()
    >>> plt.show()

    """
    P = []
    if X.ndim==1: X = X[:,None]
    for i in range(X.shape[1]):
        xi = X[:,i].copy()
        if add_sig:
            Ax = xi + 1j*signal.hilbert(xi)
        else:
            Ax = signal.hilbert(xi)
        Px = np.arctan2(Ax.imag,Ax.real)
        P.append(Px)
    P = np.array(P).T
    return np.squeeze(P)

def clean_phase(xp,w=1,thr=-np.pi/2, low=-np.pi, high=np.pi):
    r"""Cleaning Phase

    Cleaning phase to capture dominent phase information and removing small flatuations.

    Parameters
    ----------
    xp: array, [-pi, pi]
       - raw instanious phase from signal
    w: float [0,1]
       - weight for combining original + new cleaned pahse
       - xpc  = (1-w) * xp + w * xp_new
    
    thr: float, default=-np.pi/2
       - threshold for detecting phase boundaries.
    (low,high): -np.pi, np.pi
       - lower and upper value of phase value
    

    Returns
    -------
    xpr :  cleaned phase

    References
    ----------
    * wikipedia - 
    

    Notes
    -----
    #TODO

    See Also
    --------
    phase_map, dominent_freq, phase_map_reconstruction, dominent_freq_win, amplitude_equalizer

    Examples
    --------
    #sp.clean_phase
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs = sp.data.optical_sample(sample=1)
    x = x[:int(1.5*fs)]
    x = sp.filterDC_sGolay(x,window_length=fs//2+1)
    t = np.arange(len(x))/fs
    xp = sp.phase_map(x,add_sig=False)
    xpc = sp.clean_phase(xp,w=1,thr=-np.pi/2, low=-np.pi, high=np.pi)
    plt.figure(figsize=(10,3))
    plt.subplot(211)
    plt.plot(t,x,label=f'$x(t)$')
    plt.grid()
    plt.legend(bbox_to_anchor=(1,1))
    plt.xlim([t[0],t[-1]])
    plt.xticks(fontsize=0)
    plt.title('Cleaning Phase of a signal')
    plt.subplot(212)
    plt.plot(t,xp,color='C1',label=r'$\phi (t)$')
    plt.plot(t,xpc,color='C2',label=r'$\phi_c(t)$')
    plt.xlim([t[0],t[-1]])
    plt.ylim([-np.pi,np.pi])
    plt.yticks([-np.pi,0,np.pi],[r'$-\pi$','0',r'$\pi$'])
    plt.ylabel('')
    plt.xlabel('time (s)')
    plt.grid()
    plt.legend(bbox_to_anchor=(1,1))
    plt.subplots_adjust(hspace=0)
    plt.tight_layout()
    plt.show()
    """
    assert w>=0 and w<=1
    xpr = np.zeros_like(xp) + xp
    idx = np.where(np.gradient(xp)<thr)[0]
    t0, t1 = idx[0], idx[-1]
    if t0>2:
        yi = xp[:t0+1]
        xi = np.arange(len(yi))
        bi1, bi0 = np.polyfit(xi,yi,deg=1)
        xpr[:t0+1]  = bi0+bi1*xi
    if len(xp)-t1>2:
        yi = xp[t1-1:]
        xi = np.arange(len(yi))
        bi1, bi0 = np.polyfit(xi,yi,deg=1)
        xpr[t1-1:]  = bi0+bi1*xi
    for i in range(len(idx)-1):
        n = idx[i+1]-idx[i]
        xpr[idx[i]:idx[i+1]] = np.linspace(low,high,n)
    xpr = (1-w)*xp + w*xpr
    xpr = np.clip(xpr,low, high)
    return xpr

def amplitude_equalizer(x,add_sig=False,amp_mult=0, amp_shift=1,cleaning_phase=True, w=1,mollier=True, window_length=51, iterations=1,s=None,p=None,r=None,return_all=False):
    r"""Equalizing the Amplitude variation to enhance phase and frequency of a signal

    One of the reason phase map is prefered over voltage map, because voltage
    changes over the time, even of there is a periodic cycles. To mitigate this issue
    :func:`amplitude_equalizer` impliments a method to recover original signal back from
    phase map, after nuetralizing amplitude effect.


    .. math::

        X_a (t) &=  W_m \times X_{0a}(t) + W_s

        X_e(t) &=  X_a (t) \times exp^{-j\phi_c(t)}

    where:
        
        * :math:`X_a` is New Amplitude of singal
        * :math:`W_m` is amplitude multiplication weight `amp_mult`
        * :math:`W_s` is amplitude shifting factor `amp_shift`

    if `amp_mult` = 0, then amplitude of input signal is completely ignored.

    Parameters
    ----------
    x: 1d-array
      - input signal
    
    add_sig: bool,
      -  if True, signal is added to compute phase
      -  check :func:`phase_map`  for details

    amp_mult: scalar [0,1]
       - multiplication factor for original amplitude
    amp_shift: scalar
         - shifting factor for amplitude
         - new_amplitude = (amp_mult*old_amplitude + amp_shift)
    
    cleaning_phase: bool, default=True
        - if True, phase is cleaned using :func:`clean_phase`
    w: float
       - used for phase cleaning,
       - if w<=0, then no phase cleaning is applied
    
    Mollifier parameters:
       - mollier: bool,default=True,
            -  If True, recostructed signal is smooth out using mollier
            - mollier is good to removed suddon peaks and artifacts.
            - check :func:`filter_smooth_mollifier` for details
       - if True, following parameters are used
            * window_length:int, deatult=51,
            * iterations: deault=1,
            * (s,p,r): default None

    return_all: bool, default=False
       - if True, all the phase maps are returned
       - check Returns section

     
    Returns
    -------
    xe: 1d-array,
      -  reconstructed signal, same size as x

    (xp,xp_clean,xep): if return_all=True
       - xp: phase map
       - xp_clean: cleaned phase map
       - xep - phase of reconstructed signal

    References
    ----------
    * wikipedia - 
    
    
    Notes
    -----
    NOTE: Under testing and development


    See Also
    --------
    clean_phase, phase_map, dominent_freq, phase_map_reconstruction, dominent_freq_win

    Examples
    --------
    #sp.amplitude_equalizer
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs = sp.data.optical_sample(sample=1)
    x = x[int(0*fs):int(2*fs)]
    x = sp.filterDC_sGolay(x,window_length=fs//2+1)
    t = np.arange(len(x))/fs
    xe = sp.amplitude_equalizer(x)
    plt.figure(figsize=(10,5))
    plt.subplot(311)
    plt.plot(t,x)
    plt.ylabel(r'$x(t)$')
    plt.xlim([t[0],t[-1]])
    plt.grid()
    plt.title('Amplitude Equalization')
    plt.subplot(312)
    plt.plot(t,sp.phase_map(x))
    plt.xlim([t[0],t[-1]])
    plt.ylabel(r'$\phi(t)$')
    plt.grid()
    plt.subplot(313)
    plt.plot(t,xe)
    plt.xlim([t[0],t[-1]])
    plt.ylabel(r'$x_e(t)$')
    plt.xlabel('time (s)')
    plt.grid()
    plt.subplots_adjust(hspace=0)
    plt.tight_layout()
    plt.show()
    """
    
    assert x.ndim==1

    #xp = phase_map(x, add_sig=add_sig)
    xa = signal.hilbert(x)
    if add_sig: xa = x + 1j*xa
    xp = np.arctan2(xa.imag,xa.real)

    xp_clean = xp.copy()

    if cleaning_phase and w>0:
        xp_clean = clean_phase(xp_clean,w=w,thr=-np.pi/2, low=-np.pi, high=np.pi)

    xm = np.abs(xa)

    xe = (xm*amp_mult+amp_shift)*np.exp(1j*xp_clean).real
    if mollier:
        xe = filter_smooth_mollifier(xe, window_length=window_length,iterations=iterations,r=r,s=s,p=p,mode='same')

    if return_all:
        #xra = signal.hilbert(xr)
        #xpr = np.arctan2(xra.imag,xra.real)
        xep = phase_map(xe, add_sig=add_sig)
        return xe,(xp,xp_clean,xep)
    return xe

def phase_map_reconstruction(X,add_sig=False,amp_shift=1,amp_mult=0,cleaning_phase=True, w=1,mollier=True, window_length=51, iterations=1,verbose=False):
    r"""Phase Mapping and Amplitude Equalization

    Phase Mapping of multi channel signals X along with reconstruction of signal by amplitude substraction

        .. note:: for more details
            
                Check :func:`amplitude_equalizer`


    
    Parameters
    ----------
    X : (n,ch) 
      - multichannel signal, shape= (samples, n_channels)
       
    add_sig: bool,
      -  if True, signal is added to compute phase
      -  check :func:`phase_map`  for details

    amp_mult: scalar [0,1]
       - multiplication factor for original amplitude
    amp_shift: scalar
         - shifting factor for amplitude
         - new_amplitude = (amp_mult*old_amplitude + amp_shift)
    
    cleaning_phase: bool, default=True
        - if True, phase is cleaned using :func:`clean_phase`
    w: float
       - used for phase cleaning,
       - if w<=0, then no phase cleaning is applied
    
    Mollifier parameters:
       - mollier: bool,default=True,
            -  If True, recostructed signal is smooth out using mollier
            - mollier is good to removed suddon peaks and artifacts.
            - check :func:`filter_smooth_mollifier` for details
       - if True, following parameters are used
            * window_length:int, deatult=51,
            * iterations: deault=1,
            * (s,p,r): default None

    Returns
    -------

    XP : (n,ch) 
       - Instantenious Phase, shape=(samples, n_channels), 
    XE : (n,ch)
       - Reconstructed Equalized Signal shape=(samples, n_channels) 


    Notes
    -----
    NOTE: Under testing and development

    See Also
    --------
    clean_phase, phase_map, dominent_freq, dominent_freq_win, amplitude_equalizer

    Examples
    --------
    #sp.phase_map_reconstruction
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x,fs = sp.data.optical_sample(sample=1)
    x = x[int(0*fs):int(2*fs)]
    x = sp.filterDC_sGolay(x,window_length=fs//2+1)
    t = np.arange(len(x))/fs
    xp, xe = sp.phase_map_reconstruction(x)
    plt.figure(figsize=(10,5))
    plt.subplot(311)
    plt.plot(t,x)
    plt.ylabel(r'$x(t)$')
    plt.xlim([t[0],t[-1]])
    plt.grid()
    plt.title('Amplitude Equalization')
    plt.subplot(312)
    plt.plot(t,xp)
    plt.xlim([t[0],t[-1]])
    plt.ylabel(r'$\phi(t)$')
    plt.grid()
    plt.subplot(313)
    plt.plot(t,xe)
    plt.xlim([t[0],t[-1]])
    plt.ylabel(r'$x_e(t)$')
    plt.xlabel('time (s)')
    plt.grid()
    plt.subplots_adjust(hspace=0)
    plt.tight_layout()
plt.show()
    """
    XE, XP = [],[]
    if X.ndim==1: X = X[:,None]
    for i in range(X.shape[1]):
        if verbose: ProgBar_JL(i,X.shape[1])
        xi = X[:,i].copy()

        xe,(xp,xp_clean,xep) = amplitude_equalizer(xi,add_sig=add_sig,amp_shift=amp_shift,amp_mult=amp_mult,return_all=True,
                               cleaning_phase=cleaning_phase, w=w,mollier=mollier, window_length=window_length, iterations=iterations)
        XE.append(xe)
        XP.append(xp)
    XP = np.array(XP).T
    XE = np.array(XE).T
    XP = np.squeeze(XP)
    XE = np.squeeze(XE)
    return XP,XE

def _dominent_freq(x,fs,method='welch',window='hann',exclude_lower_fr=None,refine_peak=False,nfft=None,nperseg=None,return_spectrum=False,**kwargs):
    r"""
    Dominent Frequency Analysis
    """
    if isinstance(method,str) and method=='welch':
        fq, Px = signal.welch(x,fs,nperseg=nperseg,nfft=nfft,window=window,**kwargs)
    
    elif isinstance(method,str) and method.lower()=='fft':
        #dft_analysis(x, window='blackmanharris', N=None,scaling_dB=True,normalize_win=True, plot=False, fs=None)
        Px,_, N = dft_analysis(x,window=window,scaling_dB=False)
        fq = fs*np.arange(len(Px))/N
    else:
        fq, Px = signal.periodogram(x,fs,nfft=nfft,window=window,**kwargs)
    
    if exclude_lower_fr is not None:
        Px = Px[np.where(fq>exclude_lower_fr)]
        fq = fq[np.where(fq>exclude_lower_fr)]

    loc = np.argmax(Px)
    if refine_peak:
        iploc,ipmag,ipphase = peak_interp(mX = 20*np.log10(Px), pX=fq, ploc=np.array([loc]))
        iloc =  iploc[0]
        dfq= ipphase[0]
    else:
        dfq = fq[loc]
    
    if return_spectrum:
        return dfq, (Px,fq)
    return dfq

def dominent_freq(X,fs,method='welch',window='hann',exclude_lower_fr=None,refine_peak=False,nfft=None,nperseg=None,return_spectrum=False,**kwargs):
    r"""Dominent Frequency Analysis
    
    Dominent Frequency Analysis is one of the approach used to discern the different physiology from
    a biomedical signals [1]_.

    This function compute dominent frequency as the maximum peak in the spectrum. The location of the peak is returned in Hz

    The value of dominent frquency depends on the computing method for spectrum. Using Welch method is prefered as it computes
    Other posible approaches are 'FFT' and 'Periodogram'

    Computed peak can further be refined using parobolic interpolation.

    Parameters
    ----------
    X: 1d-array 2d-array
      - input signal, for 2d, channel axis is 1, e.g., (n,ch)
      - single channel or multichannel signal
    
    fs: int, 
      - sampling frequency
    
    method: str, {'welch','fft',None}, deafult='welch'
      - method to compute spectrum
      - welch method is prefered, (see example), uses scipy.signal.welch
      - if None or other than 'welch' and 'fft', periodogram is used to compute spectum ( scipy.signal.peridogram)
      - if 'fft' or 'FFT', :func:`dft_analysis` is used.

    window: str, default='hann'
      - windowing function

    exclude_lower_fr: None or scalar
      - if not None, any peak before exclude_lower_fr is excluded
      - useful to avoid dc component or any known low-frequency component
      - example: exclude_lower_fr=2, will exclude all the peaks before 2Hz
    
    refine_peak: bool, default=False
      - if True, peak is refined using parobolic interpolation
    
    return_spectrum: bool, default=False
      - if True, Magnitude spectrum and frequency values are returned along with dominent frequency
      - useful to analyse

    (nfft,nperseg): parameters for 'welch' and 'periodogram' method

    **kwargs: 
       - any aaditional keywords to pass to scipy.signal.welch or scipy.signal.periodogram
          method.
    
    Returns
    -------
    DF: scalar or 1d-array
      - scalar of X is 1d
      - array of scalers of length ch, if X is 2d
    
    (mX, Frq) :  Spectrum, if return_spectrum is True

    References
    ----------

    .. [1] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5673341/
    
    
    Notes
    -----
    Refining peak achive better localization of dominence in spectrum. See Example.


    See Also
    --------
    clean_phase, phase_map, phase_map_reconstruction, dominent_freq_win, amplitude_equalizer

    Examples
    --------
    #sp.dominent_freq
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    import scipy
    x,fs = sp.data.optical_sample(sample=1)
    x = x[:int(5*fs)]
    x = sp.filterDC_sGolay(x,window_length=fs//2+1)
    t = np.arange(len(x))/fs
    dfq = sp.dominent_freq(x,fs,method='welch',refine_peak=True)
    print('Dominent Frequency: ',dfq,'Hz')
    dfq1, (mX1, frq1) = sp.dominent_freq(x,fs,method='welch',nfft=512,return_spectrum=True)
    dfq2, (mX2, frq2) = sp.dominent_freq(x,fs,method='welch',nfft=2048,return_spectrum=True)
    dfq3, (mX3, frq3) = sp.dominent_freq(x,fs,method='welch',nfft=512,return_spectrum=True,refine_peak=True)
    dfq4, (mX4, frq4) = sp.dominent_freq(x,fs,method='welch',nfft=2048,return_spectrum=True,refine_peak=True)

    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.plot(frq1, 20*np.log10(mX1), label='nfft=512')
    plt.plot(frq2, 20*np.log10(mX2), label='nfft=2048')
    plt.axvline(dfq1,color='C0',ls='--',lw=1,label=f'df={dfq1:.3f} Hz')
    plt.axvline(dfq2,color='C1',ls='--',lw=1,label=f'df={dfq2:.3f} Hz')
    plt.xlim([0,20])
    plt.ylim([-100, None])
    plt.grid(alpha=0.1)
    plt.ylabel('Spectra (dB)')
    plt.xlabel('Frquency (Hz)')
    plt.title('Dominent Frq.')
    plt.legend()
    plt.subplot(122)
    plt.plot(frq3, 20*np.log10(mX3), label='nfft=512')
    plt.plot(frq4, 20*np.log10(mX4), label='nfft=2048')
    plt.axvline(dfq3,color='C0',ls='--',lw=1,label=f'df={dfq3:.3f} Hz')
    plt.axvline(dfq4,color='C1',ls='--',lw=1,label=f'df={dfq4:.3f} Hz')
    plt.xlim([0,20])
    plt.ylim([-100, None])
    plt.grid(alpha=0.1)
    plt.xlabel('Frquency (Hz)')
    plt.title('Dominent Frq. with refinement')
    plt.legend()
    plt.show()
    """
    DF = []
    if X.ndim>1:
        if return_spectrum: PXF = []
        for i in range(X.shape[1]):
            xi = X[:,i].copy()
            if return_spectrum:
                    
                dfi,PXFi =  _dominent_freq(xi,fs=fs,method=method,window=window,exclude_lower_fr=exclude_lower_fr,
                                refine_peak=refine_peak,
                                nfft=nfft,nperseg=nperseg,return_spectrum=True,**kwargs)
                PXF.append(PXFi)
            else:
                
                dfi =  _dominent_freq(xi,fs=fs,method=method,window=window,exclude_lower_fr=exclude_lower_fr,
                        refine_peak=refine_peak,
                        nfft=nfft,nperseg=nperseg,return_spectrum=False,**kwargs)
            
            DF.append(dfi)
    else:
        if return_spectrum:
            
            DF, PXF =  _dominent_freq(X,fs=fs,method=method,window=window,exclude_lower_fr=exclude_lower_fr,
                            refine_peak=refine_peak, nfft=nfft,nperseg=nperseg,return_spectrum=True,**kwargs)

        else:
            DF =  _dominent_freq(X,fs=fs,method=method,window=window,exclude_lower_fr=exclude_lower_fr,
                            refine_peak=refine_peak,
                            nfft=nfft,nperseg=nperseg,return_spectrum=False,**kwargs)
    
    if return_spectrum:
        return DF, PXF
    return DF

def dominent_freq_win(X,fs,win_len=100,overlap=None,method='welch',refine_peak=False,nfft=None,nperseg=None,exclude_lower_fr=None,window='hann',use_joblib=False,verbose=1,**kwargs):
    r"""Dominent Frequency Analysis Window-wise
    

    This function computes dominent frequency in moving window, to analyse the dynamics
    dominent frequency.


    Parameters
    ----------

    X: 1d-array 2d-array
      - input signal, for 2d, channel axis is 1, e.g., (n,ch)
      - single channel or multichannel signal
    
    fs: int, 
      - sampling frequency

    win_len: int, deault=100
       -  length of window
    overlap: int, or None
       - if None, overlap=win_len//2
    
    method: str, {'welch','fft',None}, deafult='welch'
      - method to compute spectrum
      - welch method is prefered, (see example), uses scipy.signal.welch
      - if None or other than 'welch' and 'fft', periodogram is used to compute spectum ( scipy.signal.peridogram)
      - if 'fft' or 'FFT', :func:`dft_analysis` is used.

    window: str, default='hann'
      - windowing function

    exclude_lower_fr: None or scalar
      - if not None, any peak before exclude_lower_fr is excluded
      - useful to avoid dc component or any known low-frequency component
      - example: exclude_lower_fr=2, will exclude all the peaks before 2Hz
    
    refine_peak: bool, default=False
      - if True, peak is refined using parobolic interpolation
    
    return_spectrum: bool, default=False
      - if True, Magnitude spectrum and frequency values are returned along with dominent frequency
      - useful to analyse

    (nfft,nperseg): parameters for 'welch' and 'periodogram' method

    **kwargs: 
       - any aaditional keywords to pass to scipy.signal.welch or scipy.signal.periodogram
          method.
    
    verbose: int
       - verbosity level, 0: silence

    Returns
    -------

    DF_win: array
      -  Dominent Frequencies of each window, size of (nw, ch), 

    References
    ----------
    * **[1]** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5673341/
    
    
    Notes
    -----
    For details, chekc :func:`dominent_freq`


    See Also
    --------
    clean_phase, phase_map, dominent_freq, phase_map_reconstruction, amplitude_equalizer

    Examples
    --------
    #sp.dominent_freq_win
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    fs=500
    t = np.linspace(0,3,7*fs)
    f=3
    x = np.sin(2*np.pi*f*t) + np.sin(2*np.pi*2*f*t**2)
    x = sp.add_noise(x,snr_db=20)
    df_win = sp.dominent_freq_win(x,fs,win_len=fs//2,refine_peak=True,verbose=0)
    tx = t[-1]*np.arange(len(df_win))/(len(df_win)-1)
    plt.figure(figsize=(10,4))
    plt.subplot(211)
    plt.plot(t,x)
    plt.ylabel('x')
    plt.xlim([t[0],t[-1]])
    plt.grid()
    plt.title('Dominent Frequency - temporal variation')
    plt.subplot(212)
    plt.plot(tx,df_win,marker='.')
    plt.xlim([t[0],t[-1]])
    plt.ylabel('DF (Hz)')
    plt.xlabel('time (s)')
    plt.grid()
    plt.subplots_adjust(hspace=0)
    plt.tight_layout()
    plt.show()
    """

    if overlap is None: overlap=win_len//2
    overlap_shift = overlap
    DF_win =[]

    verbose_win=False
    if X.ndim==1:
        X = X[:,None]
        verbose_win=True if verbose else False
        verbose=False
    for i in range(X.shape[1]):
        if verbose: ProgBar_JL(i,X.shape[1],style=2,selfTerminate=False)
        xi = X[:,i].copy()
        if use_joblib:
            win = []
            win_i = np.arange(win_len)
            while win_i[-1]<xi.shape[0]:
                win.append(win_i.copy())
                win_i+=overlap_shift

            dfq_temp = Parallel(n_jobs=-1)( delayed(_dominent_freq) (xi[win[kk]],fs,method=method,refine_peak=refine_peak,nfft=nfft,nperseg=nperseg,return_spectrum=False,
                                                                 exclude_lower_fr=exclude_lower_fr,window=window,**kwargs) for kk in range(len(win)))
        else:
            win = np.arange(win_len)
            dfq_temp =[]
            while win[-1]<xi.shape[0]:
                if verbose_win: ProgBar_JL(win[-1],xi.shape[0],style=2,selfTerminate=False)
                dfq_i = _dominent_freq(xi[win],fs=fs,method=method,refine_peak=refine_peak,nfft=nfft,return_spectrum=False,
                                      nperseg=nperseg,exclude_lower_fr=exclude_lower_fr,window=window,**kwargs)
                dfq_temp.append(dfq_i)
                win+=overlap_shift

            DF_win.append(dfq_temp)
    DF_win = np.squeeze(np.array(DF_win).T)
    return DF_win


def temp_function():
    r"""

    spatial_filter_adj
    -----------------

    Adjustancy Matrix,  Connection Matrix

    Given values X corresponding to spatial locations V, whose adjacency matrix is given as AdjM,
         applying filter `mean` or `median` etc

    Parameters
    ----------

    X    : 1d-array of size m, values of points, shape=(m,), can include `NaN`, if missing
    AdjM : 2d-array of size (m,m), Adjacency matrix

    ftype: str, or callable function, default = mean
           filter type, str = {`mean`, `median` or any :term:`np.fun` or a callable function}
           All functions should be able to handle NaN values

    exclude_self: bool, default=False,
           If True, while estimating new value at location i, self value is excluded

    default_value: scalar, default= :term:`np.nan`
           If no value is calculated, deafult value is used
    
    

    spatial_filter_dist
    -----------------
    Given values X corresponding to spatial locations V, in n-Dimentional Space applying a filter 'mean' or 'median' of radius `r`
        X: 1d-array of size m, values of points, shape=(m,), can include `NaN`, if missing

    
    Parameters
    ----------
    V: 2d-array of size (m,n), locations of points
    
    r: scaler, positive
     -  Euclidean distance


    ftype: str, or callable function:, default = mean
           - filter type, str = {'mean', 'median' or any :term:'np.fun' or a callable function}
           - All functions should be able to handle 'NaN' values

    exclude_self: bool, default=False,
            - If True, while estimating new value at location i, self value is excluded

    default_value: scalar, default=:term:'np.nan'
            - If no value is calculated, deafult value is used
    
    
    Returns
    --------
    Y: Filtered values

    show compass
    -------------






    """


"""
Signal Simulation
"""

def create_signal_1d(n=100,seed=None,circular=False,bipolar=True,sg_winlen=11,sg_polyorder=1,max_dxdt=0.1, iterations=2,max_itr=10, **kwargs):
    r""" Generate 1D arbitary signal

    Simulating arbitary signal
    
    Parameters
    ----------
    n: int, default=100
      -  number of sampels

    seed: int, defualt=None, 
       - random seed, to re-generate same signal

    circular: bool, deault=False,
       - if True, then starting and ending value of the signal will be roughly same.
       - if will signal starts from 0, it will end near to 0.
       - useful for geometric operations.
    
    bipolar: bool, default=True
       -  if True, signal will vary from -1 to +1
       -  else from 0 to 1

    sg_winlen: int,deault=11
       - window length, heigher the window size, smoother the signal will be

    sg_polyorder: int, >0
       -  order of polynomial, 
       - higher the order, more curvy the signal will be
       - smaller the order, more smoother and flat-like signal will be

    max_dxdt: float, deafult=0.1
       - maximum derivative in the signal, 
       - generated signal will be iteratively smoothen, untill, maximum derivative is <=`max_dxdt`
       - or maximum number of iterations `max_itr` are achived, when (`max_dxdt`) is not None
    
    max_itr: int, deafult=10
       - maximum number of iterations to achive `max_dxdt`
       - if None, then `iterations` are used, regardless of `max_dxdt` value.

    iterations: int, deafault=2
       - if `max_itr` is None, then `iterations` many iterations are used to smoothen the signal.

    Returns
    -------
    x: 1d-array of n samples
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    create_signal_2d

    Examples
    --------
    #sp.create_signal_1d
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    seed=11
    x1 = sp.create_signal_1d(n=100,sg_winlen=5,seed=seed)
    x2 = sp.create_signal_1d(n=100,sg_winlen=11,seed=seed)
    x3 = sp.create_signal_1d(n=100,sg_winlen=31,seed=seed)
    plt.figure(figsize=(10,4))
    plt.subplot(311)
    plt.plot(x1,label='winlen=5')
    plt.legend(loc='upper right')
    plt.grid()
    plt.subplot(312)
    plt.plot(x2,label='winlen=11')
    plt.legend(loc='upper right')
    plt.grid()
    plt.subplot(313)
    plt.plot(x3,label='winlen=31')
    plt.legend(loc='upper right')
    plt.grid()
    plt.subplots_adjust(hspace=0)
    plt.show()
    ############################################
    #sp.create_signal_1d
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    seed=1
    x1 = sp.create_signal_1d(n=100,sg_winlen=11,sg_polyorder=1,seed=seed)
    x2 = sp.create_signal_1d(n=100,sg_winlen=11,sg_polyorder=5,seed=seed)
    x3 = sp.create_signal_1d(n=100,sg_winlen=11,sg_polyorder=7,seed=seed)
    plt.figure(figsize=(10,4))
    plt.subplot(311)
    plt.plot(x1,label='order=1')
    plt.legend(loc='upper right')
    plt.grid()
    plt.subplot(312)
    plt.plot(x2,label='order=5')
    plt.legend(loc='upper right')
    plt.grid()
    plt.subplot(313)
    plt.plot(x3,label='order=7')
    plt.legend(loc='upper right')
    plt.grid()
    plt.subplots_adjust(hspace=0)
    plt.show()
    """

    if 'sg_nwin' in kwargs:
        sg_winlen = kwargs['sg_nwin']

    np.random.seed(seed)
    window_length = (n//sg_winlen)
    if window_length%2==0: window_length+=1

    #print(window_length)
    x = np.random.rand(n+window_length)
    #if circular: x = np.r_[np.zeros(window_length*3)+x[0],x,np.zeros(window_length*3)+x[0]]
    if circular:
        x[:window_length]  = x[:window_length]*0 + x[0]
        x[-window_length-1:] = x[-window_length-1:]*0 + x[0]
    xm = x.copy()

    if max_dxdt is None:
        #print('max_dxdt:',np.max(np.abs(np.diff(xm))))
        for _ in range(iterations):
            _,xm = filterDC_sGolay(xm,window_length=window_length,polyorder=sg_polyorder,return_background=True)
            #print('max_dxdt:',np.max(np.abs(np.diff(xm))))
    else:
        itr=0
        while np.max(np.abs(np.diff(xm)))>max_dxdt:
            _,xm = filterDC_sGolay(xm,window_length=window_length,polyorder=sg_polyorder,return_background=True)
            itr+=1
            if itr>max_itr:break
    
    np.random.seed(None)
    
    xm = xm[window_length//2:-window_length//2]
    #if circular: xm = xm[1*window_length:-1*window_length]
    xm -= xm.min()
    xm /= xm.max()
    if bipolar: xm = 2*(xm - 0.5)
    return xm

def create_signal_2d(n=100,seed=None,sg_winlen=11,sg_polyorder=1,iterations=1,max_dxdt=0.1,max_itr=None):
    r"""Generate 2D arbitary signal/image patch
    
    Generate a 2D Grid of (n, n)


    Parameters
    ----------
    n: int, default=100
      -  number of pixels, height and width
      - (n,n) matrix grid

    seed: int, defualt=None, 
       - random seed, to re-generate same signal

    sg_winlen: int,deault=11
       - window length, heigher the window size, smoother the image will be

    sg_polyorder: int, >0
       -  order of polynomial, 
       - higher the order, more patchy the image will be
       - smaller the order, more smoother and flat-like image will be

    max_dxdt: float, deafult=0.1
       - maximum derivative in of the each row, 
       - generated row will be iteratively smoothen, untill, maximum derivative is <=`max_dxdt`
       - or maximum number of iterations `max_itr` are achived, when (`max_dxdt`) is not None
    
    max_itr: int, deafult=10
       - maximum number of iterations to achive `max_dxdt`
       - if None, then `iterations` are used, regardless of `max_dxdt` value.

    iterations: int, deafault=1
       - if `max_itr` is None, then `iterations` many iterations are used to smoothen the image.



    Returns
    -------
    I: (n,n) array



    References
    ----------
    * wikipedia - 
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    spkit: #TODO

    Examples
    --------
    #sp.create_signal_2d
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    seed=11
    I1 = sp.create_signal_2d(n=100,sg_winlen=5,seed=seed,iterations=1)
    I2 = sp.create_signal_2d(n=100,sg_winlen=11,seed=seed,iterations=1)
    I3 = sp.create_signal_2d(n=100,sg_winlen=31,seed=seed,iterations=1)
    I = np.stack([I1,I2,I3],axis=-1)
    plt.figure(figsize=(12,4))
    plt.subplot(141)
    plt.imshow(I1)
    plt.title('I1: winlen=5')
    plt.subplot(142)
    plt.imshow(I2)
    plt.title('I2: winlen=11')
    plt.subplot(143)
    plt.imshow(I3)
    plt.title('I3: winlen=31')
    plt.subplot(144)
    plt.title('I: combined: RBG')
    plt.imshow(I)
    plt.show()

    #################################
    #sp.create_signal_2d
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    seed=11
    I1 = sp.create_signal_2d(n=100,sg_polyorder=1,seed=seed,iterations=1)
    I2 = sp.create_signal_2d(n=100,sg_polyorder=3,seed=seed,iterations=1)
    I3 = sp.create_signal_2d(n=100,sg_polyorder=5,seed=seed,iterations=1)
    I = np.stack([I1,I2,I3],axis=-1)
    plt.figure(figsize=(12,4))
    plt.subplot(141)
    plt.imshow(I1)
    plt.title('order=1')
    plt.subplot(142)
    plt.imshow(I2)
    plt.title('order=3')
    plt.subplot(143)
    plt.imshow(I3)
    plt.title('order=5')
    plt.subplot(144)
    plt.title('I: combined: RBG')
    plt.imshow(I)
    plt.show()
    """

    assert (iterations is not None) or (max_itr is not None)

    np.random.seed(seed)
    X = np.random.rand(n,n)

    for i in range(n):
        if max_dxdt is None: 
            itr=0
            while np.max(np.abs(np.diff(X[i])))>max_dxdt:
                _,X[i] = filterDC_sGolay(X[i],window_length=sg_winlen,polyorder=sg_polyorder,return_background=True)
                itr+=1
                if itr>max_itr: break
            
        else: #if iterations is not None:
            for _ in range(iterations):
                #win = np.random.randint(3,window_length)
                #if win%2==0: win+=1
                _,X[i] = filterDC_sGolay(X[i],window_length=sg_winlen,polyorder=sg_polyorder,return_background=True)

    X -= X.min()
    X /=X.max()
    for i in range(n):
        if max_dxdt is None: 
            itr=0
            while np.max(np.abs(np.diff(X[:,i])))>max_dxdt:
                _,X[:,i] = filterDC_sGolay(X[:,i],window_length=sg_winlen,polyorder=sg_polyorder,return_background=True)
                itr+=1
                if itr>max_itr: break
        else: #if iterations is not None:
            for _ in range(iterations):
                #win = np.random.randint(3,window_length)
                #if win%2==0: win+=1
                _,X[:,i] = filterDC_sGolay(X[:,i],window_length=sg_winlen,polyorder=sg_polyorder,return_background=True)

    np.random.seed(None)
    X -= X.min()
    X /=X.max()
    return X


# def create_signal_1d(n=100,seed=None,sg_polyorder=1,sg_nwin=10,max_dxdt=0.1,iterations=2,max_itr=10,circular=False):
#     r""" Generate 1D arbitary signal

#     sg_wid: window_length = (n//sg_wid) for sg_filter
#     sg_polyorder: polyorder for sg_filter

#     """
#     np.random.seed(seed)
#     x = np.random.rand(n)
#     window_length = (n//sg_nwin)
#     if window_length%2==0: window_length+=1

#     if circular: x = np.r_[np.zeros(window_length*2)+x[0],x,np.zeros(window_length*2)+x[0]]

#     xm = x.copy()

#     if max_dxdt is None:
#         #print('max_dxdt:',np.max(np.abs(np.diff(xm))))
#         for _ in range(iterations):
#             _,xm = filterDC_sGolay(xm,window_length=window_length,polyorder=sg_polyorder,return_background=True)
#             #print('max_dxdt:',np.max(np.abs(np.diff(xm))))
#     else:
#         itr=0
#         while np.max(np.abs(np.diff(xm)))>max_dxdt:
#             _,xm = filterDC_sGolay(xm,window_length=window_length,polyorder=sg_polyorder,return_background=True)
#             itr+=1
#             if itr>max_itr:break
#     np.random.seed(None)
#     xm -= xm.min()
#     xm /= xm.max()
#     return xm

# def create_signal_2d(n=100,sg_winlen=11,sg_polyorder=1,iterations=2,max_dxdt=0.1,max_itr=10,seed=None):

#     r""" Generate 2D arbitary signal/image patch"""

#     np.random.seed(seed)
#     X = np.random.rand(n,n)

#     for i in range(n):
#         if iterations is not None:
#             for _ in range(iterations):
#                 #win = np.random.randint(3,window_length)
#                 #if win%2==0: win+=1
#                 _,X[i] = filterDC_sGolay(X[i],window_length=sg_winlen,polyorder=sg_polyorder,return_background=True)
#         else:
#             itr=0
#             while np.max(np.abs(np.diff(X[i])))>max_dxdt:
#                 _,X[i] = filterDC_sGolay(X[i],window_length=sg_winlen,polyorder=sg_polyorder,return_background=True)
#                 itr+=1
#                 if itr>max_itr: break

#     X -= X.min()
#     X /=X.max()
#     for i in range(n):
#         if iterations is not None:
#             for _ in range(iterations):
#                 #win = np.random.randint(3,window_length)
#                 #if win%2==0: win+=1
#                 _,X[:,i] = filterDC_sGolay(X[:,i],window_length=sg_winlen,polyorder=sg_polyorder,return_background=True)
#         else:
#             itr=0
#             while np.max(np.abs(np.diff(X[:,i])))>max_dxdt:
#                 _,X[:,i] = filterDC_sGolay(X[:,i],window_length=sg_winlen,polyorder=sg_polyorder,return_background=True)
#                 itr+=1
#                 if itr>max_itr: break

#     np.random.seed(None)
#     X -= X.min()
#     X /=X.max()
#     return X


"""
Metrices and Evaluations
------------------------
"""

def show_compass(Ax_theta,Ax_bad=[],arr_agg='mean',figsize=(10,6),all_arrow_prop =dict(facecolor='C0',lw=1,zorder=10,alpha=0.2,edgecolor='C0',width=0.05),
                                                 avg_arrow_prop =dict(facecolor='C3',lw=4,zorder=100,edgecolor='C3',width=0.045), title='CV'):
    r"""Display Compass with Arrows and aggregated Arrow

    **Display Compass with Arrows and aggregated Arrow**

    Plot two compasses:
      * (1) With all the arrows of angles as provided in `Ax_theta`, along with averaged direction
      * (2) Only if `Ax_bad` is provided;
          With arrows excluding one indicated as bad (:func:`np.nan`, `NaN`) in `Ax_bad`
          along with averaged direction
    
    Parameters
    ----------
        Ax_theta: 2D-array,
                - e.g. MEA Feature matrix, of Angles of directions
                - 1D-array should work as well.
                - It can include NaN (:func:`np.nan`) values, which will be ignored

        Ax_bad  : 2D-array,
                - Same size array as Ax_theta, with 1 and :term:`np.nan` (`NaNs`) value
                - 1 for good channel, `NaN` for bad channel
                - default = [], in which case only one plot is shown


        arr_agg: str {'mean', 'media'}, default='mean'
                how to aggregate angles

    Returns
    -------

        Ax_theta_avg   : Aggregated Angle, including all the values

        Ax_theta_avg_bd: Aggregated Angle, excluding bad channels, indicated by `NaN` in Ax_bad,
                        if Ax_bad is provided, else same as Ax_theta_avg

    References
    ----------
    * wikipedia - 
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    direction_flow_map


    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import spkit as sp
    >>> np.random.seed(2)
    >>> N=8
    >>> Ax_theta= np.arctan2(np.random.randn(N)+1, np.random.randn(N)-1)
    >>> Ax_bad   = 1*(np.random.rand(N)>0.4).astype(float)
    >>> Ax_bad[Ax_bad==0]=np.nan
    >>> sp.show_compass(Ax_theta, Ax_bad,arr_agg='mean',title='D')
    >>> plt.show()
    """

    theta_f = (Ax_theta).reshape(-1)
    Ax_theta_avg = agg_angles(theta_f, agg=arr_agg)
    Ax_theta_avg_bd = Ax_theta_avg


    if len(Ax_bad):
        theta_bd_f = (Ax_theta*Ax_bad).reshape(-1)
        Ax_theta_avg_bd = agg_angles(theta_bd_f, agg=arr_agg)

    if len(Ax_bad):
        fig, ax = plt.subplots(1,2,figsize=figsize,subplot_kw={'projection': 'polar'})
    else:
        fig, ax = plt.subplots(1,figsize=figsize,subplot_kw={'projection': 'polar'})
        ax = [ax]

    for theta in theta_f:
        if not(np.isnan(theta)):
            ax[0].arrow(theta,0,0,1,length_includes_head=True,**all_arrow_prop)
    arr = ax[0].arrow(Ax_theta_avg,0,0,1,length_includes_head=True,**avg_arrow_prop)


    ax[0].add_patch(arr)
    plt.ylim([0,1])
    plt.ylim([0,1.1])

    ax[0].set_title(f'{title}: Compass plot, including all')

    if len(Ax_bad):
        for theta in theta_bd_f:
            if not(np.isnan(theta)):
                ax[1].arrow(theta,0,0,1,length_includes_head=True,**all_arrow_prop)

        arr = ax[1].arrow(Ax_theta_avg_bd,0,0,1,length_includes_head=True,**avg_arrow_prop)

        ax[1].add_patch(arr)

        plt.ylim([0,1.1])
        ax[1].set_title(f'{title}: Compass plot, excluding bad channels')

    plt.tight_layout()
    plt.show()

    return Ax_theta_avg, Ax_theta_avg_bd

def direction_flow_map(X_theta,X=None,upsample=1,square=True,cbar=False,arr_pivot='mid',
                       stream_plot=True,title='',heatmap_prop =dict(),
                       arr_prop =dict(color='w',scale=None),figsize=(12,8),fig=None,ax=[],show=True,
                       stream_prop =dict(density=1,color='k'),
                       **kwargs):

    r"""Displaying Directional flow map of a 2D grid

    **Displaying Directional flow map of a 2D grid**

    Displaying Directional Flow of a 2D grid, as spatial flow pattern.

    This plot two figures:
        (1) A 2D grid with arrows with a backgound of a heatmap
        (2) Streamline plot , if stream_plot=True


    Parameters
    ----------
    X_theta : 2D-array of Angles in Radian
              2D Grid of Angles of direction at a point in a grid as spatial location,
              it can include NaN for missing information, or noisy location

    X : 2D-array of float, as size as X_theta, optional (default X=None)
        These values are used as heatmap, behind the arrows of X_theta,
        if Not provided (X=None), X is created a grid of same size as X_theta with all values=0
        to create a black background.
        X can be used as showing any related metrics of each directional arrow, such as confidance
        of computation, uncertainity of estimation, etc.

    square: bool, default=True,
            Keep it True for correct aspect ratio of a Grid

    cbar: bool, Default=False
          To show colorbar of heatmap, default is set to False

    arr_pivot: str, arrows location default='mid'
          This is used in plotting arrows, where location of arrow in a grid is plotted at the center
          Keeping it 'mid' is best to look on a uniform grid

    stream_plot: bool, Default=True
            If set to True, a streamline plot of Directional flow is produced

    title: str, default=''
           To prepand a string on titles on both figures


    upsample: float, if greater than 1, X_theta is upsampled to new size of grid
              that will entail:: new_size_of_grid = upsample*old_size_of_grid


    Properties to configure: Heatmap, Arrows and Streamline plot

    heatmap_prop: dict, properties for heatmap, default: heatmap_prop =dict()
                properties of heatmap at background can be configured by using 'heatmap_prop'
                except for two keywords (square=True,cbar=False,) which are passed seprately, see above

                Some examples of settings
                    heatmap_prop =dict(vmin=0,vmax=20,cmap='jet',linewidth=0.01,alpha=0.1)
                    heatmap_prop =dict(vmin=0,vmax=20,cmap='jet')
                    heatmap_prop =dict(center=0)

                Some useful properties that can be changed are
                    (data, *, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt='.2g', annot_kws=None,
                    linewidths=0, linecolor='white', cbar_kws=None, cbar_ax=None,
                    xticklabels='auto', yticklabels='auto')

                for more details on heatmap_prop properties check https://seaborn.pydata.org/generated/seaborn.heatmap.html
                or help(seaborn.heatmap)

    arr_prop: dict, properties for arrow, default: arr_prop =dict(color='w',scale=None),
              Properties of arrows can be configured using 'arr_prop', default color is set to white: 'w'
              except for 'pivot' keyword (pivot='mid'), which is passed seprately, see above

              Some examples of settings
                arr_prop =dict(color='k',scale=None)
                arr_prop =dict(color='C0',scale=30)

              Some useful properties that can be changed are
                color: color of arrow
                width: Shaft width
                headwidthfloat, default: 3
                headlengthfloat, default: 5
                headaxislengthfloat, default: 4.5
                minshaftfloat, default: 1
                minlengthfloat, default: 1

            for more details on heatmap_prop properties check https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html
            or help(plt.quiver)

    stream_prop:dict, properties for default: stream_prop=dict(density=1,color='k')


            Some examples of settings
               stream_prop=dict(density=1,color='k')
               stream_prop=dict(density=2,color='C0')

            Some useful properties that can be changed are
               (linewidth=None, color=None, cmap=None, norm=None, arrowsize=1, arrowstyle='-|>', minlength=0.1,
               transform=None, zorder=None, start_points=None, maxlength=4.0, integration_direction='both',
               broken_streamlines=True),

            for more details on heatmap_prop properties check  https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.streamplot.html
            or help(plt.streamplot)


    figsize: tuple, default=(12,8): size of figure

    show: bool, default=True
           If True, plt.show() is executed
           setting it false can be useful to change the figures propoerties

    ax: list of axis to use for plot, default =[]
       Alternatively, list of axies can be passed to plot.
       if stream_plot=True, list of atleast two axes are expected, else atleast one
       if choosing stream_plot=False, still pass one axis as a list ax=[ax], rather than single axis itself
       If passed, ax[0] will be used for directional plot, and ax[1] will be used for streamplot, if  stream_plot=True

    fig: deafult=None, if passing axes to plot, fig can be passed too, however, it is not used for anything other than returing


    Returns
    -------
    X_theta: Same as input, if upsample>1, then new upsampled X_theta

    X  : Same as input, if upsample>1, then new upsampled X
         If input was X=None, then returns X of 0


    (fig,ax): fig and ax, useful to change axis properties if show=False

    See Also
    --------
    spkit: #TODO

    Notes
    ------
    arr_prop = dict(scale=None,headwidth=None, headlength=None, headaxislength=None, minshaft=None, minlength=None,color='w')


    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import spkit as sp
    >>> np.random.seed(100)
    >>> N,M = 8,8
    >>> Ax_theta= np.arctan2(np.random.randn(N,M)+1, np.random.randn(N,M)+1)
    >>> Ax_bad   = 1*(np.random.rand(N,M)>0.1).astype(float)
    >>> Ax_bad[Ax_bad==0]=np.nan
    >>> XY = sp.direction_flow_map(Ax_theta)
    >>> plt.show()
    """

    assert X_theta.ndim==2
    N,M = X_theta.shape

    if X is not None:
        assert X_theta.shape==X.shape
        Xi = X.copy()
    else:
        Xi = X_theta*0

    Xi_theta = X_theta.copy()
    if upsample>1:
        # try:
        #     from skimage.transform import resize
        # except Exception as e:
        #     print(f"Unexpected {e=}, {type(e)=}")
        #     warnings.warn("To use upsample install 'skimage' ",stacklevel=2)
        #     raise

        N, M = int(upsample*N), int(upsample*M)
        Xi_theta = resize(Xi_theta, [N,M])
        Xi = resize(Xi, [N,M])

    tht_x, tht_y = np.cos(Xi_theta), np.sin(Xi_theta)

    xi,yi = np.meshgrid(np.arange(M),np.arange(N))
    #print(xi.shape,yi.shape,tht_x.shape,tht_y.shape)

    if stream_plot:
        if len(ax)==0:
            fig, ax = plt.subplots(1,2,figsize=figsize)
        else:
            # if ax is passed, and stream_plot=True, number of axis should be atleast 2
            # if passing list of one axis, ax= [ax[0]], set stream_plot=False
            assert len(ax)>1

    else:
        if len(ax)==0:
            fig, ax = plt.subplots(figsize=figsize)
            ax = [ax]
    #cmap=cmap,vmin=vmin,vmax=vmax,cbar=cbar,
    #annot=False,square=True,cmap='jet',vmin=0,vmax=20
    sns.heatmap(data=Xi,ax=ax[0],square=True,cbar=False,**heatmap_prop)
    q = ax[0].quiver(xi+0.5,yi+0.5,tht_x,tht_y,pivot=arr_pivot,**arr_prop)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    if title!='': title = title+ ': '
    ax[0].set_title(f'{title}Directional flow Map')

    if stream_plot:
        ax[1].streamplot(xi,yi,tht_x,-tht_y,**stream_prop)
        ax[1].invert_yaxis()
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_box_aspect(N/M)
        ax[1].set_title(f'{title}Streamlined flow')
    if show: plt.show()
    return Xi_theta, Xi, (fig,ax)

def agg_angles(thetas, agg='mean'):
    r"""Aggregating Angles (Directional Arrows) (thetas)
    
    **Aggregating Directional Arrows (thetas)**

    Converting polar coordinates to cartesian aaverging there and converting back to polar

    Parameters
    ----------
    thetas: list/array of theta (angles)
    agg: str {'mean', 'median'}, default='mean'
     - method to use averaging, mean or median

    Returns
    -------

    theta_avg: scalar, aggregated angle value


    References
    ----------
    * wikipedia - 
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    show_compass: # Show Compass


    Examples
    --------
    >>> import numpy as np
    >>> import spkit as sp
    >>> np.random.seed(2)
    >>> N=8
    >>> # +1 -1, direction, top left
    >>> thetas= np.arctan2(np.random.randn(N)+1, np.random.randn(N)-1)
    >>> thetas_avg = sp.agg_angles(thetas, agg='mean')
    >>> print(thetas_avg)
    
    2.9525995459333507
    """
    if agg=='median':
        xy = np.nanmedian(np.c_[np.cos(thetas),np.sin(thetas)],axis=0)
    else:
        xy = np.nanmean(np.c_[np.cos(thetas),np.sin(thetas)],axis=0)

    theta_avg = np.arctan2(xy[1],xy[0])

    return theta_avg

def elbow_knee_point(x,dx=None, plot=False,show=False,show_lines=False,lw=1):
    r"""Finding Elbow-knee point of a curve
    
    The algorithm computes the perpendicular distance between each curve point to base vector.
    where base vector is line between first and last point of curve.
    
    
    .. math::

       dist = |p - (p \cdot b') b'|

       b' = b/|b|

       idx = argmax(dist)
    
    Parameters
    ----------
    x: 1d array
     - input curve, 
    dx: None, or 1d-array, default=None
     - if None, curve points are assumed to be evenly spaced.
     - if not None, then dx is the points for x is computed on.
    plot: bool, False
     - if True, plot the curve, and knee point
    show_dist: bool, default=False,
     - if True, show distance computed to use for knee point
    show: bool, default=False
     - if True, plt.show() is excecuted, else not, to be used for addtional plt commands to be affective
    
    Returns
    -------
    idx: int, 
     - index of the knee point in the curve


    References
    ----------
    * wikipedia - https://en.wikipedia.org/wiki/Knee_of_a_curve
    
    
    Notes
    -----
    * Knee point is not very accurate if curve x is noisy. Apply this algorithm with smooth curve x

    Examples
    --------
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    t = np.arange(100)/99
    x = np.exp(-2*t)
    idx = sp.elbow_knee_point(x,plot=True,show_lines=False)
    x = np.exp(-5*t)
    idx = sp.elbow_knee_point(x,plot=True,show_lines=False)
    x = np.exp(-10*t)
    idx = sp.elbow_knee_point(x,plot=True,show_lines=False)
    x = np.exp(-20*t)
    idx = sp.elbow_knee_point(x,plot=True,show_lines=False)
    x = np.exp(-50*t)
    idx = sp.elbow_knee_point(x,plot=True,show_lines=False)
    plt.grid()
    plt.show()
    """
    n = len(x)
    if dx is not None:
        x_coord = np.vstack([dx, x]).T
    else:
        x_coord = np.vstack((np.arange(n), x)).T
    
    p0 = x_coord[0]
    line_vec = x_coord[-1] - x_coord[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    all_vec = x_coord - p0
    scalar_ = np.sum(all_vec * np.tile(line_vec_norm, [n, 1]), axis=1)
    all_vec_parallel = np.outer(scalar_, line_vec_norm)
    vec_line = all_vec - all_vec_parallel
    dist_line = np.sqrt(np.sum(vec_line ** 2, axis=1))
    idx = np.argmax(dist_line).astype(int)
    
    if plot:
        plt.plot(x_coord[:,0],x_coord[:,1])
        if show_lines:
            plt.plot(x_coord[:,0],dist_line-dist_line[0]+x_coord[0,1])
            b = x_coord[[0,-1]].T
            plt.plot(b[0],b[1],color='C0',lw=0.5,ls='--')
        plt.axvline(x_coord[idx,0],color='k',ls='--',lw=lw)
        plt.plot(x_coord[idx,0],x_coord[idx,1],'or')
        plt.xlabel('idx')
        plt.ylabel('x')
        plt.title(f'')
        if show: plt.show()
    return idx

def total_variation(x,normalise=False,method='npdiff',**kwargs):
    r"""Total Variation of a signal

    .. math::

        TV(x) = \sum_n |x_{n+1}-x_{n}|

        TV(x) = \sum |\frac{dx}{dt}|

    Bounds

    .. math::

        min \{ max(x) - min(x) \} \ge TV(x) \ge (n-1) \times min \{ max(x) - min(x) \}  

    where :math:`n=length(x)`


    Parameteres
    -----------
    x: 1d-array
       - input signal, or sequence

    normalise: bool, default=False
       - if True, Total Variation is normalised with (maximum - minimum TV)
    
    method: differentiation method, default='np.diff'
    
    kwargs: kwargs for :func:`signal_diff`


    Returns
    -------
    tv: float
     -  Total Variation of x

    References
    ----------
    * wikipedia - https://en.wikipedia.org/wiki/Total_variation
    
    Examples
    --------
    #sp.total_variation
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x1, fs = sp.data.optical_sample(sample=1)
    x2, fs = sp.data.optical_sample(sample=2)
    tv_x1 = sp.total_variation(x1,normalise=False)
    tv_x2 = sp.total_variation(x2,normalise=False)
    print(f'TV(x1) = {tv_x1}')
    print(f'TV(x2) = {tv_x2}')
    """
    dx = signal_diff(x,method=method,**kwargs)
    tv = np.nansum(np.abs(dx))
    if normalise:
        n = len(x)
        min_tv = np.nanmax(x) - np.nanmin(x)
        tv = tv/(min_tv*(n-2))
    return tv

def total_variation_win(x,winlen=11,overlap=None,method='npdiff',**kwargs):
    r"""Total Variation of a signal window-wise


    Parameteres
    -----------
    x: 1d-array
       - input signal, or sequence

    winlen: int, odd, default=11
       - length of window to analyse
    overlap: int, default=None
       -  if None, overlap=winlen//2
       
    method: differentiation method, default='np.diff'
    
    kwargs: kwargs for :func:`signal_diff`


    Returns
    -------
    tv_win: 1d-array, 
      -  Total Variation of x with window-wise

    References
    ----------
    * wikipedia - https://en.wikipedia.org/wiki/Total_variation
    
    Examples
    --------
    #sp.total_variation_win
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x, fs = sp.data.optical_sample(sample=2)
    x = x[int(0.0*fs):int(2*fs)]
    tv_win = sp.total_variation_win(x,winlen=11)
    plt.figure(figsize=(10,3))
    plt.subplot(211)
    plt.plot(x)
    plt.subplot(212)
    plt.plot(tv_win)
    plt.tight_layout()
    plt.show()
    """
    
    dx = signal_diff(x,method=method,**kwargs)
    
    if overlap is None:
        overlap = winlen//2

    dx_ = np.r_[np.zeros(overlap),dx,np.zeros(overlap)]

    win = np.arange(winlen)

    tv_win =[]
    while win[-1]<len(dx_):
        tv_win.append(np.nansum(np.abs(dx_[win])))
        win +=overlap
    tv_win = np.array(tv_win)
    return tv_win

def mean_minSE(x,y,W=5,show=False,compare_mse=True,plot_log=False, esp=1e-3,show_legend=True):
    r"""Mean of Minimum Squared Error (MMSE) under temporal shift

    Mean of Minimum Squared Error (MMSE) is the metric computed the closeness of
    two signals. MMSE ignores the a little 


    for t = -W to W

    .. math:: 
       MminSE = 1/K \sum_0^K-1 min( (x(k) - y(k-t))**2  ) 
    
    Parameters
    ----------
    x: 1d-array
    y: 1d-array



    Returns
    -------

    References
    ----------
    * wikipedia - 
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    spkit: #TODO

    Examples
    --------
    >>> import numpy as np
    >>> import spkit as sp
    >>> #TODO

    """

    Es = []
    for k in range(-W,W+1):
        yi = np.roll(y,k)
        if k<0:
            yi[k:] +=np.nan
        elif k>0:
            yi[:k] +=np.nan

        Es.append((x-yi)**2)

    Es = np.array(Es)
    Mmse = np.nanmean(np.nanmin(Es,axis=0))

    if show:
        Es_min =  np.nanmin(Es,axis=0)
        SE = (x-y)**2
        plt.figure(figsize=(6,2))
        #plt.plot(Es.T,alpha=0.4)
        if plot_log:
            plt.plot(np.log(Es_min+esp),label='minSE')
            if compare_mse: plt.plot(np.log(SE+esp),label='SE')
            plt.ylabel('log Error')
        else:
            plt.plot(Es_min,label='minSE')
            if compare_mse: plt.plot(SE,label='SE')
            plt.ylabel('Error')
        if show_legend: plt.legend()
        plt.title(f'MminSE = {Mmse.round(6)} | W = {W}')
        plt.show()
    return Mmse, Es

def minMSE(x,y,W=5,show=False):
    r"""Minimum Mean Squared Error under temporal shift

    for t = -W to W

    .. math:: minMSE = min \{ 1/K \sum_0^K-1 ( (x(k) - y(k-t))**2  ) \}
       
    
    Parameters
    ----------

    Returns
    -------

    References
    ----------
    * wikipedia - 
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    spkit: #TODO

    Examples
    --------
    >>> import numpy as np
    >>> import spkit as sp
    >>> #TODO
    """
    Es = []
    for k in range(-W,W+1):
        if k>0:
            yi = np.r_[y[k:],np.zeros(k)+np.nan]
        elif k<0:
            yi = np.r_[np.zeros(np.abs(k))+np.nan,y[:k]]
        else:
            yi = y.copy()
        mse = np.nanmean((x-yi)**2)
        Es.append(mse)
    Es = np.array(Es)
    mMSE = np.nanmin(Es)
    if show:
        plt.figure(figsize=(6,2))
        #plt.plot(Es.T,alpha=0.4)
        plt.plot(range(-W,W+1),Es)
        plt.title(f'minMSE = {mMSE.round(6)} | W = {W}')
        plt.xlabel('W (shift)')
        plt.ylabel('Min Error')
        plt.grid()
        plt.show()
    return mMSE, Es


"""
Signal Differentiation
----------------------
"""


def signal_diff(x,method='fdiff',sg_window=101,sg_polyorder=1,gauss_window=0,gauss_itr=1,gauss_sigscale=2.7,dt=1):

    r"""Derivative of a signal
    
    **Derivative of a signal**

    Computing Derivating of a signal is very useful in signal processing. In some of the physiological signals
    derivative of signal allows to compute events such as activation and repolarisation time.  

    This function implements some of the well known approaches to compute derivative

    'fdiff':
        - Finite differentiation using first principle
        
        intermediate values

        .. math::
            
            dx(t) =  [x(t+d) - x(t-d)]/2  \quad  \text{when t+d and t-d in D (Domain)}

        
        boundries
        
        -  first point t=0, when t+d in D, but t-d not in D
        
        .. math::

            dx(t) &= x(t+d)-x(t)

            dx(0) &= x(d)-x(0)

        - last point t=l, when t-d in D, but t+d not in D

        .. math::

            dx(t) &= x(t)-x(t-d)

            dx(l) &= x(l)-x(l-d)


    'fgrad':
        - Finite differentiation
        - Using finite gradient (`np.gradient`)

        .. note::
            `fgrad` is essentially the same as `fdiff` computed in numpy using `np.gradient`

    *for following methods*:
        - parameters for Savitzky-Golay filter (`sg_window` , `sg_polyorder`)
        - parameters for gaussian kernal+ConvFB (`gauss_window`, `gauss_itr` ), applied only if `gauss_window`>0

    'sgdiff':
        - Computing derivetive of signal using Savitzky-Golay filter, 
        - then with gaussian kernal using Forward-Backward-Convolution (ConvFB).

    'sgdrift_diff':
        - First remove drift using Savitzky-Golay filter,
        - then apply gaussian smoothing apply `fgrad`.

    'sgsmooth_diff':
           - First smooth the input signal x with Savitzky-Golay filter then apply  
           - then apply gaussian smoothing (unnecessary, set `gauss_window =0` to avoid) then apply `fgrad`

    'gauss_diff':
            - First smooth the input signal x using gaussian smoothing
            - then apply `fgrad`

    'npdiff':
            - npdiff uses numpy's `np.diff`, which computes x[n+1] - x[n]
            - for compleness, npdiff is included.


    Parameters
    ----------
    x: 1d-array signal

    method: method to compute derivative, one of the above

    (sg_window,sg_polyorder,gauss_window,gauss_itr): Savitzky-Golay filter, Gaussian smoothing using Forward-Backward-Convolution (ConvFB)


    Returns
    -------
    dx: derivative of signal

    References
    ----------
    * For Noisy signal, one of the other approach is using Total Variation - https://arxiv.org/pdf/1701.00439
    
    
    Notes
    -----
    - Using fdiff and fgrad results in same output.
    - for noisy signal, applying smoothing first is always recommonded, (see example below)

    See Also
    --------
    filter_smooth_sGolay, filter_smooth_gauss, filter_with_kernel


    Examples
    --------
    #sp.signal_diff
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    x, fs = sp.data.optical_sample(sample=2)
    x = x[int(0.023*fs):int(0.36*fs)]
    x = x - x.mean()
    x_noise = sp.add_noise(x,snr_db=10)
    t = np.arange(len(x))/fs

    methods = ['fdiff','fgrad','sgdrift_diff','sgdiff','gauss_diff']
    DX1 = [sp.signal_diff(x.copy(),method=method,sg_window=11,gauss_window=11) for method in methods]
    DX2 = [sp.signal_diff(x_noise.copy(),method=method,sg_window=11,gauss_window=11) for method in methods]

    figsize = (8,6)
    K = len(DX1)

    plt.figure(figsize=figsize)
    plt.subplot(K+1,2,1)
    plt.plot(t,x,'C1')
    plt.xlim([t[0],t[-1]])
    plt.ylabel(f'x',rotation=0)
    plt.yticks([])
    plt.xticks(fontsize=0)
    plt.grid()
    plt.title(f'x: singal')

    plt.subplot(K+1,2,2)
    plt.plot(t,x_noise,'C1')
    plt.xlim([t[0],t[-1]])
    #plt.ylabel(f'x')
    plt.yticks([])
    plt.xticks(fontsize=0)
    plt.grid()
    plt.title(f'x_noise: noisy singal')

    for i in range(K):
        plt.subplot(K+1,2,i*2+3)
        plt.plot(t[:len(DX1[i])],DX1[i],color='C0')
        plt.xlim([t[0],t[-1]])
        plt.grid()
        plt.yticks([]) 
        #plt.xticks(labels='')
        plt.ylabel(methods[i],rotation=0, ha='right')
        if i==K-1:
            plt.xlabel('time (s)')
        else:
            plt.xticks(fontsize=0)

        plt.subplot(K+1,2,i*2+4)
        plt.plot(t[:len(DX2[i])],DX2[i],color='C0')
        plt.xlim([t[0],t[-1]])
        plt.grid()
        #if i<K-1:plt.xticks([])
        plt.yticks([])
        #plt.ylabel(f'$N={N[i]}$')
        #plt.legend(frameon=False,bbox_to_anchor=(1,1))
        if i==K-1:
            plt.xlabel('time (s)')
        else:
            plt.xticks(fontsize=0)

    plt.suptitle('Derivative of signal: dx')
    plt.subplots_adjust(hspace=0,wspace=0.05)
    plt.tight_layout()
    plt.show()
    """

    if method =='fdiff':
        dx = np.diff(x) / dt
        # Pad the data
        dx = np.hstack((dx[0], dx, dx[-1]))
        # Re-finite dxdt_hat using linear interpolation
        dx = (dx[0:-1]+dx[1:])/2
    elif method =='fgrad':
        dx = np.gradient(x)
    elif method =='npdiff':
        dx = np.diff(x)

    elif method == 'sgdiff':
        #dx = scipy.signal.savgol_filter(x, sg_win, sg_polyorder, deriv=1) / dt
        dx = filterDC_sGolay(x,window_length=sg_window,polyorder=sg_polyorder,deriv=1,return_background=True)[1]/dt
        if gauss_window>0:
            kernel   = gaussian_kernel(gauss_window,sigma_scale=gauss_sigscale)
            dx = conv1d_fb(dx, kernel,iterations=gauss_itr)
    elif method =='sgdrift_diff':
        x_h= filterDC_sGolay(x,window_length=sg_window,polyorder=sg_polyorder)
        if gauss_window>0:
            kernel   = gaussian_kernel(gauss_window,sigma_scale=gauss_sigscale)
            x_h = conv1d_fb(x_h, kernel,iterations=gauss_itr)
        dx = np.gradient(x_h)
    elif method =='sgsmooth_diff':
        x_s= filter_smooth_sGolay(x,window_length=sg_window,polyorder=sg_polyorder)
        #x_s= sp.filterDC_sGolay(x,window_length=sg_window,polyorder=sg_polyorder,return_background=True)[1]
        if gauss_window>0:
            kernel   = gaussian_kernel(gauss_window,sigma_scale=gauss_sigscale)
            x_s = conv1d_fb(x_s, kernel,iterations=gauss_itr)
        dx = np.gradient(x_s)
    elif method =='gauss_diff':
        kernel   = gaussian_kernel(gauss_window,sigma_scale=gauss_sigscale)
        xi = conv1d_fb(x, kernel,iterations=gauss_itr)
        dx = np.gradient(xi)
    else:
        raise NameError('Unknown method name, available names of methodes are ("fdiff", "fgrad", "npdiff","sgdiff","sgdrift_diff","sgsmooth_diff", "gauss_diff") ')
    return dx