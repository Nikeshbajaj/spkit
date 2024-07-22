'''
Basic signal processing methods
--------------------------------
Author @ Nikesh Bajaj
updated on Date: 27 March 2023. Version : 0.0.5
updated on Date: 26 Sep 2021, Version : 0.0.4
Github :  https://github.com/Nikeshbajaj/spkit
Contact: n.bajaj@qmul.ac.uk | n.bajaj@imperial.ac.uk
'''

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
#from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
from scipy import stats as scipystats
from copy import deepcopy
import pywt as wt
from scipy.interpolate import interp1d, interp2d
import seaborn as sns


sys.path.append("..")
#sys.path.append(".")
from .infotheory import entropy
from ..utils import ProgBar
from ..utils_misc.borrowed import resize

def filterDC_(x,alpha=256):
    '''
    TO BE DEPRECIATED  - use filterDC instead
    ----------------

    Filter out DC component - Remving drift using Recursive (IIR type) filter
    -------------------------------------
          y[n] = ((alpha-1)/alpha) * ( x[n] - x[n-1] -y[n-1])

          where y[-1] = x[0], x[-1] = x[0]
          resulting y[0] = 0
    input
    -----
    x    : (vecctor) input signal

    alpha: (scalar) filter coefficient, higher it is, more suppressed dc component (0 frequency component)
         : with alpha=256, dc component is suppressed by 20 dB

    initialize_zero: (bool): If True, running backgrpund b will be initialize it with x[0], resulting y[0] = 0
          if False, b = 0, resulting y[0] ~ x[0], and slowly drifting towards zeros line
          - recommended to set True
    output
    -----
    y : output vector

    '''
    b = x[0]
    y = np.zeros(len(x))
    for i in range(len(x)):
        b = ((alpha - 1) * b + x[i]) / alpha
        y[i] = x[i]-b
    return y

def filterDC_X(X,alpha=256,return_background=False,initialize_zero=True):
    '''
    TO BE DEPRECIATED   - use filterDC instead
    ----------------

    Filter out DC component - Remving drift using Recursive (IIR type) filter
    -------------------------------------
          y[n] = ((alpha-1)/alpha) * ( x[n] - x[n-1] -y[n-1])

          where y[-1] = x[0], x[-1] = x[0]
          resulting y[0] = 0
    input
    -----
    x    : (vecctor) input signal

    alpha: (scalar) filter coefficient, higher it is, more suppressed dc component (0 frequency component)
         : with alpha=256, dc component is suppressed by 20 dB

    initialize_zero: (bool): If True, running backgrpund b will be initialize it with x[0], resulting y[0] = 0
          if False, b = 0, resulting y[0] ~ x[0], and slowly drifting towards zeros line
          - recommended to set True
    output
    -----
    y : output vector

    '''
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

def filterDC(X,alpha=256,return_background=False,initialize_zero=True):
    '''
    Filter out DC component - Remving drift using Recursive (IIR type) filter
    -------------------------------------
          y[n] = ((alpha-1)/alpha) * ( x[n] - x[n-1] -y[n-1])

          where y[-1] = x[0], x[-1] = x[0]
          resulting y[0] = 0
    implemenatation works for single (1d array) or multi-channel (2d array)
    input
    -----
    X : (vecctor) input signal single channel (n,) or multi-channel, channel axis should be 1 shape ~ (n,ch)

    alpha: (scalar) filter coefficient, higher it is, more suppressed dc component (0 frequency component)
         : with alpha=256, dc component is suppressed by 20 dB

    initialize_zero: (bool): If True, running backgrpund b will be initialize it with x[0], resulting y[0] = 0
          if False, b = 0, resulting y[0] ~ x[0], and slowly drifting towards zeros line
          - recommended to set True
    output
    -----
    Y : output vector, shape same as input X (n,) or (n,ch)

    '''
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
    '''
    Filter out DC component - Remving drift using Savitzky-Golay filter
    -------------------------------------------------------------------
    Savitzky-Golay filter for multi-channels signal: From Scipy library

    input
    -----
    X : (vecctor) input signal single channel (n,) or multi-channel, channel axis should be 1 shape ~ (n,ch)
    window_length: should be an odd number
    others input parameters as same as in scipy.signal.savgol_filter
              :(polyorder=3, deriv=0, delta=1.0, mode='interp', cval=0.0)

    output
    ------
    Y : corrected signal
    Xm: background removed -  return only if return_background is True
    '''
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

def filter_X_(X,fs=128.0,band =[0.5],btype='highpass',order=5,ftype='filtfilt',verbose=1,use_joblib=False):
    '''
    Buttorworth filtering -  basic filtering
    ---------------------
    X : (vecctor) input signal single channel (n,) or multi-channel, channel axis should be 1 shape ~ (n,ch)

    band: cut of frequency, for lowpass and highpass, band is list of one, for bandpass list of two numbers
    btype: filter type
    order: order of filter
    ftype: filtering approach type, 'filtfilt', 'lfilter', 'SOS',
         : lfilter is causal filter, which introduces delay, filtfilt does not introduce any delay, but it is non-causal filtering
          SOS:  Filter a signal using IIR Butterworth SOS method. A forward-backward digital filter using cascaded second-order sections:
    Xf: filtered signal of same size as X
    '''
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

#TOBE TESTEST SOS
def filter_X(X,fs=128.0,band =[0.5],btype='highpass',order=5,ftype='filtfilt',verbose=1,use_joblib=False,filtr_keywors=dict()):
    '''
    Buttorworth filtering -  basic filtering
    ----------------------------------------

    Parameteres
    -----------
    X : (vecctor) input signal single channel (n,) or multi-channel, channel axis should be 1 shape ~ (n,ch)

    band: cut of frequency, for lowpass and highpass, band is list of one, for bandpass list of two numbers
    btype: filter type
    order: order of filter
    ftype: filtering approach type, 'filtfilt', 'lfilter', 'SOS', 'sosfilt','sosfiltfilt',
          'SOS' is mapped to 'sosfiltfilt'
         : lfilter is causal filter, which introduces delay, filtfilt does not introduce any delay, but it is non-causal filtering
          SOS:  Filter a signal using IIR Butterworth SOS method. A forward-backward digital filter using cascaded second-order sections.
         NOTE: 'SOS' is Recommended

    Returns
    --------
    Xf: filtered signal of same size as X
    '''
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

def Periodogram(x,fs=128,method ='welch',win='hann',nfft=None,scaling='density',average='mean',detrend='constant',nperseg=None, noverlap=None):
    '''
    Computing Periodogram using Welch or Periodogram method
    ------------------------------------------------------
    #scaling = 'density'--V**2/Hz 'spectrum'--V**2
    #average = 'mean', 'median'
    #detrend = False, 'constant', 'linear'
    nfft    = None, n-point FFT
    '''
    if method ==None:
        f, Pxx = scipy.signal.periodogram(x,fs,win,nfft=nfft,scaling=scaling,detrend=detrend)
    elif method =='welch':
        #f, Pxx = scipy.signal.welch(x,fs,win,nperseg=np.clip(len(x),0,256),scaling=scaling,average=average,detrend=detrend)
        f, Pxx = scipy.signal.welch(x,fs,win,nperseg=nperseg,noverlap=noverlap,nfft=nfft,scaling=scaling,average=average,detrend=detrend)
    return np.abs(Pxx)

def getStats(x,detail_level=1,return_names=False):
    '''
    Statistics of a given sequence x, excluding NaN values
    ------------------------------------------------------
    returns stats and names of statistics measures

    '''
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

def getQuickStats(x):
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

def OutLiers(x, method='iqr',k=1.5, include_lower=True,include_upper=True,return_lim=False):
    '''
    Identyfying outliers
    --------------------
    using
    1. Interquartile Range: below Q1 - k*IQR and above Q3 + k*IQR
    2. Stander Deviation:   below Mean -k*SD(x) above Mean + k*SD(x)

    input
    -----
    x :  1d array or nd-array

    method = 'iqr' or 'sd'
    k : (default 1.5), factor for range, for SD k=2 is widely used
    include_lower: if False, excluding lower outliers
    include_upper: if False excluding upper outliers
     - At least one of (include_lower, include_upper) should be True
    return_lim: if True, return includes lower and upper limits (lt, ul)

    output
    -----
    idx: index of outliers in x
    idx_bin: binary array of same size as x, indicating outliers
    (lt,ut): lower and upper limit for outliers, if  return_lim is True

    '''
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


'''
BASIC WAVELET FILTERING
------------------------
'''
def get_theta(w,N,k=1.5,method='optimal',IPR=[0.25,0.75]):
    '''
    Threshold for wavelet filtering
    -------------------------------------
    input
    -----
    w: wavelet coeeficients
    N: length of signal x for noise eastimation
    method:  method to compute threshold
          : 'optimal' - optimal threshold based on noise estimation
          : 'sd'      - mu ± k*sd
          : 'iqr'     - Q1 - k*IQR, Q3 + k*IQR
    k: for outlier computation as above
    IPR   : Inter-percentile range: quartile to be considers for inter-quartile range IPR = [0.25, 0.75]
          : could be [0.3, 0.7] for more aggressive threshold

    output
    -----
    theta_l, theta_u = lower and upper threshold for wavelet coeeficients

    '''
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
    '''
    Wavelet Filtering
    ------------------

    input
    -----

    x - 1d array


    Threshold Computation method:
    threshold: 'str' or float
             : if str, method to compute threshold, example : 'optimal', 'sd', 'iqr'

             'optimal': threshold = sig*sqrt(2logN), sig = median(|w|)/0.6745
             'sd' : threshold = k*SD(w)
             'iqr': threshold = q3+kr, threshold_l =q1-kr, where r = IQR(w)  #Tukey's fences
             'ttt': Modified Thompson Tau test (ttt) #TODO
             default - optimal

    mode: str, 'elim' - remove the coeeficient (by zering out), 'clip' - cliping the coefficient to threshold
         default 'elim'

    below: bool, if true, wavelet coefficient below threshold are eliminated else obove threshold


    Wavelet Decomposition modes:
    wpd_mode = ['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization']
                default 'symmetric'

    wpd_maxlevel: level of decomposition, if None, max level posible is used

    Wavelet family:
    wv = ['db3'.....'db38', 'sym2.....sym20', 'coif1.....coif17', 'bior1.1....bior6.8', 'rbio1.1...rbio6.8', 'dmey']
         :'db3'(default)

    packetwise: if true, thresholding is applied to each packet/level individually, else globally
    WPD: if true, WPD is applied as wavelet transform
    lvl: list of levels/packets apply the thresholding, if empty, applied to all the levels/packets

    show: bool, deafult=False, if to plot figure, it True, following are used
        figsize: default=(11,6), size of figure
        plot_abs_coef: bool, deafult=False,
                       if True,plot abs coefficient value, else signed

    output
    ------
    xR:  filtered signal, same size as x
    '''

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
    '''
    Wavelet Filtering applied to smaller windows
    --------------------------------------------

    Same as wavelet_filtering fumction, applied to smaller overlapping windows and reconstructed by overlap-add method

    for documentation, check help(wavelet_filtering)

    '''
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

def WPA_coeff(x,wv='db3',mode='symmetric',maxlevel=None, verticle_stacked=False):
    '''
    Wavelet Packet Decomposition
    ----------------------------
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
    '''
    wp = wt.WaveletPacket(x, wavelet=wv, mode=mode,maxlevel=maxlevel)
    wr = [wp[node.path].data for node in wp.get_level(wp.maxlevel, 'natural') ]
    WK = np.vstack(wr) if verticle_stacked else np.hstack(wr)
    return WK

def WPA_temporal(x,winsize=128,overlap=64,wv='db3',mode='symmetric',maxlevel=None,verticle_stacked=True,pad=True,verbose=0):
    '''
    Wavelet Packet Decomposition -  for each window and stacked together
    -------------------------------------
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

    '''
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

def WPA_plot(x,winsize=128,overlap=64,verticle_stacked=True,wv='db3',mode='symmetric',maxlevel=None,inpterp='sinc',
             fs=128,plot=True,pad=True,verbose=0, plottype='abs',figsize=(15,8)):
    '''
    Wavelet Packet Decomposition -  temporal - Plot
    -------------------------------------

    return Wavelet coeeficients packet vs time


    '''
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

def Wavelet_decompositions(x,wv='db3',L = 6,threshold=np.inf,show=True,WPD=False):
    '''
    Decomposing signal into different level of wavalet based signals
    '''
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


def sinc_interp(x):
    #sinc interpolation
    N = len(x)
    y = np.zeros(2*N-1) + 1j*0
    y[::2] = x.copy()
    t0 = np.arange(-(2*N-3),(2*N-3)+1)/2
    x_intp = conv1d_fft(y, np.sinc(t0))
    #x_intp = x_intp[2*N-2:-2*N+3]
    x_intp = x_intp[2*N-2-1:-2*N+3]
    return x_intp

def conv1d_fft(x,y):
    N = len(x) + len(y)-1
    P = 2**np.ceil(np.log2(N)).astype(int)
    z = np.fft.fft(x,P)*np.fft.fft(y,P)
    z = np.fft.ifft(z)
    z = z[:N]
    return z

'''
NEW UPDATES 16/03/2023    TOBE TESTES ALL
----------------------
'''

#TOBE TESTED
def add_noise(x, snr_db=10,return_noise=False):
    r"""

    ADD Gaussian Noise to Signal
    ----------------------------

    SNR =  sig_pw/noise_pw
    SNR_db = sig_pw_db - noise_pw_db

    noise_pw = sig_pw/SNR
    noise_pw = 10**( (sig_pw_db-SNR_db)/10 )

    noise ~ N(0, sqrt(noise_pw))

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
        return x+noise, noise
    return x+noise

def filter_powerline(X,fs=1000, powerline=50):
    '''
    Equivilent to lowpass butterworth filter of order 1
    '''
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

def filtering_pipeline(X,fs,trim=[0,0],iir_alpha=0,sg_window=1001,sg_polyorder=1,sg_itr=1,filter_lpf=None, filter_pwrline=None,verbose=1):
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

    if verbose: print('output: ',X.shape)
    return X

def filtering_pipeline_v2(X,fs,trim=[0,0],iir_alpha=0,sg_window=1001,sg_polyorder=1,filter_lpf =None,filter_pwrline=None,verbose=1):
    #if axis==1: X = X.T
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

def fill_nans_1d(x, pkind='linear'):
    r"""
    Fill nan values with interpolation/exterpolation for 1D
    -----------------------------------------

    Parameters
    ----------
    x :  1d array, with NaN values
    pkind : kind of interpolation
           {'linear', 'nearest', 'nearest-up', 'zero',
            'slinear', 'quadratic', 'cubic', 'previous', or 'next'. 'zero'}
            default='linear

    Returns
    -------
    y : 1d array resulting array with interpolated values instead of nans
        same shape-size as x

    Example
    -------
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

def fill_nans_2d(X,pkind='linear',filter_size=3,method='conv',clip_range=[None,None]):
    """
    Fill nan values with interpolation/exterpolation for 2D
    ----------------------------------------------------------------

    This function uses 'fill_nans_1d' for each column and each row.
    This results two inter/exter-polated values for each missing value

    To fill the missing value, funtion takes average of both, which
    reduces the variability along both axis.

    Further to remove any artifacts created by new values, smoothing is applied.
    However, original values are restored.

    Finally, if clip_range is passed, values in new matrix are clipped using it.

    Parameters
    ----------
    X: 2d-array with missing values, denoted by np.nan

    pkind : kind of interpolation
           {'linear', 'nearest', 'nearest-up', 'zero',
            'slinear', 'quadratic', 'cubic', 'previous', or 'next'. 'zero'}
            default='linear

    filter_size: int
                A 2d-filter size to apply smoothing, a kernel of filter_size x filter_size
                is created and convoled with matrix


    method: convolution method {'conv', 'conv_nan'}, default = 'conv'
           if method='conv', conventional convolution is applied, in this case,
           'filter_size' can be of any length>1

           if  method='conv_nan', a convolution operation, that can handle NaN values is used.
              For this, filter_size should be an odd number, if even number is passed, 1 is added
              to make it odd

    clip_range: list of two [l0,l1]
              After (1) inter-exter-polation, (2) applied smotthing,(3) restoring original values
              matrix values are clipped with clip_range.
              This is only applied if at least of the clip_range values is not None.


    Returns
    -------
    XI : New Matrix, where NaN are filled, but original values are left un-changed, except clipping
    Xk : New Matrix, same as XI, except, original values are not restored,

    Example
    -------
    >>> import numpy as np
    >>> import spkit as sp
    >>> np.random.seed(seed=2)
    >>> X  = np.random.randint(0,10,[5,5])
    >>> r = 1*(np.abs(np.random.randn(5,5))<1.4).astype(float)
    >>> r[r==0]=np.nan
    >>> X_nan = X*r
    >>> X_nan

        array([[ 8.,  8.,  6.,  2., nan],
               [ 7.,  2., nan,  5.,  4.],
               [nan,  5.,  7.,  3.,  6.],
               [ 4.,  3.,  7.,  6.,  1.],
               [nan,  5.,  8.,  4., nan]])


    >>> X_filled, X_smooth = sp.fill_nans_2d(X_nan)
    >>> X_filled.round(1)

        array([[8. , 8. , 6. , 2. , 1.9],
               [7. , 2. , 4.8, 5. , 4. ],
               [4.5, 5. , 7. , 3. , 6. ],
               [4. , 3. , 7. , 6. , 1. ],
               [3.3, 5. , 8. , 4. , 0.9]])



    >>> X_smooth.round(1)

        array([[7.1, 6.4, 4.9, 3.3, 1.9],
               [5.9, 5.8, 4.8, 4.2, 3.3],
               [4.5, 4.9, 4.8, 4.9, 4. ],
               [3.8, 5.1, 5.3, 4.4, 2.6],
               [3.3, 4.9, 5.6, 3.8, 0.9]])

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
            Wx = np.ones([filter_size,filter_size])
            Xk = conv2d_nan(Xk,Wx)
        else:
            Wx = np.ones([filter_size,filter_size])/(filter_size*filter_size)
            Xk = signal.convolve2d(Xk,Wx,boundary='symm',mode='same')

    Xl = Xk*1*(np.isnan(X))
    XI = np.nansum(np.array([X,Xl]),axis=0)
    if clip_range[0] is not None or clip_range[1] is not None:
        XI = np.clip(XI, clip_range[0],clip_range[1])
    return XI, Xk

def denorm_kernel(kernel,mode='mid',keep_scale=False):
    r"""
    De-normalise 1d/2d Kernel
    -----------------------

    Example
    --------
    >>> kernel = np.ones([3,3])/9
    >>> kernel

        array([[0.11111111, 0.11111111, 0.11111111],
               [0.11111111, 0.11111111, 0.11111111],
               [0.11111111, 0.11111111, 0.11111111]])


    >>> denorm_kernel_2d(kernel)

        array([[1., 1., 1.],
               [1., 1., 1.],
               [1., 1., 1.]])


    """
    S = np.nansum(kernel) if keep_scale else 1.

    M = S/np.nanmax(np.abs(kernel))

    #assert kernel.ndim ==2 or kernel.ndim==1

    if kernel.ndim ==2:
        r,c = kernel.shape
    else:
        r = kernel.shape[0]
        c = r
    if mode=='mid':
        if r%2==1 and r==c:
            r1 = r//2
            m_value = kernel[r1+1, r1+1] if kernel.ndim ==2 else kernel[r1+1]
            #m_valu*M = 1
            if m_value!=0: M = S/m_value
        elif r%2==1 and c%2==1:
            r1,c1 = r//2, c//2
            m_value = kernel[r1+1, c1+1] if kernel.ndim ==2 else kernel[r1+1]
            if m_value!=0: M = S/m_value
    return kernel*M

def conv2d_nan(x,kernel, boundary='constant', fillvalue=np.nan, denormalise_ker=False):
    r"""
    2D Convolution with NaN values
    ------------------------------

    In conventional Convolution funtions, if any of the value in
    input x or in kernel is NaN (np.nan), then NaN values are propogated and corrupt other values too.

    To avoid this, this funtion does the convolution in same fashion as conventional
    except, it allows NaN values to exists in input, without propogating them.

    while computation, it simply ignores the NaN value, as it doen not exist, and adjust the computation
    accordingly.

    If No NaN value exist, result is same as conventional convolution


    Parameters
    ----------

    x: 2D-array with NaN values.

    kernel: a 2D kernel to use for convolution

    IMPORTANT NOTE: kernel passed should NOT be normalised. If normalised kernel is used,
    the results will be very different than conventional convolution.

    For example:
       kernel_unnorm = np.ones([3,3])
       kernel_norm = np.ones([3,3])/9

    kernel_unnorm should be passed, not kernel_norm.

    To de-normalise a kernel, used  'denorm_kernel(kernel)'
    or set denormalise_ker=True

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



    denormalise_ker: bool, default=False
                     If True, first de-normalise kernel



    Returns
    -------

    y: 2D-arrray of same size as input x with no NaN propogation

    Example
    -------
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

        array([[0.11111111, 0.11111111, 0.11111111],
               [0.11111111, 0.11111111, 0.11111111],
               [0.11111111, 0.11111111, 0.11111111]])

    >>> kernel = np.ones([3,3])
    >>> kernel

        array([[1., 1., 1.],
               [1., 1., 1.],
               [1., 1., 1.]])

    # Or use denormaliser
    >>> kernel = denorm_kernel(kernel_norm)
    >>> kernel
        array([[1., 1., 1.],
               [1., 1., 1.],
               [1., 1., 1.]])


    >>> signal.convolve2d(X_nan,kernel_norm,boundary='symm',mode='same').round(1)

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

    See Also
    --------
    conv1d_nan, fill_nans_1d, fill_nans_2d

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

def conv1d_nan(x,kernel, boundary='constant', fillvalue=np.nan, denormalise_ker=False):
    r"""
    1D Convolution with NaN values
    ------------------------------

    In conventional Convolution funtions, if any of the value in
    input x or in kernel is NaN (np.nan), then NaN values are propogated and corrupt other values too.

    To avoid this, this funtion does the convolution in same fashion as conventional
    except, it allows NaN values to exists in input, without propogating them.

    while computation, it simply ignores the NaN value, as it doen not exist, and adjust the computation
    accordingly.

    If No NaN value exist, result is same as conventional convolution


    Parameters
    ----------

    x: 1D-array with NaN values.

    kernel: a 1D kernel to use for convolution

    IMPORTANT NOTE: kernel passed should NOT be normalised. If normalised kernel is used,
    the results will be very different than conventional convolution.

    For example:
       kernel_unnorm = np.ones(9)
       kernel_norm = np.ones(9)/9

    kernel_unnorm should be passed, not kernel_norm.

    To de-normalise a kernel, used  'denorm_kernel(kernel)'
    or set denormalise_ker=True

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



    denormalise_ker: bool, default=False
                     If True, first de-normalise kernel



    Returns
    -------

    y: 1D-arrray of same size as input x with no NaN propogation

    Example
    -------
    >>> import numpy as np
    >>> import spkit as sp
    >>> from scipy import signal

    >>> N = 10
    >>> np.random.seed(seed=200)
    >>> X  = np.random.randint(0,5,N)
    >>> r = 1*(np.abs(np.random.randn(N))<1.4).astype(float)
    >>> r[r==0]=np.nan
    >>> X_nan = X*r
    >>> X_nan

        array([nan,  1.,  0.,  4.,  2.,  4.,  1.,  3.,  1.,  1.,  3.,  2.,  0.,
                1.,  3., nan,  1.,  1.,  0.,  0.,  3., nan, nan,  0.,  4.,  0.,
                3.,  4., nan, nan,  4.,  0.,  3.,  0.,  0.,  3.,  1.,  0.,  0.,
               nan,  2.,  3.,  4.,  2.,  4.,  4.,  2.,  0.,  3.,  4.,  0.,  1.,
               nan,  2.,  1.,  0., nan,  1., nan,  0.,  2.,  1.,  1.,  1., nan,
                0.,  2.,  3.,  4.,  3.,  2.,  1., nan,  1.,  0.,  3.,  0.,  3.,
                0.,  4., nan,  0.,  3.,  1., nan,  4.,  1., nan, nan,  4., nan,
                2.,  0., nan,  1.,  2.,  0.,  2.,  4.,  3.])

    >>> kernel_norm = np.ones(9)/9
    >>> kernel_norm

        array([0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111,
               0.11111111, 0.11111111, 0.11111111, 0.11111111])

    >>> kernel = np.ones(9)
    >>> kernel

        array([1., 1., 1., 1., 1., 1., 1., 1., 1.])

    # Or use denormaliser
    >>> kernel = denorm_kernel(kernel_norm)
    >>> kernel

        array([1., 1., 1., 1., 1., 1., 1., 1., 1.])


    >>> signal.convolve(X_nan,kernel_norm, method='auto',mode='same').round(1)

        array([nan, nan, nan, nan, nan, 1.9, 2.1, 2.3, 1.9, 1.8, 1.7, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, 1.2, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, 2.7, 2.9, 2.6, 2.2, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, 1.3, 1.2])

    >>> sp.conv1d_nan(X_nan,kernel).round(1)

        array([1.8, 2.2, 2. , 2.1, 2. , 1.9, 2.1, 2.3, 1.9, 1.8, 1.7, 1.8, 1.5,
               1.5, 1.4, 1. , 1.1, 1.3, 1.3, 0.8, 1.3, 1.1, 1.4, 2. , 2.3, 2.2,
               2.5, 2.1, 2.6, 2. , 2. , 2. , 1.6, 1.4, 1.2, 0.9, 1.1, 1.1, 1.6,
               1.9, 2. , 2.4, 2.6, 2.6, 2.7, 2.9, 2.6, 2.2, 2.2, 2. , 1.6, 1.4,
               1.6, 1.3, 0.8, 0.8, 1. , 1. , 0.9, 0.9, 1. , 0.9, 1. , 1.2, 1.8,
               1.9, 2. , 2. , 2.1, 2. , 2. , 2.1, 1.8, 1.6, 1.2, 1.5, 1.6, 1.4,
               1.6, 1.8, 1.6, 2.1, 1.9, 2.2, 1.8, 2.2, 2.6, 2.4, 2.2, 2.2, 1.6,
               1.8, 1.5, 1.6, 1.6, 1.8, 1.7, 2. , 2. , 2.2])

    See Also
    --------
    conv2d_nan, fill_nans_1d, fill_nans_2d

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

def gaussian_kernel(window_length, sigma_scale=2.7,sigma=None):
    r"""
    Gaussian Kernel
    --------------
    Generating Gaussian kernel of given window length and sigma.

    sigma = window_length / 6

    Parameters
    ----------
    window_length: int, length of window


    sigma_scale: float, to control the width and spread of gaussian curve

    Returns
    -------

    ker: gaussian kernel of given window


    Example
    -------
    #TODO


    See Also
    --------
    #TODO


    """
    if sigma is None: sigma = window_length / 6.
    if sigma_scale is None: sigma_scale=2.7
    t = np.linspace(-sigma_scale*sigma, sigma_scale*sigma, window_length)
    gaussian_func = lambda t, sigma: 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(t**2)/(2*sigma**2))
    ker = gaussian_func(t, sigma)
    return ker / np.sum(ker)

def friedrichs_mollifier_kernel(window_size, s=1, p=2, r=0.999):

    """
    Mollifier: Kurt Otto Friedrichs
    --------------------------------
    Generalized function

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
       Spread of the middle width, heigher the value of s, narrower the width

    p: scaler, p>0, default=2,
       Order of flateness of the peak at the top,
       p=2, smoother, p=1, triangulare type
       Higher it is, more flat the peak.

    r: float, 0<r<1, default=0.999,
        it is used to compute x = [-r, r]
        recommonded to keep it r=0.999


    Returns
    -------
    ker_mol: mollifier kernel

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Mollifier
    [2] https://en.wikipedia.org/wiki/Kurt_Otto_Friedrichs

    See also
    --------
    gaussian_kernel
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

def conv1d_fb(x, kernel, iterations=1,mode='same', return_full=False):
    r"""
    1D Forward-Backward-Convolution (ConvFB)
    ----------------------------------------

    Parameters
    ----------
    x: (np.array of floats, 1xN) signal
    kernel: kernel to be used
    iterations >=1, applying conv_fb recursively
    return_full: if true, it will return 3 times of length of signal

    Returns
    -------
    y: output signal


    Example
    -------
    #TODO

    See Also
    --------
    #TODO

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

def filter_smooth_sGolay(X, window_length=127, polyorder=3, deriv=0, delta=1.0, mode='interp', cval=0.0):
    r"""
    Smoothing filter using Savitzky-Golay filter
    -------------------------------------------------------------------
    Savitzky-Golay filter for multi-channels signal: From Scipy library

    input
    -----
    X : (vecctor) input signal single channel (n,) or multi-channel, channel axis should be 1 shape ~ (n,ch)
    window_length: should be an odd number
    others input parameters as same as in scipy.signal.savgol_filter
              :(polyorder=3, deriv=0, delta=1.0, mode='interp', cval=0.0)

    output
    ------
    Y : corrected signal
    """

    if np.ndim(X)>1:
        Y = savgol_filter(X, window_length, polyorder,deriv=deriv, delta=delta, axis=0, mode=mode, cval=cval)
    else:
        Y = savgol_filter(X, window_length, polyorder,deriv=deriv, delta=delta, axis=-1, mode=mode, cval=cval)
    return Y

def filter_smooth_gauss(X, window_length=11, sigma_scale=2.7,iterations=1,mode='same'):
    r"""
    Smoothing filter using Gaussian Kernel and 1d-ConvFB
    -----------------------------------------------------
    sigma : sigma for gaussian kernel, if None, sigma=window_length/6

    input
    -----
    X : (vecctor) input signal single channel (n,) or multi-channel, channel axis should be 1 shape ~ (n,ch)
    window_length: int >1, length of gaussian kernel
    sigma_scale: float, deafult=2.7
                 To control width/spread of gauss

    iterations: int, >=1, repeating gaussian smoothing iterations times
    mode: convolution mode in {'same','valid','full'}, 'same make sense'



    output
    ------
    Y : Smoothed signal
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
    r"""
    Smoothing filter using Mollifier kernel and 1d-ConvFB
    ---------------------------------------------------
    Mollifier: Kurt Otto Friedrichs
    --------------------------------
    Generalized function

    f(x) =  exp(-s/(1-|x|**p))    for |x|<1,   x \in [-r, r]

    Convolving with a mollifier, signals's sharp features are smoothed, while still remaining close
    to the original nonsmooth (generalized) signals.

    Intuitively, given a function which is rather irregular, by convolving it with a mollifier the function gets "mollified".

    This function is infinitely differentiable, non analytic with vanishing derivative for |x| = 1,
    can be therefore used as mollifier as described in [1]. This is a positive and symmetric mollifier.[15]


    input
    -----
    X : (vecctor) input signal single channel (n,) or multi-channel, channel axis should be 1 shape ~ (n,ch)
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


    output
    ------
    Y : Mollified signal, of same shape as input X
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
    r"""
    Smoothing/Sharpening using given kernel and 1d-ConvFB
    -----------------------------------------------------
    Smoothing/Sharpening depends on kernel

    input
    -----
    X : (vecctor) input signal single channel (n,) or multi-channel, channel axis should be 1 shape ~ (n,ch)
    kernel :  custamised kernel for

    iterations: int, >=1, repeating gaussian smoothing iterations times
    mode: convolution mode in {'same','valid','full'}, 'same make sense'

    output
    ------
    Y : procesed signal

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

def signal_diff(x,method='fdiff',sg_window=101,sg_polyorder=1,gauss_window=0,gauss_itr=1,gauss_sigscale=2.7,dt=1):

    r"""
    Derivative of a signal
    -----------------------


    differentiaion methods

    'fdiff':  Finite differentiation
        intermediate values
        df(x) =  [f(x+d) - f(x-d)]/2  when x+d and x-d in D (Domain)

        boundries
        df(x) = f(x+d)-f(x)   when x+d in D, but x-d not in D  | first point x=0
        df(0) = f(d)-f(0)     when x+d in D, but x-d not in D  | first point x=0


        df(x) = f(x)-f(x-d)   when x-d in D, but x+d not in D  | last point x=l
        df(l) = f(l)-f(l=d)   when x-d in D, but x+d not in D  | last point x=l


    'fgrad':  Finite differentiation using finite gradient (np.gradient)



     for following methods:
        parameters for Savitzky-Golay filter (sg_window,sg_polyorder)
        parameters for gaussian kernal+ConvFB (gauss_window,gauss_itr), applied only if gauss_window>0


    'sgdiff': Computing derivetive of signal using Savitzky-Golay filter, then with gaussian kernal using Forward-Backward-Convolution (ConvFB).


    'sgdrift_diff': First remove drift using Savitzky-Golay filter, then apply gaussian smoothing apply fgrad.

    'sgsmooth_diff': First smooth the input signal x with Savitzky-Golay filter then apply then apply gaussian smoothing (unnecessary)
                     then apply fgrad

    'gauss_diff': First smooth the input signal x using gaussian smoothing
                     then apply fgrad


    Parameters
    ----------
    x: 1d-array signal

    method: method to compute derivative, one of the above

    (sg_window,sg_polyorder,gauss_window,gauss_itr): Savitzky-Golay filter, Gaussian smoothing using Forward-Backward-Convolution (ConvFB)


    Returns
    -------
    dx: derivative of signal


    Example
    -------
    #TODO

    See Also
    --------
    filter_smooth_sGolay, filter_smooth_gauss, filter_smooth_kernel

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

def get_activation_time(x,fs=1,method='min_dvdt',gradient_method='fdiff',sg_window=11,sg_polyorder=3,gauss_window=0,gauss_itr=1,**kwargs):
    r"""
    Get Activation Time based on Gradient
    -------------------------------------

    Activation Time in cardiac electrograms refered to as time at which depolarisation of cells/tissues/heart occures.

    For biological signals (e.g. cardiac electorgram), an activation time in signal is reflected by maximum negative deflection,
    which is equal to min-dvdt, if signal is a volatge signal and function of time x = v(t)
    However, simply computing derivative of signal is sometime misleads the activation time locatio, due to noise, so derivative of
    a given signal has be computed after pre-processing



    Parameters
    ----------
    x   : 1d-array of signal
    fs  : sampling frequency, default fs=1, in case only interested in loc

    method : Method to compute activation time, one of ("max_dvdt", "min_dvdt", "max_abs_dvdt")


    gradient_method: Method to compute gradient of signal
                    one of ("fdiff", "fgrad", "npdiff","sgdiff","sgdrift_diff","sgsmooth_diff", "gauss_diff")
                    check help(signal_diff) from sp.signal_diff

    Parameters for gradient_method:
    used if gradient_method in one of ("sgdiff","sgdrift_diff","sgsmooth_diff", "gauss_diff")
        sg_window: sgolay-filter's window length
        sg_polyorder: sgolay-filter's polynomial order
        gauss_window: window size of gaussian kernel for smoothing,
        check help(signal_diff) from sp.signal_diff

    Returns
    -------
    at : activation time (ms)
    loc: index
    mag: magnitude of deflection at loc
    dx : derivative of signal x

    Example
    -------
    #TODO

    See Also
    --------
    get_repolarisation_time, activation_time_loc, activation_repol_time_loc


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
    r"""
    Get Repolarisation Time based on Gradient
    -------------------------------------

    In contrast to 'Activation Time' in cardiac electrograms, Repolarisation Time, also refered as Recovery Time,
    indicates a time at which repolarisation of cells/tissues/heart occures.

    Repolarisation time in signal is again a reflected by maximum deflection (mostly negative), after activation occures.
    That is equal to min-dvdt, after activation time, if signal is a volatge signal and function of time x = v(t)

    However, Repolarisation Time is very hard to detect reliably, due to very small electrogram, which is mostly lost in noise.



    Parameters
    ----------
    x   : 1d-array of signal
    fs  : sampling frequency

    at_loc  : int,  location (index) of activation time, this is used any deflections before at_loc

    t_range: list of [t0 (ms),t1 (ms)]
             range of time to restrict the search of repolarisation time.
             Search start from (1000*at_loc/fs + t0) to (1000*at_loc/fs + t1) ms of given signal

             during t0 ms to t1 ms
             if t_range=[None,None], whole input signal after at_loc is considered for search
             if t_range=[t0,None], excluding signal before 1000*at_loc/fs + t0 ms
             if t_range=[None,t1], limiting search to 1000*at_loc/fs to t1 ms

        NOTE: It is recommonded to add a small time gap after at_loc,  t_range = [0.5, None] or [1, None]
              As next max deflection is almost always very near (within a few sample) to at_loc, which will leads to rt_loc very close to at_loc

    method : Method to compute Repolarisation Time, one of ("max_dvdt", "min_dvdt", "max_abs_dvdt")
             Some literation suggests to use max_dvdt instead of min_dvdt, but mostly agree on min_dvdt


    gradient_method: Method to compute gradient of signal
                    one of ("fdiff", "fgrad", "npdiff","sgdiff","sgdrift_diff","sgsmooth_diff", "gauss_diff")
                    check help(signal_diff) from sp.signal_diff

    Parameters for gradient_method:
    used if gradient_method in one of ("sgdiff","sgdrift_diff","sgsmooth_diff", "gauss_diff")
        sg_window: sgolay-filter's window length
        sg_polyorder: sgolay-filter's polynomial order
        gauss_window: window size of gaussian kernel for smoothing,
        check help(signal_diff) from sp.signal_diff


    Returns
    -------
    rt : activation time (ms)
    loc: index
    mag: magnitude of deflection
    dx : derivative of signal x


    Example
    -------
    #TODO

    See Also
    --------
    get_activation_time, activation_time_loc, activation_repol_time_loc

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

def agg_angles(thetas, agg='mean'):
    """
    Aggregating Directional Arrows (thetas)
    -----------------------------------

    Converting polar coordinates to cartesian aaverging there and converting back to polar

    Parameters
    ----------
    thetas: list/array of theta (angles)
    agg   : str {'mean', 'median'}, default='mean'
           method to use averaging, mean or median

    Returns
    -------

    theta_avg: scalar, aggregated angle value

    Example
    -------
    >>> import numpy as np
    >>> import spkit as sp

    # 1D Example
    >>> np.random.seed(2)
    >>> N=8
    >>> # +1 -1, direction, top left
    >>> thetas= np.arctan2(np.random.randn(N)+1, np.random.randn(N)-1)
    >>> thetas_avg = sp.agg_angles(thetas, agg='mean')
    >>> thetas_avg

        2.9525995459333507


    See Also
    --------
    show_compass


    """
    if agg=='median':
        xy = np.nanmedian(np.c_[np.cos(thetas),np.sin(thetas)],axis=0)
    else:
        xy = np.nanmean(np.c_[np.cos(thetas),np.sin(thetas)],axis=0)

    theta_avg = np.arctan2(xy[1],xy[0])

    return theta_avg

def show_compass(Ax_theta,Ax_bad=[],arr_agg='mean',figsize=(10,6),all_arrow_prop =dict(facecolor='C0',lw=1,zorder=10,alpha=0.2,edgecolor='C0',width=0.05),
                                                 avg_arrow_prop =dict(facecolor='C3',lw=4,zorder=100,edgecolor='C3',width=0.045), title='CV'):
    """
    Display Compass with Arrows and aggregated Arrow
    ------------------------------------------------

    Plot two compasses
      (1) With all the arrows of angles as provided in Ax_theta, along with averaged direction
      (2) Only if Ax_bad is provided;
          With arrows excluding one indicated as bad (np.nan, NaN) in Ax_bad
          along with averaged direction


    Parameters
    ----------
    Ax_theta: 2D-array (e.g. MEA Feature matrix) of Angles of directions
              1D-array should work as well.
              It can include NaN (np.nan) values, which will be ignored

    Ax_bad  : Same size array as Ax_theta, with 1 and np.nan (NaNs) value
              1 for good channel, NaN for bad channel
              default = [], in which case only one plot is shown


    arr_agg: str {'mean', 'media'}, default='mean'
             how to aggregate angles


    Returns
    -------

    Ax_theta_avg   : Aggregated Angle, including all the values

    Ax_theta_avg_bd: Aggregated Angle, excluding bad channels, indicated by NaN in Ax_bad,
                     if Ax_bad is provided, else same as Ax_theta_avg

    Examples
    -------
    >>> import numpy as np
    >>> import spkit as sp

    # 1D Example
    >>> np.random.seed(2)
    >>> N=8
    >>> # +1 -1, direction, top left
    >>> Ax_theta= np.arctan2(np.random.randn(N)+1, np.random.randn(N)-1)
    >>> Ax_bad   = 1*(np.random.rand(N)>0.4).astype(float)
    >>> Ax_bad[Ax_bad==0]=np.nan
    >>> Ax_bad

        array([ 1.,  1.,  1., nan,  1., nan,  1., nan])

    >>> sp.show_compass(Ax_theta, Ax_bad,arr_agg='mean',title='D')

    # or just one plot
    >>> sp.show_compass(Ax_theta,title='D')

    # 2D Example
    >>> np.random.seed(100)
    >>> Ax_theta= np.arctan2(np.random.randn(8,8)+1, np.random.randn(8,8)+1)
    >>> Ax_bad   = 1*(np.random.rand(8,8)>0.5).astype(float)
    >>> Ax_bad[Ax_bad==0]=np.nan
    >>> Ax_bad

        array([[ 1.,  1., nan, nan,  1.,  1., nan, nan],
               [nan, nan,  1.,  1.,  1., nan, nan,  1.],
               [ 1.,  1., nan, nan, nan,  1., nan,  1.],
               [ 1., nan, nan,  1.,  1.,  1., nan,  1.],
               [ 1.,  1., nan, nan, nan,  1., nan,  1.],
               [ 1., nan, nan,  1., nan, nan,  1.,  1.],
               [nan,  1., nan,  1.,  1.,  1., nan, nan],
               [nan,  1., nan,  1.,  1.,  1.,  1., nan]])

    >>> sp.show_compass(Ax_theta, Ax_bad,arr_agg='mean',title='D')
    >>> sp.show_compass(Ax_theta,arr_agg='median',title='D')

    See Also
    --------

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
    '''
    arr_prop = dict(scale=None,headwidth=None, headlength=None, headaxislength=None, minshaft=None, minlength=None,color='w')

    '''
    r"""

    Displaying Directional flow map of a 2D grid
    --------------------------------------------

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

    Example
    -------
    >>> import numpy as np
    >>> import spkit as sp
    >>> np.random.seed(100)
    >>> N,M = 8,8
    >>> Ax_theta= np.arctan2(np.random.randn(N,M)+1, np.random.randn(N,M)+1)
    >>> Ax_bad   = 1*(np.random.rand(N,M)>0.1).astype(float)
    >>> Ax_bad[Ax_bad==0]=np.nan
    >>> XY = sp.direction_flow_map(Ax_theta)

    # with missing values

    >>> XY = sp.direction_flow_map(Ax_theta*Ax_bad,Ax_theta*0 )

    # with upsampling by 2

    >>> XY = sp.direction_flow_map(Ax_theta.copy(),upsample=2)

    # with upsampling by 5
    >>> XY = sp.direction_flow_map(Ax_theta.copy(),upsample=5,arr_prop =dict(color='w',scale=30))

    # Non-square Grid

    >>> np.random.seed(10)
    >>> N,M = 10,20
    >>> Ax_theta= np.arctan2(np.random.randn(N,M)+1, np.random.randn(N,M)+1)
    >>> Ax_bad   = 1*(np.random.rand(N,M)>0.1).astype(float)
    >>> Ax_bad[Ax_bad==0]=np.nan

    >>> XY = sp.direction_flow_map(Ax_theta)
    >>> XY = sp.direction_flow_map(Ax_theta*Ax_bad, Ax_theta*0)

    See Also
    --------

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

def phase_map(X, add_sig=False):
    r"""
    Phase Mapping of multi channel signals X
    input  :  X : (n,ch) shape= (samples, n_channels)
    output :  PM : (n,ch) shape= (samples, n_channels)

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
    r"""
    Clean Phase:
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

def amplitude_equalizer(x,add_sig=False,amp_shift=1,am_mult=0, cleaning_phase=True, w=1,mollier=True, window_length=51, iterations=1,s=None,p=None,r=None):
    r"""
    Equalizing the Amplitude variation to Enhance the Phase and the Frequency of a signal
    """
    xa = signal.hilbert(x)

    if add_sig: xa = x + 1j*xa

    xp = np.arctan2(xa.imag,xa.real)
    xp_clean = xp.copy()
    if cleaning_phase and w>0:
        xp_clean = clean_phase(xp_clean,w=w,thr=-np.pi/2, low=-np.pi, high=np.pi)

    xm = np.abs(xa)

    xr = (xm*am_mult+amp_shift)*np.exp(1j*xp_clean).real
    if mollier:
        xr = filter_smooth_mollifier(xr, window_length=window_length,iterations=iterations,r=r,s=s,p=p,mode='same')

    xra = signal.hilbert(xr)
    xpr = np.arctan2(xra.imag,xra.real)
    return xr,xp,xp_clean,xpr

def phase_map_reconstruction(X,add_sig=False,amp_shift=1,am_mult=0,cleaning_phase=True, w=1,mollier=True, window_length=51, iterations=1,verbose=False):
    r"""
    Phase Mapping of multi channel signals X along with reconstruction of signal by amplitude substraction
    input
    -----
    X : (n,ch) shape= (samples, n_channels)
    add_sig, bool, default=False: if to add signal to analytical siganl
    amp_shift, float, shift of amplitude while amplitude substruction
    am_mult=0, float, reduction of original amplitude of signal, 0 mean complete reduction

    output
    ------
    XP : (n,ch) shape= (samples, n_channels) Instantenious Phase
    XR : (n,ch) shape= (samples, n_channels) Reconstructed Signal

    """
    XR, XP = [],[]
    if X.ndim==1: X = X[:,None]
    for i in range(X.shape[1]):
        if verbose: ProgBar(i,X.shape[1])
        xi = X[:,i].copy()

        xr,xp,xp_clean,xpr = amplitude_equalizer(xi,add_sig=add_sig,amp_shift=amp_shift,am_mult=am_mult,
                               cleaning_phase=cleaning_phase, w=w,mollier=mollier, window_length=window_length, iterations=iterations)
        XR.append(xr)
        XP.append(xp)
    XP = np.array(XP).T
    XR = np.array(XR).T
    XP = np.squeeze(XP)
    XR = np.squeeze(XR)
    return XP,XR

def _dominent_freq(x,fs,method='welch',nfft=None,nperseg=None,exclude_lower_fr=None,window='hann',**kwargs):
    r"""
    Dominent Frequency Analysis
    """
    if method=='welch':
        fq, Px = signal.welch(x,fs,nperseg=nperseg,nfft=nfft,window=window,**kwargs)
    else:
        fq, Px = signal.periodogram(x,fs,nfft=nfft,window=window,**kwargs)
    if exclude_lower_fr is not None:
        Px = Px[np.where(fq>exclude_lower_fr)]
        fq = fq[np.where(fq>exclude_lower_fr)]
    dfq = fq[np.argmax(Px)]
    return dfq

def dominent_freq(X,fs,method='welch',nfft=None,nperseg=None,exclude_lower_fr=None,window='hann',**kwargs):
    r"""
    Dominent Frequency Analysis
    """
    DF = []

    if X.ndim>1:
        for i in range(X.shape[1]):
            xi = X[:,i].copy()
            dfi =  _dominent_freq(xi,fs=fs,method=method,nfft=nfft,nperseg=nperseg,exclude_lower_fr=exclude_lower_fr,window=window,**kwargs)
            DF.append(dfi)
    else:
        DF =  _dominent_freq(X,fs,method=method,nfft=nfft,nperseg=nperseg,exclude_lower_fr=exclude_lower_fr,window=window,**kwargs)
    return DF

def dominent_freq_win(X,fs,win_len=100,overlap=None,method='welch',nfft=None,nperseg=None,exclude_lower_fr=None,window='hann',use_joblib=False,verbose=1,**kwargs):
    r"""
    Dominent Frequency Analysis Window-wise
    """

    if overlap is None: overlap=win_len//2
    overlap_shift = win_len-overlap
    DF_temp =[]

    verbose_win=False
    if X.ndim==1:
        X = X[:,None]
        verbose=False
        verbose_win=True
    for i in range(X.shape[1]):
        if verbose: ProgBar(i,X.shape[1],style=2,selfTerminate=False)
        xi = X[:,i].copy()
        if use_joblib:
            win = []
            win_i = np.arange(win_len)
            while win_i[-1]<xi.shape[0]:
                win.append(win_i.copy())
                win_i+=overlap_shift

            dfq_temp = Parallel(n_jobs=-1)( delayed(_dominent_freq) (xi[win[kk]],fs,method=method,nfft=nfft,nperseg=nperseg,
                                                                 exclude_lower_fr=exclude_lower_fr,window=window) for kk in range(len(win)))
        else:
            win = np.arange(win_len)
            dfq_temp =[]
            while win[-1]<xi.shape[0]:
                if verbose_win: ProgBar(win[-1],xi.shape[0],style=2,selfTerminate=False)
                dfq_i = _dominent_freq(xi[win],fs=fs,method=method,nfft=nfft,nperseg=nperseg,exclude_lower_fr=exclude_lower_fr,window=window)
                dfq_temp.append(dfq_i)
                win+=overlap_shift

            DF_temp.append(dfq_temp)

    return np.array(DF_temp)

def mean_minSE(x,y,W=5,show=False,compare_mse=True,plot_log=False, esp=1e-3,show_legend=True):
    r"""
    Mean of Minimum Squared Error

    MminSE = 1/K \sum_0^K-1 min( (x(k) - y(k-t))**2  )    for t = -W to W

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
    r"""
    Minimum Mean Squared Error under temporal shift

    minMSE = min [ 1/K \sum_0^K-1 ( (x(k) - y(k-t))**2  ) ]    for t = -W to W

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

def create_signal_1d(n=100,seed=None,circular=False,bipolar=True,sg_polyorder=1,sg_nwin=10,max_dxdt=0.1, iterations=2,max_itr=10):

    r""" Generate 1D arbitary signal

    sg_wid: window_length = (n//sg_wid) for sg_filter
    sg_polyorder: polyorder for sg_filter

    """
    np.random.seed(seed)
    window_length = (n//sg_nwin)
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
            _,xm = sp.filterDC_sGolay(xm,window_length=window_length,polyorder=sg_polyorder,return_background=True)
            #print('max_dxdt:',np.max(np.abs(np.diff(xm))))
    else:
        itr=0
        while np.max(np.abs(np.diff(xm)))>max_dxdt:
            _,xm = sp.filterDC_sGolay(xm,window_length=window_length,polyorder=sg_polyorder,return_background=True)
            itr+=1
            if itr>max_itr:break
    np.random.seed(None)
    xm = xm[window_length//2:-window_length//2]
    #if circular: xm = xm[1*window_length:-1*window_length]
    xm -= xm.min()
    xm /= xm.max()
    if bipolar: xm = 2*(xm - 0.5)
    return xm

def create_signal_2d(n=100,sg_winlen=11,sg_polyorder=1,iterations=2,max_dxdt=0.1,max_itr=10,seed=None):

    r""" Generate 2D arbitary signal/image patch"""

    np.random.seed(seed)
    X = np.random.rand(n,n)

    for i in range(n):
        if iterations is not None:
            for _ in range(iterations):
                #win = np.random.randint(3,window_length)
                #if win%2==0: win+=1
                _,X[i] = filterDC_sGolay(X[i],window_length=sg_winlen,polyorder=sg_polyorder,return_background=True)
        else:
            itr=0
            while np.max(np.abs(np.diff(X[i])))>max_dxdt:
                _,X[i] = filterDC_sGolay(X[i],window_length=sg_winlen,polyorder=sg_polyorder,return_background=True)
                itr+=1
                if itr>max_itr: break

    X -= X.min()
    X /=X.max()
    for i in range(n):
        if iterations is not None:
            for _ in range(iterations):
                #win = np.random.randint(3,window_length)
                #if win%2==0: win+=1
                _,X[:,i] = filterDC_sGolay(X[:,i],window_length=sg_winlen,polyorder=sg_polyorder,return_background=True)
        else:
            itr=0
            while np.max(np.abs(np.diff(X[:,i])))>max_dxdt:
                _,X[:,i] = filterDC_sGolay(X[:,i],window_length=sg_winlen,polyorder=sg_polyorder,return_background=True)
                itr+=1
                if itr>max_itr: break

    np.random.seed(None)
    X -= X.min()
    X /=X.max()
    return X

def spatial_filter_dist(X,V,r=0.1,ftype='mean',exclude_self=False,default_value=np.nan,esp=0):
    r"""
    Spatial Filter based on Distance
    --------------------------------

    Given values X corresponding to spatial locations V, in n-Dimentional Space
         applying a filter 'mean' or 'median' of radius r

    Parameters
    ----------

    X: 1d-array of size m, values of points, shape=(m,), can include NaN, if missing
    V: 2d-array of size (m,n), locations of points

    ftype: str, or callable function:, default = mean
           filter type, str = {'mean', 'median' or any np.fun or a callable function}
           All functions should be able to handle NaN values

    exclude_self: bool, default=False,
                 If True, while estimating new value at location i, self value is excluded

    default_value: scalar, default=np.nan
                  If no value is calculated, deafult value is used

    Return
    ------
    Y: Filtered values

    """

    assert X.shape[0]==V.shape[0]

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
    for i,p in enumerate(V):
        dist = np.sqrt(np.sum((V-p)**2,1))

        if exclude_self:
            xi = X[ (dist<r) & (dist>esp)].copy()
        else:
            xi  = X[dist<r].copy()
        if len(xi) and len(xi[~np.isnan(xi)]):
            Y[i] = FILTER_(xi)
    return Y

def spatial_filter_adj(X,AdjM,ftype='mean',exclude_self=False,default_value=np.nan):

    r"""
    Spatial Filter based on Connection Matrix: AdjM
    -----------------------------------------------


    Given values X corresponding to spatial locations V, whose adjacency matrix is given as AdjM,
         applying filter 'mean' or 'median' etc


    Parameters
    ----------

    X    : 1d-array of size m, values of points, shape=(m,), can include NaN, if missing
    AdjM : 2d-array of size (m,m), Adjacency matrix

    ftype: str, or callable function:, default = mean
           filter type, str = {'mean', 'median' or any np.fun or a callable function}
           All functions should be able to handle NaN values

    exclude_self: bool, default=False,
                 If True, while estimating new value at location i, self value is excluded

    default_value: scalar, default=np.nan
                  If no value is calculated, deafult value is used

    Return
    ------
    Y: Filtered values
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
