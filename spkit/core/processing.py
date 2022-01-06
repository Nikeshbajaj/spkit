'''
Basic signal processing methods
--------------------------------
Author @ Nikesh Bajaj
updated on Date: 26 Sep 2021
Version : 0.0.4
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
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
from scipy import stats
from copy import deepcopy
from .infotheory import entropy
import pywt as wt

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
    if np.ndim(X)>1:
        Xm = savgol_filter(X, window_length, polyorder,deriv=deriv, delta=delta, axis=0, mode=mode, cval=cval)
    else:
        Xm = savgol_filter(X, window_length, polyorder,deriv=deriv, delta=delta, axis=-1, mode=mode, cval=cval)
    Y = X - Xm
    if return_background: return Y, Xm
    return Y

def filter_X(X,fs=128.0,band =[0.5],btype='highpass',order=5,ftype='filtfilt',verbose=1,use_joblib=False):
    '''
    Buttorworth filtering -  basic filtering
    ---------------------
    X : (vecctor) input signal single channel (n,) or multi-channel, channel axis should be 1 shape ~ (n,ch)

    band: cut of frequency, for lowpass and highpass, band is list of one, for bandpass list of two numbers
    btype: filter type
    order: order of filter
    ftype: filtering approach type, filtfilt or lfilter,
         : lfilter is causal filter, which introduces delaye, filtfilt does not introduce any delay, but it is non-causal filtering
    Xf: filtered signal of same size as X
    '''
    if verbose: print(X.shape, 'channels axis = 1')
    b,a = butter(order,np.array(band)/(0.5*fs),btype=btype)
    if ftype=='lfilter':
        if np.ndim(X)>1:
            if use_joblib:
                try:
                    Xf  = np.array(Parallel(n_jobs=-1)(delayed(lfilter)(b,a,X[:,i]) for i in range(X.shape[1]))).T
                except:
                    print('joblib paraller failed computing with loops- turn off --> use_joblib=False')
                    Xf  = np.array([lfilter(b,a,X[:,i]) for i in range(X.shape[1])]).T
            else:
                Xf  = np.array([lfilter(b,a,X[:,i]) for i in range(X.shape[1])]).T
        else:
            Xf  = lfilter(b,a,X)
    elif ftype=='filtfilt':
        if np.ndim(X)>1:
            if use_joblib:
                try:
                    Xf  = np.array(Parallel(n_jobs=-1)(delayed(filtfilt)(b,a,X[:,i]) for i in range(X.shape[1]))).T
                except:
                    print('joblib paraller failed computing with loops- turn off --> use_joblib=False')
                    Xf  = np.array([filtfilt(b,a,X[:,i]) for i in range(X.shape[1])]).T
            else:
                Xf  = np.array([filtfilt(b,a,X[:,i]) for i in range(X.shape[1])]).T
        else:
            Xf  = filtfilt(b,a,X)
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
    iqr = stats.iqr(x[~np.isnan(x)])
    kur = stats.kurtosis(x,nan_policy='omit')
    skw = stats.skew(x[~np.isnan(x)])
    if detail_level==2:
        return np.r_[mn,sd,md,min0,max0,n,q25,q75,iqr,kur,skw], stats_names[:11]

    gmn = stats.gmean(x[~np.isnan(x)])
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

# def Mu_law(x,Mu=255,encoding=True):
#     '''
#     Ref: https://en.wikipedia.org/wiki/M-law_algorithm
#     '''
#     assert np.max(np.abs(x))<=1
#
#     if encoding:
#         #Companding ~ compression ~ encoding
#         y = np.sign(x)*np.log(1 + Mu*np.abs(x))/np.log(1+Mu)
#
#     else:
#         #Expanding ~ uncompression/expension ~ decoding
#         y = np.sign(x)*((1 + Mu)**np.abs(x) - 1)/Mu
#
#     return y
#
# def A_law(x,A=255,encoding=True):
#     '''
#     Ref: https://en.wikipedia.org/wiki/A-law_algorithm
#     '''
#     assert np.max(np.abs(x))<=1
#
#     y = np.zeros_like(x)
#
#     if encoding:
#         #Companding ~ compression ~ encoding
#         idx = np.abs(x)<1/A
#         y[idx]  = A*np.abs(x[idx])
#         y[~idx] = 1 + np.log(A*np.abs(x[~idx]))
#         y /= (1 + np.log(A))
#     else:
#         #Expanding ~ uncompression/expension ~ decoding
#         idx = np.abs(x)<(1/(1+np.log(A)))
#         y[idx]   = np.abs(x[idx])*(1+np.log(A))
#         y[~idx]  = np.exp(-1+np.abs(x[~idx])*(1+np.log(A)))
#         y /= A
#
#     y *= np.sign(x)
#
#     return y
#

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
        r = stats.iqr(w)
        q1 = np.quantile(w,IPR[0])
        q3 = np.quantile(w,IPR[1])
        #assert r ==q3-q1
        theta_u = q3 + k*r
        theta_l = q1 - k*r
    return theta_l, theta_u

def wavelet_filtering(x,wv='db3',threshold='optimal',filter_out_below=True,k=1.5,mode='elim',show=False,wpd_mode='symmetric',
        wpd_maxlevel=None,packetwise=False,WPD=True,lvl=[],verbose=False,fs=128.0,sf=1,IPR=[0.25,0.75]):
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
        plt.figure(figsize=(11,6))
        plt.subplot(211)
        plt.plot(WR,alpha=0.8,label='Coef.',color='C0')
        plt.ylabel('Wavelete Coefficients')
        ytiW =[np.min(WR),np.max(WR)]
        #print('maxlevel :',wp.maxlevel)
        if WPD: wr = [wp[node.path].data for node in wp.get_level(wp.maxlevel, 'natural') ]
        WRi = np.hstack(wr)
        plt.plot(WRi,color='C3',alpha=0.9,label='Filtered Coff.')
        ki = 0
        for i in range(len(wr)):
            ki+=len(wr[i])
            plt.axvline(ki,color='r',ls='-',lw=1)
        ytiW = ytiW+[np.min(WRi),np.max(WRi)]
        if not(packetwise):
            ytiW = ytiW+[theta_l, theta_u]
        plt.yticks(ytiW)
        plt.grid()
        plt.legend()
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
        plt.legend()
        plt.grid()
        plt.show()
    return xR

def wavelet_filtering_win(x,winsize=128,wv='db3',threshold='optimal',below=True,k=1.5,mode='elim',wpd_mode='symmetric',
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

        xr = wavelet_filtering(xi,wv=wv,threshold=threshold,below=below,k=k,mode=mode,wpd_mode=wpd_mode,wpd_maxlevel=wpd_maxlevel,
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
             fs=128,plot=True,pad=True,verbose=0, plottype='abs'):
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
        plt.figure(figsize=(15,8))
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
