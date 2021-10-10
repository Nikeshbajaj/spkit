import numpy as np
import matplotlib.pyplot as plt
import scipy, spkit, copy
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
from scipy import stats
from copy import deepcopy

def filterDC(x,alpha=256):
    b = x[0]
    xf = np.zeros(len(x))
    for i in range(len(x)):
        b = ((alpha - 1) * b + x[i]) / alpha
        xf[i] = x[i]-b
    return xf

def filterDC_X(X,alpha=256,return_background=True):
    B = X[0]
    if return_background:Bg = np.zeros_like(X)
    Xf = np.zeros_like(X)

    for i in range(X.shape[0]):
        B = ((alpha - 1) * B + X[i]) / alpha
        Xf[i] = X[i]-B
        if return_background: Bg[i]= copy.copy(B)
    if return_background: return Xf, Bg
    return Xf

def filterDC_sGolay(X, window_length=127, polyorder=3, deriv=0, delta=1.0, mode='interp', cval=0.0):
    '''
    Savitzky-Golay filter for multi-channels signal: From Scipy library

    X: input multichannel signal - shape (n,ch)
     : for single channel signal, use X[:,None] to make it two dimensional signal.
    window_length: should be an odd number
    others input parameters as same as in scipy.signal.savgol_filter
    '''
    Xm = np.array([savgol_filter(x, window_length, polyorder,deriv=deriv, delta=delta, axis=-1, mode=mode, cval=cval) for x in X])
    Xf = X - Xm
    return Xf, Xm

def filter_X(X,band =[0.5],btype='highpass',order=5,fs=128.0,ftype='filtfilt',verbose=1):
    '''
    Buttorworth filtering -  basic filtering
    ---------------------
    X: input multichannel signal - shape (n,ch)
     : for single channel signal, use X[:,None] to make it two dimensional signal.
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
        Xf  = np.array(Parallel(n_jobs=-1)(delayed(lfilter)(b,a,X[:,i]) for i in range(X.shape[1])))
    elif ftype=='filtfilt':
        Xf  = np.array(Parallel(n_jobs=-1)(delayed(filtfilt)(b,a,X[:,i]) for i in range(X.shape[1])))
    return Xf

def getStats(x,detail_level=1,return_names=False):
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
    entropy = spkit.entropy(x[~np.isnan(x)])
    names =['mean','sd','median','min','max','n','q25','q75','iqr','kur','skw','gmean','entropy']
    return np.r_[mn,sd,md,min0,max0,n,q25,q75,iqr,kur,skw,gmn,entropy], stats_names

def RhythmicDecomposition(E,fs=128.0,order=5,method='welch',win='hann',Sum=True,Mean=False,SD =False,
                    scaling='density', average='mean',detrend='constant',nperseg=None, noverlap=None,fBands=None):
    #average :  method to average the periodograms, mean or median
    # Delta, Theta, Alpha, Beta, Gamma1, Gamma2
    if fBands is None:
        fBands =[[4],[4,8],[8,14],[14,30],[30,47],[47]]
        #delta=[0.2-4] else filter is unstable-------------------------UPDATED 19feb2019

    Px = np.zeros([len(fBands),E.shape[1]])
    Pm = np.zeros([len(fBands),E.shape[1]])
    Pd = np.zeros([len(fBands),E.shape[1]])
    if Sum or Mean or SD:
        k=0
        for freqs in fBands:
            #print(np.array(freqs)/(0.5*fs))
            btype='bandpass'
            if len(freqs)==1:
                btype='lowpass' if freqs[0]==4 else 'highpass'
                b,a = butter(order,np.array(freqs[0])/(0.5*fs),btype=btype)
            else:
                b,a = butter(order,np.array(freqs)/(0.5*fs),btype=btype)

            #b,a = butter(order,np.array(freqs)/(0.5*fs),btype='bandpass')
            B = np.array(Parallel(n_jobs=-1)(delayed(lfilter)(b,a,E[:,i]) for i in range(E.shape[1])))
            P = np.array(Parallel(n_jobs=-1)(delayed(Periodogram)(B[i,:],fs=fs,method=method,win=win,scaling=scaling,
                                 average=average,detrend=detrend,nperseg=nperseg, noverlap=noverlap) for i in range(B.shape[0])))
            if Sum: Px[k,:] = np.sum(np.abs(P),axis=1).astype(float)
            if Mean: Pm[k,:] = np.mean(np.abs(P),axis=1).astype(float)
            if SD: Pd[k,:] = np.std(np.abs(P),axis=1).astype(float)
            k+=1

    return Px,Pm,Pd

def Periodogram(x,fs=128,method ='welch',win='hann',scaling='density', average='mean',detrend='constant',nperseg=None, noverlap=None):
    '''
    #scaling = 'density'--V**2/Hz 'spectrum'--V**2
    #average = 'mean', 'median'
    #detrend = False, 'constant', 'linear'
    '''
    if method ==None:
        f, Pxx = scipy.signal.periodogram(x,fs,win,scaling=scaling,detrend=detrend)
    elif method =='welch':
        #f, Pxx = scipy.signal.welch(x,fs,win,nperseg=np.clip(len(x),0,256),scaling=scaling,average=average,detrend=detrend)
        f, Pxx = scipy.signal.welch(x,fs,win,nperseg=nperseg,noverlap=noverlap,scaling=scaling,average=average,detrend=detrend)
    return np.abs(Pxx)
