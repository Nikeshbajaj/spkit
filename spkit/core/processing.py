'''
Basic signal processing methods
----------------------------------------------------------
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
import scipy, spkit, copy
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
from scipy import stats
from copy import deepcopy
from .infotheory import entropy
import pywt as wt

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
    Xm = np.array([savgol_filter(X[:,i], window_length, polyorder,deriv=deriv, delta=delta, axis=-1, mode=mode, cval=cval) for i in range(X.shape[1])]).T
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
    assert (include_upper+include_lower)
    xi = x.copy()
    if method =='iqr':
        q1 = np.nanquantile(xi,0.25)
        q3 = np.nanquantile(xi,0.75)
        ut = q3 + k*(q3-q1)
        lt = q1 - k*(q3-q1)
    elif method =='sd':
        sd = np.nanstd(xi)
        ut = k*sd
        lt = -k*sd
    else:
        print('Define method')
        return None

    if not(include_lower): lt = -np.inf

    idx_bin = (xi>=ut) | (xi<=lt)
    idx = np.where(idx_bin)[0]

    if return_lim:
        return idx, idx_bin, (lt,ut)

    return idx, idx_bin


'''
BASIC WAVELET FILTERING
------------------------
'''
def get_theta(w,N,k=1.5,method='optimal',IPR=[0.25,0.75]):
    if method =='optimal':
        sig = np.median(abs(w))/0.6745
        theta_u = sig*np.sqrt(2*np.log(N))
        theta_l = -theta_u
    elif method =='sd':
        theta_u = k*np.std(w)
        theta_l = -theta_u
    elif method=='iqr':
        r = stats.iqr(w)
        q1 = np.quantile(w,IPR[0])
        q3 = np.quantile(w,IPR[1])
        #assert r ==q3-q1
        theta_u = q3 + k*r
        theta_l = q1 - k*r
    return theta_l, theta_u

def wavelet_filtering(x,wv='db3',threshold='optimal',filter_out_below=True,k=1.5,mode='elim',show=False,wpd_mode='symmetric',wpd_maxlevel=None,
           packetwise=False,WPD=True,lvl=[],verbose=False,fs=128.0,sf=1,IPR=[0.25,0.75]):
    '''
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


def WPA_coeff(x,wv='db3',mode='symmetric',maxlevel=None, V=False):
    wp = wt.WaveletPacket(x, wavelet=wv, mode=mode,maxlevel=maxlevel)
    wr = [wp[node.path].data for node in wp.get_level(wp.maxlevel, 'natural') ]
    WR = np.vstack(wr) if V else np.hstack(wr)
    return WR

def WPA_temporal(x,winsize=128,overlap=64,wv='db3',V=True,mode='symmetric',maxlevel=None):
    win =np.arange(winsize)
    W =[]
    while win[-1]<x.shape[0]:
        Wi = WPA_coeff(x[win],V=V,wv=wv,mode=mode,maxlevel=maxlevel)
        W.append(Wi)
        win +=overlap
    Wtemp = np.hstack(W) if V else np.vstack(W).T
    return Wtemp

def WPA_plot(x,winsize=128,overlap=64,V=True,wv='db3',mode='symmetric',maxlevel=None,inpterp='sinc',fs=128,plot=True):
    if fs is None: fs =1
    t = np.arange(len(x))/fs

    win =np.arange(winsize)
    W =[]
    while win[-1]<x.shape[0]:
        Wi = WPA_coeff(x[win],V=V,wv=wv,mode=mode,maxlevel=maxlevel)
        W.append(Wi)
        win +=overlap

    if V:
        Wp = np.hstack(W)
    else:
        Wp = np.vstack(W).T
    if plot:
        plt.figure(figsize=(15,8))
        plt.subplot(211)
        plt.imshow(Wp,aspect='auto',origin='lower',interpolation=inpterp,cmap='jet',extent=[t[0], t[-1], 1, Wp.shape[0]])
        plt.subplot(212)
        plt.plot(t,x)
        plt.xlim([t[0], t[-1]])
        plt.grid()
        plt.show()
    return Wp
