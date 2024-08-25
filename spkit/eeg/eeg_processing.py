import numpy as np
import matplotlib.pyplot as plt
import scipy, copy
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
from scipy import stats
from copy import deepcopy

import warnings
warnings.filterwarnings('once')

from ..utils import deprecated
from ..utils import ProgBar_JL
from ..core.processing import filter_X, periodogram

# for backword compatibility
from ..core.processing import periodogram as Periodogram 

def rhythmic_powers(X,fs=128.0,fBands=[[4],[8,14]],Sum=True,Mean=False,SD =False,use_joblib=False,
                    filter_kwargs=dict(), periodogram_kwargs=dict(),filter_warn=True, periodogram_warn=True,
                    verbose=0):
    r"""Compute powers of different Rhythms of EEG Signal
    
    Decompose EEG Signal(s)-all the channels in Rhythms and compute power in each band for each channel

    Parameters
    ----------
    X: array,
      -  EEG segment of shape (n,ch) or (n,)
      -  where ch is number of channels
    fs: int,
       - sampling rate
    fBands: list
       - list of frequency bands
       - if None: fBands =[[4],[4,8],[8,14],[14,30],[30,47],[47]]
       - default [[4],[8,14]], 'delta' and 'alpha'

    (Sum,Mean,SD): bool, default (True, False, False)
       - if Sum=True, then Total power spectrum in the band computed,  default=True
       - if Mean=True, then Average power of the band is computed,  default=False
       - if SD=True, then Standard Deviation, (variation) in power for band is computed,  default=False

    filter_kwargs: dict, default=dict()
       - arguments for filtering, check :func:`filter_X` for details
       - default arguments setting is: dict(order=5,ftype='SOS',verbose=False)
       - To override any of the argument or suply additional argument based of :func:`filter_X`, 
         provide the in `filter_kwargs`.
       - For example, filter_kwargs = dict(ftype='filtfilt'), will overide the `ftype`.
    
    periodogram_kwargs:dict, default=dict()
       - arguments for periodogram, check :func:`periodogram` for details
       - default arguments setting is: 
         dict(method='welch',win='hann',scaling='density',nfft=None, average='mean',
                    detrend='constant',nperseg=None,noverlap=None,show_plot=False)
       - To override any of the argument or suply additional argument based of :func:`periodogram`, 
         provide the in `periodogram_kwargs`.
       - For example, periodogram_kwargs = dict(win='ham'), will overide the `win`.
    
    filter_warn: bool, default=True
       - It will show warning, if any additional argument other than deafult setting for filter, is provided
       - To turn warning statement off, set filter_warn=False
    
    periodogram_warn=True,
       - It will show warning, if any additional argument other than deafult setting for periodogram is provided
       - To turn warning statement off, set filter_warn=False
                    
    verbose: int, default=0
       - verbosity mode

    Returns
    -------

    Px: 2d-array.
      - sum of the power in a band  -  shape (number of bands,nch)
    Pm: 2d-array.
      - mean power in a band       -  shape (number of bands,nch)
    Pd: 2d-array.
      - standard deviation of power in a band  -  shape (number of bands,nch)

    References
    ----------
    
    * https://en.wikipedia.org/wiki/Electroencephalography#:~:text=%5B73%5D-,Comparison%20of%20EEG%20bands,-%5Bedit%5D

    See Also
    --------
    rhythmic_powers_win

    Examples
    --------
    #sp.eeg.rhythmic_powers
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    X,fs, ch_names = sp.data.eeg_sample_14ch()
    X = sp.filterDC_sGolay(X, window_length=fs//3+1)

    Px,Pm,Pd = sp.eeg.rhythmic_powers(X=X,fs=fs,fBands=[[4],[8,14]],Sum=True,Mean=True,SD =True)

    bind = np.arange(len(ch_names))
    plt.figure(figsize=(10,4))
    plt.subplot(211)
    plt.bar(bind*3,Px[0],label=r'$\delta$')
    plt.bar(bind*3+1,Px[1],label=r'$\alpha$')
    plt.xticks(bind*3+0.5,ch_names)
    plt.legend()
    plt.ylabel('Total Power')
    plt.subplot(212)
    plt.bar(bind*3,Pd[0],label=r'$\delta$')
    plt.bar(bind*3+1,Pd[1],label=r'$\alpha$')
    plt.xticks(bind*3+0.5,ch_names)
    plt.legend()
    plt.ylabel('Variation of Power \n within a band')
    plt.show()

    """

    #average :  method to average the periodograms, mean or median
    # Delta, Theta, Alpha, Beta, Gamma1, Gamma2

    # Atleast one of them should be True
    assert (Sum or Mean or SD)

    if fBands is None:
        fBands =[[4],[4,8],[8,14],[14,30],[30,47],[47]]

    filter_kwargs_default = dict(order=5,ftype='SOS',verbose=False,use_joblib=use_joblib)

    periodogram_kwargs_default = dict(method='welch',win='hann',scaling='density',nfft=None, average='mean',
                                      detrend='constant',nperseg=None,noverlap=None,show_plot=False)

    if len(filter_kwargs):
        for key in filter_kwargs:
            if filter_warn:
                if key not in filter_kwargs_default:
                    warnings.warn(f"new argument {key} for filtering is provided, check :func:`filter_X`, to make sure it is the correct one. To turn-off this warnign, set `filter_warn=False`")
            
            filter_kwargs_default[key] = filter_kwargs[key]

    if len(periodogram_kwargs):
        for key in periodogram_kwargs:
            if periodogram_warn:
                if key not in periodogram_kwargs_default:
                    warnings.warn(f"new argument {key} for periodogram computation is provided, check :func:`periodogram`, to make sure it is the correct one. To turn-off this warnign, set `periodogram_warn=False`")
            
            periodogram_kwargs_default[key] = periodogram_kwargs_default[key]

    if verbose:
        print('Arguments for Filtering are:')
        print('  --',filter_kwargs_default)
        print('Arguments for Periodogram computations are:')
        print('  --',periodogram_kwargs_default)

    Px = np.zeros([len(fBands),X.shape[1]])
    Pm = np.zeros([len(fBands),X.shape[1]])
    Pd = np.zeros([len(fBands),X.shape[1]])

    if Sum or Mean or SD:
        for k, freqs in enumerate(fBands):
            btype='bandpass'
            if len(freqs)==1:
                btype='lowpass' if freqs[0]<=4 else 'highpass'
                freqs=freqs[0]
            
            if verbose>1:
                #ProgBar_JL(pin,N=pend,title='',style=2,L=50)
                ProgBar_JL(k,N=len(fBands),style=2,L=50,title=f' - {freqs} | {btype}'+' '*8)

            Xf = filter_X(X.copy(),fs=fs,band=freqs,btype=btype,**filter_kwargs_default)

            if use_joblib:
                try:
                    P = np.array(Parallel(n_jobs=-1)(delayed(periodogram)(Xf[:,i],fs=fs,return_frq=False,**periodogram_kwargs_default) for i in range(Xf.shape[1])))
                except:
                    P = np.array([periodogram(Xf[:,i],fs=fs,return_frq=False,**periodogram_kwargs_default) for i in range(Xf.shape[1])])
            else:
                P = np.array([periodogram(Xf[:,i],fs=fs,return_frq=False,**periodogram_kwargs_default) for i in range(Xf.shape[1])])
        
            if Sum : Px[k,:] = np.sum(np.abs(P),axis=1).astype(float)
            if Mean: Pm[k,:] = np.mean(np.abs(P),axis=1).astype(float)
            if SD  : Pd[k,:] = np.std(np.abs(P),axis=1).astype(float)
    return Px,Pm,Pd


def rhythmic_powers_win(X,winsize=128, overlap=None,fs=128.0,fBands=[[4],[8,14]],Sum=True,Mean=False,SD =False,verbose=0, **kwargs):
    r"""Rhythmic powers-window-wise

        
    Decompose EEG Signal(s)-all the channels in Rhythms and compute power in each band for each channel

    Parameters
    ----------
    X: array,
      -  EEG segment of shape (n,ch) or (n,)
      -  where ch is number of channels

    winsize: int, default=128
       -  size of window for computing powers

    overlap: int, default=None
       - overlap shift,
       - number of samples to be shift
       - if None, overlap=winsize//2
    

    fs: int,
       - sampling rate
    fBands: list
       - list of frequency bands
       - if None: fBands =[[4],[4,8],[8,14],[14,30],[30,47],[47]]
       - default [[4],[8,14]], 'delta' and 'alpha'

    (Sum,Mean,SD): bool, default (True, False, False)
       - if Sum=True, then Total power spectrum in the band computed,  default=True
       - if Mean=True, then Average power of the band is computed,  default=False
       - if SD=True, then Standard Deviation, (variation) in power for band is computed,  default=False
    
    verbose: int, default=0
        - verbosity mode

    kwargs:
        - all the parameters as they are for :func:`rhythmic_powers`
        
        filter_kwargs: dict, default=dict()
            - arguments for filtering, check :func:`filter_X` for details
            - default arguments setting is: dict(order=5,ftype='SOS',verbose=False)
            - To override any of the argument or suply additional argument based of :func:`filter_X`, 
                provide the in `filter_kwargs`.
            - For example, filter_kwargs = dict(ftype='filtfilt'), will overide the `ftype`.
        
        periodogram_kwargs:dict, default=dict()
            - arguments for periodogram, check :func:`periodogram` for details
            - default arguments setting is: 
                dict(method='welch',win='hann',scaling='density',nfft=None, average='mean',
                            detrend='constant',nperseg=None,noverlap=None,show_plot=False)
            - To override any of the argument or suply additional argument based of :func:`periodogram`, 
                provide the in `periodogram_kwargs`.
            - For example, periodogram_kwargs = dict(win='ham'), will overide the `win`.
        
        filter_warn: bool, default=True
            - It will show warning, if any additional argument other than deafult setting for filter, is provided
            - To turn warning statement off, set filter_warn=False
        
        periodogram_warn=True,
            - It will show warning, if any additional argument other than deafult setting for periodogram is provided
            - To turn warning statement off, set filter_warn=False
                        
    Returns
    -------

    Pxt: 3d-array.
      - sum of the power in a band  -  shape (nt, number of bands,nch)
    Pmt: 3d-array.
      - mean power in a band       -  shape (nt, number of bands,nch)
    Pdt: 3d-array.
      - standard deviation of power in a band  -  shape (nt, number of bands,nch)

    See Also
    --------
    rhythmic_powers

    Examples
    --------
    #sp.eeg.rhythmic_powers_win
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    X,fs, ch_names = sp.data.eeg_sample_14ch()
    X = sp.filterDC_sGolay(X, window_length=fs//3+1)
    t = np.arange(X.shape[0])/fs
    Pxt, _, _ = sp.eeg.rhythmic_powers_win(X, winsize=128,overlap=32,fBands=[[4],[8,14],[32,47]],Sum=True)
    Pxt = np.log10(Pxt)
    plt.figure(figsize=(8,5))
    plt.subplot(211)
    plt.plot(Pxt[:,:,0], label=[r'$\delta$ (<4 Hz)',r'$\alpha$ (8-14 Hz)',r'$\gamma$ (32 Hz <)',])
    plt.xlabel('window #')
    plt.ylabel('power (dB)')
    plt.title(f'Power of Channel: {ch_names[0]}')
    plt.xlim([0,Pxt.shape[0]-1])
    plt.grid()
    plt.legend(ncol=2,frameon=False)
    plt.subplot(212)
    plt.plot(t,X[:,0])
    plt.grid()
    plt.xlim([t[0],t[-1]])
    plt.xlabel('time (s)')
    plt.ylabel(ch_names[0])
    plt.tight_layout()
    plt.show()

    """

    win = np.arange(winsize)
    if overlap is None: overlap=winsize//2
    Pxt = []
    Pmt = []
    Pdt  = []

    while win[-1]<X.shape[0]:
        Xi = X[win]
        if verbose: ProgBar_JL(win[-1],X.shape[0],style=2,color='blue')
        Px,Pm,Pd = rhythmic_powers(X=Xi,fs=fs,fBands=fBands,Sum=Sum,Mean=Mean,SD=SD,**kwargs)

        if Sum: Pxt.append(Px)
        if Mean:Pmt.append(Pm)
        if SD  :Pdt.append(Pd)

        win +=overlap

    Pxt = np.array(Pxt)
    Pmt = np.array(Pmt)
    Pdt = np.array(Pdt)

    return Pxt,Pmt,Pdt 








@deprecated("due to naming consistency, please use 'rhythmic_powers' for updated/improved functionality. [spkit-0.0.9.7]")
def RhythmicDecomposition(E,fs=128.0,order=5,method='welch',win='hann',Sum=True,Mean=False,SD =False,
                    scaling='density',nfft=None, average='mean',detrend='constant',nperseg=None, noverlap=None,fBands=None,use_joblib=False):
    r"""Compute powers of different Rhythms of EEG Signal


    Decompose EEG Signal(s)-all the channels in Rhythms and compute power in each band for each channel

        .. deprecated:: 0.0.9.7
            USE :func:`rhythmic_powers` for updated version

    .. warning:: DEPRECATED
       USE :func:`rhythmic_powers` for updated version


    Parameters
    ----------

    E: EEG segment of shape (n,nch)
    fs: sampling rate
    fBands: list of frequency bands - if None: fBands =[[4],[4,8],[8,14],[14,30],[30,47],[47]]


    Returns
    -------

    Px: sum of the power in a band  -  shape (number of bands,nch)
    Pm: mean power in a band       -  shape (number of bands,nch)
    Pd: standard deviation of power in a band  -  shape (number of bands,nch)

    References
    ----------
    * https://en.wikipedia.org/wiki/Electroencephalography#:~:text=%5B73%5D-,Comparison%20of%20EEG%20bands,-%5Bedit%5D

    See Also
    --------
    rhythmic_powers

    Examples
    --------
    #sp.eeg.RhythmicDecomposition
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    X,fs, ch_names = sp.data.eeg_sample_14ch()
    X = sp.filterDC_sGolay(X, window_length=fs//3+1)

    Px,Pm,Pd = sp.eeg.RhythmicDecomposition(E=X,fs=fs,fBands=[[4],[8,14]],Sum=True,Mean=True,SD =True)

    bind = np.arange(len(ch_names))
    plt.figure(figsize=(10,4))
    plt.subplot(211)
    plt.bar(bind*3,Px[0],label=r'$\delta$')
    plt.bar(bind*3+1,Px[1],label=r'$\alpha$')
    plt.xticks(bind*3+0.5,ch_names)
    plt.legend()
    plt.ylabel('Total Power')
    plt.subplot(212)
    plt.bar(bind*3,Pd[0],label=r'$\delta$')
    plt.bar(bind*3+1,Pd[1],label=r'$\alpha$')
    plt.xticks(bind*3+0.5,ch_names)
    plt.legend()
    plt.ylabel('Variation of Power \n within a band')
    plt.show()

    """

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
            if use_joblib:
                try:
                    B = np.array(Parallel(n_jobs=-1)(delayed(lfilter)(b,a,E[:,i]) for i in range(E.shape[1])))
                    P = np.array(Parallel(n_jobs=-1)(delayed(_periodogram)(B[i,:],fs=fs,method=method,win=win,scaling=scaling,nfft=nfft,
                                         average=average,detrend=detrend,nperseg=nperseg, noverlap=noverlap) for i in range(B.shape[0])))
                except:
                    B = [lfilter(b,a,E[:,i]) for i in range(E.shape[1])]
                    P = np.array([_periodogram(B[i],fs=fs,method=method,win=win,scaling=scaling,average=average,detrend=detrend,
                                 nperseg=nperseg, nfft=nfft,noverlap=noverlap) for i in range(len(B))])
            else:
                B = [lfilter(b,a,E[:,i]) for i in range(E.shape[1])]
                P = np.array([_periodogram(B[i],fs=fs,method=method,win=win,scaling=scaling,average=average,detrend=detrend,
                             nperseg=nperseg, nfft=nfft,noverlap=noverlap) for i in range(len(B))])
            if Sum: Px[k,:] = np.sum(np.abs(P),axis=1).astype(float)
            if Mean: Pm[k,:] = np.mean(np.abs(P),axis=1).astype(float)
            if SD: Pd[k,:] = np.std(np.abs(P),axis=1).astype(float)
            k+=1

    return Px,Pm,Pd

#@deprecated("due to naming consistency, please use 'periodogram' for updated/improved functionality. [spkit-0.0.9.7]")
def _periodogram(x,fs=128,method ='welch',win='hann',nfft=None,scaling='density',average='mean',detrend='constant',nperseg=None, noverlap=None):
    '''Computing Periodogram using Welch or Periodogram method


    **Computing Periodogram using Welch or Periodogram method**
    
    Parameters
    ----------
    #scaling = 'density'--V**2/Hz 'spectrum'--V**2
    #average = 'mean', 'median'
    #detrend = False, 'constant', 'linear'
    nfft    = None, n-point FFT
    
    Returns
    -------

    Px: |periodogram|
    
    '''
    if method ==None:
        f, Pxx = scipy.signal.periodogram(x,fs,win,nfft=nfft,scaling=scaling,detrend=detrend)
    elif method =='welch':
        #f, Pxx = scipy.signal.welch(x,fs,win,nperseg=np.clip(len(x),0,256),scaling=scaling,average=average,detrend=detrend)
        f, Pxx = scipy.signal.welch(x,fs,win,nperseg=nperseg,noverlap=noverlap,nfft=nfft,scaling=scaling,average=average,detrend=detrend)
    return np.abs(Pxx)





# def filterDC(x,alpha=256):
#     b = x[0]
#     xf = np.zeros(len(x))
#     for i in range(len(x)):
#         b = ((alpha - 1) * b + x[i]) / alpha
#         xf[i] = x[i]-b
#     return xf
#
# def filterDC_X(X,alpha=256,return_background=True):
#     B = X[0]
#     if return_background:Bg = np.zeros_like(X)
#     Xf = np.zeros_like(X)
#
#     for i in range(X.shape[0]):
#         B = ((alpha - 1) * B + X[i]) / alpha
#         Xf[i] = X[i]-B
#         if return_background: Bg[i]= copy.copy(B)
#     if return_background: return Xf, Bg
#     return Xf
#
# def filterDC_sGolay(X, window_length=127, polyorder=3, deriv=0, delta=1.0, mode='interp', cval=0.0):
#     '''
#     Savitzky-Golay filter for multi-channels signal: From Scipy library
#
#     X: input multichannel signal - shape (n,ch)
#      : for single channel signal, use X[:,None] to make it two dimensional signal.
#     window_length: should be an odd number
#     others input parameters as same as in scipy.signal.savgol_filter
#     '''
#     Xm = np.array([savgol_filter(x, window_length, polyorder,deriv=deriv, delta=delta, axis=-1, mode=mode, cval=cval) for x in X])
#     Xf = X - Xm
#     return Xf, Xm
#
# def filter_X(X,band =[0.5],btype='highpass',order=5,fs=128.0,ftype='filtfilt',verbose=1):
#     '''
#     Buttorworth filtering -  basic filtering
#     ---------------------
#     X: input multichannel signal - shape (n,ch)
#      : for single channel signal, use X[:,None] to make it two dimensional signal.
#     band: cut of frequency, for lowpass and highpass, band is list of one, for bandpass list of two numbers
#     btype: filter type
#     order: order of filter
#     ftype: filtering approach type, filtfilt or lfilter,
#          : lfilter is causal filter, which introduces delaye, filtfilt does not introduce any delay, but it is non-causal filtering
#     Xf: filtered signal of same size as X
#     '''
#     if verbose: print(X.shape, 'channels axis = 1')
#     b,a = butter(order,np.array(band)/(0.5*fs),btype=btype)
#     if ftype=='lfilter':
#         Xf  = np.array(Parallel(n_jobs=-1)(delayed(lfilter)(b,a,X[:,i]) for i in range(X.shape[1])))
#     elif ftype=='filtfilt':
#         Xf  = np.array(Parallel(n_jobs=-1)(delayed(filtfilt)(b,a,X[:,i]) for i in range(X.shape[1])))
#     return Xf

# def getStats(x,detail_level=1,return_names=False):
#     stats_names =['mean','sd','median','min','max','n','q25','q75','iqr','kur','skw','gmean','entropy']
#     esp=1e-5
#     if isinstance(x,int) or isinstance(x,float): x =  [x]
#     if isinstance(x,list):x = np.array(x)
#     assert len(x.shape)==1
#     #logsum = self.get_exp_log_sum(x)
#     x = x+esp
#     mn = np.nanmean(x)
#     sd = np.nanstd(x)
#     md = np.nanmedian(x)
#     min0 = np.nanmin(x)
#     max0 = np.nanmax(x)
#
#     n = len(x) - sum(np.isnan(x))
#
#     if detail_level==1:
#         return np.r_[mn,sd,md,min0,max0,n], stats_names[:6]
#
#     q25 = np.nanquantile(x,0.25)
#     q75 = np.nanquantile(x,0.75)
#     iqr = stats.iqr(x[~np.isnan(x)])
#     kur = stats.kurtosis(x,nan_policy='omit')
#     skw = stats.skew(x[~np.isnan(x)])
#     if detail_level==2:
#         return np.r_[mn,sd,md,min0,max0,n,q25,q75,iqr,kur,skw], stats_names[:11]
#
#     gmn = stats.gmean(x[~np.isnan(x)])
#     entropy = spkit.entropy(x[~np.isnan(x)])
#     names =['mean','sd','median','min','max','n','q25','q75','iqr','kur','skw','gmean','entropy']
#     return np.r_[mn,sd,md,min0,max0,n,q25,q75,iqr,kur,skw,gmn,entropy], stats_names
