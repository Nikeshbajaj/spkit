'''
Automatic and Tunable Artifact Removal  (ATAR) algorithm
----------------------------------------------------------
Author @ Nikesh Bajaj
updated on Date: 26 Sep 2021
Version : 0.0.4
Github :  https://github.com/Nikeshbajaj/spkit
Contact: n.bajaj@qmul.ac.uk | n.bajaj@imperial.ac.uk


For more details, check this:
Bajaj, Nikesh, et al. "Automatic and tunable algorithm for EEG artifact removal using wavelet decomposition with applications in
predictive modeling during auditory tasks." Biomedical Signal Processing and Control 55 (2020): 101624.


'''
from __future__ import absolute_import, division, print_function
name = "Signal Processing toolkit | EEG | ATAR Algorith"
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
from scipy import signal
from scipy.signal import butter, lfilter#, convolve, boxcar
from joblib import Parallel, delayed
from scipy import stats
import pywt as wt


def SoftThresholding(w,theta_a,theta_g):
    w1 = w.copy()
    if theta_g>=theta_a:
        print('thresholds are not satisfying t2>t1')
        print('Correcting: with default setting theta_g = 0.8*theta_a')
        theta_g = theta_a*0.8

    alpha = -(1.0/theta_g)*np.log((theta_a-theta_g)/(theta_a+theta_g))
    #to avoid +inf value... np.exp(710)-->inf
    w1[-(alpha*w1)>709]=708.0/(-alpha)
    w2 = (1-np.exp(-alpha*w1))/(1+np.exp(-alpha*w1))*theta_a
    w2[np.abs(w)<theta_g]=w[np.abs(w)<theta_g]
    return  w2

def LinearAttenuanating(w,theta_a,theta_b):
    w1 = w.copy()
    w1 = np.sign(w1)*theta_a*(1 -  (np.abs(w1)-theta_a)/(theta_b - theta_a))
    w1[abs(w)<=theta_a]=w[abs(w)<=theta_a]
    w1[abs(w)>theta_b]=0
    return w1

def Elimination(w,theta_a):
    w1 = w.copy()
    w1[abs(w1)>theta_a]=0
    return w1

def Outliers(WR):
    #IQR = stats.iqr(WR)
    #Q1 = np.median(WR)-IQR/2.0
    #Q3 = np.median(WR)+IQR/2.0
    Q1 = np.quantile(WR,0.25)
    Q3 = np.quantile(WR,0.75)
    IQR = Q3-Q1
    ll = Q1-1.5*IQR
    ul = Q3+1.5*IQR
    return ll,ul

def ipr2thr(r,beta=0.1,k1=None,k2=100,c=100):
    '''
    theta_a = k2*np.exp(-beta*c*r/(2.0*k2)) for c=100
    '''
    theta_a = k2*np.exp(-beta*c*r/(2.0*k2))
    if k1 is not None and theta_a<k1:
        theta_a = k1
    return theta_a

def Wfilter(x,wv='db3',thr_method='ipr',IPR=[25,75],beta=0.1,k1=10,k2=100,theta_a=np.inf,est_wmax=100,
            bf=2,gf=0.8,OptMode ='soft',factor=1.0,show_plot=False,wpd_mode='symmetric',wpd_maxlevel=None,
            WPD=True,packetwise=False,lvl=[],fs=128.0):
    '''
    Wavelet filtering using ATAR Algorithm
    for more details, check:
    Bajaj, Nikesh, et al. "Automatic and tunable algorithm for EEG artifact removal using wavelet decomposition with applications in
    predictive modeling during auditory tasks." Biomedical Signal Processing and Control 55 (2020): 101624.
    ------------------

    input
    -----
    x: single channel EEG signal shape - (n,)

    Threshold Computation method:
    thr_method : None (default), 'ipr'
           : provided with theta_a, bf , gf
           : where:-
           : theta_b = bf*theta_a  -- used for Linear Attenuation
           : theta_g = gf*theta_a  -- used for Soft thresholding

    Operating modes:
    OptMode = ['soft','elim','linAtten']
             : default 'soft'

    Wavelet Decomposition modes:
    wpd_mode = ['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization']
                default 'symmetric'

    Wavelet family:
    wv = ['db3'.....'db38', 'sym2.....sym20', 'coif1.....coif17', 'bior1.1....bior6.8', 'rbio1.1...rbio6.8', 'dmey']
         :'db3'(default)

    IPR: Inter-percentile range - [25,75] is interquartile range, a special case of IPR

    output
    ------
    xR: Corrected signal of same shape (n,)
    '''
    verbose=False
    if verbose:
        print('WPD:',WPD,' wv:',wv,' IPR:',IPR,' beta:',beta,' method:',method,' OptMode:',OptMode)
        print('k1-k2:',[k1,k2])


    if WPD: # Wavelet Packet Decomposition
        wp = wt.WaveletPacket(x, wavelet=wv, mode=wpd_mode,maxlevel=wpd_maxlevel)
        wr = [wp[node.path].data for node in wp.get_level(wp.maxlevel, 'natural') ]
        WR = np.hstack(wr)
        nodes = [node for node in wp.get_level(wp.maxlevel, 'natural')]
    else:  # Wavelet Transform
        wr = wt.wavedec(x,wavelet=wv, mode=wpd_mode,level=wpd_maxlevel)
        WR = np.hstack(wr)
        nodes = np.arange(len(wr))

    if not(packetwise):
        if thr_method=='ipr':
            if k2 is None: k2=100
            r = stats.iqr(WR,rng=IPR)
            theta_a = ipr2thr(r,beta=beta,k1=k1,k2=k2,c=est_wmax)
        elif thr_method is not None:
            print('Method for computing threshold is not defined')
        theta_b = bf*theta_a
        theta_g = gf*theta_a

    removList=[]

    for i in range(len(nodes)):
    #for node in wp.get_level(wp.maxlevel, 'natural'):
        c = wp[nodes[i].path].data if WPD else wr[i]

        if len(lvl)==0 or i in lvl:
            if packetwise:
                if thr_method=='ipr':
                    if k2 is None: k2=100
                    r = stats.iqr(c,rng=IPR)
                    theta_a = ipr2thr(r,beta=beta,k1=k1,k2=k2,c=est_wmax)
                elif thr_method is not None:
                    print('Method for computing threshold is not defined')
                theta_b = bf*theta_a
                theta_g = gf*theta_a


            if OptMode=='soft':
                c = SoftThresholding(c,theta_a,theta_g)
            elif OptMode=='linAtten':
                c = LinearAttenuanating(c,theta_a,theta_b)
            elif OptMode=='elim':
                c = Elimination(c,theta_a)
            else:
                print('Operating mode was not in list..\n No wavelet filtering is applied')
                pass
            if WPD:
                wp[nodes[i].path].data = c
            else:
                wr[i] = c
    #Reconstruction
    if WPD:
        xR = wp.reconstruct(update=False)
    else:
        xR = wt.waverec(wr, wavelet = wv)

    if show_plot:
        plt.figure(figsize=(11,6))
        plt.subplot(211)
        plt.plot(WR,'b',alpha=0.8,label='Coef.')
        plt.ylabel('Wavelete Coefficients')
        ytiW =[np.min(WR),np.max(WR)]
        #print('maxlevel :',wp.maxlevel)
        if WPD: wr = [wp[node.path].data for node in wp.get_level(wp.maxlevel, 'natural') ]
        WRi = np.hstack(wr)
        plt.plot(WRi,'r',alpha=0.9,label='Filtered Coff.')
        ytiW = ytiW+[np.min(WRi),np.max(WRi)]
        plt.yticks(ytiW)
        plt.grid()
        plt.legend()
        plt.xlim([0,len(WRi)])
        plt.subplot(212)
        if WPD:
            t = np.arange(len(wp.data))/fs
            plt.plot(t,wp.data,'b',alpha=0.8,label='signal')
        else:
            t = np.arange(len(x))/fs
            plt.plot(t,x,'b',alpha=0.8,label='signal')
        plt.plot(t,xR,'r',alpha=0.8,label='corrected')
        plt.ylabel('Signal')
        plt.yticks([np.min(xR),np.min(x),0,np.max(xR),np.max(x)])
        plt.xlim([t[0],t[-1]])
        plt.legend()
        plt.grid()
        plt.show()
    return xR

def ATAR_1Ch(x,wv='db3',winsize=128,thr_method='ipr',IPR=[25,75],beta=0.1,k1=None,k2 =100,est_wmax=100,
            theta_a=np.inf,bf=2,gf=0.8,OptMode ='soft',factor=1.0,wpd_mode='symmetric',wpd_maxlevel=None,
            verbose=True, window=['hamming',True],hopesize=None, ReconMethod='custom',packetwise=False,WPD=True,lvl=[],fs=128.0):
    '''
    Apply ATAR on short windows of signal (single channel):
    Signal is decomposed in smaller overlapping windows and reconstructed after correcting using overlap-add method.
    ----
    for more details, check:
    Bajaj, Nikesh, et al. "Automatic and tunable algorithm for EEG artifact removal using wavelet decomposition with applications in
    predictive modeling during auditory tasks." Biomedical Signal Processing and Control 55 (2020): 101624.
    ------------------

    Wfilter(x,wv='db3',method=None,IPR=[25,75],beta=0.1,k1=None,k2 =100,
     theta_a=np.inf,bf=2,gf=0.8,OptMode ='soft',factor=1.0,showPlot=False,wpd_mode='symmetric',wpd_maxlevel=None)

    input
    -----
    X: input multi-channel signal of shape (n,ch)

    Threshold Computation method:
    method : None (default), 'ipr'
           : provided with theta_a, bf , gf
           : theta_b = bf*theta_a
           : theta_g = gf*theta_a

    Operating modes:
    OptMode = ['soft','elim','linAtten']
             : default 'soft'
             : use 'elim' with global

    Wavelet Decomposition modes:
    wpd_mode = ['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization']
                default 'symmetric'

    Wavelet family:
    wv = ['db3'.....'db38', 'sym2.....sym20', 'coif1.....coif17', 'bior1.1....bior6.8', 'rbio1.1...rbio6.8', 'dmey']
         :'db3'(default)


    Reconstruction Methon
    ReconMethod :  None, 'custom', 'HamWin'
    for 'custom': window[0] is used and applied after denoising is window[1] is True else
    windowing applied before denoising

    output
    ------
    XR: corrected signal of same shape as input X
    '''
    if ReconMethod is None:
        win=np.arange(winsize)
        xR=[]
        pf=0
        while win[-1]<=x.shape[0]:
            if verbose:
                if 100*win[-1]/float(x.shape[0])>=pf+1:
                    pf = 100*win[-1]/float(x.shape[0])
                    pbar = '|'+'#'*int(pf)+' '*(99-int(pf))+'|'
                    print(str(np.round(pf,2))+'%'+pbar,end='\r', flush=True)
            xi = x[win]
            xr = Wfilter(x,wv=wv,thr_method=thr_method,IPR=IPR,beta=beta,k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,
                 OptMode =OptMode,show_plot=False,wpd_mode=wpd_mode,wpd_maxlevel=wpd_maxlevel,
                         packetwise=False,WPD=WPD,lvl=lvl,fs=fs)

            xR.append(xr)
            win+=winsize
        xR = np.hstack(xR)

    elif ReconMethod =='HamWin':
        xt  = np.hstack([np.zeros(winsize//2),x,np.zeros(winsize//2)])
        xR  = np.zeros(xt.shape)
        wh  = signal.windows.hamming(winsize+1)[:winsize]
        win = np.arange(winsize)

        while win[-1]<=xt.shape[0]:
            if verbose:
                pf = 100*win[-1]/float(x.shape[0])
                pbar = '|'+'#'*int(pf)+' '*(99-int(pf))+'|'
                print(str(np.round(pf,2))+'%'+pbar,end='\r', flush=True)
            xi = xt[win]*wh
            xr = Wfilter(xi,wv=wv,thr_method=thr_method,IPR=IPR,beta=beta,k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,
                 OptMode =OptMode,factor=factor,show_plot=False,wpd_mode=wpd_mode,wpd_maxlevel=wpd_maxlevel,
                        packetwise=packetwise,WPD=WPD,lvl=lvl,fs=fs)
            xR[win]+= xr
            win+=winsize//2
        xR = xR/1.08
        xR = xR[winsize//2:-winsize//2]

    elif ReconMethod =='custom':
        if hopesize is None: hopesize = winsize//2
        M   = winsize
        H   = hopesize
        hM1 = (M+1)//2
        hM2 = M//2

        xt  = np.hstack([np.zeros(hM2),x,np.zeros(hM1)])

        pin  = hM1
        pend = xt.size-hM1
        wh   = signal.get_window(window[0],M)
        #if len(window)>1: AfterApply = window[1]
        #else: AfterApply = False
        AfterApply = window[1] if len(window)>1 else False

        if verbose: print('Windowing after apply : ',AfterApply)


        xR   = np.zeros(xt.shape)
        pf=0
        while pin<=pend:
            if verbose:
                if 100*pin/float(pend)>=pf+1:
                    pf = 100*pin/float(pend)
                    pbar = '|'+'#'*int(pf)+' '*(99-int(pf))+'|'
                    print(str(np.round(pf,2))+'%'+pbar,end='\r', flush=True)
            xi = xt[pin-hM1:pin+hM2]
            if not(AfterApply): xi *=wh

            xr = Wfilter(xi,wv=wv,thr_method=thr_method,IPR=IPR,beta=beta,k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,
                 OptMode =OptMode,factor=factor,show_plot=False,wpd_mode=wpd_mode,wpd_maxlevel=wpd_maxlevel,
                        packetwise=packetwise,WPD=WPD,lvl=lvl,fs=fs)

            if AfterApply: xr *=wh

            xR[pin-hM1:pin+hM2] += H*xr  ## Overlap Add method
            pin += H
        xR = xR[hM2:-hM1]/sum(wh)

    if verbose:
        pf = 100
        pbar = '|'+'#'*int(pf)+' '*(99-int(pf))+'|'
        print(str(np.round(pf,2))+'%'+pbar,end='\r', flush=True)
        print('\n')
    return xR

def ATAR_mCh(X,wv='db3',winsize=128,thr_method ='ipr',IPR=[25,75],beta=0.1,k1=10,k2 =100,est_wmax=100,
              theta_a=np.inf,bf=2,gf=0.8,OptMode ='soft',wpd_mode='symmetric',wpd_maxlevel=None,factor=1.0,
              verbose=True, window=['hamming',True],hopesize=None, ReconMethod='custom',packetwise=False,WPD=True,lvl=[],fs=128.0):
    '''
    Apply ATAR on short windows of signal (multiple channels:):
    Signal is decomposed in smaller overlapping windows and reconstructed after correcting using overlap-add method.
    ------
    for more details, check:
    Bajaj, Nikesh, et al. "Automatic and tunable algorithm for EEG artifact removal using wavelet decomposition with applications in
    predictive modeling during auditory tasks." Biomedical Signal Processing and Control 55 (2020): 101624.
    ----------------

    input
    -----
    X: input multi-channel signal of shape (n,ch)

    Wavelet family:
    wv = ['db3'.....'db38', 'sym2.....sym20', 'coif1.....coif17', 'bior1.1....bior6.8', 'rbio1.1...rbio6.8', 'dmey']
         :'db3'(default)

    Threshold Computation method:
    thr_method : None (default), 'ipr'
           : None: fixed threshold theta_a is applied
           : ipr : applied with theta_a, bf , gf, beta, k1, k2 and OptMode
           : theta_b = bf*theta_a
           : theta_g = gf*theta_a

    Operating modes:
    OptMode = ['soft','elim','linAtten']
             : default 'soft'
             : use 'elim' with globalgood

    Wavelet Decomposition modes:
    wpd_mode = ['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization']
                default 'symmetric'

    Reconstruction Method - Overlap-Add method
    ReconMethod :  None, 'custom', 'HamWin'
    for 'custom': window[0] is used and applied after denoising is window[1] is True else
    windowing applied before denoising

    output
    ------
    XR: corrected signal of same shape as input X
    '''
    if hopesize is None: hopesize=winsize//2

    assert thr_method in [ None, 'ipr']
    assert OptMode in ['soft','linAtten','elim']


    if verbose:
        print('WPD Artifact Removal')
        print('WPD:',WPD,' Wavelet:',wv,', Method:',thr_method,', OptMode:',OptMode)
        if thr_method=='ipr': print('IPR=',IPR,', Beta:',beta, ', [k1,k2]=',[k1,k2])
        if thr_method is None: print('theta_a: ',theta_a)
        print('Reconstruction Method:',ReconMethod, ', Window:',window,', (Win,Overlap)=',(winsize,hopesize))

    if len(X.shape)>1:
        XR =np.array(Parallel(n_jobs=-1)(delayed(ATAR_1Ch)(X[:,i],wv=wv,winsize=winsize, thr_method=thr_method, IPR=IPR,
              beta=beta, k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,OptMode=OptMode, factor=factor, wpd_mode=wpd_mode,
              wpd_maxlevel=wpd_maxlevel,verbose=verbose, window=window,hopesize=hopesize,
              ReconMethod=ReconMethod,packetwise=packetwise,WPD=WPD,lvl=lvl,fs=fs) for i in range(X.shape[1]))).T
    else:
        XR =ATAR_1Ch(X,wv=wv,winsize=winsize, thr_method=thr_method, IPR=IPR,
              beta=beta, k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,OptMode=OptMode, factor=factor, wpd_mode=wpd_mode,
              wpd_maxlevel=wpd_maxlevel, verbose=verbose, window=window,hopesize=hopesize,
              ReconMethod=ReconMethod,packetwise=packetwise,WPD=WPD,lvl=lvl,fs=fs)
    return XR

def ATAR_mCh_noParallel(X,wv='db3',winsize=128,thr_method ='ipr',IPR=[25,75],beta=0.1,k1=10,k2 =100,est_wmax=100,
              theta_a=np.inf,bf=2,gf=0.8,OptMode ='soft',wpd_mode='symmetric',wpd_maxlevel=None,factor=1.0,
              verbose=True, window=['hamming',True],hopesize=None, ReconMethod='custom',packetwise=False,WPD=True,lvl=[],fs=128.0):
    '''
    Apply ATAR on short windows of signal (multiple channels:): - Without using Joblib - in case that creates issue in some systems and IDE
    Signal is decomposed in smaller overlapping windows and reconstructed after correcting using overlap-add method.
    ------
    for more details, check:
    Bajaj, Nikesh, et al. "Automatic and tunable algorithm for EEG artifact removal using wavelet decomposition with applications in
    predictive modeling during auditory tasks." Biomedical Signal Processing and Control 55 (2020): 101624.
    ----------------

    input
    -----
    X: input multi-channel signal of shape (n,ch)

    Wavelet family:
    wv = ['db3'.....'db38', 'sym2.....sym20', 'coif1.....coif17', 'bior1.1....bior6.8', 'rbio1.1...rbio6.8', 'dmey']
         :'db3'(default)

    Threshold Computation method:
    thr_method : None (default), 'ipr'
           : None: fixed threshold theta_a is applied
           : ipr : applied with theta_a, bf , gf, beta, k1, k2 and OptMode
           : theta_b = bf*theta_a
           : theta_g = gf*theta_a

    Operating modes:
    OptMode = ['soft','elim','linAtten']
             : default 'soft'
             : use 'elim' with globalgood

    Wavelet Decomposition modes:
    wpd_mode = ['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization']
                default 'symmetric'

    Reconstruction Method - Overlap-Add method
    ReconMethod :  None, 'custom', 'HamWin'
    for 'custom': window[0] is used and applied after denoising is window[1] is True else
    windowing applied before denoising

    output
    ------
    XR: corrected signal of same shape as input X
    '''
    if hopesize is None: hopesize=winsize//2

    assert thr_method in [ None, 'ipr']
    assert OptMode in ['soft','linAtten','elim']


    if verbose:
        print('WPD Artifact Removal')
        print('WPD:',WPD,' Wavelet:',wv,', Method:',thr_method,', OptMode:',OptMode)
        if thr_method=='ipr': print('IPR=',IPR,', Beta:',beta, ', [k1,k2]=',[k1,k2])
        if thr_method is None: print('theta_a: ',theta_a)
        print('Reconstruction Method:',ReconMethod, ', Window:',window,', (Win,Overlap)=',(winsize,hopesize))

    if len(X.shape)>1:

        XR =np.array([
              ATAR_1Ch(X[:,i],wv=wv,winsize=winsize, thr_method=thr_method, IPR=IPR,
              beta=beta, k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,OptMode=OptMode, factor=factor, wpd_mode=wpd_mode,
              wpd_maxlevel=wpd_maxlevel,verbose=verbose, window=window,hopesize=hopesize,
              ReconMethod=ReconMethod,packetwise=packetwise,WPD=WPD,lvl=lvl,fs=fs) for i in range(X.shape[1])]).T
    else:
        XR =ATAR_1Ch(X,wv=wv,winsize=winsize, thr_method=thr_method, IPR=IPR,
              beta=beta, k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,OptMode=OptMode, factor=factor, wpd_mode=wpd_mode,
              wpd_maxlevel=wpd_maxlevel, verbose=verbose, window=window,hopesize=hopesize,
              ReconMethod=ReconMethod,packetwise=packetwise,WPD=WPD,lvl=lvl,fs=fs)
    return XR


def Wfilter_dev(x,wv='db3',thr_method='ipr',IPR=[25,75],beta=0.1,k1=10,k2=100,theta_a=np.inf,est_wmax=100,
            bf=2,gf=0.8,OptMode ='soft',factor=1.0,show_plot=False,wpd_mode='symmetric',wpd_maxlevel=None,
           WPD=True,packetwise=False,lvl=[],fs=128.0):

    '''
    ---- IN DEVELOPMENT MODE ----- AVOID USING IT FOR NOW -----
    ------------------------------------------------------------
    Threshold Computation method:
    thr_method : None (default), 'ipr', 'global','outliers','std'
           : provided with theta_a, bf , gf
           : where:-
           : theta_b = bf*theta_a  -- used for Linear Attenuation
           : theta_g = gf*theta_a  -- used for Soft thresholding

    Operating modes:
    OptMode = ['soft','elim','linAtten']
             : default 'soft'
             : use 'elim' with global

    Wavelet Decomposition modes:
    wpd_mode = ['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization']
                default 'symmetric'

    Wavelet family:
    wv = ['db3'.....'db38', 'sym2.....sym20', 'coif1.....coif17', 'bior1.1....bior6.8', 'rbio1.1...rbio6.8', 'dmey']
         :'db3'(default)


    '''
    verbose=False
    if verbose:
        print('WPD:',WPD,' wv:',wv,' IPR:',IPR,' beta:',beta,' method:',method,' OptMode:',OptMode)
        print('k1-k2:',[k1,k2])


    if WPD: # Wavelet Packet Decomposition
        wp = wt.WaveletPacket(x, wavelet=wv, mode=wpd_mode,maxlevel=wpd_maxlevel)
        wr = [wp[node.path].data for node in wp.get_level(wp.maxlevel, 'natural') ]
        WR = np.hstack(wr)
        nodes = [node for node in wp.get_level(wp.maxlevel, 'natural')]
    else:  # Wavelet Transform
        wr = wt.wavedec(x,wavelet=wv, mode=wpd_mode,level=wpd_maxlevel)
        WR = np.hstack(wr)
        nodes = np.arange(len(wr))

    if not(packetwise):
        if thr_method=='ipr':
            if k2 is None: k2=100
            r = stats.iqr(WR,rng=IPR)
            theta_a = ipr2thr(r,beta=beta,k1=k1,k2=k2,c=est_wmax)
        elif thr_method =='global':
            sig = np.median(abs(WR))/0.6745
            theta_a = sig*np.sqrt(2*np.log(len(x)))
        elif thr_method =='outliers':
            ll,ul = Outliers(WR)
            thrlw = ll
            thrup = ul
            _,theta_a = Outliers(np.abs(WR))
        elif thr_method =='std':
            theta_a = 1.5*np.std(WR)
        elif thr_method is not None:
            print('Method for computing threshold is not defined')
        theta_b = bf*theta_a
        theta_g = gf*theta_a

    removList=[]

    for i in range(len(nodes)):
    #for node in wp.get_level(wp.maxlevel, 'natural'):
        c = wp[nodes[i].path].data if WPD else wr[i]

        if len(lvl)==0 or i in lvl:
            if packetwise:
                if thr_method=='ipr':
                    if k2 is None: k2=100
                    r = stats.iqr(c,rng=IPR)
                    theta_a = ipr2thr(r,beta=beta,k1=k1,k2=k2,c=est_wmax)
                elif thr_method =='global':
                    sig = np.median(abs(c))/0.6745
                    theta_a = sig*np.sqrt(2*np.log(len(x)))
                elif thr_method =='outliers':
                    ll,ul = Outliers(c)
                    thrlw = ll
                    thrup = ul
                    _,theta_a = Outliers(np.abs(c))
                elif thr_method =='std':
                    theta_a = 1.5*np.std(c)

                theta_b = bf*theta_a
                theta_g = gf*theta_a


            if OptMode=='soft':
                c = SoftThresholding(c,theta_a,theta_g)
            elif OptMode=='linAtten':
                c = LinearAttenuanating(c,theta_a,theta_b)
            elif OptMode=='elim':
                if thr_method =='outliers':
                    c[c>thrup]=0
                    c[c<thrlw]=0
                elif thr_method not in ['std','global']:
                    c = Elimination(c,theta_a)
                else:
                    #method is None, -- apply elimination with given theta_a
                    c = Elimination(c,theta_a)
            else:
                print('Operating mode was not in list..\n No wavelet filtering is applied')
                pass
            if WPD:
                wp[nodes[i].path].data = c
            else:
                wr[i] = c
    #Reconstruction
    if WPD:
        xR = wp.reconstruct(update=False)
    else:
        xR = wt.waverec(wr, wavelet = wv)

    if show_plot:
        plt.figure(figsize=(11,6))
        plt.subplot(211)
        plt.plot(WR,'b',alpha=0.8,label='Coef.')
        plt.ylabel('Wavelete Coefficients')
        ytiW =[np.min(WR),np.max(WR)]
        #print('maxlevel :',wp.maxlevel)
        if WPD: wr = [wp[node.path].data for node in wp.get_level(wp.maxlevel, 'natural') ]
        WRi = np.hstack(wr)
        plt.plot(WRi,'r',alpha=0.9,label='Filtered Coff.')
        ytiW = ytiW+[np.min(WRi),np.max(WRi)]
        plt.yticks(ytiW)
        plt.grid()
        plt.legend()
        plt.xlim([0,len(WRi)])
        plt.subplot(212)
        if WPD:
            t = np.arange(len(wp.data))/fs
            plt.plot(t,wp.data,'b',alpha=0.8,label='signal')
        else:
            t = np.arange(len(x))/fs
            plt.plot(t,x,'b',alpha=0.8,label='signal')
        plt.plot(t,xR,'r',alpha=0.8,label='corrected')
        plt.ylabel('Signal')
        plt.yticks([np.min(xR),np.min(x),0,np.max(xR),np.max(x)])
        plt.xlim([t[0],t[-1]])
        plt.legend()
        plt.grid()
        plt.show()
    return xR

def ATAR_1Ch_dev(x,wv='db3',winsize=128,thr_method='ipr',IPR=[25,75],beta=0.1,k1=None,k2 =100,est_wmax=100,
              theta_a=np.inf,bf=2,gf=0.8,OptMode ='soft',factor=1.0,wpd_mode='symmetric',wpd_maxlevel=None,
              verbose=True, window=['hamming',True],hopesize=None, ReconMethod='custom',packetwise=False,WPD=True,
                  lvl=[],fs=128.0):
    '''
    ---- IN DEVELOPMENT MODE ----- AVOID USING IT FOR NOW -----
    ------------------------------------------------------------
    Apply Wfilter on short windows:

    Wfilter(x,wv='db3',method=None,IPR=[25,75],beta=0.1,k1=None,k2 =100,
     theta_a=np.inf,bf=2,gf=0.8,OptMode ='soft',factor=1.0,showPlot=False,wpd_mode='symmetric',wpd_maxlevel=None)


    Threshold Computation method:
    method : None (default), 'ipr', 'global','outliers'
           : provided with theta_a, bf , gf
           : theta_b = bf*theta_a
           : theta_g = gf*theta_a

    Operating modes:
    OptMode = ['soft','elim','linAtten','clip','logfy','log10fy','sqrtfy','reduceBY','energy', 'removeNode']
             : default 'soft'
             : use 'elim' with global

    Wavelet Decomposition modes:
    wpd_mode = ['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization']
                default 'symmetric'

    Wavelet family:
    wv = ['db3'.....'db38', 'sym2.....sym20', 'coif1.....coif17', 'bior1.1....bior6.8', 'rbio1.1...rbio6.8', 'dmey']
         :'db3'(default)


    Reconstruction Methon
    ReconMethod :  None, 'custom', 'HamWin'
    for 'custom': window[0] is used and applied after denoising is window[1] is True else
    windowing applied before denoising
    '''
    if ReconMethod is None:
        win=np.arange(winsize)
        xR=[]
        pf=0
        while win[-1]<=x.shape[0]:
            if verbose:
                if 100*win[-1]/float(x.shape[0])>=pf+1:
                    pf = 100*win[-1]/float(x.shape[0])
                    pbar = '|'+'#'*int(pf)+' '*(99-int(pf))+'|'
                    print(str(np.round(pf,2))+'%'+pbar,end='\r', flush=True)
            xi = x[win]
            xr = Wfilter_dev(x,wv=wv,thr_method=thr_method,IPR=IPR,beta=beta,k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,
                 OptMode =OptMode,show_plot=False,wpd_mode=wpd_mode,wpd_maxlevel=wpd_maxlevel,
                         packetwise=False,WPD=WPD,lvl=lvl,fs=fs)

            xR.append(xr)
            win+=winsize
        xR = np.hstack(xR)

    elif ReconMethod =='HamWin':
        xt  = np.hstack([np.zeros(winsize//2),x,np.zeros(winsize//2)])
        xR  = np.zeros(xt.shape)
        wh  = signal.windows.hamming(winsize+1)[:winsize]
        win = np.arange(winsize)

        while win[-1]<=xt.shape[0]:
            if verbose:
                pf = 100*win[-1]/float(x.shape[0])
                pbar = '|'+'#'*int(pf)+' '*(99-int(pf))+'|'
                print(str(np.round(pf,2))+'%'+pbar,end='\r', flush=True)
            xi = xt[win]*wh
            xr = Wfilter_dev(xi,wv=wv,thr_method=thr_method,IPR=IPR,beta=beta,k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,
                 OptMode =OptMode,factor=factor,show_plot=False,wpd_mode=wpd_mode,wpd_maxlevel=wpd_maxlevel,
                        packetwise=packetwise,WPD=WPD,lvl=lvl,fs=fs)
            xR[win]+= xr
            win+=winsize//2
        xR = xR/1.08
        xR = xR[winsize//2:-winsize//2]

    elif ReconMethod =='custom':
        if hopesize is None: hopesize = winsize//2
        M   = winsize
        H   = hopesize
        hM1 = (M+1)//2
        hM2 = M//2

        xt  = np.hstack([np.zeros(hM2),x,np.zeros(hM1)])

        pin  = hM1
        pend = xt.size-hM1
        wh   = signal.get_window(window[0],M)
        #if len(window)>1: AfterApply = window[1]
        #else: AfterApply = False
        AfterApply = window[1] if len(window)>1 else False

        if verbose: print('Windowing after apply : ',AfterApply)


        xR   = np.zeros(xt.shape)
        pf=0
        while pin<=pend:
            if verbose:
                if 100*pin/float(pend)>=pf+1:
                    pf = 100*pin/float(pend)
                    pbar = '|'+'#'*int(pf)+' '*(99-int(pf))+'|'
                    print(str(np.round(pf,2))+'%'+pbar,end='\r', flush=True)
            xi = xt[pin-hM1:pin+hM2]
            if not(AfterApply): xi *=wh

            xr = Wfilter_dev(xi,wv=wv,thr_method=thr_method,IPR=IPR,beta=beta,k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,
                 OptMode =OptMode,factor=factor,showPlot=False,wpd_mode=wpd_mode,wpd_maxlevel=wpd_maxlevel,
                        packetwise=packetwise,WPD=WPD,lvl=lvl,fs=fs)

            if AfterApply: xr *=wh

            xR[pin-hM1:pin+hM2] += H*xr  ## Overlap Add method
            pin += H
        xR = xR[hM2:-hM1]/sum(wh)

    if verbose:
        pf = 100
        pbar = '|'+'#'*int(pf)+' '*(99-int(pf))+'|'
        print(str(np.round(pf,2))+'%'+pbar,end='\r', flush=True)
        print('\n')
    return xR

def ATAR_mCh_dev(X,wv='db3',winsize=128,thr_method ='ipr',IPR=[25,75],beta=0.1,k1=10,k2 =100,est_wmax=100,
              theta_a=np.inf,bf=2,gf=0.8,OptMode ='soft',wpd_mode='symmetric',wpd_maxlevel=None,
              verbose=True, window=['hamming',True],hopesize=None, ReconMethod='custom',packetwise=False,WPD=True,lvl=[],fs=128.0):

    '''
    ---- IN DEVELOPMENT MODE ----- AVOID USING IT FOR NOW -----
    ------------------------------------------------------------
    Apply Wfilter on short windows for multiple channels:

    Wfilter(x,wv='db3',thr_method='ipr',IPR=[25,75],beta=0.1,k1=None,k2 =100,
     theta_a=np.inf,bf=2,gf=0.8,OptMode ='soft',factor=1.0,showPlot=False,wpd_mode='symmetric',wpd_maxlevel=None)


    Wavelet family:
    wv = ['db3'.....'db38', 'sym2.....sym20', 'coif1.....coif17', 'bior1.1....bior6.8', 'rbio1.1...rbio6.8', 'dmey']
         :'db3'(default)

    Threshold Computation method:
    thr_method : None (default), 'ipr', 'global','outliers','std'
           : None: fixed threshold theta_a is applied
           : ipr : applied with theta_a, bf , gf, beta, k1, k2 and OptMode
           : global: sig = np.median(abs(WR))/0.6745, theta_a = sig*np.sqrt(2*np.log(len(x)))
           : outliers: remove wavelet co-efficients wf> Q3+1.5IQR and wf<Q1-1.5IQR
           : theta_b = bf*theta_a
           : theta_g = gf*theta_a

    Operating modes:
    OptMode = ['soft','elim','linAtten']
             : default 'soft'
             : use 'elim' with globalgood

    Wavelet Decomposition modes:
    wpd_mode = ['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization']
                default 'symmetric'

    Reconstruction Method
    ReconMethod :  None, 'custom', 'HamWin'

    '''
    if hopesize is None: hopesize=winsize//2

    assert method in [ None, 'ipr', 'global','outliers','std']
    assert OptMode in ['soft','linAtten','elim']


    if verbose:
        print('WPD Artifact Removal')
        print('WPD:',WPD,' Wavelet:',wv,', Method:',method,', OptMode:',OptMode)
        if method=='ipr': print('IPR=',IPR,', Beta:',beta, ', [k1,k2]=',[k1,k2])
        if method is None: print('theta_a: ',theta_a)
        print('Reconstruction Method:',ReconMethod, ', Window:',window,', (Win,Overlap)=',(winsize,hopesize))

    if len(X.shape)>1:
        XR =np.array(Parallel(n_jobs=-1)(delayed(ATAR_1Ch_dev)(X[:,i],wv=wv,winsize=winsize, thr_method=thr_method, IPR=IPR,
              beta=beta, k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,OptMode=OptMode, factor=factor, wpd_mode=wpd_mode,
              wpd_maxlevel=wpd_maxlevel,verbose=verbose, window=window,hopesize=hopesize,
              ReconMethod=ReconMethod,packetwise=packetwise,WPD=WPD,lvl=lvl,fs=fs) for i in range(X.shape[1]))).T
    else:
        XR =ATAR_1Ch_dev(X,wv=wv,winsize=winsize, thr_method=thr_method, IPR=IPR,
              beta=beta, k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,OptMode=OptMode, factor=factor, wpd_mode=wpd_mode,
              wpd_maxlevel=wpd_maxlevel, verbose=verbose, window=window,hopesize=hopesize,
              ReconMethod=ReconMethod,packetwise=packetwise,WPD=WPD,lvl=lvl,fs=fs)
    return XR
