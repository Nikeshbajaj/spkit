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
import warnings
warnings.filterwarnings('once')


from ..utils import ProgBar, ProgBar_JL
from ..stats import outliers
from ..core.processing import wavelet_filtering, wavelet_filtering_win


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

def _wfilter(x,wv='db3',thr_method='ipr',IPR=[25,75],beta=0.1,k1=10,k2=100,theta_a=np.inf,est_wmax=100,
            bf=2,gf=0.8,OptMode ='soft',factor=1.0,show_plot=False,wpd_mode='symmetric',wpd_maxlevel=None,
            WPD=True,packetwise=False,lvl=[],fs=128.0,verbose=0,**kwargs):
    '''Wavelet filtering using ATAR Algorithm
    
    For more details, check:
    Bajaj, Nikesh, et al. "Automatic and tunable algorithm for EEG artifact removal using wavelet decomposition with applications in
    predictive modeling during auditory tasks." Biomedical Signal Processing and Control 55 (2020): 101624.
    https://doi.org/10.1016/j.bspc.2019.101624
    

    Parameters
    ----------
    x: 1d-array (n,)
       single channel EEG signal shape - 

    *Threshold Computation method*

    thr_method : None (default), 'ipr'
           : provided with theta_a, bf, gf
           : where:-
              * theta_b = bf*theta_a  -- used for Linear Attenuation
              * theta_g = gf*theta_a  -- used for Soft thresholding

    *Operating modes*

    OptMode : {'soft','elim','linAtten'}
            : default 'soft'

    *Wavelet Decomposition modes*

    wpd_mode : {'zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization'}
              default 'symmetric'

    *Wavelet family*
    
    wv : str, 'db3'(default)
       {'db3'.....'db38', 'sym2.....sym20', 'coif1.....coif17', 'bior1.1....bior6.8', 'rbio1.1...rbio6.8', 'dmey'}

    IPR: list of two, (default =[25,75])
      - Inter-percentile range - [25,75] is interquartile range, a special case of IPR

    Returns
    -------
    xR: 1d-array (n,)
        - Corrected signal of same shape 


    References
    ----------
    * Bajaj, Nikesh, et al. "Automatic and tunable algorithm for EEG artifact removal using wavelet decomposition with applications in
    predictive modeling during auditory tasks." Biomedical Signal Processing and Control 55 (2020): 101624.
    https://doi.org/10.1016/j.bspc.2019.101624

    
    
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


    '''

    #verbose=False
    if verbose:
        print('WPD:',WPD,' wv:',wv,' IPR:',IPR,' beta:',beta,' method:',thr_method,' OptMode:',OptMode)
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
            if verbose: print('Method for computing threshold is not defined')
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
                    if verbose: print('Method for computing threshold is not defined')
                theta_b = bf*theta_a
                theta_g = gf*theta_a


            if OptMode=='soft':
                c = SoftThresholding(c,theta_a,theta_g)
            elif OptMode=='linAtten':
                c = LinearAttenuanating(c,theta_a,theta_b)
            elif OptMode=='elim':
                c = Elimination(c,theta_a)
            else:
                if verbose: print('Operating mode was not in list..\n No wavelet filtering is applied')
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

def ATAR_1Ch(x,wv='db3',winsize=128,thr_method='ipr',OptMode ='soft',
            IPR=[25,75],beta=0.1,k1=None,k2 =100,est_wmax=100,
            theta_a=np.inf,bf=2,gf=0.8,
            wpd_mode='symmetric',wpd_maxlevel=None,
            recons_method='atar-style',window=['hamming',True], hopesize=None,
            packetwise=False,WPD=True,lvl=[],verbose=False, **kwargs):
    
    
    r"""ATAR : Automatic and Tunable Artifact Removal Algorithm (single channel)
    
    .. raw:: html

        <h2 style="text-align:center">ATAR: - Automatic and Tunable Artifact Removal Algorithm :  Single Channel</h2>

    Apply ATAR on short windows of signal (single channel):

    Signal is decomposed in smaller overlapping windows and reconstructed after correcting using overlap-add method.

    For more details, check [1]_

    **Operating Modes**

    **1. Soft Thresholding**

    .. math ::

       \lambda_s (w)  &=  w  & \quad \text{if } |w|<\theta_{\gamma}

          &=  \theta_{\alpha} \frac{1 - e^{\alpha w}}{1 + e^{\alpha w}}   & \quad \text{otherwise}


    **2. Elimination**

    .. math ::

       \lambda_e (w)  &=  w  & \quad \text{if } |w| \le \theta_{\alpha}

         &=  0   & \quad \text{otherwise}


    **3. Linear Attenuation**
    
    .. math ::

       \lambda_a (w)  &=  w  & \quad \text{if } |w| \le \theta_{\alpha}

              &=  sgn(w) \theta_{\alpha} \Big( 1 -  \frac{|w| - \theta_{\alpha}}{\theta_{\beta}-\theta_{\alpha}} \Big) & \quad \text{if } \theta_{\alpha} < |w| \le \theta_{\beta}
              
              &=  0   & \quad \text{otherwise}


    **Computing Threshold**

    * :math:`f_{\beta}(r)  = k_2 \cdot exp \Big(-\beta \frac{w_{max}}{k_2} \times \frac{r}{2} \Big)`

    * :math:`\theta_{\alpha}  =  f_{\beta}(r)  \quad \text{if } f_{\beta}(r) \ge k_1` otherwise :math:`\theta_{\alpha}  =  k_1`

    * :math:`\theta_{\gamma}  = g_f \times \theta_{\alpha}` ,  where a default value for 'g_f = 0.8' **For Soft-threshold**
    
    * :math:`\theta_{\beta}  = b_f \times \theta_{\alpha}` , where a default value for 'b_f = 2' **For Linear Attenuation**




    ..
        _wfilter(x,wv='db3',thr_method='ipr',IPR=[25,75],beta=0.1,k1=10,k2=100,theta_a=np.inf,est_wmax=100,
            bf=2,gf=0.8,OptMode ='soft',factor=1.0,show_plot=False,wpd_mode='symmetric',wpd_maxlevel=None,
            WPD=True,packetwise=False,lvl=[],fs=128.0,verbose=0)

    Parameters
    ----------
    x : 1d-array
      - input single-channel signal of shape (n,)

    wv = str, 'db3'(default) wavelet
      - one of wavelet family {'db3'.....'db38', 'sym2.....sym20', 
                     'coif1.....coif17', 'bior1.1....bior6.8', 
                    'rbio1.1...rbio6.8', 'dmey'}

    winsize: int, deafult=128
       -  window size to apply ATAR
    
    hopesize: int, None, default=None,
       - overlap shift for next window
       - if None, hopesize=winsize//2
       - only used when recons_method={'atar-style','atar','custom'}


    Operating modes

    OptMode: str, {'soft','elim','linAtten'}, default 'soft'
        - Operating Modes:
        - soft: soft-thresholding (need theta_a, and theta_g)
        - elim: Elimination mode  (need theta_a)
        - linAtten: Linear Attenuation mode (need theta_a and theta_b)
            - given `bf` and `gf` (default bf = 2, gf=0.8)
            -  where:-
                * theta_b = bf*theta_a  -- used for Linear Attenuation
                * theta_g = gf*theta_a  -- used for Soft thresholding

    *Threshold Computation method*

    thr_method: str, {'ipr', None}
        - Computing method for threshold
        - if 'ipr'  'theta_a' is computed as per ATAR's approach
        - if None, passed value of 'theta_a' is used along with (bf, gf)
    
    IPR: list of two default=[25,75]
       - To compute Interpercentile range r
       - e.g. [10,90], [30,70]
       - Higher the range is, threshold is more affected by outliers
       - Interpercentile range r is mapped to threshold, using `beta`, `k1` and `k2`
    
    beta: float (0,1] default=0.1
       - beta as tuning parameter for threshold
       - higher the value, more aggressive is the algorithm to remove artifacts

    (k1,k2) :scalars, defualt (10,100)
       - lower and upper bounds on the threshold value
       - should be of same order as signal magnitude.
       - if signal is in volts, values will be of order 1e-6, then k1 and k2 should be around those order
       - defult values k1=10, k2=100 are chosen for signal with unit of microVolt (uV), of order of 100s
       

       .. warning:: k1,k2 bounds
          if k2 is very high (e.g. 100) and signal amplitude is in 1e-3, 1e-6, 
          ATAR will have no effect on the signal, as theshold to identify and remove artifacts will be so high.

    
    est_wmax: int, default=100
        - est_wmax is the value in the expression (15 in paper) or above Computing Threshold equations `w_max`
    

    *Wavelet Decomposition*

    wpd_mode: str,  default 'symmetric'
        -  one of the {'zero', 'constant', 'symmetric', 'periodic',
            'smooth', 'periodization'}

    wpd_maxlevel: int, defualt=None
        - maximum number of levels for decomposition,
        - if None, max possible number of level are used.

    verbose: int
       - verbosity mode

    Experimental Settings

        .. note:: NOT RECOMMONEDED TO CHANGE
            Following parameters are experimental,
            they are not recommonded to change, and leave as default

    WPD: bool, default=True
       - if False Wavelet Transform is used for decompisiton of signal,
         else Wavelet Packet Decomposition is used
    
    packetwise: bool, deafult=False
       -  if True, threshold is computed and applied to each packet independently. 

    lvl: list defualt=[]
       -  if provided, ATAR is applied to provided level numbers only.
     
    recons_method: str deafult='atar-style'
       - reconstruction method after applying atar to each window.
       - one of {'atar-style', 'atar', 'custom', 'HamWin','Hamming'}

       .. note:: NOT RECOMMONEDED TO CHANGE
          KEEP IT TO DEFAULT
    
    
    window: list of two default=['hamming',True]
        -  window function, and if windowing is applied before or after atar


       
    Returns
    -------
    XR: corrected signal of same shape as input X

    Reference
    ---------
    .. [1] Bajaj, Nikesh, et al. "Automatic and tunable algorithm for EEG artifact removal using wavelet decomposition with applications in predictive modeling during auditory tasks." Biomedical Signal Processing and Control 55 (2020): 101624. https://doi.org/10.1016/j.bspc.2019.101624 


    See Also
    --------
    ATAR: Automatic and Tuanable Artifact Removal Algorithm
    ATAR_mCh: ATAR for multiple channel
    ICA_filtering: ICA based Artifact Removal Algorithm


    Examples
    --------
    #sp.eeg.ATAR_1Ch
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    X,fs, ch_names = sp.data.eeg_sample_14ch()
    X = sp.filterDC_sGolay(X, window_length=fs//3+1)
    t = np.arange(X.shape[0])/fs
    # Single Channel
    x = X[:,0] 
    xc1 = sp.eeg.ATAR_1Ch(x,wv='db3', winsize=128, thr_method='ipr',beta=0.1, k1=10, k2=100,OptMode='soft',verbose=1)
    xc2 = sp.eeg.ATAR_1Ch(x,wv='db3', winsize=128, thr_method='ipr',beta=0.1, k1=10, k2=100,OptMode='linAtten',verbose=1)
    xc3 = sp.eeg.ATAR_1Ch(x,wv='db3', winsize=128, thr_method='ipr',beta=0.1, k1=10, k2=100,OptMode='elim',verbose=1)
    plt.figure(figsize=(10,6))
    plt.subplot(211)
    plt.plot(t,x, label='$x$: raw EEG - single channel')
    plt.plot(t,xc1,label=r'$x_{c1}$: Soft Thresholding')
    plt.plot(t,xc2,label=r'$x_{c2}$: Linear Attenuation')
    plt.plot(t,xc3,label=r'$x_{c3}$: Elimination')
    #plt.xlim([9,12])
    #plt.ylim([-200,200])
    plt.legend(bbox_to_anchor=(0.5,0.99),ncol=2,fontsize=8)
    plt.grid()
    plt.title(r'ATAR Algorithm')
    plt.subplot(212)
    plt.plot(t,x, label='$x$: raw EEG - single channel')
    plt.plot(t,xc1,label=r'$x_{c1}$: Soft Thresholding')
    plt.plot(t,xc2,label=r'$x_{c2}$: Linear Attenuation')
    plt.plot(t,xc3,label=r'$x_{c3}$: Elimination')
    plt.xlim([9,12])
    plt.ylim([-200,200])
    plt.legend(bbox_to_anchor=(0.5,0.99),ncol=2,fontsize=8)
    plt.grid()
    plt.title(r'ATAR Algorithm')
    plt.xlabel('time (s)')
    plt.tight_layout()
    plt.show()
    """

    warn_str = ' This will be removed in future versions. To turn of this warning, set `warn=False`. [0.0.9.7]'
    
    WARN = True
    if 'warn' in kwargs:
        WARN = kwargs['warn']

    if 'ReconMethod' in kwargs:
        recons_method = kwargs['ReconMethod']
        if WARN:
            warnings.warn('Argument `ReconMethod` is changed to `recons_method`' + warn_str )


    #ReconMethod='custom',

    if (k1 is not None and np.max(x)<k1) or np.max(x)<k2:
        warnings.warn('Make sure the upper and lower bound values (k1,k2) are of same order as signal amplitude. If amplitude of signal is much lower than k2 or even k1, ATAR algorithm will have no affect on signal. For example, k2=100, and/or k1=10 is setting for amplitude in micro-volt (in order of 100s). If provided signal is in volt (1e-6), multiply signal with 1e6 (X*1e6) and then apply ATAR')

    if np.max(x)*100<k2:
        warnings.warn('Upper bound k2 is set to very high. ATAR might have no impact of signal. Either change amplitude unit of signal by multiplying 1e3, or 1e6, or lower the value of k2 and respectively, k1.  One of the straightforward way to set k2 is k2 = np.std(X).')

    if hopesize is None: hopesize = winsize//2

    if recons_method is None:
        #if ReconMethod is None:
        win=np.arange(winsize)
        xR=[]
        pf=0
        while win[-1]<=x.shape[0]:
            if verbose:
                ProgBar_JL(win[-1], x.shape[0], style=2,L=50,selfTerminate=True,color='red',title=f' Mode : {OptMode}    ')
                # if 100*win[-1]/float(x.shape[0])>=pf+1:
                #     pf = 100*win[-1]/float(x.shape[0])
                #     pbar = '|'+'#'*int(pf)+' '*(99-int(pf))+'|'
                #     print(str(np.round(pf,2))+'%'+pbar,end='\r', flush=True)
            xi = x[win]
            xr = _wfilter(x,wv=wv,thr_method=thr_method,IPR=IPR,beta=beta,k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,
                 OptMode =OptMode,show_plot=False,wpd_mode=wpd_mode,wpd_maxlevel=wpd_maxlevel,
                         packetwise=False,WPD=WPD,lvl=lvl)

            xR.append(xr)
            win+=winsize
        xR = np.hstack(xR)

    elif recons_method in ['HamWin','Hamming']:
        #elif ReconMethod =='HamWin':
        xt  = np.hstack([np.zeros(winsize//2),x,np.zeros(winsize//2)])
        xR  = np.zeros(xt.shape)
        wh  = signal.windows.hamming(winsize+1)[:winsize]
        win = np.arange(winsize)

        while win[-1]<=xt.shape[0]:
            if verbose:
                ProgBar_JL(win[-1], x.shape[0], style=2,L=50,selfTerminate=True,color='blue', title=f' Mode : {OptMode}    ')
                # pf = 100*win[-1]/float(x.shape[0])
                # pbar = '|'+'#'*int(pf)+' '*(99-int(pf))+'|'
                # print(str(np.round(pf,2))+'%'+pbar,end='\r', flush=True)
            xi = xt[win]*wh
            xr = _wfilter(xi,wv=wv,thr_method=thr_method,IPR=IPR,beta=beta,k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,
                 OptMode =OptMode,show_plot=False,wpd_mode=wpd_mode,wpd_maxlevel=wpd_maxlevel,
                        packetwise=packetwise,WPD=WPD,lvl=lvl)
            xR[win]+= xr
            win+=winsize//2
        xR = xR/1.08
        xR = xR[winsize//2:-winsize//2]

    elif recons_method in ['atar-style','atar','custom']:
        #elif ReconMethod =='custom':
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
        if verbose>1:
            print('Windowing after apply : ',AfterApply)
        
        xR   = np.zeros(xt.shape)
        pf=0
        while pin<=pend:
            if verbose:
                #ProgBar_JL(i,N,title='',style=2,L=50,selfTerminate=True,delta=None,sym='▓',color='blue')
                ProgBar_JL(pin, pend, style=2,L=50,selfTerminate=True,color='green',title=f' Mode : {OptMode}    ')
                # if 100*pin/float(pend)>=pf+1:
                #     pf = 100*pin/float(pend)
                #     pbar = '|'+'#'*int(pf)+' '*(99-int(pf))+'|'
                #     print(str(np.round(pf,2))+'%'+pbar,end='\r', flush=True)
            
            xi = xt[pin-hM1:pin+hM2]
            if not(AfterApply): xi *=wh

            xr = _wfilter(xi,wv=wv,thr_method=thr_method,IPR=IPR,beta=beta,k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,
                 OptMode =OptMode,show_plot=False,wpd_mode=wpd_mode,wpd_maxlevel=wpd_maxlevel,
                        packetwise=packetwise,WPD=WPD,lvl=lvl)

            if AfterApply: xr *=wh

            xR[pin-hM1:pin+hM2] += H*xr  ## Overlap Add method
            pin += H
        xR = xR[hM2:-hM1]/sum(wh)

    #if verbose:
        # pf = 100
        # pbar = '|'+'#'*int(pf)+' '*(99-int(pf))+'|'
        # print(str(np.round(pf,2))+'%'+pbar,end='\r', flush=True)
        # print('\n')
    return xR

def ATAR_mCh(X,wv='db3',winsize=128,thr_method ='ipr',OptMode ='elim',
              IPR=[25,75],beta=0.1,k1=10,k2 =100,est_wmax=100,
              theta_a=np.inf,bf=2,gf=0.8,wpd_mode='symmetric',
              wpd_maxlevel=None,
              verbose=False, window=['hamming',True],hopesize=None, 
              recons_method='atar-style',packetwise=False,WPD=True,lvl=[],
              use_joblib=False, **kwargs):
    r"""ATAR: - Automatic and Tunable Artifact Removal Algorithm

    .. raw:: html

        <h2 style="text-align:center">ATAR: - Automatic and Tunable Artifact Removal Algorithm: Multi-channels</h2>

    Apply ATAR on short windows of signal

    Signal is decomposed in smaller overlapping windows and reconstructed after correcting using overlap-add method.

    For more details, check [1]_

    **Operating Modes**

    **1. Soft Thresholding**

    .. math ::

       \lambda_s (w)  &=  w  & \quad \text{if } |w|<\theta_{\gamma}

          &=  \theta_{\alpha} \frac{1 - e^{\alpha w}}{1 + e^{\alpha w}}   & \quad \text{otherwise}


    **2. Elimination**

    .. math ::

       \lambda_e (w)  &=  w  & \quad \text{if } |w| \le \theta_{\alpha}

         &=  0   & \quad \text{otherwise}


    **3. Linear Attenuation**
    
    .. math ::

       \lambda_a (w)  &=  w  & \quad \text{if } |w| \le \theta_{\alpha}

              &=  sgn(w) \theta_{\alpha} \Big( 1 -  \frac{|w| - \theta_{\alpha}}{\theta_{\beta}-\theta_{\alpha}} \Big) & \quad \text{if } \theta_{\alpha} < |w| \le \theta_{\beta}
              
              &=  0   & \quad \text{otherwise}


    **Computing Threshold**

    * :math:`f_{\beta}(r)  = k_2 \cdot exp \Big(-\beta \frac{w_{max}}{k_2} \times \frac{r}{2} \Big)`

    * :math:`\theta_{\alpha}  =  f_{\beta}(r)  \quad \text{if } f_{\beta}(r) \ge k_1` otherwise :math:`\theta_{\alpha}  =  k_1`

    * :math:`\theta_{\gamma}  = g_f \times \theta_{\alpha}` ,  where a default value for 'g_f = 0.8' **For Soft-threshold**
    
    * :math:`\theta_{\beta}  = b_f \times \theta_{\alpha}` , where a default value for 'b_f = 2' **For Linear Attenuation**


    Parameters
    ----------
    x : 1d-array
      - input single-channel signal of shape (n,)

    wv = str, 'db3'(default) wavelet
      - one of wavelet family {'db3'.....'db38', 'sym2.....sym20', 
                     'coif1.....coif17', 'bior1.1....bior6.8', 
                    'rbio1.1...rbio6.8', 'dmey'}

    winsize: int, deafult=128
       -  window size to apply ATAR
    
    hopesize: int, None, default=None,
       - overlap shift for next window
       - if None, hopesize=winsize//2
       - only used when recons_method={'atar-style','atar','custom'}


    Operating modes

    OptMode: str, {'soft','elim','linAtten'}, default 'soft'
        - Operating Modes:
        - soft: soft-thresholding (need theta_a, and theta_g)
        - elim: Elimination mode  (need theta_a)
        - linAtten: Linear Attenuation mode (need theta_a and theta_b)
            - given `bf` and `gf` (default bf = 2, gf=0.8)
            -  where:-
                * theta_b = bf*theta_a  -- used for Linear Attenuation
                * theta_g = gf*theta_a  -- used for Soft thresholding

    *Threshold Computation method*

    thr_method: str, {'ipr', None}
        - Computing method for threshold
        - if 'ipr'  'theta_a' is computed as per ATAR's approach
        - if None, passed value of 'theta_a' is used along with (bf, gf)
    
    IPR: list of two default=[25,75]
       - To compute Interpercentile range r
       - e.g. [10,90], [30,70]
       - Higher the range is, threshold is more affected by outliers
       - Interpercentile range r is mapped to threshold, using `beta`, `k1` and `k2`
    
    beta: float (0,1] default=0.1
       - beta as tuning parameter for threshold
       - higher the value, more aggressive is the algorithm to remove artifacts

    (k1,k2) :scalars, defualt (10,100)
       - lower and upper bounds on the threshold value
       - should be of same order as signal magnitude.
       - if signal is in volts, values will be of order 1e-6, then k1 and k2 should be around those order
       - defult values k1=10, k2=100 are chosen for signal with unit of microVolt (uV), of order of 100s
       

       .. warning:: k1,k2 bounds
          if k2 is very high (e.g. 100) and signal amplitude is in 1e-3, 1e-6, 
          ATAR will have no effect on the signal, as theshold to identify and remove artifacts will be so high.

    
    est_wmax: int, default=100
        - est_wmax is the value in the expression (15 in paper) or above Computing Threshold equations `w_max`
    

    *Wavelet Decomposition*

    wpd_mode: str,  default 'symmetric'
        -  one of the {'zero', 'constant', 'symmetric', 'periodic',
            'smooth', 'periodization'}

    wpd_maxlevel: int, defualt=None
        - maximum number of levels for decomposition,
        - if None, max possible number of level are used.

    use_joblib: bool, default=False
        -  If True, joblib is used for parallel processing of the channels

    verbose: int
       - verbosity mode

    Experimental Settings

        .. note:: NOT RECOMMONEDED TO CHANGE
            Following parameters are experimental,
            they are not recommonded to change, and leave as default

    WPD: bool, default=True
       - if False Wavelet Transform is used for decompisiton of signal,
         else Wavelet Packet Decomposition is used
    
    packetwise: bool, deafult=False
       -  if True, threshold is computed and applied to each packet independently. 

    lvl: list defualt=[]
       -  if provided, ATAR is applied to provided level numbers only.
     
    recons_method: str deafult='atar-style'
       - reconstruction method after applying atar to each window.
       - one of {'atar-style', 'atar', 'custom', 'HamWin','Hamming'}

       .. note:: NOT RECOMMONEDED TO CHANGE
          KEEP IT TO DEFAULT
    
    
    window: list of two default=['hamming',True]
        -  window function, and if windowing is applied before or after atar

    Returns
    -------
    XR: corrected signal of same shape as input X
    
    
    References
    ----------
    .. [1] Bajaj, Nikesh, et al. "Automatic and tunable algorithm for EEG artifact removal using wavelet decomposition with applications in predictive modeling during auditory tasks." Biomedical Signal Processing and Control 55 (2020): 101624. https://doi.org/10.1016/j.bspc.2019.101624
    


    See Also
    --------
    ATAR: Automatic and Tuanable Artifact Removal Algorithm
    ATAR_1Ch: ATAR for single channel
    ICA_filtering: ICA based Artifact Removal Algorithm

    Examples
    --------
    #sp.eeg.ATAR_mCh
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    X,fs, ch_names = sp.data.eeg_sample_14ch()
    X = sp.filterDC_sGolay(X, window_length=fs//3+1)
    t = np.arange(X.shape[0])/fs
    # Multi-Channel
    Xc = sp.eeg.ATAR_mCh(X,wv='db3', winsize=128, thr_method='ipr',beta=0.2, k1=10, k2=100,OptMode='elim',verbose=1)
    sep=200
    plt.figure(figsize=(10,6))
    plt.subplot(211)
    plt.plot(t,x, label='$x$: raw EEG - single channel')
    plt.plot(t,xc1,label=r'$x_{c1}$: Soft Thresholding')
    plt.plot(t,xc2,label=r'$x_{c2}$: Linear Attenuation')
    plt.plot(t,xc3,label=r'$x_{c3}$: Elimination')
    plt.xlim([9,12])
    plt.ylim([-200,200])
    plt.legend(bbox_to_anchor=(0.5,0.99),ncol=2,fontsize=8)
    plt.grid()
    plt.title(r'ATAR Algorithm')
    plt.xlabel('time (s)')
    plt.subplot(223)
    plt.plot(t,X+np.arange(X.shape[1])*sep)
    plt.xlim([t[0],t[-1]])
    plt.yticks(np.arange(X.shape[1])*sep,ch_names)
    plt.title(r'$X$: EEG')
    plt.xlabel('time (s)')
    plt.subplot(224)
    plt.plot(t,Xc+np.arange(14)*sep)
    plt.xlim([t[0],t[-1]])
    plt.title(r'$X_c$: ATAR (Elimination Mode)')
    plt.yticks(np.arange(X.shape[1])*sep,ch_names)
    plt.xlabel('time (s)')
    plt.tight_layout()
    plt.show()
    """
    if hopesize is None: hopesize=winsize//2

    assert thr_method in [ None, 'ipr']
    assert OptMode in ['soft','linAtten','elim']

    if (k1 is not None and np.max(X)<k1) or np.max(X)<k2:
        warnings.warn('Make sure the upper and lower bound values (k1,k2) are of same order as signal amplitude. If amplitude of signal is much lower than k2 or even k1, ATAR algorithm will have no affect on signal. For example, k2=100, and/or k1=10 is setting for amplitude in micro-volt (in order of 100s). If provided signal is in volt (1e-6), multiply signal with 1e6 (X*1e6) and then apply ATAR')

    if np.max(X)*100 < k2:
        warnings.warn('Upper bound k2 is set to very high. ATAR might have no impact of signal. Either change amplitude unit of signal by multiplying 1e3, or 1e6, or lower the value of k2 and respectively, k1.')



    if verbose>1:
        print('WPD Artifact Removal')
        print('WPD:',WPD,' Wavelet:',wv,', Method:',thr_method,', OptMode:',OptMode)
        if thr_method=='ipr': print('IPR=',IPR,', Beta:',beta, ', [k1,k2]=',[k1,k2])
        if thr_method is None: print('theta_a: ',theta_a)
        print('Reconstruction Method:',recons_method, ', Window:',window,', (Win,Overlap)=',(winsize,hopesize))

    if len(X.shape)>1:
        if use_joblib:
            XR = np.array(Parallel(n_jobs=-1)(delayed(ATAR_1Ch)(X[:,i],wv=wv,winsize=winsize, thr_method=thr_method, IPR=IPR,
                  beta=beta, k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,OptMode=OptMode,wpd_mode=wpd_mode,
                  wpd_maxlevel=wpd_maxlevel,verbose=verbose, window=window,hopesize=hopesize,
                  recons_method=recons_method,packetwise=packetwise,WPD=WPD,lvl=lvl) for i in range(X.shape[1]))).T
        else:
            XR =np.array([ATAR_1Ch(X[:,i],wv=wv,winsize=winsize, thr_method=thr_method, IPR=IPR,
              beta=beta, k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,OptMode=OptMode,wpd_mode=wpd_mode,
              wpd_maxlevel=wpd_maxlevel,verbose=0, window=window,hopesize=hopesize,
              recons_method=recons_method,packetwise=packetwise,WPD=WPD,lvl=lvl) for i in range(X.shape[1])]).T
    else:
        XR =ATAR_1Ch(X,wv=wv,winsize=winsize, thr_method=thr_method, IPR=IPR,
              beta=beta, k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,OptMode=OptMode,wpd_mode=wpd_mode,
              wpd_maxlevel=wpd_maxlevel, verbose=verbose, window=window,hopesize=hopesize,
              recons_method=recons_method,packetwise=packetwise,WPD=WPD,lvl=lvl)
    return XR

def ATAR(X,wv='db3',winsize=128,thr_method ='ipr',OptMode ='soft',
              IPR=[25,75],beta=0.1,k1=10,k2=100,est_wmax=100,
              theta_a=np.inf,bf=2,gf=0.8,
              wpd_mode='symmetric',wpd_maxlevel=None,
              recons_method='atar-style', window=['hamming',True],hopesize=None, 
              packetwise=False,WPD=True,lvl=[],use_joblib=False,verbose=False,**kwargs):

    r"""ATAR - Automatic and Tunable Artifact Removal Algorithm

    .. raw:: html

        <h2 style="text-align:center">ATAR: - Automatic and Tunable Artifact Removal Algorithm</h2>

    Apply ATAR on short windows of signal (single channel):

    Signal is decomposed in smaller overlapping windows and reconstructed after correcting using overlap-add method.

    For more details, check [1]_

    **Operating Modes**

    **1. Soft Thresholding**

    .. math ::

       \lambda_s (w)  &=  w  & \quad \text{if } |w|<\theta_{\gamma}

          &=  \theta_{\alpha} \frac{1 - e^{\alpha w}}{1 + e^{\alpha w}}   & \quad \text{otherwise}


    **2. Elimination**

    .. math ::

       \lambda_e (w)  &=  w  & \quad \text{if } |w| \le \theta_{\alpha}

         &=  0   & \quad \text{otherwise}


    **3. Linear Attenuation**
    
    .. math ::

       \lambda_a (w)  &=  w  & \quad \text{if } |w| \le \theta_{\alpha}

              &=  sgn(w) \theta_{\alpha} \Big( 1 -  \frac{|w| - \theta_{\alpha}}{\theta_{\beta}-\theta_{\alpha}} \Big) & \quad \text{if } \theta_{\alpha} < |w| \le \theta_{\beta}
              
              &=  0   & \quad \text{otherwise}


    **Computing Threshold**

    * :math:`f_{\beta}(r)  = k_2 \cdot exp \Big(-\beta \frac{w_{max}}{k_2} \times \frac{r}{2} \Big)`

    * :math:`\theta_{\alpha}  =  f_{\beta}(r)  \quad \text{if } f_{\beta}(r) \ge k_1` otherwise :math:`\theta_{\alpha}  =  k_1`

    * :math:`\theta_{\gamma}  = g_f \times \theta_{\alpha}` ,  where a default value for 'g_f = 0.8' **For Soft-threshold**
    
    * :math:`\theta_{\beta}  = b_f \times \theta_{\alpha}` , where a default value for 'b_f = 2' **For Linear Attenuation**


    Parameters
    ----------
    x : 1d-array
      - input single-channel signal of shape (n,)

    wv = str, 'db3'(default) wavelet
      - one of wavelet family {'db3'.....'db38', 'sym2.....sym20', 
                     'coif1.....coif17', 'bior1.1....bior6.8', 
                    'rbio1.1...rbio6.8', 'dmey'}

    winsize: int, deafult=128
       -  window size to apply ATAR
    
    hopesize: int, None, default=None,
       - overlap shift for next window
       - if None, hopesize=winsize//2
       - only used when recons_method={'atar-style','atar','custom'}


    Operating modes

    OptMode: str, {'soft','elim','linAtten'}, default 'soft'
        - Operating Modes:
        - soft: soft-thresholding (need theta_a, and theta_g)
        - elim: Elimination mode  (need theta_a)
        - linAtten: Linear Attenuation mode (need theta_a and theta_b)
            - given `bf` and `gf` (default bf = 2, gf=0.8)
            -  where:-
                * theta_b = bf*theta_a  -- used for Linear Attenuation
                * theta_g = gf*theta_a  -- used for Soft thresholding

    *Threshold Computation method*

    thr_method: str, {'ipr', None}
        - Computing method for threshold
        - if 'ipr'  'theta_a' is computed as per ATAR's approach
        - if None, passed value of 'theta_a' is used along with (bf, gf)
    
    IPR: list of two default=[25,75]
       - To compute Interpercentile range r
       - e.g. [10,90], [30,70]
       - Higher the range is, threshold is more affected by outliers
       - Interpercentile range r is mapped to threshold, using `beta`, `k1` and `k2`
    
    beta: float (0,1] default=0.1
       - beta as tuning parameter for threshold
       - higher the value, more aggressive is the algorithm to remove artifacts

    (k1,k2) :scalars, defualt (10,100)
       - lower and upper bounds on the threshold value
       - should be of same order as signal magnitude.
       - if signal is in volts, values will be of order 1e-6, then k1 and k2 should be around those order
       - defult values k1=10, k2=100 are chosen for signal with unit of microVolt (uV), of order of 100s
       

       .. warning:: k1,k2 bounds
          if k2 is very high (e.g. 100) and signal amplitude is in 1e-3, 1e-6, 
          ATAR will have no effect on the signal, as theshold to identify and remove artifacts will be so high.

    
    est_wmax: int, default=100
        - est_wmax is the value in the expression (15 in paper) or above Computing Threshold equations `w_max`
    

    *Wavelet Decomposition*

    wpd_mode: str,  default 'symmetric'
        -  one of the {'zero', 'constant', 'symmetric', 'periodic',
            'smooth', 'periodization'}

    wpd_maxlevel: int, defualt=None
        - maximum number of levels for decomposition,
        - if None, max possible number of level are used.

    use_joblib: bool, default=False
        -  If True, joblib is used for parallel processing of the channels

    verbose: int
       -  vebosity level


    Experimental Settings

        .. note:: NOT RECOMMONEDED TO CHANGE
            Following parameters are experimental,
            they are not recommonded to change, and leave as default

    WPD: bool, default=True
       - if False Wavelet Transform is used for decompisiton of signal,
         else Wavelet Packet Decomposition is used
    
    packetwise: bool, deafult=False
       -  if True, threshold is computed and applied to each packet independently. 

    lvl: list defualt=[]
       -  if provided, ATAR is applied to provided level numbers only.
     
    recons_method: str deafult='atar-style'
       - reconstruction method after applying atar to each window.
       - one of {'atar-style', 'atar', 'custom', 'HamWin','Hamming'}

       .. note:: NOT RECOMMONEDED TO CHANGE
          KEEP IT TO DEFAULT
    
    
    window: list of two default=['hamming',True]
        -  window function, and if windowing is applied before or after atar


    Returns
    -------
    XR: corrected signal of same shape as input X
    
    
    References
    ----------
    .. [1] Bajaj, Nikesh, et al. "Automatic and tunable algorithm for EEG artifact removal using wavelet decomposition with applications in predictive modeling during auditory tasks." Biomedical Signal Processing and Control 55 (2020): 101624. https://doi.org/10.1016/j.bspc.2019.101624
    



    See Also
    --------
    ATAR_1Ch: ATAR for single channel
    ATAR_mCh: ATAR for multiple channel
    ICA_filtering: ICA based Artifact Removal Algorithm


    Examples
    --------
    #sp.eeg.ATAR
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    X,fs, ch_names = sp.data.eeg_sample_14ch()
    X = sp.filterDC_sGolay(X, window_length=fs//3+1)
    t = np.arange(X.shape[0])/fs
    # Single Channel
    x = X[:,0] 
    xc1 = sp.eeg.ATAR(x,wv='db3', winsize=128, thr_method='ipr',beta=0.1, k1=10, k2=100,OptMode='soft',verbose=1)
    xc2 = sp.eeg.ATAR(x,wv='db3', winsize=128, thr_method='ipr',beta=0.1, k1=10, k2=100,OptMode='linAtten',verbose=1)
    xc3 = sp.eeg.ATAR(x,wv='db3', winsize=128, thr_method='ipr',beta=0.1, k1=10, k2=100,OptMode='elim',verbose=1)
    # Multi-Channel
    Xc = sp.eeg.ATAR(X,wv='db3', winsize=128, thr_method='ipr',beta=0.2, k1=10, k2=100,OptMode='elim')
    sep=200
    plt.figure(figsize=(10,6))
    plt.subplot(211)
    plt.plot(t,x, label='$x$: raw EEG - single channel')
    plt.plot(t,xc1,label=r'$x_{c1}$: Soft Thresholding')
    plt.plot(t,xc2,label=r'$x_{c2}$: Linear Attenuation')
    plt.plot(t,xc3,label=r'$x_{c3}$: Elimination')
    plt.xlim([9,12])
    plt.ylim([-200,200])
    plt.legend(bbox_to_anchor=(0.5,0.99),ncol=2,fontsize=8)
    plt.grid()
    plt.title(r'ATAR Algorithm')
    plt.xlabel('time (s)')
    plt.subplot(223)
    plt.plot(t,X+np.arange(X.shape[1])*sep)
    plt.xlim([t[0],t[-1]])
    plt.yticks(np.arange(X.shape[1])*sep,ch_names)
    plt.title(r'$X$: EEG')
    plt.xlabel('time (s)')
    plt.subplot(224)
    plt.plot(t,Xc+np.arange(14)*sep)
    plt.xlim([t[0],t[-1]])
    plt.title(r'$X_c$: ATAR (Elimination Mode)')
    plt.yticks(np.arange(X.shape[1])*sep,ch_names)
    plt.xlabel('time (s)')
    plt.tight_layout()
    plt.show()
    """

    warn_str = ' This will be removed in future versions. To turn of this warning, set `warn=False`. [0.0.9.7]'
    
    WARN = True
    if 'warn' in kwargs:
        WARN = kwargs['warn']

    if 'ReconMethod' in kwargs:
        recons_method = kwargs['ReconMethod']
        if WARN:
            warnings.warn('Argument `ReconMethod` is changed to `recons_method`' + warn_str )

    assert thr_method in [ None, 'ipr']
    assert OptMode in ['soft','linAtten','elim']

    if (k1 is not None and np.max(X)<k1) or np.max(X)<k2:
        warnings.warn('Make sure the upper and lower bound values (k1,k2) are of same order as signal amplitude. If amplitude of signal is much lower than k2 or even k1, ATAR algorithm will have no affect on signal. For example, k2=100, and/or k1=10 is setting for amplitude in micro-volt (in order of 100s). If provided signal is in volt (1e-6), multiply signal with 1e6 (X*1e6) and then apply ATAR')

    if np.max(X)*100 < k2:
        warnings.warn('Upper bound k2 is set to very high. ATAR might have no impact of signal. Either change amplitude unit of signal by multiplying 1e3, or 1e6, or lower the value of k2 and respectively, k1.')



    if verbose>1:
        print('WPD Artifact Removal')
        print('WPD:',WPD,' Wavelet:',wv,', Method:',thr_method,', OptMode:',OptMode)
        if thr_method=='ipr': print('IPR=',IPR,', Beta:',beta, ', [k1,k2]=',[k1,k2])
        if thr_method is None: print('theta_a: ',theta_a)
        print('Reconstruction Method:',recons_method, ', Window:',window,', (Win,Overlap)=',(winsize,hopesize))

    if len(X.shape)>1:
        if use_joblib:
            XR = np.array(Parallel(n_jobs=-1)(delayed(ATAR_1Ch)(X[:,i],wv=wv,winsize=winsize, thr_method=thr_method, IPR=IPR,
                  beta=beta, k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,OptMode=OptMode, wpd_mode=wpd_mode,
                  wpd_maxlevel=wpd_maxlevel,verbose=verbose, window=window,hopesize=hopesize,
                  recons_method=recons_method,packetwise=packetwise,WPD=WPD,lvl=lvl) for i in range(X.shape[1]))).T
        else:
            XR =np.array([ATAR_1Ch(X[:,i],wv=wv,winsize=winsize, thr_method=thr_method, IPR=IPR,
              beta=beta, k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,OptMode=OptMode,wpd_mode=wpd_mode,
              wpd_maxlevel=wpd_maxlevel,verbose=0, window=window,hopesize=hopesize,
              recons_method=recons_method,packetwise=packetwise,WPD=WPD,lvl=lvl) for i in range(X.shape[1])]).T
    else:
        XR =ATAR_1Ch(X,wv=wv,winsize=winsize, thr_method=thr_method, IPR=IPR,
              beta=beta, k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,OptMode=OptMode,wpd_mode=wpd_mode,
              wpd_maxlevel=wpd_maxlevel, verbose=verbose, window=window,hopesize=hopesize,
              recons_method=recons_method,packetwise=packetwise,WPD=WPD,lvl=lvl)
    return XR





def ATAR_mCh_noParallel(X,wv='db3',winsize=128,thr_method ='ipr',IPR=[25,75],beta=0.1,k1=10,k2 =100,est_wmax=100,
              theta_a=np.inf,bf=2,gf=0.8,OptMode ='soft',wpd_mode='symmetric',wpd_maxlevel=None,factor=1.0,
              verbose=False, window=['hamming',True],hopesize=None, ReconMethod='custom',packetwise=False,WPD=True,lvl=[],fs=128.0):
    '''
    ''
    ATAR: - Automatic and Tunable Artifact Removal Algorithm
    ========================================================

    Apply ATAR on short windows of signal (multiple channels:): - Without using Joblib - in case that creates issue in some systems and IDE

    Signal is decomposed in smaller overlapping windows and reconstructed after correcting using overlap-add method.
    
    For more details, check:
    Ref: Bajaj, Nikesh, et al. "Automatic and tunable algorithm for EEG artifact removal using wavelet decomposition with applications in predictive modeling during auditory tasks." Biomedical Signal Processing and Control 55 (2020): 101624.
    https://doi.org/10.1016/j.bspc.2019.101624

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


    if verbose >1:
        print('WPD Artifact Removal')
        print('WPD:',WPD,' Wavelet:',wv,', Method:',thr_method,', OptMode:',OptMode)
        if thr_method=='ipr': print('IPR=',IPR,', Beta:',beta, ', [k1,k2]=',[k1,k2])
        if thr_method is None: print('theta_a: ',theta_a)
        print('Reconstruction Method:',ReconMethod, ', Window:',window,', (Win,Overlap)=',(winsize,hopesize))

    if len(X.shape)>1:

        XR =np.array([
              ATAR_1Ch(X[:,i],wv=wv,winsize=winsize, thr_method=thr_method, IPR=IPR,
              beta=beta, k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,OptMode=OptMode, factor=factor, wpd_mode=wpd_mode,
              wpd_maxlevel=wpd_maxlevel,verbose=0, window=window,hopesize=hopesize,
              ReconMethod=ReconMethod,packetwise=packetwise,WPD=WPD,lvl=lvl,fs=fs) for i in range(X.shape[1])]).T
    else:
        XR =ATAR_1Ch(X,wv=wv,winsize=winsize, thr_method=thr_method, IPR=IPR,
              beta=beta, k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,OptMode=OptMode, factor=factor, wpd_mode=wpd_mode,
              wpd_maxlevel=wpd_maxlevel, verbose=verbose, window=window,hopesize=hopesize,
              ReconMethod=ReconMethod,packetwise=packetwise,WPD=WPD,lvl=lvl,fs=fs)
    return XR

def _Wfilter_dev(x,wv='db3',thr_method='ipr',IPR=[25,75],beta=0.1,k1=10,k2=100,theta_a=np.inf,est_wmax=100,
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
        print('WPD:',WPD,' wv:',wv,' IPR:',IPR,' beta:',beta,' method:',thr_method,' OptMode:',OptMode)
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

def _ATAR_1Ch_dev(x,wv='db3',winsize=128,thr_method='ipr',IPR=[25,75],beta=0.1,k1=None,k2 =100,est_wmax=100,
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

def _ATAR_mCh_dev(X,wv='db3',winsize=128,thr_method ='ipr',IPR=[25,75],beta=0.1,k1=10,k2 =100,est_wmax=100,
              theta_a=np.inf,bf=2,gf=0.8,OptMode ='soft',wpd_mode='symmetric',wpd_maxlevel=None,factor=1,
              verbose=True, window=['hamming',True],hopesize=None, ReconMethod='custom',packetwise=False,WPD=True,lvl=[],
              fs=128.0,use_joblib=False):

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

    assert thr_method in [ None, 'ipr', 'global','outliers','std']
    assert OptMode in ['soft','linAtten','elim']


    if verbose:
        print('WPD Artifact Removal')
        print('WPD:',WPD,' Wavelet:',wv,', Method:',thr_method,', OptMode:',OptMode)
        if thr_method=='ipr': print('IPR=',IPR,', Beta:',beta, ', [k1,k2]=',[k1,k2])
        if thr_method is None: print('theta_a: ',theta_a)
        print('Reconstruction Method:',ReconMethod, ', Window:',window,', (Win,Overlap)=',(winsize,hopesize))

    if len(X.shape)>1:
        if use_joblib:
            XR =np.array(Parallel(n_jobs=-1)(delayed(ATAR_1Ch_dev)(X[:,i],wv=wv,winsize=winsize, thr_method=thr_method, IPR=IPR,
                  beta=beta, k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,OptMode=OptMode, factor=factor, wpd_mode=wpd_mode,
                  wpd_maxlevel=wpd_maxlevel,verbose=verbose, window=window,hopesize=hopesize,
                  ReconMethod=ReconMethod,packetwise=packetwise,WPD=WPD,lvl=lvl,fs=fs) for i in range(X.shape[1]))).T
        else:
            XR =np.array([ATAR_1Ch_dev(X[:,i],wv=wv,winsize=winsize, thr_method=thr_method, IPR=IPR,
                  beta=beta, k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,OptMode=OptMode, factor=factor, wpd_mode=wpd_mode,
                  wpd_maxlevel=wpd_maxlevel,verbose=verbose, window=window,hopesize=hopesize,
                  ReconMethod=ReconMethod,packetwise=packetwise,WPD=WPD,lvl=lvl,fs=fs) for i in range(X.shape[1])]).T
    else:
        XR =ATAR_1Ch_dev(X,wv=wv,winsize=winsize, thr_method=thr_method, IPR=IPR,
              beta=beta, k1=k1,k2 =k2, theta_a=theta_a,bf=bf,gf=gf,est_wmax=est_wmax,OptMode=OptMode, factor=factor, wpd_mode=wpd_mode,
              wpd_maxlevel=wpd_maxlevel, verbose=verbose, window=window,hopesize=hopesize,
              ReconMethod=ReconMethod,packetwise=packetwise,WPD=WPD,lvl=lvl,fs=fs)
    return XR
