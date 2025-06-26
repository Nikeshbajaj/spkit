'''
Continues Wavelet Transform
---------------------------
Author @ Nikesh Bajaj
Date: 22 Apr 2021
Version : 0.0.1
Github :  https://github.com/Nikeshbajaj/spkit
Contact: n.bajaj@qmul.ac.uk
'''

from __future__ import absolute_import, division, print_function
name = "Signal Processing toolkit | CWT"
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
from scipy.fftpack import fft, ifft, fftshift, ifftshift
from scipy import fftpack
from scipy.special import factorial as Fac
from scipy.special import erf
from scipy.signal import convolve
from scipy import signal
from ..utils import deprecated


#----------CWT--------------------------------------

def CTFT2(x,fs=128,nfft=None):
    if nfft is None:
        X = fftpack.fftn(x)
    else:
        X = fftpack.fftn(x,[nfft])
    N = len(X)
    f = (fs/N)*(np.arange(N)-N/2)
    return X,f
def iCTFT2(X,fs=128,nfft=None):
    if nfft is None:
        x = fftpack.ifftn(X)
    else:
        x = fftpack.ifftn(X,[nfft])
    N = len(x)
    t = np.arange(N)/fs
    return x,t
def CTFT1(x,t,axis=-1):
    N = len(t)
    Dt = t[1] - t[0]
    Df = 1. / (N * Dt)
    f = Df * (np.arange(N) - N / 2)
    X = Dt * fftshift(fft(x),axes=axis)
    return X,f
def iCTFT1(X,f,axis=-1):
    N = len(f)
    Df = f[1] - f[0]
    Dt = 1. / (N * Df)
    t = Dt * np.arange(N)
    x = ifft(fftshift(X,axes=axis))
    return x,t
def CTFT(x,t, axis=-1, method=1,ignoreWar=False):
    assert t.ndim == 1
    assert x.shape[axis] == t.shape[0]
    N = len(t)
    if not(ignoreWar):
        if N % 2 != 0:
            raise ValueError("number of samples must be even")

    Dt = t[1] - t[0]
    Df = 1. / (N * Dt)
    t0 = t[N // 2]

    f = Df * (np.arange(N) - N / 2)

    shape = np.ones(x.ndim, dtype=int)
    shape[axis] = N
    phase = np.ones(N)
    phase[1::2] = -1
    phase = phase.reshape(shape)
    if method == 1:
        X = Dt * fft(x * phase, axis=axis)
    else:
        X = Dt * fftshift(fft(x, axis=axis), axes=axis)
    X *= phase
    X *= np.exp(-2j * np.pi * t0 * f.reshape(shape))
    X *= np.exp(-1j * np.pi * N / 2)
    return X,f
def iCTFT(X,f, axis=-1, method=1,noshift=False,ignoreWar=False):
    assert f.ndim == 1
    assert X.shape[axis] == f.shape[0]
    N = len(f)
    if not(ignoreWar):
        if N % 2 != 0:
            raise ValueError("number of samples must be even")

    f0 = f[0]
    Df = f[1] - f[0]

    t0 = -0.5 / Df
    if noshift:
        t0=0
    Dt = 1. / (N * Df)
    t = t0 + Dt * np.arange(N)

    shape = np.ones(X.ndim, dtype=int)
    shape[axis] = N

    t_calc = t.reshape(shape)
    f_calc = f.reshape(shape)

    X_prime = X * np.exp(2j * np.pi * t0 * f_calc)
    x_prime = ifft(X_prime, axis=axis)
    x = N * Df * np.exp(2j * np.pi * f0 * (t_calc - t0)) * x_prime
    return x,t


#----------Continues Wavelet Functions--------------------------------------

def GaussWave(t,f,f0,Q,t0=0):
    r"""Gaussian Wavelet

    Generate time and frequency domain gaussian wavelet functions

    .. math::

        \psi(t) &= e^{-a(t-t_0)^{2}} \cdot e^{-2\pi jf_0(t-t_0)}
        
        \psi(f) &= \sqrt{\pi/a}\left( e^{-2\pi jft_0}\cdot e^{-\pi^{2}((f-f_0)^{2})/a}\right)

    where

    .. math::

        a = \left( \frac{f_0}{Q} \right)^{2}


    The main parametres of Gaussian Wavelet are:
        * f0 - center frequency
        * Q - associated with spread of bandwidth, as a = (f0/Q)^2

    Parameters
    ----------
    t: 1d array,
      - time span array corresponding to signal for analysis,
      - must be centered at 0
    f: 1d array,
      -  frquency array for gaussian wavelet 
    f0: array
       - array of center frquencies for wavelets, default: np.linspace(0.1,10,100) [scale value]
    Q: float or array 
       - array of q-factor for each wavelet, e.g. 0.5 (default) or np.linspace(0.1,5,100)
       - if array, should be of same size as f0
    t0: float, default=0, 
        - time shift of wavelet, or phase shift in frquency, Not suggeestive to change


    Returns
    -------

    wt : time-domain wavelet(s)
    wf : frequency-domain wavelet(s)

    References
    ----------
    * wikipedia - 
    
    
    Notes
    -----
    It is efficient and easy to use :func:`ScalogramCWT` with wType=='Gauss'
    
    code::
        XW,S = ScalogramCWT(x,t,fs=fs,wType='Gauss',PlotPSD=True,PlotW=True, f0=f0,Q=Q)

    See Also
    --------
    ScalogramCWT

    Examples
    --------
    #sp.cwt.GaussWave
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    fs = 128                                 #sampling frequency
    t = np.linspace(-5,5,fs*10+1)            #time
    f = np.linspace(-fs//2,fs//2,2*len(t))   #frequency range
    f0 = 2     #np.linspace(0.1,5,2)[:,None]
    Q1  = 2.5   #np.linspace(0.1,5,10)[:,None]
    Q2  = 0.5   #np.linspace(0.1,5,10)[:,None]
    wt1,wf1 = sp.cwt.GaussWave(t,f,f0=f0,Q=Q1)
    wt2,wf2 = sp.cwt.GaussWave(t,f,f0=f0,Q=Q2)
    plt.figure(figsize=(10,5))
    plt.subplot(221)
    plt.plot(t,wt1.T.real,label='real')
    plt.plot(t,wt1.T.imag,'-',label='image')
    plt.xlim(t[0],t[-1])
    #plt.xlabel('time')
    plt.ylabel('Q=2.5')
    plt.legend()
    plt.subplot(222)
    plt.plot(f,abs(wf1.T), alpha=0.9,label=f'Q={Q1},f0={f0}')
    plt.xlim(f[0],f[-1])
    plt.xlim(-10,10)
    plt.legend()
    plt.axvline(f0,color='k',lw=1,ls='--')
    plt.subplot(223)
    plt.plot(t,wt2.T.real,label='real')
    plt.plot(t,wt2.T.imag,'-',label='image')
    plt.xlim(t[0],t[-1])
    plt.xlabel('time')
    plt.ylabel('Q=0.5')
    plt.legend()
    plt.subplot(224)
    plt.plot(f,abs(wf2.T), alpha=0.9,label=f'Q={Q2},f0={f0}')
    plt.xlim(f[0],f[-1])
    plt.xlim(-10,10)
    plt.axvline(f0,color='k',lw=1,ls='--')
    plt.xlabel('Frequency')
    plt.legend()
    plt.suptitle('Gauss Wavelet')
    plt.tight_layout()
    plt.show()


    ##############################
    #sp.cwt.GaussWave
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    fs = 128                                 #sampling frequency
    t = np.linspace(-5,5,fs*10+1)            #time
    f = np.linspace(-fs//2,fs//2,2*len(t))   #frequency range
    f01  = 2     #np.linspace(0.1,5,2)[:,None]
    f02  = 5     #np.linspace(0.1,5,2)[:,None]
    Q  = 1   #np.linspace(0.1,5,10)[:,None]
    wt1,wf1 = sp.cwt.GaussWave(t,f,f0=f01,Q=Q)
    wt2,wf2 = sp.cwt.GaussWave(t,f,f0=f02,Q=Q)
    plt.figure(figsize=(10,5))
    plt.subplot(221)
    plt.plot(t,wt1.T.real,label='real')
    plt.plot(t,wt1.T.imag,'-',label='image')
    plt.xlim(t[0],t[-1])
    #plt.xlabel('time')
    plt.ylabel('Q=2.5')
    plt.legend()
    plt.subplot(222)
    plt.plot(f,abs(wf1.T), alpha=0.9,label=f'Q={Q},f0={f01}')
    plt.xlim(f[0],f[-1])
    plt.xlim(-10,10)
    plt.legend()
    plt.axvline(f01,color='k',lw=1,ls='--')
    plt.subplot(223)
    plt.plot(t,wt2.T.real,label='real')
    plt.plot(t,wt2.T.imag,'-',label='image')
    plt.xlim(t[0],t[-1])
    plt.xlabel('time')
    plt.ylabel('Q=0.5')
    plt.legend()
    plt.subplot(224)
    plt.plot(f,abs(wf2.T), alpha=0.9,label=f'Q={Q},f0={f02}')
    plt.xlim(f[0],f[-1])
    plt.xlim(-10,10)
    plt.axvline(f02,color='k',lw=1,ls='--')
    plt.xlabel('Frequency')
    plt.legend()
    plt.suptitle('Gauss Wavelet')
    plt.tight_layout()
    plt.show()

    ##############################
    #sp.cwt.GaussWave
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    fs = 100                                 #sampling frequency
    t = np.linspace(-5,5,fs*10+1)            #time
    f = np.linspace(-fs//2,fs//2,2*len(t))   #frequency range
    f0 = np.linspace(0.5,10,10)[:,None]
    Q  = np.linspace(1,5,10)[:,None]
    wt,wf = sp.cwt.GaussWave(t,f,f0=f0,Q=Q)
    LABELS = [f'Q={j[0]:,.2f}, f0={i[0]:,.2f}'  for (i, j) in zip(f0, Q)]
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(t,wt.T.real, alpha=0.8)
    plt.plot(t,wt.T.imag,'--', alpha=0.6)
    plt.xlim(t[0],t[-1])
    plt.xlabel('time')
    plt.subplot(122)
    plt.plot(f,abs(wf.T), alpha=0.6,label=LABELS)
    plt.xlim(f[0],f[-1])
    plt.xlim(-20,20)
    plt.xlabel('Frequency')
    plt.legend()
    plt.suptitle('Gauss Wavelet')
    plt.tight_layout()
    plt.show()
    """
    a = (f0 * 1. / Q) ** 2
    Wt = np.exp(-a * (t - t0) ** 2)* np.exp(2j * np.pi * f0 * (t - t0))
    Wf = np.sqrt(np.pi / a)* np.exp(-2j * np.pi * f * t0)* np.exp(-np.pi ** 2 *((f - f0) ** 2)/ a)
    return Wt,Wf
def MorlateWave(t,f=1,sig=1):
    r"""Morlate Wavelet

    Generate time and frequency domain Morlate wavelet functions

    .. math::

        \psi(t) &= C_{\sigma}\pi^{-0.25}  e^{-0.5t^2} \left(e^{j\sigma t}-K_{\sigma} \right)
        
        \psi(w) &= C_{\sigma}\pi^{-0.25} \left(e^{-0.5(\sigma-w)^2} -K_{\sigma}e^{-0.5w^2} \right)
    

    where

    .. math::

        C_{\sigma} &= \left(1+ e^{-\sigma^{2}} - 2e^{-\frac{3}{4}\sigma^{2}}   \right)^{-0.5}

        K_{\sigma} &=e^{-0.5\sigma^{2}}
        w &= 2\pi f

    
    Parameters
    ----------
    t: 1d array,
      - time span array corresponding to signal for analysis,
      - must be centered at 0
    
    sig: scalar or array 
       - array of sigma values for morlet wavelet, default=1 [scale value]
       - np.linspace(0.1,10,100)
    
    f: array
     -  array of frequency range, e.g. np.linspace(-10,10,2*N-1), where N = len(x) or len(t)

    Returns
    -------
    
    wt : time-domain wavelet(s)
    wf : frequency-domain wavelet(s)



    References
    ----------
    * wikipedia - https://en.wikipedia.org/wiki/Morlet_wavelet
    
    Notes
    -----
    It is efficient and easy to use :func:`ScalogramCWT` with wType=='Morlet'
    code::
        
        XW,S = ScalogramCWT(x,t,fs=fs,wType='Morlet',PlotPSD=True)

    See Also
    --------
    ScalogramCWT

    Examples
    --------
    #sp.cwt.MorlateWave
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    fs = 100                              #sampling frequency
    t = np.linspace(-5,5,fs*10+1)         #time
    f = np.linspace(-10,10,2*len(t)-1)    #frequency range
    sig1  = 1    #np.linspace(0.1,10,100)
    sig2  = 10
    wt1,wf1 = sp.cwt.MorlateWave(t,f,sig=sig1)
    wt2,wf2 = sp.cwt.MorlateWave(t,f,sig=sig2)
    plt.figure(figsize=(10,5))
    plt.subplot(221)
    plt.plot(t,wt1.T.real,label='real')
    plt.plot(t,wt1.T.imag,'-',label='image')
    plt.xlim(t[0],t[-1])
    #plt.xlabel('time')
    plt.ylabel(f'sig={sig1}')
    plt.legend()
    plt.subplot(222)
    plt.plot(f,abs(wf1.T))
    plt.xlim(f[0],f[-1])
    plt.xlim(-10,10)
    plt.subplot(223)
    plt.plot(t,wt2.T.real,label='real')
    plt.plot(t,wt2.T.imag,'-',label='image')
    plt.xlim(t[0],t[-1])
    plt.xlabel('time')
    plt.ylabel(f'sig={sig2}')
    plt.legend()
    plt.subplot(224)
    plt.plot(f,abs(wf2.T))
    plt.xlim(f[0],f[-1])
    plt.xlim(-10,10)
    plt.xlabel('Frequency')
    #plt.legend()
    plt.suptitle('Morlate Wavelet')
    plt.tight_layout()
    plt.show()
    
    """
    Ks = np.exp(-0.5*sig**2)
    Cs = (1.0 + np.exp(-sig**2) - 2.0*np.exp(-(3.0/4)*sig**2))**(-0.5)
    #Cs=1
    Wt = Cs*(np.pi**(-0.25))*np.exp(-0.5*(t**2))*(np.exp(1j*sig*t)-Ks)

    w=2*np.pi*f
    Wf = Cs*np.pi**(-0.25)*(np.exp(-0.5*(sig-w)**2) - Ks*np.exp(-0.5*w**2) )
    return Wt,Wf

def GaborWave(t,f=1,f0=1,a=0.1,t0=0,):
    r"""Gabor Wavelet

    The Gabor Wavelet function (technically same as Gaussian)

    .. math::

        \psi(t) &= e^{-(t-t_0)^2/a^2}e^{-jf_0(t-t_0)}

        \psi(f) &= e^{-((f-f_0)a)^2}e^{-jt_0(f-f_0)}

    where

        * a = Oscilation Rate
        * f0 = center Frequenct


    Parameters
    ----------
    t: 1d array,
      - time span array corresponding to signal for analysis,
      - must be centered at 0
    f: 1d array,
      -  frquency array for wavelet  analysis

    f0: array, default=1
       - array of center frquencies for wavelets, 
       - np.linspace(0.1,10,100) [scale value]
    
    a: float or array, default=0.1
       - array of oscillasion rate for each wavelet, e.g. 0.5 or np.linspace(0.1,5,100)
       - if array, should be of same size as f0
    t0: float, default=0, 
        - time shift of wavelet, or phase shift in frquency, Not suggeestive to change

    Returns
    -------
    wt : time-domain wavelet(s)
    wf : frequency-domain wavelet(s)


    References
    ----------
    * wikipedia - https://en.wikipedia.org/wiki/Gabor_wavelet
    
    
    Notes
    -----
    It is efficient and easy to use :func:`ScalogramCWT` with wType=='Gabor'
    
    code::
        XW,S = ScalogramCWT(x,t,fs=fs,wType='Gabor',PlotPSD=True)

    See Also
    --------
    ScalogramCWT

    Examples
    --------
    #sp.cwt.GaborWave
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    fs = 100                              #sampling frequency
    t = np.linspace(-5,5,fs*10+1)         #time
    #f = np.linspace(-10,10,2*len(t)-1)    #frequency range
    f = np.linspace(-fs//2,fs//2,2*len(t))   #frequency range
    f0 = np.linspace(1,40,10)[:,None]
    a1  = 0.5
    a2  = 2
    wt1,wf1 = sp.cwt.GaborWave(t,f,f0=f0,a=a1,t0=0)
    wt2,wf2 = sp.cwt.GaborWave(t,f,f0=f0,a=a2,t0=0)
    plt.figure(figsize=(10,5))
    plt.subplot(221)
    plt.plot(t,wt1.T.real,label='real')
    plt.plot(t,wt1.T.imag,'-',label='image')
    plt.xlim(t[0],t[-1])
    plt.ylabel(f'a={a1}')
    plt.subplot(222)
    plt.plot(f,abs(wf1.T))
    plt.xlim(f[0],f[-1])
    plt.subplot(223)
    plt.plot(t,wt2.T.real,label='real')
    plt.plot(t,wt2.T.imag,'-',label='image')
    plt.xlim(t[0],t[-1])
    plt.xlabel('time')
    plt.ylabel(f'a={a2}')
    plt.subplot(224)
    plt.plot(f,abs(wf2.T))
    plt.xlim(f[0],f[-1])
    plt.xlabel('Frequency')
    plt.suptitle('Gabor Wavelet')
    plt.tight_layout()
    plt.show()

    """
    Wt = np.exp(-((t-t0)**2)/(a**2))*np.exp(-1j*f0*(t-t0))

    Wf = np.exp(-((f-f0)*a)**2)*np.exp(-1j*t0*(f-f0))
    return Wt,Wf

def PoissonWave(t,f=1,n=1,method=3):
    r"""Poisson Wavelet


    **Type 1 (n): Method=1**

    .. math::

        \psi(t) &= \left(\frac{t-n}{n!}\right)t^{n-1} e^{-t}
        
        \psi(w) &= \frac{-jw}{(1+jw)^{n+1}}

    where Admiddibility const :math:`C_{\psi} =\frac{1}{n}` and :math:`w = 2\pi f`

    **Type 2:  Method=2**

    .. math::

        \psi(t) &= \frac{1}{\pi} \frac{1-t^2}{(1+t^2)^2}

        \psi(t) &= p(t) + \frac{d}{dt}p(t)

        \psi(w) &= |w|e^{-|w|}

    where

    .. math::

        p(t) &=\frac{1}{\pi}\frac{1}{1+t^2}
        
        w &= 2\pi f

    **Type 3 (n):  Method=3**

    .. math::

        \psi(t) &= \frac{1}{2\pi}(1-jt)^{-(n+1)}
    
        \psi(w) &= \frac{1}{\Gamma{n+1}}w^{n}e^{-w}u(w)

    where

    
    Unit step function 

    .. math::
        
        u(w) &= 1 \quad \text{ if  $w>=0$ }\quad \text{else  } 0
        
        w &= 2\pi f


    Parameters
    ----------
    t: 1d array,
      - time span array corresponding to signal for analysis,
      - must be centered at 0
    f: 1d array,
      -  frquency array for wavelet  analysis

    n: int, array, default=1
      -  array of integers
      -  np.arange(100), [scale value]

    method: {1,2,3}, default=3
      - method to select the type of Poisson wavelet computations
        method = 1,2,3, different implementation of Poisson funtion, default 3
        keep the method=3, other methods are under development and not exactly compatibile with framework yet,

    Returns
    -------
    wt : time-domain wavelet(s)
    wf : frequency-domain wavelet(s)

    References
    ----------
    * wikipedia - https://en.wikipedia.org/wiki/Poisson_wavelet
    
    
    Notes
    -----
    Method 3 produces more stable results
    It is efficient and easy to use :func:`ScalogramCWT` with wType=='Poisson'
    code::
        XW,S = ScalogramCWT(x,t,fs=fs,wType='Poisson',method = 3,PlotPSD=True)

    See Also
    --------
    ScalogramCWT

    Examples
    --------
    #sp.cwt.PoissonWave
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    fs = 100                              #sampling frequency
    t = np.linspace(-5,5,fs*10+1)         #time
    f = np.linspace(-10,10,2*len(t)-1)    #frequency range
    #f = np.linspace(-fs//2,fs//2,2*len(t))   #frequency range
    n1 = np.arange(5)
    n2 = np.arange(5,10)
    wt1,wf1 = sp.cwt.PoissonWave(t,f=f,n=n1,method=3)
    wt2,wf2 = sp.cwt.PoissonWave(t,f=f,n=n2,method=3)

    plt.figure(figsize=(10,5))
    plt.subplot(221)
    plt.plot(t,wt1.T.real,label='real')
    plt.plot(t,wt1.T.imag,'-',label='image')
    plt.xlim(t[0],t[-1])
    plt.subplot(222)
    plt.plot(f,abs(wf1.T))
    plt.xlim(f[0],f[-1])
    plt.subplot(223)
    plt.plot(t,wt2.T.real,label='real')
    plt.plot(t,wt2.T.imag,'-',label='image')
    plt.xlim(t[0],t[-1])
    plt.xlabel('time')
    plt.subplot(224)
    plt.plot(f,abs(wf2.T))
    plt.xlim(f[0],f[-1])
    plt.xlabel('Frequency')
    plt.suptitle('Poisson Wavelet')
    plt.tight_layout()
    plt.show()

    """
    n = n + np.zeros_like(n)
    if method==1:
        tx = t.copy()
        ind = np.where(t<0)[0]
        tx[ind]=0
        if len(np.asarray(n))==1:
            Wt = ((tx-n)/Fac(n))*tx**(n-1)*np.exp(-tx)
            Wt[:,ind] =0
            print(Wt.shape)
        else:
            n =np.asarray(n)
            Wt = ((tx-n[:,None])/Fac(n[:,None]))*tx**(n[:,None]-1)*np.exp(-tx)
            print(Wt.shape)
            Wt[:,ind] =0
        #Wt[ind]=0

        w = 2*np.pi*f
        Wf = -1j*w/(1+1j*w)**(n[:,None]+1)
    elif method==2:
        Wt = (1.0/np.pi)*(1-t**2)/(1+t**2)**2
        w = 2*np.pi*f
        Wf = abs(w)*np.exp(-abs(w))

    elif method==3:
        Wt = (1.0/(2*np.pi))*(1-1j*t)**(-n[:,None]-1)
        w = 2*np.pi*f
        uw = 1.0*(w>=0)
        Wf = (1.0/Fac(n[:,None]))*(w**n[:,None])*np.exp(-w)*uw
    return  Wt,Wf

def cMaxicanHatWave(t,f,f0=0,a=1):
    r"""Complex Maxican Hat Wavelet
    
    Generate Wavelet functions to be used for analysis

    .. math::

        \psi(t) &= \frac{2}{\sqrt{3}} \pi^{-\frac{1}{4}}\left(\sqrt{\pi}(1-t^2)e^{-\frac{1}{2}t^2} - \left(\sqrt{2}jt + \sqrt{\pi}erf\left[\frac{j}{\sqrt{2}}t \right] (1-t^2)e^{-\frac{1}{2}t^2}\right)\right)e^{-2\pi jf_0 t}
        
        \psi(w) &= 2\sqrt{\frac{2}{3}}\pi^{-1/4}(w-w_0)^2e^{-\frac{1}{2} (w-w_0)^2}  \quad \text{ if  $w\ge 0$,}\quad \text{ 0  else}

    where

    :math:`w = 2\pi f` and :math:`w_0 = 2\pi f_0`


    Parameters
    ----------
    t: 1d array,
      - time span array corresponding to signal for analysis,
      - must be centered at 0
    f: 1d array,
      -  frquency array for wavelet  analysis

    f0: array , default=0
       - array of center frquencies for wavelets
       - np.linspace(1,40,100) [scale value]
    
    a: float or array, default 1.0
       -  oscillation parameter, could be an array (not suggeestive)


    Returns
    -------
    wt : time-domain wavelet(s)
    wf : frequency-domain wavelet(s)

    References
    ----------
    * wikipedia - https://en.wikipedia.org/wiki/Complex_Mexican_hat_wavelet
    
    
    Notes
    -----
    It is efficient and easy to use :func:`ScalogramCWT` with wType=='cMaxican'
    code::
        XW,S = ScalogramCWT(x,t,fs=fs,wType='cMaxican',PlotPSD=True)

    See Also
    --------
    ScalogramCWT

    Examples
    --------
    #sp.cwt.cMaxicanHatWave
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    fs = 100                              #sampling frequency
    t = np.linspace(-5,5,fs*10+1)         #time
    f = np.linspace(-10,10,2*len(t)-1)    #frequency range
    #f = np.linspace(-fs//2,fs//2,2*len(t))   #frequency range
    wt1,wf1 = sp.cwt.cMaxicanHatWave(t,f=f,f0=0,a=1)
    wt2,wf2 = sp.cwt.cMaxicanHatWave(t,f=f,f0=5,a=0.1)
    plt.figure(figsize=(10,5))
    plt.subplot(221)
    plt.plot(t,wt1.T.real,label='real')
    plt.plot(t,wt1.T.imag,'-',label='image')
    plt.xlim(t[0],t[-1])
    plt.ylabel('a=1, f0=0')
    plt.subplot(222)
    plt.plot(f,abs(wf1.T))
    plt.xlim(f[0],f[-1])
    plt.subplot(223)
    plt.plot(t,wt2.T.real,label='real')
    plt.plot(t,wt2.T.imag,'-',label='image')
    plt.xlim(t[0],t[-1])
    plt.xlabel('time')
    plt.ylabel('a=0.1, f0=5')
    plt.subplot(224)
    plt.plot(f,abs(wf2.T))
    plt.xlim(f[0],f[-1])
    plt.xlabel('Frequency')
    plt.suptitle('Complex MaxicanHat Wavelet')
    plt.tight_layout()
    plt.show()

    """
    Wt = (2.0/np.sqrt(3))*(np.pi**(-0.25))*(
        np.sqrt(np.pi)*(1.0-t**2)*np.exp(-0.5*(t**2))
        -( np.sqrt(2)*1j*t + np.sqrt(np.pi)*erf(1j*t/np.sqrt(2))*(1-t**2)*np.exp(-0.5*t**2))
        )*np.exp(-2*np.pi*1j*f0*t)

    w1 = 2*np.pi*(f-f0)
    w = 1.0*(w1>=0)*w1
    #w=w1

    Wf = 2*np.sqrt(2.0/3)*np.pi**(-0.25)*(w**2)*np.exp(-0.5*a*(w**2))
    return Wt,Wf

def ShannonWave(t,f,f0=3.0/4,fw=0.5):
    r"""Complex Shannon Wavelet

    Generate Wavelet functions to be used for analysis

    .. math::

        \psi(t) &= Sinc(t/2) \cdot e^{-2j\pi f_0t}

        \psi(w) &= \prod \left( \frac{w-w_0}{\pi} \right)

    where

    :math:`\prod (x) = 1` if :math:`x \leq 0.5`, 0 else
    
    :math:`w = 2\pi f` and :math:`w_0 = 2\pi f_0`
    
    Parameters
    ----------
    t: 1d array,
      - time span array corresponding to signal for analysis,
      - must be centered at 0
    f: 1d array,
      -  frquency array for wavelet  analysis

    f0: scalar or array, default = 3/4 
       - array of center frquencies for wavelets,
       - 0.1*np.arange(10) 

    fw: scalar or array, default 0.5
      - BandWidth each wavelet, , could be an array (not suggeestive)


    Returns
    -------
    wt : time-domain wavelet(s)
    wf : frequency-domain wavelet(s)

    References
    ----------
    * wikipedia - https://en.wikipedia.org/wiki/Shannon_wavelet
    
    
    Notes
    -----
    It is efficient and easy to use :func:`ScalogramCWT` with wType=='cShannon'
    code::
        XW,S = ScalogramCWT(x,t,fs=fs,wType='cShannon',PlotPSD=True)

    See Also
    --------
    ScalogramCWT

    Examples
    --------
    #sp.cwt.ShannonWave
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    fs = 100                              #sampling frequency
    t = np.linspace(-5,5,fs*10+1)         #time
    f = np.linspace(-10,10,2*len(t)-1)    #frequency range
    #f = np.linspace(-fs//2,fs//2,2*len(t))   #frequency range
    wt1,wf1 = sp.cwt.ShannonWave(t,f,f0=0,fw=0.5)
    wt2,wf2 = sp.cwt.ShannonWave(t,f,f0=2.5,fw=2)
    plt.figure(figsize=(10,5))
    plt.subplot(221)
    plt.plot(t,wt1.T.real,label='real')
    plt.plot(t,wt1.T.imag,'-',label='image')
    plt.xlim(t[0],t[-1])
    plt.ylabel('fw=0.5, f0=0')
    plt.subplot(222)
    plt.plot(f,abs(wf1.T))
    plt.xlim(f[0],f[-1])
    plt.subplot(223)
    plt.plot(t,wt2.T.real,label='real')
    plt.plot(t,wt2.T.imag,'-',label='image')
    plt.xlim(t[0],t[-1])
    plt.xlabel('time')
    plt.ylabel('fw=2, f0=2.5')
    plt.subplot(224)
    plt.plot(f,abs(wf2.T))
    plt.xlim(f[0],f[-1])
    plt.xlabel('Frequency')
    plt.suptitle('Shannon Wavelet')
    plt.tight_layout()
    plt.show()

    """
    w  = 2*np.pi*f
    w0 = 2*np.pi*f0
    Wt = np.sinc(t/2)*np.exp(-2*np.pi*f0*1j*t)
    v1 = (w-w0)/np.pi
    v2 = (w+w0)/np.pi
    Wf = 1.0*(abs(v1)<=fw) + 0.0*(abs(v2)<=fw)
    return Wt,Wf

def WavePSD(x,t,wType='Gauss',PlotW=False,PlotPSD =True,dFFT=False,nFFT=False,reshape=True,**Parameters):
    """Wavelet Power Spectral Density 


    .. note::
        This is experimental function,
        Use :func:`ScalogramCWT` instead.

    .. warning::
        This is experimental function,
        Use :func:`ScalogramCWT` instead.

    

    1. For Gauss  : f0 =Array, Q=float, t0=float=0, f=Freq Range
    2. For Morlet : sig=Array,                      f=Freq Range
    3. For Gabor  : f0 =Array, a=float, t0=float=0, f=Freq Range
    4. For Poisson: n  =Array,                      f=Freq Range
    5. For Complex MaxicanHat, f0=freq shift        f=Freq Range
    6. For Complex Shannon   , f0=freq shift,fw=BandWidth f=Freq Range

    Parameters
    ----------

    x: array-like, input signal,
    t: array-like, time array corresponding to x, same length as x
    fs: sampling rate
    PlotPSD: bool, if True, plot Scalogram
    PlotW :  bool, if True, plot wavelets in time and frequecy with different scalling version
    fftMeth: if True, FFT method is used, else convolution method is used. FFT method is faster.
    interpolation: str, or None, interpolation while ploting Scalogram.

    Parameters for different wavelet functions
    
    Common Parameters for all the Wavelet functions

    f : array of frequency range to be analysed, e.g. np.linspace(-10,10,2*N-1), where N = len(x)
      : if None, frequency range of signal is considered from -fs/2 to fs/2
      : ( fs/n1*(np.arange(n1)-n1/2))

    A list of wavelets will be generated for each value of scale (e.g. f0, sigma, n etc)

    1. Gauss: (wType =='Gauss')
        f0 = array of center frquencies for wavelets, default: np.linspace(0.1,10,100) [scale value]
        Q  = float or array of q-factor for each wavelet, e.g. 0.5 (default) or np.linspace(0.1,5,100)
           : if array, should be of same size as f0
        t0 = float=0, time shift of wavelet, or phase shift in frquency, Not suggeestive to change

    2. For Morlet: (wType =='Morlet')
        sig = array of sigma values for morlet wavelet, default: np.linspace(0.1,10,100) [scale value]
        fw = array of frequency range, e.g. np.linspace(-10,10,2*N-1), where N = len(x)
        ref: https://en.wikipedia.org/wiki/Morlet_wavelet

    3. For Gabor: (wType =='Gabor')
        Gauss and Gabor wavelet are essentially same
        f0 = array of center frquencies for wavelets, default: np.linspace(1,40,100) [scale value]
        a  = float, oscillation parameter, default 0.5,
             could be an array (not suggeestive), similar to Gauss, e.g np.linspace(0.1,1,100) or np.logspace(0.001,0.5,100)
        t0 = float=0, time shift of wavelet, or phase shift in frquency. Not suggeestive to change

    4. For Poisson: (wType=='Poisson')
        n  = array of intergers, default np.arange(100), [scale value]
        method = 1,2,3, different implementation of Poisson funtion, default 3
        keep the method=3, other methods are under development and not exactly compatibile with framework yet,

        ref: https://en.wikipedia.org/wiki/Poisson_wavelet

    5. For Complex MaxicanHat: (wType=='cMaxican')
        f0 = array of center frquencies for wavelets, default: np.linspace(1,40,100) [scale value]
        a  = float, oscillation parameter, default 1.0, could be an array (not suggeestive)

        ref: https://en.wikipedia.org/wiki/Complex_Mexican_hat_wavelet

    6. For Complex Shannon: (wType=='cShannon')
        f0 = array of center frquencies for wavelets, default: 0.1*np.arange(10) [scale value],
        fw = BandWidth each wavelet, default 0.5, could be an array (not suggeestive)

        ref: https://en.wikipedia.org/wiki/Shannon_wavelet


    Returns
    -------

    XW: Complex-valued matrix of time-scale - Scalogram, with shape (len(S), len(x)). scale vs time
    S :  scale values


    See Also
    --------
    ScalogramCWT


    Notes
    ------
    NOTE: Use :func:`ScalogramCWT` instead.

    """
    N = len(x)
    if dFFT:
        X,f = CTFT1(x,t)
    else:
        X,f = CTFT(x,t)

    if nFFT:
        nfft = 2*N -1
        X,f = CTFT2(x,fs=128,nfft=nfft)

    #N = len(t)
    t1 = t-t[N//2]
    #t1=t
    f1 = Parameters['f'] if ('f' in Parameters.keys() and Parameters['f'] is not None) else f
    #----------Gauss Wavelet------------
    if wType =='Gauss':
        Q  = Parameters['Q']
        f0 = Parameters['f0']
        t0 = Parameters['t0'] if 't0' in Parameters.keys() else 0
        f1 = Parameters['f'] if ('f' in Parameters.keys() and Parameters['f'] is not None) else f

        Wt,Wf = GaussWave(t=t1,f=f1,t0=t0,f0=f0[:,None],Q=Q)
        S=f0

    #----------Morlet Wavelet------------
    elif wType =='Morlet':
        sig = Parameters['sig']
        f1 = Parameters['f'] if ('f' in Parameters.keys() and Parameters['f'] is not None) else f
        t0 = Parameters['t0'] if 't0' in Parameters.keys() else 0
        #t2 = t-t[len(t)//2]
        t2 = t-t0
        Wt,Wf = MorlateWave(t=t1,f=f1,sig=sig[:,None])
        S=sig

    #----------Gabor Wavelet------------
    elif wType =='Gabor':
        a  = Parameters['a']
        f0 = Parameters['f0']
        t0 = Parameters['t0'] if 't0' in Parameters.keys() else 0
        f1 = Parameters['f'] if ('f' in Parameters.keys() and Parameters['f'] is not None) else f

        Wt,Wf =GaborWave(t=t1,f=f1,f0=f0[:,None],a=a,t0=t0)
        S=f0

    #----------Poisson Wavelet------------
    elif wType=='Poisson':
        method = Parameters['method']
        n = Parameters['n']
        f1 = Parameters['f'] if ('f' in Parameters.keys() and Parameters['f'] is not None) else f
        Wt,Wf = PoissonWave(t=t1,f=f1,n=n,method=method)
        S=n

    elif wType=='cMaxican':
        f0 = Parameters['f0'] if ('f0' in Parameters.keys() and Parameters['f0'] is not None) else np.arange(5)[:,None]
        a = Parameters['a'] if 'a' in Parameters.keys() else 1.0
        print(a)
        Wt,Wf = cMaxicanHatWave(t=t1,f=f1,f0=f0,a=a)
        S = f0
    elif wType=='cShannon':
        f0 = Parameters['f0'] if ('f0' in Parameters.keys() and Parameters['f0'] is not None) else 0.1*np.arange(10)[:,None]
        fw = Parameters['fw'] if 'fw' in Parameters.keys() else 0.5
        Wt,Wf = ShannonWave(t=t1,f=f1,f0=f0,fw=fw)
        S = f0
    else:
        raise ValueError('Wavelet type was not recognized.')
        print('Wavelet type was not recognized')

    if nFFT:
        #XWf = X*np.conj(fftshift(Wf,axes=-1))
        #XW,ty = iCTFT2(XWf,fs=128,nfft=None)
        #XW = XW[:,:x.shape[0]]

        Wf1 = fftshift(Wf,axes=-1)
        xw = ifft(X*np.conj(Wf1))
        if reshape:
            XW = xw[:,:x.shape[0]]
        else:
            XW = xw
    else:
        if dFFT:
            #XW,ty = iCTFT1(X*np.conj(Wf),f)
            XW1 = X*np.conj(Wf)
            print(XW1.shape)
            XW = fftpack.ifftn(XW1,shape=[2*X.shape[0]-1],axes=[-1])
            if reshape:
                XW = XW[:,:x.shape[0]]
            ty = np.arange(XW.shape[1])/128.0

        else:
            XW,ty = iCTFT(X*np.conj(Wf),f)

    if PlotW:
        plt.figure(figsize=(13,6))
        plt.subplot(221)
        plt.plot(t,Wt.T.real)
        plt.plot(t,Wt.T.imag)
        plt.xlim([t[0],t[-1]])
        plt.subplot(222)
        plt.plot(f,abs(Wf).T)
        plt.xlim([f[0],f[-1]])
        plt.subplot(224)
        plt.plot(f,np.angle(Wf).T)
        plt.xlim([f[0],f[-1]])
        plt.show()
    if PlotPSD:
        plt.figure(figsize=(13,6))
        plt.subplot(211)
        plt.plot(t,x)
        plt.xlim([t[0],t[-1]])
        plt.subplot(212)
        plt.imshow(abs(XW),aspect='auto',origin ='lower', cmap=plt.cm.jet, extent=[t[0], t[-1], S[0], S[-1]],interpolation='sinc' )
        #plt.subplot(313)
        #plt.imshow(np.angle(XW),aspect='auto',origin ='lower', cmap=plt.cm.jet, extent=[t[0], t[-1], S[0], S[-1]],interpolation='sinc' )
        plt.show()

    return XW,S

def ScalogramCWT(x,t,wType='Gauss',fs=128,PlotPSD=False,PlotW=False,fftMeth=True,interpolation='sinc',**Parameters):
    r'''Scalogram using Continues Wavelet Transform 

    
    Compute scalogram using Continues Wavelet Transform for wavelet type (wType) and given scale range

    Parameters
    ----------

    x: array-like,
      - input signal,
    t: array-like
      - time array corresponding to x, same length as x
    fs: int,
      - sampling rate
    
    PlotPSD: bool, default=False
       - if True, plot Scalogram
    
    PlotW : bool, default=False
       - if True, plot wavelets in time and frequecy with different scalling version
    
    fftMeth:bool, default=True
       - if True, FFT method is used, else convolution method is used. FFT method is faster.
    
    interpolation: str, or None,
       -  interpolation while ploting Scalogram. Only used for plotting

    
    Parameters for different wavelet functions
    
        - Common Parameters for all the Wavelet functions

        f : array
          -  array of frequency range to be analysed, e.g. np.linspace(-10,10,2*N-1), where N = len(x)
          -  if None, frequency range of signal is considered from -fs/2 to fs/2
          -  (fs/n1*(np.arange(n1)-n1/2))

    A list of wavelets will be generated for each value of scale (e.g. f0, sigma, n etc)


    wType: {'Gauss','Morlet','Gabor','Poisson','cMaxican','cShannon'}

       - Type of Complext Wavelet Function
       - lower case str is acceptable, and a few common names are mapped to them too
       - 'Gauss', 'gauss', 'gaussian' are same
       - 'cMaxican', 'maxican', 'cmaxican' are same
       - 'cShannon' 'shannon' are same


    1. Gauss: (wType =='Gauss')
        f0 = array of center frquencies for wavelets, default: np.linspace(0.1,10,100) [scale value]
        Q  = float or array of q-factor for each wavelet, e.g. 0.5 (default) or np.linspace(0.1,5,100)
           : if array, should be of same size as f0
        t0 = float=0, time shift of wavelet, or phase shift in frquency, Not suggeestive to change

    2. For Morlet: (wType =='Morlet')
        sig = array of sigma values for morlet wavelet, default: np.linspace(0.1,10,100) [scale value]
        fw = array of frequency range, e.g. np.linspace(-10,10,2*N-1), where N = len(x)
        ref: https://en.wikipedia.org/wiki/Morlet_wavelet

    3. For Gabor: (wType =='Gabor')
        Gauss and Gabor wavelet are essentially same
        f0 = array of center frquencies for wavelets, default: np.linspace(1,40,100) [scale value]
        a  = float, oscillation parameter, default 0.5,
             could be an array (not suggeestive), similar to Gauss, e.g np.linspace(0.1,1,100) or np.logspace(0.001,0.5,100)
        t0 = float=0, time shift of wavelet, or phase shift in frquency. Not suggeestive to change

    4. For Poisson: (wType=='Poisson')
        n  = array of intergers, default np.arange(100), [scale value]
        method = 1,2,3, different implementation of Poisson funtion, default 3
        keep the method=3, other methods are under development and not exactly compatibile with framework yet,

        ref: https://en.wikipedia.org/wiki/Poisson_wavelet

    5. For Complex MaxicanHat: (wType=='cMaxican')
        f0 = array of center frquencies for wavelets, default: np.linspace(1,40,100) [scale value]
        a  = float, oscillation parameter, default 1.0, could be an array (not suggeestive)

        ref: https://en.wikipedia.org/wiki/Complex_Mexican_hat_wavelet

    6. For Complex Shannon: (wType=='cShannon')
        f0 = array of center frquencies for wavelets, default: 0.1*np.arange(10) [scale value],
        fw = BandWidth each wavelet, default 0.5, could be an array (not suggeestive)

        ref: https://en.wikipedia.org/wiki/Shannon_wavelet

    Returns
    -------

    XW: Complex-valued matrix of time-scale - Scalogram, with shape (len(S), len(x)). scale vs time
    S :  scale values

    Examples
    --------
    import numpy as np
    import matplotlib.pyplot as plt
    from spkit.cwt import ScalogramCWT

    # Example 1 - EEG Signal
    import spkit as sp
    from spkit.cwt import compare_cwt_example
    x,fs = sp.load_data.eegSample_1ch()
    t = np.arange(len(x))/fs
    print(x.shape, t.shape)
    compare_cwt_example(x,t,fs=fs)


    # Example 2.1 - different wavelets
    XW,S = ScalogramCWT(x,t,fs=fs,wType='Gauss',PlotPSD=True)

    # Example 2.2 - set scale values and number of points
    nS = 100
    f0 = np.linspace(0.1,10,nS) # range of scale values - frquency
    Q  = np.linspace(0.1,5,nS)  # different q-factor for each scale value
    # Q = 0.5
    XW,S = ScalogramCWT(x,t,fs=fs,wType='Gauss',PlotPSD=True,f0=f0,Q=Q)

    # Example 2.3  - plot scalled wavelets too
    XW,S = ScalogramCWT(x,t,fs=fs,wType='Gauss',PlotPSD=True,PlotW=True,f0=f0,Q=Q)

    # Example 3
    t = np.linspace(-5, 5, 10*100)
    x = (np.sin(2*np.pi*0.75*t*(1-t) + 2.1) + 0.1*np.sin(2*np.pi*1.25*t + 1) + 0.18*np.cos(2*np.pi*3.85*t))
    xn = x + np.random.randn(len(t)) * 0.5
    XW,S = ScalogramCWT(xn,t,fs=100,wType='Gauss',PlotPSD=True)

    # Example 4
    f0 = np.linspace(0.1,30,100)
    Q  = np.linspace(0.1,5,100) # or = 0.5
    XW,S = ScalogramCWT(xn,t,fs=128,wType='Gauss',PlotPSD=True,f0=f0,Q=Q)

    '''
    N  = len(x)
    n1 = 2*N-1   # considering both signal to be N length, convolutional legth n1 = N+N-1
    f = fs/n1*(np.arange(n1)-n1/2)

    if fftMeth: X = fft(x,n=n1)

    t1 = t-t[N//2] # time length for wavelets
    f1 = Parameters['f'] if ('f' in Parameters and Parameters['f'] is not None) else f
    t0 = Parameters['t0'] if 't0' in Parameters else 0

    #----------Gauss Wavelet------------
    if wType in ['Gauss','gauss','guassian']:
        Q     = Parameters['Q'] if 'Q' in Parameters else 0.5
        f0    = Parameters['f0'] if 'f0' in Parameters else np.linspace(0.1,10,100)

        if isinstance(Q,np.ndarray) and Q.ndim==1: Q = Q[:,None]

        Wt,Wf = GaussWave(t=t1,f=f1,t0=t0,f0=f0[:,None],Q=Q)
        S=f0

    #----------Morlet Wavelet------------
    elif wType in ['Morlet', 'morlet']:
        sig   = Parameters['sig'] if 'sig' in Parameters else np.linspace(0.1,10,100)
        fw    = Parameters['fw'] if ('fw' in Parameters and Parameters['fw'] is not None) else np.linspace(-10,10,2*N-1)
        Wt,Wf = MorlateWave(t=t1,f=fw,sig=sig[:,None])
        S=sig

    #----------Gabor Wavelet------------
    elif wType in ['Gabor', 'gabor']:
        a     = Parameters['a'] if 'a' in Parameters else 0.5
        f0    = Parameters['f0'] if 'f0' in Parameters else np.linspace(1,40,100)
        if isinstance(a,np.ndarray) and a.ndim==1: a = a[:,None]
        Wt,Wf = GaborWave(t=t1,f=f1,f0=f0[:,None],a=a,t0=t0)
        S=f0

    #----------Poisson Wavelet------------
    elif wType in ['Poisson', 'poisson']:
        method = Parameters['method'] if 'method' in Parameters else 3
        n      = Parameters['n'] if 'n' in Parameters else np.arange(100)
        Wt,Wf  = PoissonWave(t=t1,f=f1,n=n,method=method)
        S=n

    elif wType in ['cMaxican','cmaxican','maxican','maxicanhat']:
        f0 = Parameters['f0'] if ('f0' in Parameters and Parameters['f0'] is not None) else np.arange(5)
        a = Parameters['a'] if 'a' in Parameters else 1.0
        if isinstance(a,np.ndarray) and a.ndim==1: a = a[:,None]
        Wt,Wf = cMaxicanHatWave(t=t1,f=f1,f0=f0[:,None],a=a)
        S = f0
    elif wType in ['cShannon','cshannon','shannon']:
        f0 = Parameters['f0'] if ('f0' in Parameters.keys() and Parameters['f0'] is not None) else 0.1*np.arange(10)
        fw = Parameters['fw'] if 'fw' in Parameters.keys() else 0.5
        Wt,Wf = ShannonWave(t=t1,f=f1,f0=f0[:,None],fw=fw)
        S = f0
    else:
        raise ValueError('Wavelet type was not recognized.')
        print('Wavelet type was not recognized')


    if fftMeth:
        #print(X.shape,x.shape,Wf.shape)
        Wf = fftshift(Wf,axes=-1)
        xw = ifft(X*np.conj(Wf))
        XW = xw[:,:x.shape[0]]
    else:
        XW = Centered(convolve(x[None,:],Wt),[Wt.shape[0],x.shape[0]])


    #print(XW.shape)

    if PlotW:
        plt.figure(figsize=(13,3))
        plt.subplot(121)
        plt.plot(t,Wt.T.real)
        plt.plot(t,Wt.T.imag)
        plt.xlim([t[0],t[-1]])
        plt.xlabel('time')
        plt.title('Wavelets at diff. scale')
        plt.subplot(122)
        #plt.plot(f,abs(Wf).T)
        #plt.xlim([f[0],f[-1]])
        plt.plot(f-f[0],abs(Wf).T)
        plt.xlim([0,f[-1]])
        #plt.subplot(224)
        #plt.plot(f,np.angle(Wf).T)
        #plt.xlim([f[0],f[-1]])
        plt.xlabel('frequency')
        plt.title('Wavelets at diff. scale')
        plt.show()
    if PlotPSD:
        plt.figure(figsize=(13,6))
        plt.subplot(211)
        plt.plot(t,x)
        plt.xlim([t[0],t[-1]])
        plt.subplot(212)
        plt.imshow(abs(XW),aspect='auto',origin ='lower', cmap=plt.cm.jet, extent=[t[0], t[-1], S[0], S[-1]],interpolation=interpolation )
        #plt.subplot(313)
        #plt.imshow(np.angle(XW),aspect='auto',origin ='lower', cmap=plt.cm.jet, extent=[t[0], t[-1], S[0], S[-1]],interpolation='sinc' )
        plt.ylabel('scale')
        plt.xlabel('time')
        plt.show()

    return XW,S

def Centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def compare_cwt_example(x,t,fs=128,sLog =False):
    """Compare different CWT Functions

    Parameters
    ----------
    x: array-like,
      - input signal,
    t: array-like, 
      - time array corresponding to x, same length as x
    fs: int, 
      - sampling frequency

    Returns
    -------
    display plot

    References
    ----------
    * wikipedia - 
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    ScalogramCWT

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import spkit as sp
    >>> x,fs = sp.data.eeg_sample_1ch()
    >>> t = np.arange(len(x))/fs
    >>> sp.cwt.compare_cwt_example(x,t,fs=fs)
    """

    t1 = t
    t0 = 0

    #print('Gauss')
    f0 = np.linspace(0.1,40,100)
    Q  = np.linspace(0.1,5,100)#[:,None]
    XW1,S1 = ScalogramCWT(x,t,wType='Gauss',fs =fs,PlotW=False,PlotPSD =False,fftMeth=True,f0=f0,Q=Q)

    #print('Morlet')
    sig = np.linspace(0.1,10,100)
    f = np.linspace(-10,10,2*len(x)-1)
    XW2,S2 = ScalogramCWT(x,t,wType='Morlet',fs =fs,PlotW=False,PlotPSD =False,fftMeth=True,sig=sig,f=f)

    #print('Gabor')
    f0 = np.linspace(1,40,100)
    a  = 0.5
    XW3,S3 = ScalogramCWT(x,t,wType='Gabor',fs =fs,PlotW=False,PlotPSD =False,fftMeth=True,f0=f0,a=a)

    #print('Poisson')
    n = np.arange(100)
    XW4,S4 = ScalogramCWT(x,t,wType='Poisson',fs =fs,PlotW=False,PlotPSD =False,fftMeth=True,n=n,method=3)

    #print('cMaxican')
    f0 = np.linspace(0,40,80)#[:,None]
    a  = 0.005
    XW5,S5 = ScalogramCWT(x,t,wType='cMaxican',fs =fs,PlotW=False,PlotPSD =False,fftMeth=True,f0=f0,a=a)

    #print('cShannon')
    f0 = np.linspace(0,40,40)#[:,None]
    fw=5
    XW6,S6 = ScalogramCWT(x,t,wType='cShannon',fs = 128,PlotW=False,PlotPSD =False,fftMeth=True,f0=f0,fw=fw)

    N=32
    win = signal.get_window('hann', N)
    fx, tx, Sxx = signal.spectrogram(x, fs=fs,nfft=2*N,nperseg=N,noverlap=N//2,window=win)

    plt.figure(figsize=(15,15))
    plt.subplot(811)
    #print(x.shape,t.shape)
    plt.plot(t,x)
    plt.xlim([t[0],t[-1]])
    plt.grid()

    plt.subplot(812)
    plt.imshow(abs(XW1),aspect='auto',origin ='lower', cmap=plt.cm.jet, extent=[t[0], t[-1], S1[0], S1[-1]],interpolation='sinc')
    plt.ylabel('Gauss')

    plt.subplot(813)
    plt.imshow(abs(XW2),aspect='auto',origin ='lower', cmap=plt.cm.jet, extent=[t[0], t[-1], S2[0], S2[-1]],interpolation='sinc')
    plt.ylabel('Morlet')

    plt.subplot(814)
    plt.imshow(abs(XW3),aspect='auto',origin ='lower', cmap=plt.cm.jet, extent=[t[0], t[-1], S3[0], S3[-1]],interpolation='sinc')
    plt.ylabel('Gabor')

    plt.subplot(815)
    plt.imshow(abs(XW4),aspect='auto',origin ='lower', cmap=plt.cm.jet, extent=[t[0], t[-1], S4[0], S4[-1]],interpolation='sinc')
    plt.ylabel('Poisson')

    plt.subplot(816)
    plt.imshow(abs(XW5),aspect='auto',origin ='lower', cmap=plt.cm.jet, extent=[t[0], t[-1], S5[0], S5[-1]],interpolation='sinc')
    plt.ylabel('cMaxican')

    plt.subplot(817)
    plt.imshow((abs(XW6)),aspect='auto',origin ='lower', cmap=plt.cm.jet, extent=[t[0], t[-1], S6[0], S6[-1]],interpolation='sinc')
    plt.ylabel('cShannon')

    plt.subplot(818)
    if sLog:
        plt.imshow(np.log10(Sxx),aspect='auto',  origin='lower',cmap=plt.cm.jet, extent=[t[0], t[-1], fx[0], fx[-1]],interpolation='sinc')
    else:
        plt.imshow(Sxx,aspect='auto',  origin='lower',cmap=plt.cm.jet, extent=[t[0], t[-1], fx[0], fx[-1]],interpolation='sinc')
    plt.ylabel('Spectrogram')

    plt.subplots_adjust(hspace=0.05)
    plt.show()

def cMaxicanHatWave_(t,f):
    """Complex Maxican Hat Wavelet
    
    Parameters
    ----------

    Returns
    -------
    wt : time-domain wavelet(s)
    wf : frequency-domain wavelet(s)

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
    Wt = (2.0/np.sqrt(3))*(np.pi**(-0.25))*(
        np.sqrt(np.pi)*(1.0-t**2)*np.exp(-0.5*(t**2))
        -( np.sqrt(2)*1j*t + np.sqrt(np.pi)*erf(1j*t/np.sqrt(2))*(1-t**2)*np.exp(-0.5*t**2))
        )

    w1 = 2*np.pi*f
    w = 1.0*(w1>=0)*w1
    #w=w1
    Wf = 2*np.sqrt(2.0/3)*np.pi**(-0.25)*(w**2)*np.exp(-0.5*(w**2))

    #w0 = 2*np.pi*f0
    #w0 = w0 + np.zeros_like(f0)
    #Wf = 2*np.sqrt(2.0/3)*np.pi**(-0.25)*(w0*w**2)*np.exp(-0.25*(w0*w**2))
    #Wf = 2*np.sqrt(2.0/3)*np.pi**(-0.25)*((w-w0[:, None])**2)*np.exp(-0.25*((w-w0[:, None])**2))
    #Wf = 2*np.sqrt(2.0/3)*np.pi**(-0.25)*((w+w0[:, None])**2)*np.exp(-0.25*((w+w0[:, None])**2))
    #Wf = 0.5*Wf1+0.0*Wf2
    return Wt,Wf

def ShannonWave_v0(t,f):
    """Complex Maxican Hat Wavelet
    
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
    Wt = np.sinc(t/2)*np.cos(3*np.pi*t/2)
    w =2*np.pi*f
    v1 = (w-3*np.pi/2.0)/np.pi
    v2 = (w+3*np.pi/2.0)/np.pi
    Wf = 1.0*(abs(v1)<=0.5) + 1.0*(abs(v2)<=0.5)
    return Wt,Wf

def ShannonWave_v1(t,f,f0=3.0/4):
    """Complex Maxican Hat Wavelet
    
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
    w  = 2*np.pi*f
    w0 = 2*np.pi*f0
    Wt = np.sinc(t/2)*np.cos(w0*t)
    v1 = (w-w0)/np.pi
    v2 = (w+w0)/np.pi
    Wf = 1.0*(abs(v1)<=0.5) + 1.0*(abs(v2)<=0.5)
    return Wt,Wf
