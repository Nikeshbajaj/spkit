'''
Fractional Processes and methods
--------------------------------
Author @ Nikesh Bajaj
updated on Date: 1 Jan 2022
Version : 0.0.1
Github :  https://github.com/Nikeshbajaj/spkit
Contact: n.bajaj@qmul.ac.uk | n.bajaj@imperial.ac.uk
'''

from __future__ import absolute_import, division, print_function
name = "Signal Processing toolkit | Fractional Processes"
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
from scipy.fftpack import fft, ifft, fftshift

try:
    from scipy.signal import blackmanharris, triang
except:
    # for scipy>1.12, location has changed blackmanharris, triang 
    from scipy.signal.windows import blackmanharris, triang

from scipy import signal

from .processing import sinc_interp, conv1d_fft
from ..utils import deprecated


def frft(x,alpha=0.1,method=1,verbose=0):
    r"""Fractional Fourier Transform
    
    **Fractional Fourier Transform**

    .. math::
        F^{\alpha}(x) = FRFT(x)

    Parameters
    ----------
    x:  real signal
    alpha: scalar, 0<a<4
    method=1, other methods to be implemented

    Returns
    -------
    Y: complex signal


    References
    ----------
    * wikipedia - https://en.wikipedia.org/wiki/Fractional_Fourier_transform
    
    
    Notes
    -----
    #TODO

    See Also
    --------
    ifrft: Inverse Fractional Fourier Transform
    ffrft: Fast Fractional Fourier Transform
    iffrft: Inverse Fast Fractional Fourier Transform

    Examples
    --------
    #sp.frft
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    t = np.linspace(0,2,500)
    x = np.cos(2*np.pi*5*t)
    xf = sp.frft(x,alpha=0.5)
    plt.figure(figsize=(10,4))
    plt.subplot(211)
    plt.plot(t,x,label='x: input signal')
    plt.xlim([t[0],t[-1]])
    plt.xlabel('time (s)')
    plt.ylabel('x')
    plt.legend(loc='upper right')
    plt.subplot(212)
    plt.plot(t,xf.real,label='xf.real',alpha=0.9)
    plt.plot(t,xf.imag,label='xf.imag',alpha=0.9)
    plt.plot(t,np.abs(xf),label='|xf|',alpha=0.9)
    plt.xlim([t[0],t[-1]])
    plt.ylabel(r'xf: FRFT(x) $\alpha=0.5$')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    """
    verbose=0
    x0 = x.copy() + 1j*0
    N     = x0.shape[0]
    shft  = (np.arange(N) + int(N//2))%N
    sN    = np.sqrt(N)
    alpha = alpha%4

    Xa = np.zeros(x.shape) + 1j*np.zeros(x0.shape)

    # if solution is related to fft
    if alpha==0: Xa = x0
    if alpha==2: Xa = np.flipud(x0)
    if alpha==1: Xa[shft] = np.fft.fft(x0[shft])/sN
    if alpha==3: Xa[shft] = np.fft.ifft(x0[shft])*sN

    if alpha in [0,1,2,3]: return Xa

    # reduce to interval 0.5 < a < 1.5
    if alpha>2.0:
        alpha = alpha-2
        x0 = np.flipud(x0)
    if alpha>1.5:
        alpha = alpha-1
        x0[shft] = np.fft.fft(x0[shft])/sN
    if alpha<0.5:
        alpha = alpha+1
        x0[shft] = np.fft.ifft(x0[shft])*sN

    alpha_r = alpha*np.pi/2

    if method==1:
        if verbose: print(x0.shape)
        tanf = np.tan(alpha_r/2)
        sinf = np.sin(alpha_r)
        x0 = np.r_[np.zeros(N-1), sinc_interp(x0), np.zeros(N-1)]
        if verbose:print(alpha_r,tanf,sinf)
        if verbose: print(x0.shape)
        if verbose>1:print(x0.real)

        #chirp premultiplication
        t0 = np.arange(-2*N+2,2*N-2+1)
        chrp = np.exp((-1j*np.pi/N)*(tanf/4)*(t0**2))
        x0 *= chrp

        if verbose: print(x0.shape)
        if verbose>1:print(x0.real)


        #chirp convolution
        coef = np.pi/N/sinf/4

        #print(coef)

        t0 = np.arange(-(4*N-4),4*N-4+1)
        chrp1 = np.exp(1j*coef*(t0**2))
        Xa = conv1d_fft(chrp1,x0)
        if verbose: print(x0.shape,Xa.shape)
        Xa = np.sqrt(coef/np.pi)*Xa[4*N-3-1:8*N-7]
        #Faf = Faf(4*N-3:8*N-7)*sqrt(c/pi);
        #print(np.sqrt(coef/np.pi))
        if verbose>1:print(Xa)


        if verbose: print(x0.shape, Xa.shape)
        #chirp post multiplication
        Xa = chrp*Xa

        if verbose: print(x0.shape, Xa.shape)
        #normalizing constant
        #print(np.exp(-1j*(1-alpha)*np.pi/4))
        if verbose>1:print(np.exp(-1j*(1-alpha)*np.pi/4)*Xa)
        Xa = np.exp(-1j*(1-alpha)*np.pi/4)*Xa[N-1:-N+1:2]
        #Faf = exp(-i*(1-a)*pi/4)*Faf(N:2:end-N+1)
        if verbose: print(x0.shape, Xa.shape)
        return Xa

    elif method==2:
        #sin_x = np.pi/(N+1)/np.sin(alpha)/4
        #t = np.pi/(N+1)*np.tan(alpha/2)/4
        #Cs = np.sqrt(s/np.pi)*np.exp(-1j*(1-a)*np.pi/4)
        raise NotImplementedError('Not implemented yet')

def ifrft(x,alpha=0.1,method=1, verbose=0):
    r"""Inverse Fractional Fourier Transform

    **Inverse Fractional Fourier Transform**
    
    Parameters
    ----------
    x: complex-signal
    alpha: scalar, 0<a<4
    method=1, other methods to be implemented
    
    Returns
    -------
    y: complex signal
    - reconstruction using IFRFT
    - imaginary part is mostly zero

    References
    ----------
    * wikipedia
    
    Notes
    -----
    # Recostruction of the signal is has some artifact to be removed.

    See Also
    --------
    frft: Fractional Fourier Transform
    ffrft: Fast Fractional Fourier Transform
    iffrft: Inverse Fast Fractional Fourier Transform

    Examples
    --------
    #sp.ifrft
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    t = np.linspace(0,2,500)
    x = np.cos(2*np.pi*5*t)
    xf = sp.frft(x,alpha=0.5)
    x1 = sp.ifrft(xf,alpha=0.5)
    plt.figure(figsize=(10,5))
    plt.subplot(311)
    plt.plot(t,x,label='x: input signal')
    plt.xlim([t[0],t[-1]])
    plt.xlabel('time (s)')
    plt.ylabel('x')
    plt.legend(loc='upper right')
    plt.subplot(312)
    plt.plot(t,xf.real,label='xf.real',alpha=0.9)
    plt.plot(t,xf.imag,label='xf.imag',alpha=0.9)
    plt.plot(t,np.abs(xf),label='|xf|',alpha=0.9)
    plt.xlim([t[0],t[-1]])
    plt.ylabel(r'xf: FRFT(x) $\alpha=0.5$')
    plt.legend(loc='upper right')
    plt.subplot(313)
    plt.plot(t,x1.real,label='x1.real',alpha=0.9)
    plt.plot(t,x1.imag,label='x1.imag',alpha=0.9)
    plt.plot(t,np.abs(x1),label='|x1|',alpha=0.9)
    plt.xlim([t[0],t[-1]])
    plt.ylabel(r'x1: IFRFT(xf) $\alpha=0.5$')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    """
    return frft(x,alpha=-alpha,method=method, verbose=verbose)

def ffrft(x, alpha):
    r"""Fast Fractional Fourier Transform
    
    **Fast Fractional Fourier Transform**

      - perfect reconstruction with iffrft

    Parameters
    ----------
    x:  real signal
    alpha: value

    Returns
    -------
    Y: complex signal

    References
    ----------
    * wikipedia - 
    
    
    Notes
    -----
    * FRFT :func:`frft` Fractional Fourier Transform, is classic and more accepted approach. Compared to FFRFT, :func:`ffrft`


    See Also
    --------
    frft: Fractional Fourier Transform
    ifrft: Inverse Fractional Fourier Transform
    iffrft: Inverse Fast Fractional Fourier Transform

    Examples
    --------
    #sp.ffrft
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    t = np.linspace(0,2,500)
    x = np.cos(2*np.pi*5*t)
    xf = sp.ffrft(x,alpha=0.5)
    plt.figure(figsize=(10,5))
    plt.subplot(211)
    plt.plot(t,x,label='x: input signal')
    plt.xlim([t[0],t[-1]])
    plt.xlabel('time (s)')
    plt.ylabel('x')
    plt.legend(loc='upper right')
    plt.subplot(212)
    plt.plot(t,xf.real,label='xf.real',alpha=0.9)
    plt.plot(t,xf.imag,label='xf.imag',alpha=0.9)
    plt.plot(t,np.abs(xf),label='|xf|',alpha=0.9)
    plt.xlim([t[0],t[-1]])
    plt.ylabel(r'xf: FFRFT(x) $\alpha=0.5$')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    """


    def un_permut(x):
        x0, x_end = x[:1], x[1:]
        return np.r_[x0, x_end[::-1]]
    x0 = np.array(x)
    x1 = np.fft.fft(x0, norm='ortho')
    x2 = un_permut(x0)
    x3 = un_permut(x1)

    # Derive eigenbasis
    tMat = np.array([[1, 1 , 1, 1],
                     [1, 1j,-1,-1j],
                     [1,-1 , 1,-1],
                     [1,-1j,-1, 1j]])

    Y = np.c_[x0,x1,x2,x3]@tMat

    # Phase eigenbasis
    alpha_ph = np.array([1, 1j**alpha, 1j**(2*alpha), 1j**(3*alpha)])

    Y = Y*alpha_ph

    return np.sum(Y,1) / 4

def iffrft(x, alpha):
    r"""Inverse Fast Fractional Fourier Transform
    
    **Inverse Fast Fractional Fourier Transform**

    Parameters
    ----------
    x: complex-signal
    alpha: scalar, 0<a<4
    method: default=1
       - other methods to be implemented
    
    Returns
    -------
    y: complex signal
    - reconstruction using IFRFT
    - imaginary part is mostly zero

    References
    ----------
    * wikipedia
    
    
    Notes
    -----
    * FRFT: :func:`frft` Fractional Fourier Transform, is classic and more accepted approach. Compared to FFRFT

    See Also
    --------
    frft: Fractional Fourier Transform
    ifrft: Inverse Fractional Fourier Transform
    ffrft: Fast Fractional Fourier Transform
    
    Examples
    --------
    #sp.iffrft
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    t = np.linspace(0,2,500)
    x = np.cos(2*np.pi*5*t)
    xf = sp.ffrft(x,alpha=0.5)
    x1 = sp.iffrft(xf,alpha=0.5)
    plt.figure(figsize=(10,5))
    plt.subplot(311)
    plt.plot(t,x,label='x: input signal')
    plt.xlim([t[0],t[-1]])
    plt.xlabel('time (s)')
    plt.ylabel('x')
    plt.legend(loc='upper right')
    plt.subplot(312)
    plt.plot(t,xf.real,label='xf.real',alpha=0.9)
    plt.plot(t,xf.imag,label='xf.imag',alpha=0.9)
    plt.plot(t,np.abs(xf),label='|xf|',alpha=0.9)
    plt.xlim([t[0],t[-1]])
    plt.ylabel(r'xf: FRFT(x) $\alpha=0.5$')
    plt.legend(loc='upper right')
    plt.subplot(313)
    plt.plot(t,x1.real,label='x1.real',alpha=0.9)
    plt.plot(t,x1.imag,label='x1.imag',alpha=0.9)
    plt.plot(t,np.abs(x1),label='|x1|',alpha=0.9)
    plt.xlim([t[0],t[-1]])
    plt.ylabel(r'x1: IFRFT(xf) $\alpha=0.5$')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    """
    return ffrft(x, -alpha)
