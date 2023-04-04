'''
Advance signal processing methods
--------------------------------
Author @ Nikesh Bajaj
updated on Date: 27 March 2023. Version : 0.0.5
updated on Date: 1 Jan 2022. Version : 0.0.1
Github :  https://github.com/Nikeshbajaj/spkit
Contact: n.bajaj@qmul.ac.uk | n.bajaj@imperial.ac.uk
'''

from __future__ import absolute_import, division, print_function
name = "Signal Processing toolkit | Advance"
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
from scipy.signal import blackmanharris, triang
from scipy import signal

def isPower2(n):
    """
    Check if num is power of two
    """
    return ((n & (n - 1)) == 0) and n > 0

def dft_analysis(x, window='blackmanharris', N=None,scaling_dB=True,normalize_win=True, plot=False, fs=None):

    """
    Analysis of a signal x using the Discrete Fourier Transform
    ----------------------------------------------------------

    input
    -----
    x  : 1d-array input signal ofshape (n,)
    window: window-type (default = 'blackmanharris')
          : in None, qrectangular window is used

    N : FFT size should be >= len(x) and power of 2
      : if None then N = 2**np.ceil(np.log2(len(n)))

    scaling_dB: bool, if false, then linear scale of spectrum is returned, else in dB
              : default True
    normalize_win: bool (default True), if to normalize wondow (recommended)
    plot: int, (default: 0) for no plot
        : 1 for plotting magnitude and phse spectrum
        : 2 for ploting signal along with spectrum

    fs : sampling frequency, only used to plot the signal when plot=2
       : if not provided, fs=1 is used
       : it does not affect any computations

    output
    ------
    mX: magnitude spectrum (of shape=int((N/2)+1)) # positive spectra
    pX: phase spectrum  same shape as mX
    N : N-point FFT used for computation
    """

    n = x.shape[0]
    #FFT size (N) is not a power of 2
    if N is None:  N = int(2**np.ceil(np.log2(n)))
    assert isPower2(N)
    #FFT size is smaller than signal - will crop the spectrum information of beyond
    assert N>=n

    #window
    if window is None: window='boxcar'
    win = signal.get_window(window, n)
    # normalize analysis window
    if normalize_win: win = win / win.sum()

    # positive spectrum, including 0
    hN  = int((N/2)+1)
    # half analysis window size by rounding
    # half analysis window size by floor
    hM1 = int(np.floor((win.size+1)/2))
    hM2 = int(np.floor(win.size/2))

    fftbuffer = np.zeros(N)

    xw = x*win  #windowing singal

    # zero-phase window in fftbuffer
    fftbuffer[:hM1] = xw[hM2:]
    fftbuffer[-hM2:] = xw[:hM2]

    # FFT
    X = fft(fftbuffer)

    # magnitude spectrum of positive frequencies
    absX = np.abs(X[:hN])
    absX[absX<np.finfo(float).eps] = np.finfo(float).eps
    if scaling_dB:
        mX = 20 * np.log10(absX) # in dB
    else:
        mX = absX.copy()         # abs

    # phase calculation
    tol = 1e-14
    X[:hN].real[np.abs(X[:hN].real) < tol] = 0.0
    X[:hN].imag[np.abs(X[:hN].imag) < tol] = 0.0
    pX = np.unwrap(np.angle(X[:hN]))    # unwrapped phase spectrum

    if plot:
        if fs is None: fs=1
        freq =  fs*np.arange(mX.shape[0])/N

        if plot>1:
            plt.figure(figsize=(12,6))
            plt.subplot(221)
        else:
            plt.figure(figsize=(12,4))
            plt.subplot(121)
        plt.plot(freq, mX)
        plt.title('Magnitude Spectrum')
        plt.xlabel(f'Frequency{" (normalized)" if fs==1 else " (Hz)"}')
        plt.ylabel(f'|X|{" (dB)" if scaling_dB else ""}')
        plt.grid()
        plt.xlim([freq[0], freq[-1]])

        if plot>1:
            plt.subplot(222)
        else:
            plt.subplot(122)
        plt.plot(freq, pX)
        plt.title('Phase Spectrum')
        plt.xlabel(f'Frequency{" (normalized)" if fs==1 else " (Hz)"}')
        plt.ylabel('<|X|')
        plt.grid()
        plt.xlim([freq[0], freq[-1]])
        if plot>1:
            tx = np.arange(len(x))/fs
            plt.subplot(313)
            plt.plot(tx,x,label='signal')
            plt.plot(tx,xw/np.linalg.norm(xw,2)*np.linalg.norm(x,2),label='windowed and scaled')
            plt.legend()
            plt.grid()
            plt.xlim([tx[0],tx[-1]])
            plt.ylabel('Amiplitude')
            plt.title('signal: x')
            plt.xlabel('time(s)')
        plt.show()
    return mX, pX, N

def dft_synthesis(mX, pX, M=None,scaling_dB=True,window=None):
    """
    Synthesis of a signal using the Discrete Fourier Transform from positive spectra
    ----------------------------------------------------------
    input
    -----
    mX: magnitude spectrum - 1d-array  (of shape=int((N/2)+1)) for N-point FFT
    pX: phase spectrum     - same size as mX
    M : length of signal: x, if None, then M = N = 2*(len(mX)-1)
    window: if provided, synthesized signal is rescalled with corresponding window function
          : undoing the scaling

    output
    -------
    y: output signal of shape (M,)
    """
    # size of positive spectrum including sample 0
    hN = mX.size
    # FFT size
    N = (hN-1)*2

    #size of mX is not (N/2)+1
    assert isPower2(N)

    if M is None: M = N
    # half analysis window size by rounding
    hM1 = int(np.floor((M+1)/2))
    # half analysis window size by floor
    hM2 = int(np.floor(M/2))
    # buffer for FFT
    fftbuffer = np.zeros(N)
    y = np.zeros(M)
    Y = np.zeros(N, dtype = complex)

    if not(scaling_dB): mX = 20*np.log10(mX)

    # generate positive frequencies (postive side of spectrum)
    Y[:hN] = 10**(mX/20) * np.exp(1j*pX)

    # generate negative frequencies (negative side of spectrum)
    Y[hN:] = 10**(mX[-2:0:-1]/20) * np.exp(-1j*pX[-2:0:-1])

    # compute inverse FFT
    fftbuffer = np.real(ifft(Y))

    # undo zero-phase window
    y[:hM2] = fftbuffer[-hM2:]
    y[hM2:] = fftbuffer[:hM1]

    if window is not None:
        win = signal.get_window(window, y.shape[0])
        win.sum()
        y *=win.sum()

    return y

def stft_analysis(x, winlen, window='blackmanharris',nfft=None, overlap=None):
    """
    Analysis of a signal using the Short-Time Fourier Transform
    ------------------------------------------------------------
    input
    -----
    x: 1d-array signal - shape (n,)
    winlen : window length for analysis (good choice is a odd number)
           : window size is chosen based on the frequency resolution required
           : winlen >= Bs*fs/del_f
           : where Bs=4 for hamming window, Bs=6 for blackman harris
           : def_f is different between two frequecies (to be resolve)
           : higher the window length better the frequency resolution, but poor time resolution
    overlap: overlap of windows
           : if None then winlen//2 is used (50% overlap)
           : shorter overlap can improve time resoltion - upto an extend
    window: analysis window (default = blackmanharris)
          : if None, rectangular window is used
    nfft: FFT size, should be >=winlen and power of 2
        : if None -  nfft = 2**np.ceil(np.log2(len(n)))

    output
    ------
    mXt : magnitude spectra of shape (number of frames, int((nfft/2)+1))
    pXt : phase spectra of same shape as mXt
    """

    if nfft is None: nfft = int(2**np.ceil(np.log2(winlen)))
    if overlap is None: overlap=winlen//2

    # half analysis window size by rounding
    hM1 = int(np.floor((winlen+1)/2))
    # half analysis window size by floor
    hM2 = int(np.floor(winlen/2))

    # padding zeros at beginning and at the end (haft size of window)
    x = np.append(np.zeros(hM2),x)
    x = np.append(x,np.zeros(hM2))
    win = np.arange(winlen)
    mXt, pXt = [],[]
    while win[-1]<x.size:
        xi = x[win]
        mX, pX,_ = dft_analysis(xi, window=window, N=nfft)
        mXt.append(mX)
        pXt.append(pX)
        win += overlap

    mXt = np.vstack(mXt)
    pXt = np.vstack(pXt)
    return mXt, pXt

def stft_synthesis(mXt, pXt, winlen, overlap):
    """
    Synthesis of signal from Short-Time Fourier Transform
    ------------------------------------------------------
    input
    -----
    mXt: magnitude spectra of signal - 2d-array of shape (number of frames, int((nfft/2)+1))
    pXt: phase spectra of same size as mXt
    winlen: window length used while analysing
    overlap: overlap of windows used while analysing

    output
    ------
    y : 1d-array - synthesized signal shape = (nFrames*overlap + winlen)
    """
    # half -sides of analysis window
    hM1 = int(np.floor((winlen+1)/2))
    hM2 = int(np.floor(winlen/2))
    nFrames = mXt.shape[0]
    y = np.zeros(nFrames*overlap + winlen)
    for i in range(nFrames):
        yi = dft_synthesis(mXt[i,:], pXt[i,:], M=winlen)
        # overlap-add to generate output sound
        y[i*overlap:i*overlap+winlen] += overlap*yi

    #cropping
    y = y[hM2:-hM1]
    return y

def peak_detection(mX, thr):
    """
    Detect spectral peaks
    ---------------------
    input
    -----
    mX  : magnitude spectrum(in dB)
    thr : thresholdc(dB)

    output
    ------
    ploc: peak locations
    """
    # locations above threshold
    thresh = np.where(mX[1:-1]>thr, mX[1:-1], 0)

    # locations higher than the next one
    next_minor = np.where(mX[1:-1]>mX[2:], mX[1:-1], 0)

    # locations higher than the previous one
    prev_minor = np.where(mX[1:-1]>mX[:-2], mX[1:-1], 0)

    # locations fulfilling the three criteria
    ploc = thresh * next_minor * prev_minor

    # add 1 to compensate for previous steps
    ploc = ploc.nonzero()[0] + 1
    return ploc

def peak_interp(mX, pX, ploc):
    """
    Interpolate peak values using parabolic interpolation
    -----------------------------------------------------
    refined loction:
        kp_new =  kp + 0.5*(X[kp-1] - X[kp+1])/(X[kp-1] -2*X[kp]+X[kp+1])
    refined value:
        X_new  =  X[kp] + 0.25*(kp_new - kp)*(X[kp-1] - X[kp+1])

    input
    -----
    mX  : magnitude spectrum
    pX  : phase spectrum
    ploc: locations of peaks

    output
    ------
    interpolated - refined locations of frequencies and corresponding magnitude and phase:
    iploc  : peak location
    ipmag  : magnitude values
    ipphase: phase values
    """
    # magnitude of peak bin
    val  = mX[ploc]
    lval = mX[ploc-1]  # at left
    rval = mX[ploc+1]  # at right
    # center of parabola
    iploc = ploc + 0.5*(lval-rval)/(lval-2*val+rval)
    # magnitude of peaks
    ipmag = val - 0.25*(lval-rval)*(iploc-ploc)
    # phase of peaks by linear interpolation
    ipphase = np.interp(iploc, np.arange(0, pX.size), pX)
    return iploc, ipmag, ipphase

def TWM_f0(pfreq,pmag,f0min=0.01,f0max=None,f0_pre=0,f0max_err=1,verbose=0):
    """
    Two-Way Mismatch algorithn: Selecting f0 from possible candidates
    -----------------------------------------------------------------

    input
    -----
    pfreq : peak frequencies
    pmag  : magnitudes of respective frequencies
    f0_pre: f0 of previous frame, (default=0, when not present)
    f0min : minimum f0 (default f0=0.01)
    f0max : maximum f0 (if None f0max = max(pfreq), max from candidates f0)
    f0max_err: maximum error allowed

    output
    ------
    if stable:
       f0: fundamental frequency in Hz
    else
       f0 = 0 (no f0 found)
    """
    if (pfreq.size < 3) & (f0_pre == 0): return 0

    #Minumum fundamental frequency (minf0) smaller than 0
    assert f0min>=0

    # use only peaks within given range
    f0inx = np.argwhere((pfreq>f0min) & (pfreq<f0max))[:,0]

    # return 0 if no peaks within range
    if (f0inx.size == 0): return 0

    # candidates frequencies of peak
    f0cf = pfreq[f0inx]
    f0cm = pmag[f0inx]

    # if stable f0 in previous frame
    if f0_pre>0:
        # use only peaks close to it
        shortlist = np.argwhere(np.abs(f0cf-f0_pre)<f0_pre/2.0)[:,0]
        maxc   = np.argmax(f0cm)
        maxcfd = f0cf[maxc]%f0_pre
        if maxcfd > f0_pre/2:
            maxcfd = f0_pre - maxcfd

        # or the maximum magnitude peak is not a harmonic
        if (maxc not in shortlist) and (maxcfd>(f0_pre/4)):
            shortlist = np.append(maxc, shortlist)

        # frequencies of candidates
        f0cf   = f0cf[shortlist]
        #f0cm_i = pmag[shortlist]
    # return 0 if no peak candidates
    if (f0cf.size == 0): return 0

    # call the TWM function with peak candidates
    #f0, f0error = UF_C.twm(pfreq, pmag, f0cf)
    if verbose: print('TWM_in:',pfreq, pmag, f0cf)
    f0, f0error = TWM_algo(pfreq, pmag, f0cf)
    if verbose: print('TWM_out:',f0, f0error, 'er_thr',f0max_err)

    # accept and return f0 if below max error allowed
    if (f0>0) and (f0error<f0max_err):
        #if return_cand: return f0,(f0cf,f0cm_i)
        return f0
    #if return_cand: return 0,f0cf
    return 0

def TWM_algo(pfreq, pmag, f0c,verbose=0):
    """
    Two-way mismatch algorithm for f0 detection
    -------------------------------------------

    input
    -----
    pfreq: peak frequencies
    pmag : magnitudes of peak frequnecies
    f0c  : frequencies of f0 candidates

    output
    ------
    f0: fundamental frequency detected
    f0Error: corresponding two-way mismatch error

    Ref: by Beauchamp&Maher - https://github.com/MTG/sms-tools
    """

    #default parameter setting by author
    p = 0.5    # weighting by frequency value
    q = 1.4    # weighting related to magnitude of peaks
    r = 0.5    # scaling related to magnitude of peaks
    rho = 0.33 # weighting of MP error

    # maximum number of peaks to test
    maxnpeaks = 10


    Amax = max(pmag)
    harmonic = np.matrix(f0c)

    # PM errors
    ErrorPM = np.zeros(harmonic.size)

    MaxNPM = min(maxnpeaks, pfreq.size)

    if verbose: print(Amax, harmonic, ErrorPM, MaxNPM)

    # PM Error: predicted to measured mismatch error
    for i in range(MaxNPM) :
        difmatrixPM = harmonic.T * np.ones(pfreq.size)
        difmatrixPM = abs(difmatrixPM - np.ones((harmonic.size, 1))*pfreq)
        FreqDistance = np.amin(difmatrixPM, axis=1)    # minimum along rows
        peakloc = np.argmin(difmatrixPM, axis=1)
        Ponddif = np.array(FreqDistance) * (np.array(harmonic.T)**(-p))
        PeakMag = pmag[peakloc]
        MagFactor = 10**((PeakMag-Amax)/20)
        ErrorPM = ErrorPM + (Ponddif + MagFactor*(q*Ponddif-r)).T
        harmonic = harmonic+f0c


    # MP errors
    ErrorMP = np.zeros(harmonic.size)
    MaxNMP = min(maxnpeaks, pfreq.size)
    if verbose: print('--', ErrorMP, MaxNMP, f0c.size)

    # MP error: measured to predicted mismatch error
    for i in range(f0c.size) :
        nharm = np.round(pfreq[:MaxNMP]/f0c[i])
        nharm = (nharm>=1)*nharm + (nharm<1)
        FreqDistance = abs(pfreq[:MaxNMP] - nharm*f0c[i])
        Ponddif = FreqDistance * (pfreq[:MaxNMP]**(-p))
        PeakMag = pmag[:MaxNMP]
        MagFactor = 10**((PeakMag-Amax)/20)
        ErrorMP[i] = sum(MagFactor * (Ponddif + MagFactor*(q*Ponddif-r)))

    # total error
    Error = (ErrorPM[0]/MaxNPM) + (rho*ErrorMP/MaxNMP)
    if verbose: print('Total Error',Error)

    # get the smallest error # f0 with the smallest error
    f0index = np.argmin(Error)
    f0 = f0c[f0index]
    return f0, Error[f0index]

def f0_detection(x, fs, winlen,nfft=None,overlap=None,window='hann',thr=-10,f0min=0.01,f0max=None,f0err=1,
                 return_cand=False):
    """
    Fundamental frequency detection of a signal using TWM algorithm
    ---------------------------------------------------------------

    input
    -----
    x : 1d-array - input signal
    fs: sampling rate
    winlen: window length for analysis
    overlap: overlap of windows
           : if None then winlen//4 is used (25% overlap)
           : shorter overlap can improve time resoltion - upto an extend
    window: analysis window (default = hann)
          : if None, rectangular window is used
    nfft: FFT size, should be >=winlen and power of 2
        : if None -  nfft = 2**np.ceil(np.log2(len(n)))

    thr: threshold for selecting frequencies (in negative dB) -default =-10
    f0min: minimum f0 frequency in Hz (default=0.01)
    f0max: maximim f0 frequency in Hz - if None (default)=fs/2

    f0err: error threshold in the f0 detection (default 1)
    return_cand: bool, if True, returns all candidate frequencies in given range (f0min<=f0cand<=f0max)
               : with correspoding magnitude and phase
               : default False

    output
    ------
    f0: fundamental frequencies of each window
      : f0=0 is refered to frame where no fundamental frequency is found.

    if return_cand True:
    f0_cand: candidates of fundamental frequency
    f0_mag : magnitude of candidates fundamental frequency

    """

    if f0max is None: f0max = fs/2
    #Minumum fundamental frequency shouls be positive and less than half of sampling rate
    assert (f0min>=0) and (f0max<=fs/2)

    if nfft is None: nfft = int(2**np.ceil(np.log2(winlen)))

    if overlap is None: overlap=winlen//4
    #positive overlap
    assert overlap>0


    hN  = int(nfft/2)
    hM1 = int(np.floor((winlen+1)/2)) # first half of analysis window
    hM2 = int(np.floor(winlen/2))     # second half of analysis window

    #padding zeros before and after
    x = np.append(np.zeros(hM2),x)
    x = np.append(x,np.zeros(hM1))

    fftbuffer = np.zeros(nfft)
    f0 = []
    f0t = 0
    f0stable = 0
    if return_cand: f0_cand, f0_mag = [],[]

    win = np.arange(winlen)
    while win[-1]<x.size:
        # ith frame
        xi = x[win]
        mX, pX,_ = dft_analysis(xi, window=window, N=nfft)

        # detect peak locations
        ploc = peak_detection(mX=mX, thr=thr)

        # refine peak values - with parabolic interpolation
        iploc, ipmag, ipphase = peak_interp(mX, pX, ploc)
        ipfreq = fs * iploc/nfft    # in Hz

        if return_cand:
            f0_inx = np.argwhere((ipfreq>f0min) & (ipfreq<f0max))[:,0]
            f0_cand.append(ipfreq[f0_inx].tolist())
            f0_mag.append(ipmag[f0_inx].tolist())

        #TWM Algorithm
        f0t = TWM_f0(ipfreq,ipmag,f0_pre=f0stable,f0max_err=f0err,f0min=f0min,f0max=f0max)

        # consider a stable f0 if it is close to the previous one
        if ((f0stable==0)&(f0t>0)) or ((f0stable>0)&(np.abs(f0stable-f0t)<f0stable/5.0)):
            f0stable = f0t
        else:
            f0stable = 0
        f0.append(f0t)
        win += overlap

    f0 = np.array(f0)
    if return_cand:
        return f0, f0_cand, f0_mag
    return f0

def sinc_dirichlet(x, N):
    """
    Generate the main lobe of a sinc function (Dirichlet kernel)
    ------------------------------------------------------------
    input
    -----
    x: array of indexes to compute
    N: size of FFT to simulate
    output
    ------
    y: samples of the main lobe of a sinc function
    """

    # compute the sinc function
    idx = np.where(x==0)
    x[idx]+=1e-5
    y = np.sin(N * x/2) / np.sin(x/2)
    # avoid NaN if x == 0
    #y[np.isnan(y)] = N
    y[idx] = N
    return y

def blackman_lobe(x,N=512):
    """
    Main lobe of a Blackman-Harris window
    -------------------------------------
    x: bin positions to compute (real values)

    y: main lobe of spectrum of a Blackman-Harris window
    """
    # size of fft to use
    #N = 512

    # frequency sampling
    f = x*np.pi*2/N
    df = 2*np.pi/N
    y = np.zeros(x.size)
    # window constants
    consts = [0.35875, 0.48829, 0.14128, 0.01168]
    # iterate over the four sincs to sum
    for m in range(4):
        # sum of scaled sinc functions
        y += consts[m]/2 * (sinc_dirichlet(f-df*m, N) + sinc_dirichlet(f+df*m, N))
    # normalize
    y = y/N/consts[0]
    return y

def sine_tracking(pfreq, pmag, pphase, tfreq, freqDevOffset=20, freqDevSlope=0.01):
    """
    Tracking sinusoids from one frame to the next
    ---------------------------------------------

    input
    -----
    frequencies
    pfreq, pmag, pphase: frequencies and magnitude of current frame
    tfreq: frequencies of incoming tracks from previous frame
    freqDevOffset: minimum frequency deviation at 0Hz
    freqDevSlope: slope increase of minimum frequency deviation

    output
    ------
    tfreqn, tmagn, tphasen: frequency, magnitude and phase of tracks
    """

    # initialize array for output frequencies
    tfreqn = np.zeros(tfreq.size)
    tmagn = np.zeros(tfreq.size)
    tphasen = np.zeros(tfreq.size)

    # indexes of current peaks
    pindexes = np.array(np.nonzero(pfreq), dtype=int)[0]
    incomingTracks = np.array(np.nonzero(tfreq), dtype=int)[0]
    newTracks = np.zeros(tfreq.size, dtype=int) -1 # initialize to -1 new tracks

    # order current peaks by magnitude
    magOrder = np.argsort(-pmag[pindexes])
    pfreqt = np.copy(pfreq)
    pmagt = np.copy(pmag)
    pphaset = np.copy(pphase)

    # continue incoming tracks
    if incomingTracks.size > 0:                                 # if incoming tracks exist
        for i in magOrder:                                        # iterate over current peaks
            if incomingTracks.size == 0:                            # break when no more incoming tracks
                break
            track = np.argmin(abs(pfreqt[i] - tfreq[incomingTracks]))   # closest incoming track to peak
            freqDistance = abs(pfreq[i] - tfreq[incomingTracks[track]]) # measure freq distance
            if freqDistance < (freqDevOffset + freqDevSlope * pfreq[i]):  # choose track if distance is small
                    newTracks[incomingTracks[track]] = i                      # assign peak index to track index
                    incomingTracks = np.delete(incomingTracks, track)         # delete index of track in incomming tracks
    indext = np.array(np.nonzero(newTracks != -1), dtype=int)[0]   # indexes of assigned tracks
    if indext.size > 0:
        indexp = newTracks[indext]                                    # indexes of assigned peaks
        tfreqn[indext] = pfreqt[indexp]                               # output freq tracks
        tmagn[indext] = pmagt[indexp]                                 # output mag tracks
        tphasen[indext] = pphaset[indexp]                             # output phase tracks
        pfreqt= np.delete(pfreqt, indexp)                             # delete used peaks
        pmagt= np.delete(pmagt, indexp)                               # delete used peaks
        pphaset= np.delete(pphaset, indexp)                           # delete used peaks

    # create new tracks from non used peaks
    emptyt = np.array(np.nonzero(tfreq == 0), dtype=int)[0]      # indexes of empty incoming tracks
    peaksleft = np.argsort(-pmagt)                                  # sort left peaks by magnitude
    if ((peaksleft.size > 0) & (emptyt.size >= peaksleft.size)):    # fill empty tracks
            tfreqn[emptyt[:peaksleft.size]] = pfreqt[peaksleft]
            tmagn[emptyt[:peaksleft.size]] = pmagt[peaksleft]
            tphasen[emptyt[:peaksleft.size]] = pphaset[peaksleft]
    elif ((peaksleft.size > 0) & (emptyt.size < peaksleft.size)):   # add more tracks if necessary
            tfreqn[emptyt] = pfreqt[peaksleft[:emptyt.size]]
            tmagn[emptyt] = pmagt[peaksleft[:emptyt.size]]
            tphasen[emptyt] = pphaset[peaksleft[:emptyt.size]]
            tfreqn = np.append(tfreqn, pfreqt[peaksleft[emptyt.size:]])
            tmagn = np.append(tmagn, pmagt[peaksleft[emptyt.size:]])
            tphasen = np.append(tphasen, pphaset[peaksleft[emptyt.size:]])
    return tfreqn, tmagn, tphasen

def sinetracks_cleaning(tfreq, minTrackLength=3):
    """
    Delete short fragments of a collection of sinusoidal tracks
    -----------------------------------------------------------
    input
    ------
    tfreq: frequency of tracks 2d-array of shape = (number of frames, number of tracks)
    minTrackLength: minimum duration of tracks in number of frames

    output
    ------
    tfreqn: cleaned frequency tracks of same size as tfreq
    """

    # if no tracks return input
    if tfreq.shape[1] == 0:
        return tfreq

    # number of frames and tracks
    nFrames = tfreq[:,0].size
    nTracks = tfreq[0,:].size

    for t in range(nTracks):
        trackFreqs = tfreq[:,t]

        # begining of track contours
        trackBegs = np.nonzero((trackFreqs[:nFrames-1] <= 0)
                                & (trackFreqs[1:]>0))[0] + 1
        if trackFreqs[0]>0:
            trackBegs = np.insert(trackBegs, 0, 0)

        # end of track contours
        trackEnds = np.nonzero((trackFreqs[:nFrames-1] > 0)
                                & (trackFreqs[1:] <=0))[0] + 1
        if trackFreqs[nFrames-1]>0:
            trackEnds = np.append(trackEnds, nFrames-1)

        # lengths of track contours
        trackLengths = 1 + trackEnds - trackBegs

        # delete short track contours
        for i,j in zip(trackBegs, trackLengths):
            if j <= minTrackLength:
                trackFreqs[i:i+j] = 0
    return tfreq

def sine_spectrum(ipfreq, ipmag, ipphase, N, fs):
    """
    Generate a spectrum from a series of sine values - using Blackman Harris window
    --------------------------------------------------------------------------------
    : generate a spectrum by placing main lobe(s) of blackman harris window on given
      locations of frequencies with corresponding magnitude and phases

    input
    -----
    ipfreq : frequencies of sinasodals - 1d-array of shape (m,)
    ipmag  : magnitudes of sinasodals  - same size as ipfreq
    ipphase : pahses of sinasodals      - same size as ipfreq

    N  : size of analysis window to generate complex spectrum
    fs : sampling frequency

    output
    ------
    Y: Complex spectrum generated for given sines

    """
    # Complex spectrum
    Y = np.zeros(N, dtype = complex)
    # size of positive freq. spectrum
    hN = int(N/2)
    K = ipfreq.size  # number of sine waves
    # generate all sine spectral lobes
    for i in range(K):
        loc = N * ipfreq[i] / fs  # it should be in range ]0,hN-1[
        if loc==0 or loc>hN-1: continue  # exclude zero frequency and last one

        binremainder = round(loc)-loc

        # main lobe (real value) bins to read =  8 samples
        lb = np.arange(binremainder-4, binremainder+5)

        # lobe magnitudes of the complex exponential
        lmag = blackman_lobe(lb) * 10**(ipmag[i]/20)
        b = np.arange(round(loc)-4, round(loc)+5)

        for m in range(0, 9):
            if b[m] < 0:
                # peak lobe crosses DC bin
                Y[-b[m]] += lmag[m]*np.exp(-1j*ipphase[i])
            elif b[m] > hN:
                # peak lobe croses Nyquist bin
                Y[b[m]] += lmag[m]*np.exp(-1j*ipphase[i])
            elif b[m] == 0 or b[m] == hN:
                # peak lobe in the limits of the spectrum
                Y[b[m]] += lmag[m]*np.exp(1j*ipphase[i]) + lmag[m]*np.exp(-1j*ipphase[i])
            else:
                # peak lobe in positive freq. range
                Y[b[m]] += lmag[m]*np.exp(1j*ipphase[i])

        # fill the negative part of the spectrum
        Y[hN+1:] = Y[hN-1:0:-1].conjugate()
    return Y

def sineModel_analysis(x,fs,winlen,overlap=None,window='blackmanharris', nfft=None,  thr=-10,
                       maxn_sines=100,minDur=.01,freq_devOffset=20,freq_devSlope=0.01):
    """
    Analysis of a signal x using the sinusoidal model
    -------------------------------------------------
    - Decomposing a signal x into sine waves tracks over the time


    input
    -----
    x : input signal - 1d-array of shape (n,)
    fs: sampling frequency

    winlen : window length for analysis
    overlap: overlap of windows
           : if None overlap = winlen//4
    window : window type (default = 'blackmanharris') e.g. hann,ham

    nfft: FFT-points used for analysis, should be >=winlen and should be of power of 2
        : if None, than nfft = ceil[2**log2(winlen)]

    thr : threshold in negative dB for selecting sine tracks
    maxn_sines: maximum number of sines per frame
    minDur    : minimum duration of sines in seconds
    freq_devOffset: minimum frequency deviation
    freq_devSlope : slope increase of minimum frequency deviation

    output
    ------
    fXt : frequencies
    mXt  : magnitudes
    pXt: phases of sinusoidal tracks

    fXt, mXt, pXt
    """

    #Minimum duration of sine tracks smaller than 0
    assert minDur>=0
    #if (minSineDur <0):                          # raise error if minSineDur is smaller than 0
    #    raise ValueError("Minimum duration of sine tracks smaller than 0")

    if nfft is None: nfft = int(2**np.ceil(np.log2(winlen)))
    if overlap is None: overlap=winlen//4

    hM1 = int(np.floor((winlen+1)/2))
    hM2 = int(np.floor(winlen/2))

    # appending zeros
    x = np.append(np.zeros(hM2),x)
    x = np.append(x,np.zeros(hM2))

    #pin = hM1
    #pend = x.size - hM1
    tfreq = np.array([])
    win = np.arange(winlen)
    fXt,mXt,pXt = [], [], []
    while win[-1]<x.size:
        #ith frame
        xi = x[win]
        mX, pX, _ = dft_analysis(xi, window=window, N=nfft)

        # detect locations of peaks
        ploc = peak_detection(mX, thr)

        # refine peak values
        iploc, ipmag, ipphase = peak_interp(mX, pX, ploc)
        ipfreq = fs*iploc/float(nfft)    # in Hz

        # perform sinusoidal tracking by adding peaks to trajectories
        tfreq, tmag, tphase = sine_tracking(ipfreq, ipmag, ipphase, tfreq,freqDevOffset=freq_devOffset,
                                  freqDevSlope=freq_devSlope)

        # limit number of tracks to maxnSines
        tfreq  = np.resize(tfreq, min(maxn_sines, tfreq.size))
        tmag   = np.resize(tmag,  min(maxn_sines, tmag.size))
        tphase = np.resize(tphase,min(maxn_sines, tphase.size))

        jtfreq  = np.zeros(maxn_sines)
        jtmag   = np.zeros(maxn_sines)
        jtphase = np.zeros(maxn_sines)

        jtfreq[:tfreq.size]  = tfreq
        jtmag[:tmag.size]    = tmag
        jtphase[:tphase.size]= tphase

        fXt.append(jtfreq)
        mXt.append(jtmag)
        pXt.append(jtphase)
        win += overlap

    # delete sine tracks shorter than minSineDur

    fXt = np.vstack(fXt)
    mXt = np.vstack(mXt)
    pXt = np.vstack(pXt)

    fXt = sinetracks_cleaning(fXt, round(fs*minDur/overlap))
    return fXt, mXt, pXt

def sineModel_synthesis(fXt,mXt,pXt,fs,overlap,crop_end=False):
    """
    Synthesis of signal x using the Sinusoidal Model
    ------------------------------------------------
    Synthesing signal for given frequencies with magnitude sinasodal tracks

    input
    -----
    fXt : frequency tracks - 2d-array- shape =(number of frames, number of sinasodal tracks)
    mXt : magnitude    -  same size of array as
    pXt : phases of sinusoids
    fs  :  sampling frequency
    overlap: overlap of consequitive frames (in samples)

    output
    ------
    y : 1d-array - synthesised signal

    -------
    Ref: https://en.wikipedia.org/wiki/Additive_synthesis
    Ref: https://en.wikipedia.org/wiki/Sinusoidal_model
    Ref: https://www.coursera.org/learn/audio-signal-processing
    """

    def synthWindow(H):
        N = int(H*4) # 4-times of overlap size H
        hN = int(N/2)
        syn_win = np.zeros(N)

        # triangular window
        tri_win = triang(2*H)

        # add triangular window
        syn_win[hN-H:hN+H] = tri_win

        # blackmanharris window
        bh_win = blackmanharris(N)
        bh_win /= np.sum(bh_win)

        #synthesis window - triangular/Blackman-Harris of middle half
        syn_win[hN-H:hN+H] = syn_win[hN-H:hN+H]/bh_win[hN-H:hN+H]
        return syn_win

    syn_win = synthWindow(H=overlap)

    N = int(overlap*4) # synthesis window size - 4times of overlap
    # half of synthesis window
    hN = int(N/2)

    # number of frames
    L = fXt.shape[0]

    # output of size (overlap*(number of frames + 3))
    y  = np.zeros(int(overlap*(L+3)))
    lastytfreq = fXt[0,:]

    # initialize synthesis phases
    ytphase = 2*np.pi*np.random.rand(fXt[0,:].size)

    pout = 0
    for i in range(L):
        #ith frame
        if pXt is not None and (pXt.size > 0):
            ytphase = pXt[i,:]
        else:
            # if no phases generate them
            # propagate phases - if phase is not provided
            ytphase += (np.pi*(lastytfreq + fXt[i,:])/fs)*overlap
            lastytfreq = fXt[i,:]

        # generate spectrum of sine waves for current frame
        # using blackman-harris window
        Y = sine_spectrum(fXt[i,:], mXt[i,:], ytphase, N, fs)

        # make phase inside 2*pi
        ytphase = ytphase % (2*np.pi)

        # signal for current frame
        yw = np.real(fftshift(ifft(Y)))

        # overlap-add and apply a synthesis window
        y[i*overlap:i*overlap+N] += syn_win*yw

    y = y[hN:]
    if crop_end: y = y[-overlap:]
    return y


#TOBE TESTED
def simplify_signal(x,fs,winlen,overlap,mag=-1,N=1,thr=-20,minDur=0.01,freq_devOffset=10,freq_devSlope=0.1,window='blackmanharris'):
    r"""
    Simplify a signal with Sinasodal Decomposition-Recomposition Model
    -----------------------------------------------------------------




    """
    fXst, mXst, pXst = sineModel_analysis(x,fs,winlen=winlen,overlap=overlap,
                          window=window, nfft=None, thr=thr,
                          maxn_sines=N,minDur=minDur, freq_devOffset=freq_devOffset,freq_devSlope=freq_devSlope)


    if mag<0:
        xr = sineModel_synthesis(fXst, mXst, pXst,fs,overlap=overlap)
    else:
        xr = sineModel_synthesis(fXst, mXst*0+mag, pXst,fs,overlap=overlap)

    xr = xr[:len(x)]
    return xr, (fXst, mXst, pXst)
