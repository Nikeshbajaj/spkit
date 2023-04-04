'''
Ramanujan methods
-----------------
Author @ Nikesh Bajaj
updated on Date: 1 jan 2021, Version : 0.0.1
Github :  https://github.com/Nikeshbajaj/spkit
Contact: n.bajaj@qmul.ac.uk | n.bajaj@imperial.ac.uk
'''

from __future__ import absolute_import, division, print_function
name = "Signal Processing toolkit | Ramanujan methods"
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
import sys, scipy
from scipy import linalg as LA
import warnings

def RFB(x, Pmax=10, Rcq=10, Rav=2, Th=0.2,Penalty=None,return_filters=False, apply_averaging=True):
    r'''

    Ramanujan Filter Banks for Estimation and Tracking of Periodicity
    -----------------------------------------------------------------

    input
    ------
    x    = 1d array, sequence of signal
    Pmax = the largest expected period.
    Rcq  = Number of repeats in each Ramanujan filter
    Rav  = Number of repeats in each averaging filter
    Th   = Outputs of the RFB are thresholded to zero for all values less than Th*max(output)
    Penalt = penalty for each period shape=(len(Pmax)),
             If None, then set to 1, means no penalty

    output
    ------
    y = 2d array of shape = (len(x),Pmax), time vs period matrix, normalized

    if return_filters==True: also returns

    FR = list of Ramanujan Filters
    FA = list of Averaging Filters

    References:
    [1] S.V. Tenneti and P. P. Vaidyanathan, "Ramanujan Filter Banks for Estimation
    and Tracking of Periodicity", Proc. IEEE Int. Conf. Acoust.
    Speech, and Signal Proc., Brisbane, April 2015.

    [2] P.P. Vaidyanathan and S.V. Tenneti, "Properties of Ramanujan Filter Banks",
    Proc. European Signal Processing Conference, France, August 2015.

    Python impletation is done by using matlab code version from
    - http://systems.caltech.edu/dsp/students/srikanth/Ramanujan/
    '''
    # Peanlty vector.
    if Penalty is None: Penalty = np.ones(Pmax)

    # Can be (optionally) used to set preference  to a certain set of periods in the
    # time vs period plane.

    FR = [[]]*Pmax #The set of Ramanujan Filters
    FA = [[]]*Pmax #The set of Averaging Filters
    for i in range(Pmax):
        cq = np.zeros(i+1) + 1j*0  #cq shall be the ith ramanujan sum sequnece.
        k_orig = np.arange(i+1)+1
        k = k_orig[np.gcd(k_orig,i+1)==1]
        for n in range(i+1):
            cq[n] += np.sum([np.exp(1j*2*np.pi*a*(n)/(i+1)) for a in k])

        cq = np.real(cq)
        FR[i]  = np.tile(cq,Rcq)
        FR[i] /= np.linalg.norm(FR[i])

        FA[i]  = np.tile(np.ones(i+1),Rav)
        FA[i] /= np.linalg.norm(FA[i])


    #Computing the Outputs of the Filter Bank
    y = np.zeros([len(x),Pmax])

    if np.ndim(x)>1:
        xi = x[:,0].copy()
    else:
        xi = x.copy()

    for i in range(Pmax):
        npad = len(FR[i]) - 1
        xi_padded = np.pad(xi, (npad//2, npad - npad//2), mode='constant')
        y_temp = np.convolve(xi_padded,FR[i],mode='valid')
        y_temp = (np.abs(y_temp))**2
        y_temp = y_temp/Penalty[i]
        if apply_averaging:
            npad = len(FA[i]) - 1
            y_temp_padded = np.pad(y_temp, (npad//2, npad - npad//2), mode='constant')
            y_temp = np.convolve(y_temp_padded,FA[i],mode='valid')
        y[:,i] = y_temp

    y[:,0] = 0;  # Periods 1 give strong features on the time vs period planes. Hence, zeroing them out to view the other periods better.
    y = y - np.min(y)
    y = y/np.max(y)
    y[y<Th]=0
    if return_filters:
        return y,FR, FA
    return y

def Create_Dictionary(Nmax, rowSize, method='Ramanujan'):
    '''
    Creating Dictionary
    -------------------
    input
    ----
    Nmax    : maximum expected Period,
    rowSize : number of rows (e.g. samples in signal)
    method  : 'Ramanujan' 'random', 'NaturalBasis', 'DFT'

    output
    ------
    A :  Matrix of (rowSize, q)

    The relevant paper is:
    [1] S.V. Tenneti and P. P. Vaidyanathan, "Nested Periodic Matrices and Dictionaries:
        New Signal Representations for Period Estimation", IEEE Transactions on Signal
        Processing, vol.63, no.14, pp.3736-50, July, 2015.

    Python impletation is done by using matlab code version from
    - http://systems.caltech.edu/dsp/students/srikanth/Ramanujan/
    '''
    A = []
    for N in range(Nmax):
        if method in ['Ramanujan', 'NaturalBasis', 'random']:
            if method=='Ramanujan':
                c1 = np.zeros(N+1) + 1j*0
                k_orig = np.arange(N+1)+1
                k = k_orig[np.gcd(k_orig,N+1)==1]
                for n in range(N):
                    c1[n] += np.sum([np.exp(1j*2*np.pi*a*(n)/(N+1)) for a in k])
                c1 = np.real(c1)

            elif method=='NaturalBasis':
                c1 = np.zeros(N+1)
                c1[0] = 1

            elif method=='random':
                c1 = np.random.randn(N+1)

            k_orig = np.arange(N+1)+1
            k = k_orig[np.gcd(k_orig,N+1)==1]
            CN_colSize = len(k)  #k.shape[1]

            CN=[]
            for j in range(CN_colSize):
                CN.append(np.roll(c1,j))
            CN = np.vstack(CN).T

        else: #method=='Farey'
            A_dft = LA.dft(N+1)
            a = np.arange(N+1)
            a[0] = N+1
            a = (N+1)/np.gcd(a,N+1)
            I = np.arange(N+1)
            I = I[a==N+1]
            CN = A_dft[:,I]

        CNA = np.tile(CN,(np.floor(rowSize/(N+1)).astype(int),1))
        CN_cutoff = CN[:np.remainder(rowSize,N+1),:]
        CNA =np.vstack([CNA,CN_cutoff])
        A.append(CNA)
    return np.hstack(A)

def PeriodStrength(x,Pmax,method='Ramanujan',lambd=1,L=1,cvxsol=False):
    '''
    Computing strength of periods
    -----------------------------
    for given signal x, using method and respective loss fun (e.g. l1, l2)

    inputs
    -----
    x   :  one dimentional sequence (signal)
    Pmax: largest expected period in the signal
    method: type of dictionary used to create transform matrix A
          : 'Ramanujan', 'NaturalBasis', 'random' or Farray (DFT)

    lambd: for penalty vector, to force towards lower (usually) or higher periods
         : if 0, then penalty vector is 1, means no penalization
         : if >0, then lambd is multiplied to penalty vector

    L : regularazation: L=1, minimize ||s||_1, L=2, ||s||_2

    cvxsol: bool, wether to use cvxpy solver of matrix decomposition approach
          : matrix decomposition approach works only for L=2
          : for L=1, use cvxpy as solver

    output
    ------
    period_energy: vecotor shape: (Pmax,): strength of each period


    Reference:
    [1] S.V. Tenneti and P. P. Vaidyanathan, "Nested Periodic Matrices and Dictionaries:
       New Signal Representations for Period Estimation", IEEE Transactions on Signal
       Processing, vol.63, no.14, pp.3736-50, July, 2015.

    Python impletation is done by using matlab code version from
    - http://systems.caltech.edu/dsp/students/srikanth/Ramanujan/
    '''
    if cvxsol:
        try:
            import cvxpy
        except Exception as err:
            wst =  "cvxpy is not installed! use 'pip install cvxpy --user' \n"
            wst += "install cvxpy for L1 norm minimization for PeriodStrength fun (Ramanujan methods)"
            wst += "Or set 'cvxsol=False' to use LMS\n"
            wst += f"Unexpected {err}, {type(err)}"
            warnings.warn(wst,stacklevel=2)
            raise
    assert np.ndim(x)==1
    #Nmax = Pmax
    A = Create_Dictionary(Pmax,x.shape[0],method)

    #Penalty Vector Calculation
    if lambd>0:
        penalty_vector = []
        for i in range(Pmax):
            k = np.arange(i+1)+1
            k_red = k[np.gcd(k,i+1)==1]
            k_red = len(k_red)
            penalty_vector.append((i+1)*np.ones(k_red))
        penalty_vector = np.hstack(penalty_vector)
        penalty_vector = lambd*(penalty_vector**2)
    else:
        penalty_vector = np.ones(A.shape[1]) #0*(penalty_vector**2)+1


    if cvxsol:
        s = cvxpy.Variable(A.shape[1], complex=np.sum(np.iscomplex(x)))
        cost = cvxpy.norm(cvxpy.multiply(penalty_vector, s),L)
        constraints = [x == A@s]
        prob = cvxpy.Problem(cvxpy.Minimize(cost),constraints)
        prob.solve()
        si = s.value
    else:
        #x = A@s -->  s = inv(A.T@A)@A.T@x
        D = np.diag((1./penalty_vector)**2)
        PP = (D@A.T)@LA.inv(A@D@A.T)
        s = PP@x
        si = s

    period_energy = np.zeros(Pmax)
    index_end = 0
    for i in range(Pmax):
        k_orig = np.arange(i+1)+1
        k = k_orig[np.gcd(k_orig,i+1)==1]
        index_start = index_end
        index_end   = index_end + len(k)
        period_energy[i] += np.sum(np.abs(si[index_start:index_end])**2)

    period_energy[0] = 0 #one sample period is stronger, so zeroing it out
    return period_energy

def RFB_example_1(period=10,SNR=0,seed=10):
    np.random.seed(seed)
    #period = 10
    #SNR = 0

    x1 = np.zeros(30)
    x2 = np.random.randn(period)
    x2 = np.tile(x2,10)
    x3 = np.zeros(30)
    x  = np.r_[x1,x2,x3]
    x /= LA.norm(x,2)

    noise  = np.random.randn(len(x))
    noise /= LA.norm(noise,2)

    noise_power = 10**(-1*SNR/20)

    noise *= noise_power
    x_noise = x + noise

    plt.plot(x,label='signal: x')
    plt.plot(x_noise, label='signal+noise: x_noise')
    plt.xlabel('sample (n)')
    plt.legend()
    plt.show()


    Pmax = 40  #Largest expected period in the input
    Rcq  = 10   # Number of repeats in each Ramanujan filter
    Rav  = 2    #Number of repeats in each averaging filter
    Th   = 0.2   #Outputs of the RFB are thresholded to zero for all values less than Th*max(output)

    y,FR, FA = RFB(x_noise,Pmax, Rcq, Rav, Th,return_filters=True)

    plt.figure(figsize=(15,5))
    im = plt.imshow(y.T,aspect='auto',cmap='jet',extent=[1,len(x_noise),Pmax,1])
    plt.colorbar(im)
    plt.xlabel('sample (n)')
    plt.ylabel('period (in samples)')
    plt.show()

    plt.stem(np.arange(1,y.shape[1]+1),np.sum(y,0))
    plt.xlabel('period (in samples)')
    plt.ylabel('strength')
    plt.show()

    print('top 10 periods: ',np.argsort(np.sum(y,0))[::-1][:10]+1)

def RFB_example_2(periods=[3,7,11],signal_length=100,SNR=10,seed=15):
    np.random.seed(seed)
    #periods    = [3,7,11]
    #signal_length = 100
    #SNR = 10
    x = np.zeros(signal_length)
    for period in periods:
        x_temp  = np.random.randn(period)
        x_temp  = np.tile(x_temp,int(np.ceil(signal_length/period)))
        x_temp  = x_temp[:signal_length]
        x_temp /= LA.norm(x_temp,2)
        x += x_temp

    x /= LA.norm(x,2)

    noise  = np.random.randn(len(x))
    noise /= LA.norm(noise,2)
    noise_power = 10**(-1*SNR/20)
    noise *= noise_power
    x_noise = x + noise

    plt.plot(x,label='signal: x')
    plt.plot(x_noise, label='signal+noise: x_noise')
    plt.xlabel('sample (n)')
    plt.legend()
    plt.show()


    Pmax = 90

    periodE = PeriodStrength(x_noise,Pmax=Pmax,method='Ramanujan',lambd=1, L=1, cvxsol=True)

    plt.stem(np.arange(len(periodE))+1,periodE)
    plt.xlabel('period (in samples)')
    plt.ylabel('strength')
    plt.title('L1 + penality')
    plt.show()

    print('top 10 periods: ',np.argsort(periodE)[::-1][:10]+1)


    periodE = PeriodStrength(x_noise,Pmax=Pmax,method='Ramanujan',lambd=0, L=1, cvxsol=True)

    plt.stem(np.arange(len(periodE))+1,periodE)
    plt.xlabel('period (in samples)')
    plt.ylabel('strength')
    plt.title('L1 without penality')
    plt.show()


    print('top 10 periods: ',np.argsort(periodE)[::-1][:10]+1)


    periodE = PeriodStrength(x_noise,Pmax=Pmax,method='Ramanujan',lambd=1, L=2, cvxsol=False)

    plt.stem(np.arange(len(periodE))+1,periodE)
    plt.xlabel('period (in samples)')
    plt.ylabel('strength')
    plt.title('L2 +  penalty')
    plt.show()

    print('top 10 periods: ',np.argsort(periodE)[::-1][:10]+1)


    y = RFB(x_noise,Pmax = Pmax, Rcq=10, Rav=2, Th=0.2)

    plt.figure(figsize=(15,5))
    im = plt.imshow(y.T,aspect='auto',cmap='jet',extent=[1,len(x_noise),Pmax,1])
    plt.colorbar(im)
    plt.xlabel('sample (n)')
    plt.ylabel('period (in samples)')
    plt.show()

    plt.stem(np.arange(1,y.shape[1]+1),np.sum(y,0))
    plt.xlabel('period (in samples)')
    plt.ylabel('strength')
    plt.show()

    print('top 10 periods: ',np.argsort(np.sum(y,0))[::-1][:10]+1)



    XF = np.abs(np.fft.fft(x_noise))[:1+len(x_noise)//2]
    fq = np.arange(len(XF))/(len(XF)-1)

    plt.stem(fq,XF)
    plt.title('DFT')
    plt.ylabel('| X |')
    plt.xlabel(r'frequency $\times$ ($\omega$/2)   ~   1/period ')
    plt.show()

def RFB_prange(x,Pmin=1,Pmax=10,skip=1,Rcq=10,Rav=2,thr=0.2,Penalty=None,return_filters=False,apply_averaging=True):
    '''
    Ramanujan Filter Banks for Estimation and Tracking of Periodicity
    -----------------------------------------------------------------
    - for range of period given by Pmin and Pmax.

    input
    ------
    x    = 1d array, sequence of signal
    Pmin = the smallest expected period. (default=1)
    Pmax = the largest expected period.
    skip = int >=1: if to skip period (default=1 --> no skipping) (>1 is not recomended)
    Rcq  = Number of repeats in each Ramanujan filter
    Rav  = Number of repeats in each averaging filter
    thr   = Outputs of the RFB are thresholded to zero for all values less than Th*max(output)
    Penalty = penalty for each period shape=(len(Pmax)),
             If None, then set to 1, means no penalty
    apply_averaging: bool, if False, no averaging is applied (deault=True)
    return_filters: bool, ifTrue, return FR - Ramanujan and FA - Averaging filters

    output
    ------
    y = 2d array of shape = (len(x),Pmax), time vs period matrix, normalized

    if return_filters==True: also returns

    FR = list of Ramanujan Filters
    FA = list of Averaging Filters

    References:
    [1] S.V. Tenneti and P. P. Vaidyanathan, "Ramanujan Filter Banks for Estimation
    and Tracking of Periodicity", Proc. IEEE Int. Conf. Acoust.
    Speech, and Signal Proc., Brisbane, April 2015.

    [2] P.P. Vaidyanathan and S.V. Tenneti, "Properties of Ramanujan Filter Banks",
    Proc. European Signal Processing Conference, France, August 2015.

    Python impletation is done by using matlab code version from
    - http://systems.caltech.edu/dsp/students/srikanth/Ramanujan/
    '''



    Plist = np.arange(Pmin,Pmax+1,skip).astype(int)
    nP = len(Plist)

    # Peanlty vector.
    if Penalty is None: Penalty = np.ones(nP)
    # Can be (optionally) used to set preference  to a certain set of periods in the
    # time vs period plane.


    FR = []  #*nP #The set of Ramanujan Filters
    FA = []  #*nP #The set of Averaging Filters

    for p in Plist:      #range(Pmin-1,Pmax):
        cq = np.zeros(p) + 1j*0  #cq shall be the ith ramanujan sum sequnece.
        k_orig = np.arange(p)+1
        k = k_orig[np.gcd(k_orig,p)==1]
        for n in range(p):
            cq[n] += np.sum([np.exp(1j*2*np.pi*a*(n)/(p)) for a in k])

        cq = np.real(cq)
        fr = np.tile(cq,Rcq)
        fr /=np.linalg.norm(fr)
        FR.append(fr)

        fa  = np.tile(np.ones(p),Rav)
        fa /= np.linalg.norm(fa)
        FA.append(fa)


    #Computing the Outputs of the Filter Bank
    y = np.zeros([len(x),nP])

    if np.ndim(x)>1:
        xi = x[:,0].copy()
    else:
        xi = x.copy()

    kii= 0
    for i in range(len(FR)):
        npad = len(FR[i]) - 1
        xi_padded = np.pad(xi, (npad//2, npad - npad//2), mode='constant')
        y_temp = np.convolve(xi_padded,FR[i],mode='valid')
        y_temp = (np.abs(y_temp))**2
        y_temp = y_temp/Penalty[i]
        if apply_averaging:
            npad = len(FA[i]) - 1
            y_temp_padded = np.pad(y_temp, (npad//2, npad - npad//2), mode='constant')
            y_temp = np.convolve(y_temp_padded,FA[i],mode='valid')
        y[:,i] = y_temp

    if Plist[0]==1:
        y[:,0] = 0;  # Periods 1 give strong features on the time vs period planes.
                     # Hence, zeroing them out to view the other periods better.
    y = y - np.min(y)
    y = y/np.max(y)
    y[y<thr]=0
    if return_filters:
        return y,Plist,FR, FA
    return y,Plist
