from __future__ import absolute_import, division, print_function

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


'''
Author @ Nikesh Bajaj
Date: 18 Apr 2019
Version : 0.0.1
Github :  https://github.com/Nikeshbajaj/spkit
Contact: n.bajaj@qmul.ac.uk
'''

# Probability distribuation is computed using histogram
# and optimal bin size of histogram is computed using Freedman–Diaconis rule

def entropy(x,alpha=1,ignoreZero=False, base=2):
    '''
    Rényi entropy of order α
    alpha:[0,inf]
         :0: Max-entropy
             H(x) = log(N)
         :1: Shannan entropy
             H(x) = -\sum{Px*log(Px)}
         :2: Collision entropy
             H(x) = 1/(1-α)*log{\sum{Px^α}}
         :inf:Min-entropy:
             H(x) = -log(max(Px))
    base: base of log:
        : if 2, entropy is in bits, e-nats, 10 -bans
    ignoreZero: if true, probabilities with zero value will be omited, before computations
          : It doesn't make much of difference
    '''
    if base=='e': base =np.exp(1)

    if alpha==0:
        H = np.log(len(x))
    else:
        frq,_ = np.histogram(x,bins='fd')
        Pr = frq/np.sum(frq)
        Pr = Pr[Pr>0] if ignoreZero else Pr+1e-10

        if alpha==1:
            H  = -np.sum(Pr*np.log(Pr))
        elif alpha==np.inf or alpha=='inf':
            H  = -np.log(np.max(Pr))
        else:
            H = (1.0/(1.0-alpha))*np.log(np.sum(Pr**alpha))

    H /=np.log(base)
    return H

def entropy_joint(x,y,ignoreZero=False,base=2):
    '''
	H(X,Y) = \sum {P(x,y)*np.log(P(x,y))}

    Computing joint probability using histogram2d from numpy
    '''
    # computing the optimal bin size using Freedman–Diaconis rule-
    _,bins = np.histogram(x,bins='fd')
    binx = np.ceil((np.max(x)-np.min(x))/(bins[1]-bins[0])).astype(int)

    _,bins = np.histogram(y,bins='fd')
    biny = np.ceil((np.max(y)-np.min(y))/(bins[1]-bins[0])).astype(int)

    #Computing joint probability
    frq = np.histogram2d(x,y,bins=[binx,biny])[0]
    Prxy = frq/np.sum(frq)
    Prxy = Prxy[Prxy>0] if ignoreZero else Prxy + 1e-10
    Hxy = - np.sum(Prxy*np.log(Prxy))
    if base!='e': Hxy /= np.log(base)
    return Hxy

def entropy_cond(x,y,ignoreZero=False,base=2):
    '''H(X|Y) = H(X,Y) - H(Y)'''
    Hxy = entropy_joint(x,y,ignoreZero=ignoreZero,base=base)
    Hy  = entropy(y,ignoreZero=ignoreZero,base=base)

    Hx1y = Hxy-Hy

    return Hx1y

def mutual_Info(x,y,ignoreZero=False,base=2):
    '''I(X;Y) = H(X)+H(Y)-H(X,Y)'''
    '''I(X;Y) = H(X) - H(X|Y)'''

    I = entropy(x,ignoreZero=ignoreZero,base=base)+\
        entropy(y,ignoreZero=ignoreZero,base=base)-\
        entropy_joint(x,y,ignoreZero=ignoreZero,base=base)
    return I

def entropy_kld(x,y,base=2):
    '''
    H_xy =  \sum{Px*log(Px/Py)}
    Cross entropy - Kullback–Leibler divergence
    '''
    _,bins = np.histogram(x,bins='fd')
    binx = bins[1]-bins[0]

    _,bins = np.histogram(y,bins='fd')
    biny = bins[1]-bins[0]

    binxy = np.min([binx,biny])
    xy = np.r_[x,y]

    nbins = np.ceil((max(xy)-min(xy))/binxy).astype(int)

    frq,_ = np.histogram(x,bins=nbins)
    PrX = frq/np.sum(frq)

    frq,_ = np.histogram(y,bins=nbins)
    PrY = frq/np.sum(frq)

    #ignoring to be divided by 0
    PrX += 1e-10
    PrY += 1e-10

    H  = np.sum(PrX*np.log(PrX/PrY))
    if base !='e': H = H/np.log(base)
    return H

def entropy_cross(x,y,base=2):
    '''
    Cross entropy
    H_xy = - \sum{Px*log(Py)}
    '''

    _,bins = np.histogram(x,bins='fd')
    binx = bins[1]-bins[0]

    _,bins = np.histogram(y,bins='fd')
    biny = bins[1]-bins[0]

    binxy = np.min([binx,biny])
    xy = np.r_[x,y]

    nbins = np.ceil((max(xy)-min(xy))/binxy).astype(int)

    frq,_ = np.histogram(x,bins=nbins)
    PrX = frq/np.sum(frq)

    frq,_ = np.histogram(y,bins=nbins)
    PrY = frq/np.sum(frq)

    #ignoring to be divided by 0
    PrX += 1e-10
    PrY += 1e-10

    H  = -np.sum(PrX*np.log(PrY))
    if base !='e': H = H/np.log(base)
    return H

def HistPlot(x,show=False,norm=False):
    frq,bins = np.histogram(x,bins='fd')
    if norm: frq = frq/np.sum(frq)
    plt.bar(bins[:-1],frq,width=0.8*(bins[1]-bins[0]),alpha=0.5)
    #plt.plot(bins[:-1],frq,'--k',lw=0.7)
    if show: plt.show()

def binSize_FD(x):
    IQR = stats.iqr(x)
    bw =  2.0*IQR/(len(x)**(1/3))
    return bw
