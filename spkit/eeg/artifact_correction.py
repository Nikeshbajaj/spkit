'''
ICA Based - Artifact Removal Algorithms:
------------------------------------------
Author @ Nikesh Bajaj
updated on Date: 26 Sep 2021
Version : 0.0.4
Github :  https://github.com/Nikeshbajaj/spkit
Contact: n.bajaj@qmul.ac.uk | n.bajaj@imperial.ac.uk
'''


from __future__ import absolute_import, division, print_function
name = "Signal Processing toolkit | EEG | ICA based Artifact Removal Algorith"
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
#from .ICA_methods import ICA
from core.matDecomposition import ICA, SVD
#from matplotlib.gridspec import GridSpec
from scipy.stats import kurtosis, skew
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import utils

def ICA_filtering(X,winsize=128,ICA_method='extended-infomax',kur_thr=2,corr_thr=0.8,AF_ch_index =[0,13],
                 F_ch_index=[1,2,11,12],verbose=True,window=['hamming',True],hopesize=None,winMeth='custom'):
    '''

    input
    ------
    X: input signal (n,ch) with n samples and ch channels
    winsize:  window size to process, if None, entire signal is used at once
    ICAMed = ['fastICA','infomax','extended-infomax','picard']

    (1) Kurtosis based artifacts - mostly for motion artifacts
    ------------------------------------------------------
    kur_thr: (default 2) threshold on kurtosis of IC commponents to remove, higher it is, more peaky component is selected
           : +ve int value
    (2) Correlation Based Index (CBI) for eye movement artifacts
    --------------------------------------------------------
    for applying CBI method, index of prefrontal (AF - First Layer of electrodes towards frontal lobe) and frontal lobe (F - second layer of electrodes)channels needs to be provided.
    For case of 14-channels Emotiv Epoc
    ch_names = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
    PreProntal Channels =['AF3','AF4'], Fronatal Channels = ['F7','F3','F4','F8']
    AF_ch_index =[0,13] :  (AF - First Layer of electrodes towards frontal lobe)
    F_ch_index =[1,2,11,12] : (F - second layer of electrodes)
    if AF_ch_index or F_ch_index is None, CBI is not applied

    (3) Correlation of any independent component with many EEG channels
    ---------------------------------------------------------
    If any indepentdent component is correlated fo corr_thr% (80%) of elecctrodes, is considered to be artifactual
    -- Similar like CBI, except, not comparing fronatal and prefrontal but all
    corr_thr: (deafult 0.8) threshold to consider correlation, higher the value less IC are removed and vise-versa
            : float [0-1],
            : if None, this  is not applied
    '''
    assert kur_thr>=0
    assert corr_thr>=0 and corr_thr<=1
    # CBI channels can not be all of the cchannels
    assert len(AF_ch_index)<X.shape[1] and len(F_ch_index)<X.shape[1]


    win = np.arange(winsize)
    #XR =[]
    nch = X.shape[1]
    #nSeg = X.shape[0]//winsize
    if hopesize is None: hopesize=winsize//2
    if verbose:
        print('ICA Artifact Removal : ' + ICA_method)

    if winMeth is None:
        xR = _RemoveArtftICA_CBI_Kur_Iso(X,winsize=winsize,corr_thr=corr_thr,kur_thr=kur_thr,ICA_method=ICA_method,
              AF_ch_index=AF_ch_index,F_ch_index=F_ch_index,verbose=verbose)

    elif winMeth =='custom':
        M   = winsize
        H   = hopesize
        hM1 = (M+1)//2
        hM2 = M//2

        Xt  = np.vstack([np.zeros([hM2,nch]),X,np.zeros([hM1,nch])])

        pin  = hM1
        pend = Xt.shape[0]-hM1
        wh   = get_window(window[0],M)

        if len(window)>1: AfterApply = window[1]
        else: AfterApply =False
        xR   = np.zeros(Xt.shape)

        while pin<=pend:
            if verbose:
                utils.ProgBar_float(pin,N=pend,title='',style=2,L=50)

            xi = Xt[pin-hM1:pin+hM2]
            if not(AfterApply):
                xi *=wh[:,None]
            xr = ICAremoveArtifact(xi,ICA_method=ICA_method,corr_thr=corr_thr,kur_thr=kur_thr,AF_ch_index=AF_ch_index,F_ch_index=F_ch_index)
            if AfterApply: xr *=wh[:,None]
            xR[pin-hM1:pin+hM2] += H*xr  ## Overlap Add method
            pin += H
        xR = xR[hM2:-hM1]/sum(wh)
    return xR

def _RemoveArtftICA_CBI_Kur_Iso(X,winsize=128,corr_thr=0.8,kur_thr=2,ICA_method='extended-infomax',verbose=True,AF_ch_index =[0,13],F_ch_index=[1,2,11,12]):
    '''
    ICAMed = ['fastICA','infomax','extended-infomax','picard']
    ICAMed = ['fastICA','infomax','extended-infomax','picard']

    (1) Kurtosis based artifacts - mostly for motion artifacts
    ------------------------------------------------------
    kur_thr: (default 2) threshold on kurtosis of IC commponents to remove, higher it is, more peaky component is selected
           : +ve int value
    (2) Correlation Based Index (CBI) for eye movement artifacts
    --------------------------------------------------------
    for applying CBI method, index of prefrontal (AF - First Layer of electrodes towards frontal lobe) and frontal lobe (F - second layer of electrodes)channels needs to be provided.
    For case of 14-channels Emotiv Epoc
    ch_names = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
    PreProntal Channels =['AF3','AF4'], Fronatal Channels = ['F7','F3','F4','F8']
    AF_ch_index =[0,13] :  (AF - First Layer of electrodes towards frontal lobe)
    F_ch_index =[1,2,11,12] : (F - second layer of electrodes)
    if AF_ch_index or F_ch_index is None, CBI is not applied

    (3) Correlation of any independent component with many EEG channels
    ---------------------------------------------------------
    If any indepentdent component is correlated fo corr_thr% (80%) of elecctrodes, is considered to be artifactual
    -- Similar like CBI, except, not comparing fronatal and prefrontal but all
    corr_thr: (deafult 0.8) threshold to consider correlation, higher the value less IC are removed and vise-versa
            : float [0-1],
            : if None, this  is not applied
    '''
    win = np.arange(winsize)
    XR =[]
    nch = X.shape[1]
    nSeg = X.shape[0]//winsize
    if verbose:
        print('ICA Artifact Removal : '+ ICA_method)

    while win[-1]<X.shape[0]:
        if verbose:
            pf = win[-1]*100.0/float(X.shape[0])
            pbar = '|'+'#'*int(pf)+' '*(99-int(pf))+'|'
            print(str(np.round(pf,2))+'%'+pbar,end='\r', flush=True)

        Xi = X[win,:]
        J =[]
        ica = ICA(n_components=nch,method=ICA_method)
        ica.fit(Xi.T)
        IC = ica.transform(Xi.T).T
        mu = ica.pca_mean_
        W = ica.get_sMatrix()
        #A = ica.get_tMatrix()
        sd = np.std(IC,axis=0)
        ICn = IC/sd
        Wn = W*sd
        Wnr = Wn/np.sqrt(np.sum(Wn**2,axis=1,keepdims=True))

        # Method (3)
        if corr_thr is not None:
            ICss,frqs = np.unique(np.argmax(Wnr,axis=1), return_counts=True)
            j1 = ICss[np.where(frqs/nch>=corr_thr)[0]]
            J.append(j1)
            ICss,frqs = np.unique(np.argmin(Wnr,axis=1), return_counts=True)
            j2 = ICss[np.where(frqs/nch>=corr_thr)[0]]
            J.append(j2)
        # Method (2)
        if len(AF_ch_index) and len(F_ch_index):
            CBI,j3,Fault = CBIeye(Wnr,plotW=False,AF_ch_index=AF_ch_index,F_ch_index=F_ch_index)
            if Fault:
                J.append(j3)

        # Method (1)
        kur   = kurtosis(ICn,axis=0)
        J.append(np.where(abs(kur)>=kur_thr)[0])
        J = list(set(np.hstack(J)))
        if verbose>1:
            print('IC to be removed - ',J)
        if len(J)>0:
            #print('------')
            for ji in J:
                W[:,ji]=0
            Xr = np.dot(IC,W.T)+mu
        else:
            Xr = Xi
        if win[0]==0:
            XR = Xr
        else:
            XR = np.vstack([XR,Xr])
        win +=winsize
    if verbose:
        pf = 100
        pbar = '|'+'#'*int(pf)+' '*(99-int(pf))+'|'
        print(str(np.round(pf,2))+'%'+pbar,end='\r', flush=True)
        print('')
    return XR

def CBIeye(Wnr,plotW=False,AF_ch_index =[0,13],F_ch_index=[1,2,11,12],verbose=False):
    #ch_names = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
    #f1stLayer =['AF3','AF4']
    #f1stLyInx =[0,13]
    #f2stLyInx =[1,2,11,12]
    '''
    # Correlation Based Index
    '''
    CBI = np.sum(abs(Wnr[AF_ch_index,:]),axis=0)
    j = np.argmax(CBI)
    if plotW:
        #sns.heatmap(Wnr)
        PlotICACom_mne(abs(Wnr),title=np.around(CBI,2))
    if verbose:
        print('#IC ',j)
        print('1st  ',(Wnr[AF_ch_index,j]))
        print('2nd  ',(Wnr[F_ch_index,j]))
        print([x>y for x in abs(Wnr[AF_ch_index,j]) for y in abs(Wnr[F_ch_index,j])])
        print([x>y for x in Wnr[AF_ch_index,j] for y in Wnr[F_ch_index,j]])
    isArtifact = np.prod([x>y for x in abs(Wnr[AF_ch_index,j]) for y in abs(Wnr[F_ch_index,j])])
    #Artifact = np.prod([x>y for x in Wnr[f1stLyInx,j] for y in Wnr[f2stLyInx,j]])
    return CBI,j,isArtifact

def ICAremoveArtifact(x,ICA_method='extended-infomax',corr_thr=0.8,kur_thr=2.0,AF_ch_index =[0,13],F_ch_index=[1,2,11,12]):
    nch = x.shape[1]
    J =[]
    ica = ICA(n_components=nch,method=ICA_method)
    ica.fit(x.T)
    IC = ica.transform(x.T).T
    mu = ica.pca_mean_
    W = ica.get_sMatrix()
    #A = ica.get_tMatrix()
    sd = np.std(IC,axis=0)
    ICn = IC/sd
    Wn = W*sd
    Wnr = Wn/np.sqrt(np.sum(Wn**2,axis=1,keepdims=True))

    # Method (3)
    if corr_thr is not None:
        ICss,frqs = np.unique(np.argmax(Wnr,axis=1), return_counts=True)
        j1 = ICss[np.where(frqs/nch>=corr_thr)[0]]
        J.append(j1)
        ICss,frqs = np.unique(np.argmin(Wnr,axis=1), return_counts=True)
        j2 = ICss[np.where(frqs/nch>=corr_thr)[0]]
        J.append(j2)

    # Method (2)
    if len(AF_ch_index) and len(F_ch_index):
        CBI,j3,Fault = CBIeye(Wnr,plotW=False,AF_ch_index=AF_ch_index,F_ch_index=F_ch_index)
        if Fault:
            J.append(j3)

    # Method (1)
    kur   = kurtosis(ICn,axis=0)
    J.append(np.where(abs(kur)>=kur_thr)[0])
    J = list(set(np.hstack(J)))

    if len(J)>0:
        #print('------')
        for ji in J:
            W[:,ji]=0
        xr = np.dot(IC,W.T)+mu
    else:
        xr = x
    return xr

def PlotICACom_mne(W, title=None,ch_names=['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']):
    from mne.channels import read_montage
    from mne.viz.topomap import plot_topomap
    #ch_names = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
    montage = read_montage('standard_1020',ch_names)
    epos = montage.get_pos2d()
    ch = montage.ch_names
    eOrder = [ch_names.index(c) for c in ch]
    nch = len(ch_names)
    mask = np.ones(nch).astype(int)

    fig, ax = plt.subplots(2,nch//2,figsize=(15,5))
    i,j=0,0
    for k in range(14):
        #e=np.random.randn(14)
        e = W[:,k]
        plot_topomap(e[eOrder],epos,axes=ax[i,j],show=False,cmap='jet',mask=mask)
        for kk in range(len(eOrder)):
            ax[i,j].text(epos[kk,0]/3.99,epos[kk,1]/3,ch_names[eOrder[kk]],fontsize=6)
        if title is None:
            ax[i,j].set_title(str(k))
        else:
            ax[i,j].set_title(str(title[k]))
        j+=1
        if j==7:
            i+=1
            j=0
    #plt.axis('off')
    plt.subplots_adjust(hspace=0.0,wspace=0.0)
    plt.show()
