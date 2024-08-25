import os, scipy, copy, pickle
import numpy as np
import pandas as pd
#from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy import signal

import matplotlib
try:
    MATPLOTV = float(matplotlib.__version__[:3])
except:
    MATPLOTV = None


import warnings
warnings.filterwarnings('once')
from ..utils import deprecated
from ..utils import ProgBar_JL

def cart2sph(cart):
    r"""Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    cart_pts : ndarray, shape (n_points, 3)
        Array containing points in Cartesian coordinates (x, y, z)

    Returns
    -------
    sph_pts : ndarray, shape (n_points, 3)
        Array containing points in spherical coordinates (rad, azimuth, polar)
    """
    assert cart.ndim == 2 and cart.shape[1] == 3
    cart = np.atleast_2d(cart)
    sph = np.empty((len(cart), 3))
    sph[:, 0] = np.sqrt(np.sum(cart * cart, axis=1))
    sph[:, 1] = np.arctan2(cart[:, 1], cart[:, 0])
    sph[:, 2] = np.arccos(cart[:, 2] / sph[:, 0])
    sph = np.nan_to_num(sph)
    return sph

def sph2cart(sph):
    r"""Convert spherical coordinates to Cartesion coordinates.
    
    Parameters
    ----------
    sph_pts : ndarray, shape (n_points, 3)
        Array containing points in spherical coordinates (rad, azimuth, polar)


    Returns
    -------
    cart_pts : ndarray, shape (n_points, 3)
        Array containing points in Cartesian coordinates (x, y, z)

    """
    assert sph.ndim == 2 and sph.shape[1] == 3
    sph = np.atleast_2d(sph)
    cart = np.empty((len(sph), 3))
    cart[:, 2] = sph[:, 0] * np.cos(sph[:, 2])
    xy = sph[:, 0] * np.sin(sph[:, 2])
    cart[:, 0] = xy * np.cos(sph[:, 1])
    cart[:, 1] = xy * np.sin(sph[:, 1])
    return cart

def pol2cart(pol):
    r"""Transform polar coordinates to cartesian.
    
    
    Parameters
    ----------
    pol : ndarray, shape (n_points, 3)
        Array containing points in polar coordinates ()


    Returns
    -------
    cart_pts : ndarray, shape (n_points, 3)
        Array containing points in Cartesian coordinates (x, y, z)

    
    """
    cart = np.empty((len(pol), 2))
    if pol.shape[1] == 2:  # phi, theta
        cart[:, 0] = pol[:, 0] * np.cos(pol[:, 1])
        cart[:, 1] = pol[:, 0] * np.sin(pol[:, 1])
    else:  # radial distance, theta, phi
        d = pol[:, 0] * np.sin(pol[:, 2])
        cart[:, 0] = d * np.cos(pol[:, 1])
        cart[:, 1] = d * np.sin(pol[:, 1])
    return cart

def s1020_get_epos2d_(ch_names, reorder=False):
    r"""Get 2D projection points for given channel using 10-20 System

    """
    filen ='Standard_1020.csv'
    filen = os.path.join(os.path.dirname(__file__),'files', filen)
    #print(filen)
    D = pd.read_csv(filen)

    # Check if channel names provided are valid
    assert np.prod([ch_names[i] in list(D['channel']) for i in range(len(ch_names))])

    epos = np.array(D.iloc[:,1:])
    pos2d = pol2cart(cart2sph(epos)[:, 1:][:, ::-1])
    idx = [i for i in range(len(D)) if D['channel'][i] in ch_names]

    if reorder:
        idx = [i for i in range(len(D)) if D['channel'][i] in ch_names]
    else:
        idx = [np.where(ch_names[i]==D['channel'])[0][0] for i in range(len(ch_names))]

    ch  = list(D['channel'][idx])
    pos = pos2d[idx,:]

    return pos, ch

def s1020_get_epos2d(ch_names,style='eeglab-mne',case_sensitive=False,reorder=False,clean_label=True,use_precomputed=True):
    r"""Get 2D projection points for given channel using 10-20 System

    Parameters
    ----------
    ch_names: list of str
       -  list of channel names
       -  e.g. ch_names = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
    
        
        .. versionadded:: 0.0.9.7

        .. note:: SPKIT-Style
            
            A new style, closely matching to [3]_ from wikipedia reference [2]_
            See the example below.

    style: str {'eeglab-mne','spkit'}, default= 'eeglab-mne'
       - style of the topographic map
       - default is set to convention style  'eeglab-mne'
       - however, 'spkit' follows more close to [2]_ and [3]_

    case_sensitive: bool, default=False
       - ignoring the upper and lowercase name

    reorder: bool, default=False
      - if order of the position and channel is rearraged.
      
      .. deprecated:: 0.0.9.7
          It doesn't make sense to re-order the list.

    Returns
    -------
    pos: 2d-array, size (nc,2)
    ch: channel names 
      -  as it is in standard file.

    See Also
    --------
    s1005_get_epos2d, s1010_get_epos2d

    References
    ----------

    * .. [1] https://www.bem.fi/book/13/13.htm#03
    * .. [2] https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)
    * .. [3] Mariust Hart, http://www.mariusthart.net/downloads/eeg_electrodes_10-20.pdf, http://www.beteredingen.nl

    Examples
    --------
    #sp.eeg.s1020_get_epos2d
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp

    ch_names_emotiv = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
    pos1, ch1 = sp.eeg.s1020_get_epos2d(ch_names_emotiv)
    pos2, ch2 = sp.eeg.s1020_get_epos2d(ch_names_emotiv,style='spkit')


    ch_names_all = sp.eeg.presets.standard_1020_ch
    pos3, ch3 = sp.eeg.s1020_get_epos2d(ch_names_all)

    ch_names_spkit_all = sp.eeg.presets.standard_1020_spkit_ch
    pos4, ch4 = sp.eeg.s1020_get_epos2d(ch_names_spkit_all,style='spkit')

    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.plot(pos1[:,0],pos1[:,1],'.',alpha=0.5)
    for i,ch in enumerate(ch_names_emotiv):
        plt.text(pos1[i,0],pos1[i,1],ch,va='center',ha='center')
    plt.title('Emotiv: 14 channels: mne-eeglab-style')
    plt.axvline(0,lw=0.5,color='k',ls='--')
    plt.axhline(0,lw=0.5,color='k',ls='--')
    plt.subplot(122)
    plt.plot(pos3[:,0],pos3[:,1],'.',alpha=0.5)
    for i,ch in enumerate(ch_names_all):
        plt.text(pos3[i,0],pos3[i,1],ch,va='center',ha='center')
    plt.title(f'All 10-20 Channels: (n={len(ch_names_all)})')
    plt.axvline(0,lw=0.5,color='k',ls='--')
    plt.axhline(0,lw=0.5,color='k',ls='--')
    plt.suptitle('MNE-EEGLab style topographic map')
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.plot(pos2[:,0],pos2[:,1],'.',alpha=0.5)
    for i,ch in enumerate(ch_names_emotiv):
        plt.text(pos2[i,0],pos2[i,1],ch,va='center',ha='center')
    plt.title('Emotiv: 14 channels')
    plt.axvline(0,lw=0.5,color='k',ls='--')
    plt.axhline(0,lw=0.5,color='k',ls='--')
    plt.subplot(122)
    plt.plot(pos4[:,0],pos4[:,1],'.',alpha=0.5)
    for i,ch in enumerate(ch_names_spkit_all):
        plt.text(pos4[i,0],pos4[i,1],ch,va='center',ha='center')
    plt.title(f'All 10-20 Channels: (n={len(ch_names_spkit_all)})')
    plt.suptitle('SPKIT-style topographic map')
    plt.axvline(0,lw=0.5,color='k',ls='--')
    plt.axhline(0,lw=0.5,color='k',ls='--')
    plt.tight_layout()
    plt.show()
    """
    ch_names = ch_names
    if clean_label:
        ch_names = [ch.replace('.','').upper()for ch in ch_names]
    

    if use_precomputed:
        filen ='precomputed_projections.pkl'
        filen = os.path.join(os.path.dirname(__file__),'files', filen)
        with open(filen, "rb") as f:
            data = pickle.load(f)
        if style=='spkit':
            data = data['Standard_1020_spkit']
            channels_1020 = data['ch']
            pos2d = data['2d']
        else:
            data = data['Standard_1020']
            channels_1020 = data['ch']
            pos2d = data['2d']
    else:
        if style=='spkit':
            filen ='Standard_1020_spkit.csv'
            filen = os.path.join(os.path.dirname(__file__),'files', filen)
            D = pd.read_csv(filen)
            channels_1020 = list(D['channel'])
            pos2d = np.array(D[['x','y']])

        else:
            filen ='Standard_1020.csv'
            filen = os.path.join(os.path.dirname(__file__),'files', filen)
            D = pd.read_csv(filen)
            channels_1020 = list(D['channel'])
            #epos = np.array(D.iloc[:,1:])
            epos = np.array(D[['x','y','z']])
            pos2d = pol2cart(cart2sph(epos)[:, 1:][:, ::-1])
    
    
    channels_1020_lwr = [ch.lower() for ch in channels_1020]
    ch_names_lwr = [ch.lower() for ch in ch_names]

    # Check if channel names provided are valid
    if case_sensitive:
        assert np.prod([ch_names[i] in channels_1020 for i in range(len(ch_names))])
    else: # Case-insensitive
        assert np.prod([ch_names_lwr[i] in channels_1020_lwr for i in range(len(ch_names_lwr))])

    #idx = [i for i in range(len(D)) if D['channel'][i] in ch_names]

    # if reorder:
    #     idx = [i for i in range(len(D)) if D['channel'][i] in ch_names]
    # else:
    #     idx = [np.where(ch_names[i]==D['channel'])[0][0] for i in range(len(ch_names))]

    if case_sensitive:
        idx = [channels_1020.index(ch) for ch in ch_names]
    else:
        idx = [channels_1020_lwr.index(ch) for ch in ch_names_lwr]
        
    #ch  = list(D['channel'][idx])

    ch = list(np.array(channels_1020)[idx])
    pos = pos2d[idx,:]

    return pos, ch

def s1010_get_epos2d(ch_names, case_sensitive=False,clean_label=True,reorder=False,use_precomputed=True):
    r"""Get 2D projection points for given channel using 10-10 System

    Parameters
    ----------
    ch_names: list of str
       -  list of channel names
       -  e.g. ch_names = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
    
    case_sensitive: bool, default=False
       - if False, ignoring the upper and lowercase name

    clean_label: bool, default=True
       - clean the channel names
    
    reorder: bool, default=False
      - if order of the position and channel is rearraged.
      
      .. deprecated:: 0.0.9.7
          It doesn't make sense to re-order the list.

    Returns
    -------
    pos: 2d-array, size (nc,2)
    ch: channel names
      -  as it is in standard file.

    See Also
    --------
    s1005_get_epos2d, s1020_get_epos2d

    Examples
    --------
    #sp.eeg.s1010_get_epos2d
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp

    ch_names = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
    pos, ch = sp.eeg.s1010_get_epos2d(ch_names)

    plt.figure(figsize=(12,5))

    plt.subplot(121)
    plt.plot(pos[:,0],pos[:,1],'.',alpha=0.5)
    for i,ch in enumerate(ch_names):
        plt.text(pos[i,0],pos[i,1],ch,va='center',ha='center')
    plt.title('Emotiv: 14 channels')

    ch_names = sp.eeg.presets.standard_1010_ch
    pos, ch = sp.eeg.s1010_get_epos2d(ch_names)

    plt.subplot(122)
    plt.plot(pos[:,0],pos[:,1],'.',alpha=0.5)
    for i,ch in enumerate(ch_names):
        plt.text(pos[i,0],pos[i,1],ch,va='center',ha='center')
    plt.title(f'All 10-10 Channels: (n={len(ch_names)})')
    plt.show()
    """
    ch_names_copy = ch_names
    if clean_label:
        ch_names_copy = [ch.replace('.','').upper().replace('Z','z').replace('FP','Fp') for ch in ch_names_copy]
    
    if use_precomputed:
        filen ='precomputed_projections.pkl'
        filen = os.path.join(os.path.dirname(__file__),'files', filen)
        with open(filen, "rb") as f:
            data = pickle.load(f)
        
        data = data['Standard_1010']
        channels_1010 = data['ch']
        pos2d = data['2d']
    else:
        filen ='Standard_1010_MI.csv'
        filen = os.path.join(os.path.dirname(__file__),'files', filen)
        D = pd.read_csv(filen)
        channels_1010 = list(D['channel'])
        #epos = np.array(D.iloc[:,1:])
        epos = np.array(D[['Theta','Phi']])*(np.pi/180)
        pos2d = pol2cart(epos)



    channels_1010_lwr = [ch.lower() for ch in channels_1010]
    ch_names_lwr = [ch.lower() for ch in ch_names_copy]

    # Check if channel names provided are valid
        # Check if channel names provided are valid
    if case_sensitive:
        assert np.prod([ch_names_copy[i] in channels_1010 for i in range(len(ch_names_copy))])
    else: # Case-insensitive
        assert np.prod([ch_names_lwr[i] in channels_1010_lwr for i in range(len(ch_names_lwr))])

    #epos = np.array(D.iloc[:,1:])
    #epos = np.array(D.iloc[:,1:])*(np.pi/180)
    #epos = np.array(D[['Theta','Phi']])*(np.pi/180)
    #pos2d = pol2cart(epos)

    #idx = [i for i in range(len(D)) if D['channel'][i] in ch_labels1]
    # if reorder:
    #     idx = [i for i in range(len(D)) if D['channel'][i] in ch_names]
    # else:
    #     idx = [np.where(ch_names[i]==D['channel'])[0][0] for i in range(len(ch_names))]

    if case_sensitive:
        idx = [channels_1010.index(ch) for ch in ch_names_copy]
    else:
        idx = [channels_1010_lwr.index(ch) for ch in ch_names_lwr]

    #ch  = list(D['channel'][idx])
    ch = list(np.array(channels_1010)[idx])
    pos = pos2d[idx,:]
    return pos, ch

def s1005_get_epos2d(ch_names, case_sensitive=False,reorder=False,clean_label=True,use_precomputed=True):
    r"""Get 2D projection points for given channel using 10-05 System
    
    Parameters
    ----------
    ch_names: list of str
       -  list of channel names
       -  e.g. ch_names = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
    
    case_sensitive: bool, default=False
       - if False, ignoring the upper and lowercase name

    clean_label: bool, default=True
       - clean the channel names
    
    reorder: bool, default=False
      - if order of the position and channel is rearraged.
      
      .. deprecated:: 0.0.9.7
          It doesn't make sense to re-order the list.

    Returns
    -------
    pos: 2d-array, size (nc,2)
    ch: channel names
      -  as it is in standard file.

    See Also
    --------
    s1010_get_epos2d, s1020_get_epos2d

    Examples
    --------
    #sp.eeg.s1005_get_epos2d
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp

    ch_names = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
    pos, ch = sp.eeg.s1005_get_epos2d(ch_names)

    plt.figure(figsize=(12,5))

    plt.subplot(121)
    plt.plot(pos[:,0],pos[:,1],'.',alpha=0.5)
    for i,ch in enumerate(ch_names):
        plt.text(pos[i,0],pos[i,1],ch,va='center',ha='center')
    plt.title('Emotiv: 14 channels')

    ch_names = sp.eeg.presets.standard_1005_ch
    pos, ch = sp.eeg.s1005_get_epos2d(ch_names)

    plt.subplot(122)
    plt.plot(pos[:,0],pos[:,1],'.',alpha=0.5)
    for i,ch in enumerate(ch_names):
        plt.text(pos[i,0],pos[i,1],ch,va='center',ha='center')
    plt.title(f'All 10-05 Channels: (n{len(ch_names)})')
    plt.show()
    """
    ch_names_copy = ch_names
    if clean_label:
        ch_names_copy = [ch.replace('.','').upper().replace('Z','z').replace('FP','Fp') for ch in ch_names]

    if use_precomputed:
        filen ='precomputed_projections.pkl'
        filen = os.path.join(os.path.dirname(__file__),'files', filen)
        with open(filen, "rb") as f:
            data = pickle.load(f)
        
        data = data['Standard_1005']
        channels_1005 = data['ch']
        pos2d = data['2d']
    else:
        filen ='Standard_1005.csv'
        filen = os.path.join(os.path.dirname(__file__), 'files',filen)
        D = pd.read_csv(filen)
        channels_1005 = list(D['channel'])
        #epos = np.array(D.iloc[:,1:])
        epos = np.array(D[['x','y','z']])
        pos2d = pol2cart(cart2sph(epos)[:, 1:][:, ::-1])
    

    channels_1005_lwr = [ch.lower() for ch in channels_1005]
    ch_names_lwr = [ch.lower() for ch in ch_names_copy]

    # Check if channel names provided are valid
        # Check if channel names provided are valid
    if case_sensitive:
        assert np.prod([ch_names_copy[i] in channels_1005 for i in range(len(ch_names_copy))])
    else: # Case-insensitive
        assert np.prod([ch_names_lwr[i] in channels_1005_lwr for i in range(len(ch_names_lwr))])

    #epos = np.array(D.iloc[:,1:])
    #epos = np.array(D.iloc[:,1:])*(np.pi/180)
    #pos2d = pol2cart(epos)
    #epos = np.array(D[['x','y','z']])
    #pos2d = pol2cart(cart2sph(epos)[:, 1:][:, ::-1])

    #idx = [i for i in range(len(D)) if D['channel'][i] in ch_labels1]
    # if reorder:
    #     idx = [i for i in range(len(D)) if D['channel'][i] in ch_names]
    # else:
    #     idx = [np.where(ch_names[i]==D['channel'])[0][0] for i in range(len(ch_names))]

    if case_sensitive:
        idx = [channels_1005.index(ch) for ch in ch_names_copy]
    else:
        idx = [channels_1005_lwr.index(ch) for ch in ch_names_lwr]

    #ch  = list(D['channel'][idx])
    ch = list(np.array(channels_1005)[idx])
    pos = pos2d[idx,:]

    return pos, ch


ch_names = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
pos, ch1 = s1020_get_epos2d(ch_names, reorder=False)

filen ='precomputed_projections.pkl'
filen = os.path.join(os.path.dirname(__file__),'files', filen)
with open(filen, "rb") as f:
    data = pickle.load(f)

class presets:
    emotiv_14ch_names = ('AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4')
    standard_1020_ch = tuple(copy.deepcopy(data['Standard_1020']['ch']))
    standard_1010_ch = tuple(copy.deepcopy(data['Standard_1010']['ch']))
    standard_1005_ch = tuple(copy.deepcopy(data['Standard_1005']['ch']))
    standard_1020_spkit_ch = tuple(copy.deepcopy(data['Standard_1020_spkit']['ch']))


class topo_grid(object):
    def __init__(self,pos=None,res=64,xlim=[None,None],ylim=[None,None]):
        self.res = res
        self.pos = pos
    
    def set_grid(self):
        xmin, xmax,ymin, ymax = -0.5, 0.5, -0.5, 0.5
        xi = np.linspace(xmin, xmax, res)
        yi = np.linspace(ymin, ymax, res)
        self.Xi, self.Yi = np.meshgrid(xi, yi)
        return self



class GridInter(object):
    r"""
    Grid Interpolator for 2d EEG electrodes
    """
    def __init__(self,pos,res=64):
        try:
            from scipy.spatial.qhull import Delaunay
        except:
            from scipy.spatial import Delaunay
        import itertools

        extremes = np.array([pos.min(axis=0), pos.max(axis=0)])
        diffs = extremes[1] - extremes[0]
        extremes[0] -= diffs
        extremes[1] += diffs

        eidx = np.array(list(itertools.product(*([[0] * (pos.shape[1] - 1) + [1]] * pos.shape[1]))))
        pidx = np.tile(np.arange(pos.shape[1])[np.newaxis], (len(eidx), 1))
        self.n_extra = pidx.shape[0]
        outer_pts = extremes[eidx, pidx]
        pos = np.concatenate((pos, outer_pts))
        self.tri = Delaunay(pos)

        self.set_Grid()

    def set_Grid(self, res=64):
        xmin, xmax,ymin, ymax = -0.5, 0.5, -0.5, 0.5
        xi = np.linspace(xmin, xmax, res)
        yi = np.linspace(ymin, ymax, res)
        self.Xi, self.Yi = np.meshgrid(xi, yi)
        return self

    def get_Grid(self,data):
        from scipy.interpolate import CloughTocher2DInterpolator
        idata = np.concatenate((data, np.zeros(self.n_extra)))
        Interpolator = CloughTocher2DInterpolator(self.tri, idata)
        args = [self.Xi, self.Yi]
        Zi = Interpolator(*args)
        return Zi

def GridInterpolation(pos, data, Xi,Yi):
    r"""Grid Interpolation
    
    
    """
    #try:
    #    from scipy.spatial.qhull import Delaunay
    #except:
    from scipy.spatial import Delaunay
    from scipy.interpolate import CloughTocher2DInterpolator
    import itertools

    extremes = np.array([pos.min(axis=0), pos.max(axis=0)])
    diffs = extremes[1] - extremes[0]
    extremes[0] -= diffs
    extremes[1] += diffs

    eidx = np.array(list(itertools.product(*([[0] * (pos.shape[1] - 1) + [1]] * pos.shape[1]))))
    pidx = np.tile(np.arange(pos.shape[1])[np.newaxis], (len(eidx), 1))
    n_extra = pidx.shape[0]
    outer_pts = extremes[eidx, pidx]
    pos = np.concatenate((pos, outer_pts))
    tri = Delaunay(pos)

    data = np.concatenate((data, np.zeros(n_extra)))
    Interpolator = CloughTocher2DInterpolator(tri, data)
    args = [Xi, Yi]
    Zi = Interpolator(*args)
    return Zi


#--------NEW TO BE Improved---------------
def _topo_setup(pos):
    return 

def _topopap_zi(data,pos=None,Zi=None,ch_names=None, style='spkit',system='1020',
                res=64,show=True,axes=None,returnIm = False,**kwargs):
    r"""Display Topographical Map of EEG, with given values
    """

    # At least pos is or ch_names should be provided to make a topograph
    assert (pos is not None) or (ch_names is not None)

    if pos is None:
        if system=='1020':
            pos,ch = s1020_get_epos2d(ch_names,style=style)
        elif system=='1010':
            pos,ch = s1010_get_epos2d(ch_names,style=style)
        elif system=='1005':
            pos,ch = s1005_get_epos2d(ch_names,style=style)

    # Deafult Settings
    props = dict(vmin=None,vmax=None,contours=True,interpolation=None,
            showsensors=True, shownames=True,showhead=True, showbound=True,
            shift_origin=False,show_vhlines=True,match_shed=True,fontdict=None,
            f1=0.5,f2=0.85,fx=0.5,fy=0.5,r=1,
            bound_prop=dict(r1=0.85,r2=0.85,xy=(0,0),color='k',alpha=0.5),
            head_prop =dict(),
            sensor_prop=dict(),
            font_prop=dict())

    for key in kwargs:
        props[key] = kwargs[key]
    
    ipos  = pos.copy()
    idata = data.copy()

    if style=='spkit':
        xmin, xmax,ymin, ymax = -0.5, 0.5, -0.5, 0.5
    else:
        if shift_origin:
            ipos -= f1 * (ipos.max(axis=0) + ipos.min(axis=0))
        
        ipos *= f2 / (ipos.max(axis=0) - ipos.min(axis=0))
        xmin, xmax,ymin, ymax = -props['fx'], props['fx'], -props['fy'], props['fy']



    if Zi is None:
        xi = np.linspace(xmin, xmax, res)
        yi = np.linspace(ymin, ymax, res)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = GridInterpolation(ipos, idata, Xi,Yi)
    else:
        res = Zi.shape[0]
        xi = np.linspace(xmin, xmax, res)
        yi = np.linspace(ymin, ymax, res)
        Xi, Yi = np.meshgrid(xi, yi)

    if showplot:
        ax = axes if axes else plt.gca()
        if match_shed:
            patch = patches.Ellipse(bound_prop['xy'],r,r,clip_on=True,transform=ax.transData)
        else:
            patch = patches.Ellipse((0, 0),1,1,clip_on=True,transform=ax.transData)
        #im = ax.imshow(Zi, cmap='jet', origin='lower',aspect='equal', extent=(xmin, xmax, ymin, ymax),
        #              interpolation=interpolation)
        im = ax.imshow(Zi, cmap='jet', origin='lower',aspect='equal', extent=(xmin, xmax, ymin, ymax),
                  interpolation=interpolation,vmin=vmin, vmax=vmax)
        im.set_clip_path(patch)
        ax.axis('off')
        if contours:
            contrs = ax.contour(Xi, Yi, Zi, 6, colors='k',linewidths=0.5)
            try:
                # matplotlib version 3.8 plans to remove .collections
                contrs.set_clip_path(patch)
            except:
                # old version of matplotlib works this way
                for col in contrs.collections:
                    col.set_clip_path(patch)
        if showsensors:
            ax.plot(ipos[:,0],ipos[:,1],'.k',markersize=2)
            if len(sensorprop):
                ax.scatter(ipos[:,0],ipos[:,1],**sensorprop)


        if shownames and ch_names is not None:
            for i in range(len(ipos)):
                ax.text(ipos[i,0],ipos[i,1],ch_names[i],horizontalalignment='center',
                    verticalalignment='center',fontdict=fontdict,**fontkwds)
        if showhead:
            #ax.plot(0,0.58,'^k',ms=25,  markerfacecolor="None")
            #ax.plot(0,0.54,'2k',ms=30)
            #ax.plot(0,0.54,'2k',ms=30)
            #ax.plot(0,0.54,'2k',ms=30)
            ax.plot(0,0.54,'2k',**head_prop)

        if showbound:
            xy,r1,r2 = [bound_prop[key] for key in ['xy','r1','r2']]
            patch = patches.Ellipse(xy,r1,r2, color=bound_prop['color'], fill=False,alpha=bound_prop['alpha'])
            ax.add_artist(patch)

        if show_vhlines:
            ax.vlines([0],-0.5,0.5,color='k',ls='--',lw=0.5)
            ax.hlines([0],-0.5,0.5,color='k',ls='--',lw=0.5)

        if returnIm: return Zi, im
    return Zi

def _topomap(pos,data,res=64,showplot=False,axes=None,contours=True,showsensors=True,
            interpolation=None,shownames=True, ch_names=None,showhead=True,vmin=None,vmax=None,
            returnIm = False,fontdict=None,**fontkeywords):


    return 



def topomap(data,pos=None,ch_names=None,res=64,Zi=None,show=True,return_im=False,
                    standard='1020',
                    style='spkit',
                    **kwargs):
    r"""Display Topographical Map of EEG, with given values


    Parameters
    ----------
    data: 1d-array
       -  1d-array of values for the each electrode.
    
    pos: 2d-array, deafult=None
       -  2D projection points for each electroed
       - computed using  one of either :func:`s1020_get_epos2d`, :func:`s1010_get_epos2d`, :func:`s1005_get_epos2d`
           or provided by custom system
       -  MUST BE same number of points as data points, e.g. pos.shape[0]==data.shape[0]
       - IF `None`, then computed using `ch_names` `system` `style` provided.

    ch_names: list of str, default=None
       - name of the channels/electrodes
       - should be same size as data.shape[0]
       - IF `pos` is None, this is used to compute projection points (pos), in that case, make sure to provide
         valid channel names and respective system. or compute using :func:`s1020_get_epos2d`, :func:`s1010_get_epos2d`, :func:`s1005_get_epos2d`
       - This list is also used to shownames of sensors, if `shownames` is True.
       
       .. note:: IMPORTANT
            Either `pos` or valid  `ch_names` should be provided

    res: int, default=64
       - resolution of image to be generated, (res,res)

    Zi: 2d-array, default None,
       - Pre-computed Image of topographic map. 
       - if provided, then `data` is ignored, else image is generated using data-points
       
    show: bool, defult=True
      - if False, then plot is not produced
      - useful, when interested in only generating Zi images

      .. versionchanged 0.0.9.7::
          argument name `showplot` is changed to `show`

    return_im: bool, default=False
      - if True, in addtion to Zi (generate Image), `im` object is also returned
    
      .. versionchanged 0.0.9.7::
          argument name `returnIm` is changed to `return_im`


    standard: str, default ='1020'
      - it is used ONLY if `pos` is None
      - standard to extract pos from ch_names
      - check :func:`s1020_get_epos2d`, :func:`s1010_get_epos2d`, :func:`s1005_get_epos2d`

    style: str, default = 'spkit'
      - it is used ONLY if `pos` is None and standard is '1020'
      - style to extract pos from ch_names
      - check :func:`s1020_get_epos2d`
    




    kwargs:
        
        There are other arguments which can be supplied to modify the topomap disply
        
        * warn: bool, default True
            -  set it to False, to supress the warnings of change names
        * c:scalar,default=0.5
             - shifting origin, only used if `shift_origin` is True
        * s:scalar,default=0.85
             - scale the electrod postions by the factor `s`, higher it is more spreaded they are
        * (fx, fy):scalar,default=(0.5, 0.5)
             - x-lim, y-lim of grid,
        * (rx,ry):scalar,default=(1,1)
             - radius of the ellips to clip, higher it is more outter area is covered
        * shift_origin=False
           - If True, center point of electrode poisition will be the origin = (0,0)
           - it uses `c` parameter to shift the origin
           - pos -=  c*([dx, dy]) 
           - if `c=1`, new origin will be at 0,0

        * axes=None: Axes to use for plotting, if None, then created using plt.gca()
        * For Image display, plt.imshow kwargs:
             -   (cmap='jet',origin='lower',aspect='equal',vmin=None,vmax=None,interpolation=None)
        * contours=True: contour lines
             - if True, `contours_n` contours are shown with linewidth of `contours_lw`
        * showsensors=True: sensor location
             - display sensors/eletrodes as dots
             - add cicles of sensors using `sensor_prop`
        * shownames=True
             - if True, show them, using `fontdict` and `font_prop`
             - default fontdict=None, font_prop=dict()
        * showhead=True
             - If True, show head, using `head_prop` (default head_prop =dict(markersize=30))
             - it is passed to ax.plot as kwargs
        * showbound=True:  show boundary
             - uses bound_prop=dict(rx=0.85,ry=0.85,xy=(0,0),color='k',alpha=0.5)
             - could be used to show oval shape head, with rx,ry = 0.85,0.9
        * show_vhlines=True: lines
             - show verticle and horizontal lines passing through origin
        * show_colorbar=False, colorbar
             - if True, show colorbar with label as `colorbar_label`
        * match_shed=True: boundary
             - if True, center of boundary is same as center of grid.

    Returns
    -------
    Zi: 2d-array
      -  2D full Grid as image, without circular crop.
      - Obatained from Inter/exterpolation 

    im: image object
      - returned from im = ax.imshow()
      - only of `return_im` is True
    

    See Also
    --------
    Gen_SSFI


    References
    ----------

    Notes
    -----



    Examples
    --------
    #sp.eeg.topomap
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    X,fs, ch_names = sp.data.eeg_sample_14ch()
    X = sp.filterDC_sGolay(X, window_length=fs//3+1)
    Px,Pm,Pd = sp.eeg.rhythmic_powers(X=X,fs=fs,fBands=[[4],[8,14]],Sum=True,Mean=True,SD =True)
    Px = 20*np.log10(Px)
    pos1, ch = sp.eeg.s1020_get_epos2d(ch_names,style='eeglab-mne')
    pos2, ch = sp.eeg.s1020_get_epos2d(ch_names,style='spkit')
    fig, ax = plt.subplots(1,2,figsize=(10,4))
    Z1i,im1 = sp.eeg.topomap(pos=pos1,data=Px[0],axes=ax[0],return_im=True)
    Z2i,im2 = sp.eeg.topomap(pos=pos1,data=Px[1],axes=ax[1],return_im=True)
    ax[0].set_title(r'$\delta$ : [0.5-4] Hz')
    ax[1].set_title(r'$\alpha$ : [8-14] Hz')
    plt.colorbar(im1, ax=ax[0],label='dB')
    plt.colorbar(im2, ax=ax[1],label='dB')
    plt.suptitle('Topographical map')
    plt.show()

    fig, ax = plt.subplots(1,2,figsize=(10,4))
    Z1i,im1 = sp.eeg.topomap(pos=pos2,data=Px[0],axes=ax[0],return_im=True)
    Z2i,im2 = sp.eeg.topomap(pos=pos2,data=Px[1],axes=ax[1],return_im=True)
    ax[0].set_title(r'$\delta$ : [0.5-4] Hz')
    ax[1].set_title(r'$\alpha$ : [8-14] Hz')
    plt.colorbar(im1, ax=ax[0],label='dB')
    plt.colorbar(im2, ax=ax[1],label='dB')
    plt.suptitle('Topographical map : SPKIT-style')
    plt.show()
    """
    #showplot=False,axes=None,contours=True,showsensors=True,
    #interpolation=None,shownames=True, ch_names=None,showhead=True,showbound=True,vmin=None,vmax=None,
    #returnIm = False,fontdict=None,f1=0.5,f2=0.85,fx=0.5,fy=0.5,
    #bound_prop=dict(rx=0.85,ry=0.85,xy=(0,0),color='k',alpha=0.5),match_shed=True,r=1,shift_origin=False,
    #head_prop =dict(ms=60,markeredgewidth=3),show_vhlines=True,sensorprop=dict(),
    #fontkwds={}
    
    # At least pos is or ch_names should be provided to make a topograph
    assert (pos is not None) or (ch_names is not None)

    if pos is None:
        if standard=='1020':
            pos,ch = s1020_get_epos2d(ch_names,style=style)
        elif standard=='1010':
            pos,ch = s1010_get_epos2d(ch_names)
        elif standard=='1005':
            pos,ch = s1005_get_epos2d(ch_names)

    #MUST BE same number of points as data points
    assert pos.shape[0]==data.shape[0]

    warn_str = ' This will be removed in future versions. To turn of this warning, set `warn=False`. [0.0.9.7]'
    
    WARN = True
    if 'warn' in kwargs:
        WARN = kwargs['warn']

    if 'showplot' in kwargs:
        show = kwargs['showplot']
        if WARN:
            warnings.warn('Argument `showplot` is changed to `show`' + warn_str )

    if 'returnIm' in kwargs:
        return_im = kwargs['returnIm']
        if WARN:
            warnings.warn('Argument `returnIm` is changed to `return_im`.' + warn_str)
            
    if 'sensorprop' in kwargs:
        kwargs['sensor_prop'] = kwargs['sensorprop']
        if WARN:
            warnings.warn('Argument `sensorprop` is changed to `sensor_prop`' + warn_str )
    if 'fontkwds' in kwargs:
        kwargs['font_prop'] = kwargs['fontkwds']
        if WARN:
            warnings.warn('Argument `fontkwds` is changed to `font_prop`' + warn_str )

    # Deafult Settings
    props = dict(axes=None,cmap='jet', origin='lower',aspect='equal',vmin=None,vmax=None,contours=True,interpolation=None,
            showsensors=True, shownames=True,showhead=True, showbound=True,show_colorbar=False,colorbar_label=None,
            shift_origin=False,show_vhlines=True,match_shed=True,fontdict=None,
            c=0.5,s=0.85,fx=0.5,fy=0.5,rx=1,ry=1,contours_n=6, contours_lw=0.5,
            bound_prop=dict(rx=0.85,ry=0.85,xy=(0,0),color='k',alpha=0.5),
            head_prop =dict(markersize=30),
            sensor_prop=dict(),
            font_prop=dict())

    # Updating Settings
    if 'bound_prop' in kwargs:
        for key in props['bound_prop']:
            if key not in kwargs['bound_prop']:
                kwargs['bound_prop'][key] = props['bound_prop'][key]

    if 'head_prop' in kwargs:
        for key in props['head_prop']:
            if key not in kwargs['head_prop']:
                kwargs['head_prop'][key] = props['head_prop'][key]

    for key in kwargs:
        props[key] = kwargs[key]
    


    #bound_default=dict(r1=0.85,r2=0.85,xy=(0,0),color='k',alpha=0.5)
    #for key in bound_default:
    #    if key not in bound_prop:
    #        bound_prop[key] = bound_default[key]

    ipos  = pos.copy()
    idata = data.copy()
    if props['shift_origin']:
        ipos -= props['c'] * (ipos.max(axis=0) + ipos.min(axis=0))
    
    ipos *= props['s'] / (ipos.max(axis=0) - ipos.min(axis=0))

    xmin, xmax,ymin, ymax = -props['fx'], props['fx'], -props['fy'], props['fy']


    if Zi is None:
        xi = np.linspace(xmin, xmax, res)
        yi = np.linspace(ymin, ymax, res)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = GridInterpolation(ipos, idata, Xi,Yi)
    else:
        res = Zi.shape[0]
        xi = np.linspace(xmin, xmax, res)
        yi = np.linspace(ymin, ymax, res)
        Xi, Yi = np.meshgrid(xi, yi)

    if show:
        bound_prop = props['bound_prop']
        axes = props['axes']
        ax = axes if axes else plt.gca()
        rx,ry = props['rx'],props['ry']
        if props['match_shed']:
            patch = patches.Ellipse(bound_prop['xy'],rx,ry,clip_on=True,transform=ax.transData)
        else:
            patch = patches.Ellipse((0, 0),rx,ry,clip_on=True,transform=ax.transData)
            
        #im = ax.imshow(Zi, cmap='jet', origin='lower',aspect='equal', extent=(xmin, xmax, ymin, ymax),
        #              interpolation=interpolation)


        # deafult: cmap='jet', origin='lower',aspect='equal',
        imshow_kwars = dict(cmap=props['cmap'], origin=props['origin'],aspect=props['aspect'],
                        interpolation=props['interpolation'],vmin=props['vmin'], vmax=props['vmax'])

        im = ax.imshow(Zi,extent=(xmin, xmax, ymin, ymax),**imshow_kwars)

        im.set_clip_path(patch)
        ax.axis('off')

        if props['contours']:
            contrs = ax.contour(Xi, Yi, Zi, props['contours_n'], colors='k',linewidths=props['contours_lw'])
            #for col in contrs.collections:
            #    col.set_clip_path(patch)
            try:
                # matplotlib version 3.8 plans to remove .collections
                contrs.set_clip_path(patch)
            except:
                # old version of matplotlib works this way
                for col in contrs.collections:
                    col.set_clip_path(patch)
                
        if props['showsensors']:
            ax.plot(ipos[:,0],ipos[:,1],'.k',markersize=2)
            sensor_prop = props['sensor_prop']
            if len(sensor_prop):
                ax.scatter(ipos[:,0],ipos[:,1],**sensor_prop)

        if props['shownames'] and ch_names is not None:
            for i in range(len(ipos)):
                ax.text(ipos[i,0],ipos[i,1],ch_names[i],ha='center', va='center',fontdict=fontdict,**fontkwds)

        if props['showhead']:
            #ax.plot(0,0.58,'^k',ms=25,  markerfacecolor="None")
            #ax.plot(0,0.54,'2k',ms=30)
            #ax.plot(0,0.54,'2k',ms=30)
            #ax.plot(0,0.54,'2k',ms=30)
            ax.plot(0,0.54,'2k',**props['head_prop'])

        if props['showbound']:
            xy,rx,ry = [bound_prop[key] for key in ['xy','rx','ry']]
            patch = patches.Ellipse(xy,rx,ry, color=bound_prop['color'], fill=False,alpha=bound_prop['alpha'])
            ax.add_artist(patch)

        if props['show_vhlines']:
            ax.vlines([0],-0.5,0.5,color='k',ls='--',lw=0.5)
            ax.hlines([0],-0.5,0.5,color='k',ls='--',lw=0.5)
        if props['show_colorbar']:
            plt.colorbar(im, ax=ax,label=props['colorbar_label'])

        if return_im: return Zi, im
    return Zi


def gen_ssfi(PX,pos,res=64,NormalizeEachBand=False,verbose=1):
    r"""Generating Spatio-Spectral Feature Image (SSFI)

    Spatio-Spectral Feature Images


    * Input   - PX   : (100,6,14) or (100,6*14)
    * Output  - Ximg : (100,6,64,64)


    Parameters
    ----------
    PX: 2D-array or 3D-array,
      -  Power values of the each band and each channel
      if 2D-array
        -  shape of (nt , N*M)
        -  where nt=number of segments, N=number of frequency bands, M=number of channels
        -  e.g:
                for 5 chennels, three bands, power values of each band is in sequence
                - PX[0] = [1,2,3,4,5, 1,2,3,4,5, 6,1,2,3,4,5]
                - PX.shape = (100, 3*5), 100 time instant, 3 bands and 5 channels
      if 3D-array
        - shape of (nt, N, M)
        - where nt=number of segments, N=number of frequency bands, M=number of channels
        -  e.g:
                for 5 chennels, three bands, power values of each band is in sequence
                - PX.shape = (100, 3, 14), 100 time instant, 3 bands and 14 channels


    pos: 2D projections
      - shape = (M,2)
      - number of channel M same in order as values

    res: int default=64
      -  resolution of SSFI

    NormalizeEachBand: bool, Default=False
      - If True, power values of each band are normalised
      - it is useful, if distribution if more improtant than power values
    
    Returns
    -------

    Ximg: 3D-Array
      - (nt,N, res, res)



    References
    ----------
    [1] Bajaj N, Requena Carrión J., (2023, August). Deep Representation of EEG Signals Using Spatio-Spectral Feature Images. 
    Applied Sciences. 2023; 13(17):9825. https://doi.org/10.3390/app13179825.  [Link]


    Examples
    --------
    #sp.eeg.gen_ssfi
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    X,fs, ch_names = sp.data.eeg_sample_14ch()
    X = sp.filterDC_sGolay(X, window_length=fs//3+1)
    t = np.arange(X.shape[0])/fs
    Pxt, _, _ = sp.eeg.rhythmic_powers_win(X, winsize=128,overlap=32,fBands=[[4],[8,14],[32,47]],Sum=True)
    Pxt = np.log10(Pxt)
    pos, ch = sp.eeg.s1020_get_epos2d(ch_names,style='spkit')
    Ximg = sp.eeg.gen_ssfi(Pxt,pos=pos, res=64)

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

    plt.figure(figsize=(10,3.5))
    for i in range(10):
        plt.subplot(3,10,i+1)
        plt.imshow(Ximg[i,0],cmap='jet')
        plt.title(f'n={i}')
        if i==0:
            plt.ylabel(r'$\delta$')
            plt.xticks([])
            plt.yticks([])
        else:
            plt.axis('off')
        plt.subplot(3,10,10+i+1)
        plt.imshow(Ximg[i,1],cmap='jet')
        if i==0:
            plt.ylabel(r'$\alpha$')
            plt.xticks([])
            plt.yticks([])
        else:
            plt.axis('off')
        plt.subplot(3,10,20+i+1)
        plt.imshow(Ximg[i,2],cmap='jet')
        if i==0:
            plt.ylabel(r'$\gamma$')
            plt.xticks([])
            plt.yticks([])
        else:
            plt.axis('off')
        
    plt.suptitle('SSFI: Spatio-Spectral Feature Images')
    plt.tight_layout()
    plt.show()

    """
    assert pos.shape[1]==2
    M = pos.shape[0] #number of channels
    if PX.ndim==2:
        K = PX.shape[1]
        N = K//M         #number of bands
        nt = PX.shape[0]
        assert N*M ==PX.shape[1]
    else:
        (nt, N, M) = PX.shape
        assert pos.shape[0]==M
    Ximg=[]
    for i in range(nt):
        if verbose: ProgBar_JL(i,nt,style=2, color='blue')
        img =[]
        band =np.arange(M)
        for j in range(N):
            if PX.ndim==2:
                px = PX[i,band]
            else:
                px = PX[i,j]
            if NormalizeEachBand: px /= px.sum()    #Not a good idea, loos the relativeity of bands
            Zi = topomap(data=px,pos=pos.copy(),res=res,show=False)
            #Zi = TopoMap(pos,data=f,res=res, showplot=False)
            img.append(Zi)
            if PX.ndim==2: band +=M
        Ximg.append(np.array(img))
    Ximg = np.array(Ximg)
    return Ximg


#-------------------------------


def TopoMap(pos, data,res=64, showplot=False,axes=None,contours=True,showsensors=True,
            interpolation=None,shownames=True, ch_names=None,showhead=True,vmin=None,vmax=None,
            returnIm = False,fontdict=None,**fontkeywords):

    r"""Topographical Map for given the data point and 2D projection points,


        .. deprecated:: 0.0.9.7
            USE :func:`topomap` for updated version

    .. warning:: DEPRECATED
       USE :func:`topomap` for updated version


    """
    ipos  = pos.copy()
    idata = data.copy()
    ipos -= 0.5 * (ipos.max(axis=0) + ipos.min(axis=0))
    ipos *= 0.85 / (ipos.max(axis=0) - ipos.min(axis=0))

    xmin, xmax,ymin, ymax = -0.5, 0.5, -0.5, 0.5

    xi = np.linspace(xmin, xmax, res)
    yi = np.linspace(ymin, ymax, res)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = GridInterpolation(ipos, idata, Xi,Yi)


    if showplot:
        ax = axes if axes else plt.gca()
        patch = patches.Ellipse((0, 0),1,1,clip_on=True,transform=ax.transData)
        #im = ax.imshow(Zi, cmap='jet', origin='lower',aspect='equal', extent=(xmin, xmax, ymin, ymax),
        #              interpolation=interpolation)
        im = ax.imshow(Zi, cmap='jet', origin='lower',aspect='equal', extent=(xmin, xmax, ymin, ymax),
                  interpolation=interpolation,vmin=vmin, vmax=vmax)
        im.set_clip_path(patch)
        ax.axis('off')
        if contours:
            contrs = ax.contour(Xi, Yi, Zi, 6, colors='k',linewidths=0.5)
            try:
                # matplotlib version 3.8 plans to remove .collections
                contrs.set_clip_path(patch)
            except:
                # old version of matplotlib works this way
                for col in contrs.collections:
                    col.set_clip_path(patch)
        if showsensors:
            ax.plot(ipos[:,0],ipos[:,1],'.k',markersize=2)


        if shownames and ch_names is not None:
            for i in range(len(ipos)):
                ax.text(ipos[i,0],ipos[i,1],ch_names[i],horizontalalignment='center',
                    verticalalignment='center',fontdict=fontdict,**fontkeywords)
        if showhead:
            #ax.plot(0,0.58,'^k',ms=25,  markerfacecolor="None")
            ax.plot(0,0.54,'2k',ms=30)
            ax.plot(0,0.54,'2k',ms=30)
            ax.plot(0,0.54,'2k',ms=30)

        if returnIm: return Zi, im
    return Zi

def Gen_SSFI(X,pos=pos,res=64,NormalizeEachBand=False,prebar=' ',newline=False):
    r"""Generating Spatio-Spectral Feature Image (SSFI)



    .. deprecated:: 0.0.9.7
            USE :func:`gen_ssfi` for updated version

    .. warning:: DEPRECATED
       USE :func:`gen_ssfi` for updated version

    """
    Ximages=[]
    Prg=['\\','-','/','|']
    #GI = el.GridInter(pos,res=res)
    for i in range(X.shape[0]):
        pf = int(50*i/(X.shape[0]-1))
        pbar = '|'+'#'*int(pf)+'.'*int(50-pf)+'|'+str(i+1)+'/'+str(X.shape[0])
        print(prebar+'|'+ str(Prg[i%len(Prg)])+'| '+str(2*pf)+'%'+pbar,end='\r', flush=True)
        band =np.arange(14)
        img =[]
        for j in range(6):
            f = X[i,band]
            if NormalizeEachBand: f /= f.sum()    #Not a good idea, loos the relativeity of bands
            #im,_ = plot_topomap(f[eOrder],epos,show=False,cmap='jet',res=64)
            #Zi = GI.get_Grid(f)
            Zi = TopoMap(pos,data=f,res=res, showplot=False)
            img.append(Zi)
            band +=14
        Ximages.append(np.array(img))
    if newline: print(' ')
    return np.array(Ximages)

def showTOPO(Zi,pos,ch_names,axes=None,vmin=None, vmax=None,res=64,interpolation=None,
             contours=True,showsensors=True,shownames=True,showhead=True):
    r"""Topographical Map for given the data point and 2D projection points,

    """
    ipos  = pos.copy()
    ipos -= 0.5 * (ipos.max(axis=0) + ipos.min(axis=0))
    ipos *= 0.85 / (ipos.max(axis=0) - ipos.min(axis=0))
    xmin, xmax,ymin, ymax = -0.5, 0.5, -0.5, 0.5
    xi = np.linspace(xmin, xmax, res)
    yi = np.linspace(ymin, ymax, res)
    Xi, Yi = np.meshgrid(xi, yi)

    ax = axes if axes else plt.gca()
    patch = patches.Ellipse((0, 0),1,1,clip_on=True,transform=ax.transData)
    im = ax.imshow(Zi, cmap='jet', origin='lower',aspect='equal', extent=(xmin, xmax, ymin, ymax),
                  interpolation=interpolation,vmin=vmin, vmax=vmax)
    im.set_clip_path(patch)
    ax.axis('off')
    if contours:
        contrs = ax.contour(Xi, Yi, Zi, 6, colors='k',linewidths=0.5)
        # for col in contrs.collections:
        #     col.set_clip_path(patch)
        try:
            # matplotlib version 3.8 plans to remove .collections
            contrs.set_clip_path(patch)
        except:
            # old version of matplotlib works this way
            for col in contrs.collections:
                col.set_clip_path(patch)

    if showsensors:
        ax.plot(ipos[:,0],ipos[:,1],'.k',markersize=2)

    if shownames and ch_names is not None:
        for i in range(len(ipos)):
            ax.text(ipos[i,0],ipos[i,1],ch_names[i],horizontalalignment='center',
                verticalalignment='center')
    if showhead:
        #ax.plot(0,0.58,'^k',ms=25,  markerfacecolor="None")
        ax.plot(0,0.54,'2k',ms=30)
        ax.plot(0,0.54,'2k',ms=30)
        ax.plot(0,0.54,'2k',ms=30)
    return im

def TopoMap_Zi(pos,data,Zi=None,res=64,showplot=False,axes=None,contours=True,showsensors=True,
            interpolation=None,shownames=True, ch_names=None,showhead=True,showbound=True,vmin=None,vmax=None,
            returnIm = False,fontdict=None,f1=0.5,f2=0.85,fx=0.5,fy=0.5,
               bound_prop=dict(r1=0.85,r2=0.85,xy=(0,0),color='k',alpha=0.5),match_shed=True,r=1,shift_origin=False,
               head_prop =dict(ms=60,markeredgewidth=3),show_vhlines=True,sensorprop=dict(),
               fontkwds={}):
    r"""Display Topographical Map of EEG, with given values
    """

    # Deafult Settings
    props = dict(vmin=None,vmax=None,contours=True,interpolation=None,
            showsensors=True, shownames=True,showhead=True, showbound=True,
            shift_origin=False,show_vhlines=True,match_shed=True,fontdict=None,
            f1=0.5,f2=0.85,fx=0.5,fy=0.5,r=1,
            bound_prop=dict(r1=0.85,r2=0.85,xy=(0,0),color='k',alpha=0.5),
            head_prop =dict(),
            sensor_prop=dict(),
            font_prop=dict())

    for key in kwargs:
        props[key] = kwargs[key]
    



    bound_default=dict(r1=0.85,r2=0.85,xy=(0,0),color='k',alpha=0.5)

    for key in bound_default:
        if key not in bound_prop:
            bound_prop[key] = bound_default[key]


    ipos  = pos.copy()
    idata = data.copy()
    if shift_origin: ipos -= f1 * (ipos.max(axis=0) + ipos.min(axis=0))
    ipos *= f2 / (ipos.max(axis=0) - ipos.min(axis=0))

    xmin, xmax,ymin, ymax = -fx, fx, -fy, fy


    if Zi is None:
        xi = np.linspace(xmin, xmax, res)
        yi = np.linspace(ymin, ymax, res)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = GridInterpolation(ipos, idata, Xi,Yi)
    else:
        res = Zi.shape[0]
        xi = np.linspace(xmin, xmax, res)
        yi = np.linspace(ymin, ymax, res)
        Xi, Yi = np.meshgrid(xi, yi)

    if showplot:
        ax = axes if axes else plt.gca()
        if match_shed:
            patch = patches.Ellipse(bound_prop['xy'],r,r,clip_on=True,transform=ax.transData)
        else:
            patch = patches.Ellipse((0, 0),1,1,clip_on=True,transform=ax.transData)
        #im = ax.imshow(Zi, cmap='jet', origin='lower',aspect='equal', extent=(xmin, xmax, ymin, ymax),
        #              interpolation=interpolation)
        im = ax.imshow(Zi, cmap='jet', origin='lower',aspect='equal', extent=(xmin, xmax, ymin, ymax),
                  interpolation=interpolation,vmin=vmin, vmax=vmax)
        im.set_clip_path(patch)
        ax.axis('off')
        if contours:
            contrs = ax.contour(Xi, Yi, Zi, 6, colors='k',linewidths=0.5)
            #for col in contrs.collections:
            #    col.set_clip_path(patch)
            try:
                # matplotlib version 3.8 plans to remove .collections
                contrs.set_clip_path(patch)
            except:
                # old version of matplotlib works this way
                for col in contrs.collections:
                    col.set_clip_path(patch)
        if showsensors:
            ax.plot(ipos[:,0],ipos[:,1],'.k',markersize=2)
            if len(sensorprop):
                ax.scatter(ipos[:,0],ipos[:,1],**sensorprop)


        if shownames and ch_names is not None:
            for i in range(len(ipos)):
                ax.text(ipos[i,0],ipos[i,1],ch_names[i],horizontalalignment='center',
                    verticalalignment='center',fontdict=fontdict,**fontkwds)
        if showhead:
            #ax.plot(0,0.58,'^k',ms=25,  markerfacecolor="None")
            #ax.plot(0,0.54,'2k',ms=30)
            #ax.plot(0,0.54,'2k',ms=30)
            #ax.plot(0,0.54,'2k',ms=30)
            ax.plot(0,0.54,'2k',**head_prop)

        if showbound:
            xy,r1,r2 = [bound_prop[key] for key in ['xy','r1','r2']]
            patch = patches.Ellipse(xy,r1,r2, color=bound_prop['color'], fill=False,alpha=bound_prop['alpha'])
            ax.add_artist(patch)

        if show_vhlines:
            ax.vlines([0],-0.5,0.5,color='k',ls='--',lw=0.5)
            ax.hlines([0],-0.5,0.5,color='k',ls='--',lw=0.5)

        if returnIm: return Zi, im
    return Zi

def display_topo_RGB(IR,kersize=4,interpolation = 'bilinear'):
    r""" RGB Image show of Topographic Maps.
    
    """
    if not(isinstance(IR,np.ndarray)):
        IR = IR.numpy()
    IR = IR[0]
    kernel_norm = np.ones([kersize,kersize])/(kersize*kersize)
    IR = IR.copy()
    IR = np.array([signal.convolve2d(IR[:,:,i].copy(),kernel_norm,boundary='symm',mode='same') for i in range(6)]).transpose([1,2,0])
    #Ii = signal.convolve2d(Ii,kernel_norm,boundary='symm',mode='same')
    IR = np.clip(IR,0,1)
    fig,ax = plt.subplots(1,6, figsize=(15,5))
    _ = TopoMap_Zi(pos, data=np.array([1,1]),Zi=IR[:,:,:3],res=64,showplot=True,axes=ax[0],contours=False,showsensors=True,
                interpolation=interpolation,shownames=True, ch_names=None,showhead=True,vmin=None,vmax=None,
                returnIm = False,fontdict=dict(fontsize=8))
    _ = TopoMap_Zi(pos, data=np.array([1,1]),Zi=IR[:,:,3:],res=64,showplot=True,axes=ax[1],contours=False,showsensors=True,
                interpolation=interpolation,shownames=True, ch_names=None,showhead=True,vmin=None,vmax=None,
                returnIm = False,fontdict=dict(fontsize=8))
    _ = TopoMap_Zi(pos, data=np.array([1,1]),Zi=IR[:,:,:3],res=64,showplot=True,axes=ax[2],contours=False,showsensors=True,
                interpolation=interpolation,shownames=True, ch_names=None,showhead=True,vmin=0,vmax=1,
                returnIm = False,fontdict=dict(fontsize=8))
    _ = TopoMap_Zi(pos, data=np.array([1,1]),Zi=IR[:,:,3:],res=64,showplot=True,axes=ax[3],contours=False,showsensors=True,
                interpolation=interpolation,shownames=True, ch_names=None,showhead=True,vmin=0,vmax=1,
                returnIm = False,fontdict=dict(fontsize=8))
    _ = TopoMap_Zi(pos, data=np.array([1,1]),Zi=IR.mean(2),res=64,showplot=True,axes=ax[4],contours=False,showsensors=True,
                interpolation=interpolation,shownames=True, ch_names=None,showhead=True,vmin=None,vmax=None,
                returnIm = False,fontdict=dict(fontsize=8))
    _ = TopoMap_Zi(pos, data=np.array([1,1]),Zi=IR.std(2),res=64,showplot=True,axes=ax[5],contours=False,showsensors=True,
                interpolation=interpolation,shownames=True, ch_names=None,showhead=True,vmin=None,vmax=None,
                returnIm = False,fontdict=dict(fontsize=8))
    #fig.suptitle('Listening',y=0.75)
    plt.show()

def _test_():
    #print('testing.......dependancies')
    filen ='Standard_1020.csv'
    filen = os.path.join(os.path.dirname(__file__),'files',filen)
    D = pd.read_csv(filen)
    #try:
    #    from scipy.spatial.qhull import Delaunay
    #except:
    from scipy.spatial import Delaunay
    #from scipy.spatial.qhull import Delaunay
    from scipy.interpolate import CloughTocher2DInterpolator
    from matplotlib import patches
    import itertools
    #print('..............done')

try:
    _test_()
    #print('')
    #print('Library loaded')
except:
    print('dependencis are not satisfied')
    '''
    filen ='Standard_1020.csv'
    filen = os.path.join(os.path.dirname(__file__), filen)
    D = pd.read_csv(filen)
    from scipy.spatial.qhull import Delaunay
    from scipy.interpolate import CloughTocher2DInterpolator
    from matplotlib import patches
    import itertools
    '''

# if __name__ == "__main__":
#     ch_names = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
#     pos, ch1 = s1020_get_epos2d_(ch_names, reorder=False)



# def TopoMap_Zi(pos, data,Zi=None,res=64,showplot=False,axes=None,contours=True,showsensors=True,
#             interpolation=None,shownames=True, ch_names=None,showhead=True,showbound=True,bound_alpha=0.8,
#             bound_clr='k',vmin=None,vmax=None,
#             returnIm = False,fontdict=None,f1=0.5,f2=0.85,fx=0.5,fy=0.5,r=1,**fontkeywords):
#
#     ipos  = pos.copy()
#     idata = data.copy()
#     ipos -= f1 * (ipos.max(axis=0) + ipos.min(axis=0))
#     ipos *= f2 / (ipos.max(axis=0) - ipos.min(axis=0))
#
#     xmin, xmax,ymin, ymax = -fx, fx, -fy, fy
#
#
#     if Zi is None:
#         xi = np.linspace(xmin, xmax, res)
#         yi = np.linspace(ymin, ymax, res)
#         Xi, Yi = np.meshgrid(xi, yi)
#         Zi = sp.eeg.GridInterpolation(ipos, idata, Xi,Yi)
#     else:
#         res = Zi.shape[0]
#         xi = np.linspace(xmin, xmax, res)
#         yi = np.linspace(ymin, ymax, res)
#         Xi, Yi = np.meshgrid(xi, yi)
#
#     if showplot:
#         from matplotlib import patches
#         ax = axes if axes else plt.gca()
#         patch = patches.Ellipse((0, 0),r,r,clip_on=True,transform=ax.transData)
#         #im = ax.imshow(Zi, cmap='jet', origin='lower',aspect='equal', extent=(xmin, xmax, ymin, ymax),
#         #              interpolation=interpolation)
#         im = ax.imshow(Zi, cmap='jet', origin='lower',aspect='equal', extent=(xmin, xmax, ymin, ymax),
#                   interpolation=interpolation,vmin=vmin, vmax=vmax)
#         im.set_clip_path(patch)
#         ax.axis('off')
#         if contours:
#             contrs = ax.contour(Xi, Yi, Zi, 6, colors='k',linewidths=0.5)
#             for col in contrs.collections:
#                 col.set_clip_path(patch)
#         if showsensors:
#             ax.plot(ipos[:,0],ipos[:,1],'.k',markersize=2)
#
#
#         if shownames and ch_names is not None:
#             for i in range(len(ipos)):
#                 ax.text(ipos[i,0],ipos[i,1],ch_names[i],horizontalalignment='center',
#                     verticalalignment='center',fontdict=fontdict,**fontkeywords)
#         if showhead:
#             #ax.plot(0,0.58,'^k',ms=25,  markerfacecolor="None")
#             ax.plot(0,0.54,'2k',ms=30)
#             ax.plot(0,0.54,'2k',ms=30)
#             ax.plot(0,0.54,'2k',ms=30)
#
#         if showbound:
#             patch = patches.Ellipse((0, 0),f2,f2, color=bound_clr, fill=False,alpha=bound_alpha)
#             ax.add_artist(patch)
#
#         if returnIm: return Zi, im
#     return Zi