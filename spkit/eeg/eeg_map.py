import os, scipy
import numpy as np
import pandas as pd
#from pathlib import Path
import matplotlib.pyplot as plt

def cart2sph(cart):
    """Convert Cartesian coordinates to spherical coordinates.

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
    """Convert spherical coordinates to Cartesion coordinates."""
    assert sph.ndim == 2 and sph.shape[1] == 3
    sph = np.atleast_2d(sph)
    cart = np.empty((len(sph), 3))
    cart[:, 2] = sph[:, 0] * np.cos(sph[:, 2])
    xy = sph[:, 0] * np.sin(sph[:, 2])
    cart[:, 0] = xy * np.cos(sph[:, 1])
    cart[:, 1] = xy * np.sin(sph[:, 1])
    return cart

def pol2cart(pol):
    """Transform polar coordinates to cartesian."""
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
    filen ='Standard_1020.csv'
    filen = os.path.join(os.path.dirname(__file__), filen)
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

class GridInter(object):
    '''
    Grid Interpolator for 2d EEG electrodes
    '''
    def __init__(self,pos,res=64):
        from scipy.spatial.qhull import Delaunay
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
    from scipy.spatial.qhull import Delaunay
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

def TopoMap(pos, data,res=64, showplot=False,axes=None,contours=True,showsensors=True,
            interpolation=None,shownames=True, ch_names=None,showhead=True,vmin=None,vmax=None,
            returnIm = False,fontdict=None,**fontkeywords):
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
        from matplotlib import patches
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


ch_names = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
pos, ch1 = s1020_get_epos2d_(ch_names, reorder=False)

def Gen_SSFI(X,pos=pos,res=64,NormalizeEachBand=False,prebar=' ',newline=False):
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
            Zi = TopoMap(pos,f)
            img.append(Zi)
            band +=14
        Ximages.append(np.array(img))
    if newline: print(' ')
    return np.array(Ximages)


def showTOPO(Zi,pos,ch_names,axes=None,vmin=None, vmax=None,res=64,interpolation=None,
             contours=True,showsensors=True,shownames=True,showhead=True):

    ipos  = pos.copy()
    ipos -= 0.5 * (ipos.max(axis=0) + ipos.min(axis=0))
    ipos *= 0.85 / (ipos.max(axis=0) - ipos.min(axis=0))
    xmin, xmax,ymin, ymax = -0.5, 0.5, -0.5, 0.5
    xi = np.linspace(xmin, xmax, res)
    yi = np.linspace(ymin, ymax, res)
    Xi, Yi = np.meshgrid(xi, yi)

    from matplotlib import patches
    ax = axes if axes else plt.gca()
    patch = patches.Ellipse((0, 0),1,1,clip_on=True,transform=ax.transData)
    im = ax.imshow(Zi, cmap='jet', origin='lower',aspect='equal', extent=(xmin, xmax, ymin, ymax),
                  interpolation=interpolation,vmin=vmin, vmax=vmax)
    im.set_clip_path(patch)
    ax.axis('off')
    if contours:
        contrs = ax.contour(Xi, Yi, Zi, 6, colors='k',linewidths=0.5)
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


def _test_():
    print('testing.......dependancies')
    filen ='Standard_1020.csv'
    filen = os.path.join(os.path.dirname(__file__), filen)
    D = pd.read_csv(filen)
    from scipy.spatial.qhull import Delaunay
    from scipy.interpolate import CloughTocher2DInterpolator
    from matplotlib import patches
    import itertools
    print('..............done')






_test_()
print('')
print('Library loaded')
