'''
Something Magical and Fun with Geomatry
------------------------------------------
Author @ Nikesh Bajaj
updated on Date: 27 March 2023. Version : 0.0.1
Github :  https://github.com/Nikeshbajaj/spkit
Contact: n.bajaj@qmul.ac.uk | n.bajaj@imperial.ac.uk
'''



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import sys, os

from . import basic_geo as G
from ..core.processing import filterDC_sGolay, create_signal_1d, create_signal_2d
from ..utils_misc.borrowed import resize
from scipy import signal
from scipy.special import ellipkinc,ellipj, ellipk

# #========================================
# Mapping & Transformation
# #======================================

def ellipj_clx(u, m):
    r""" Jacobi Elliptic Functions for complex variable
    """

    sn, cn, dn, ph = ellipj(np.real(u), m)

    if np.all(np.imag(u) == 0.0):
        return sn, cn, dn, ph

    # formulas (57) ff. from
    # <https://mathworld.wolfram.com/JacobiEllipticFunctions.html>
    # or
    # <https://paramanands.blogspot.com/2011/01/elliptic-functions-complex-variables.html>

    # k = np.sqrt(m)
    # k_ = np.sqrt(1 - k ** 2)
    # m_ = k_ ** 2

    m_ = 1.0 - m
    sn_, cn_, dn_, _ = ellipj(np.imag(u), m_)

    D = 1.0 - (dn * sn_) ** 2

    sni = (sn * dn_ + 1j * (cn * dn * sn_ * cn_)) / D
    cni = (cn * cn_ - 1j * (sn * dn * sn_ * dn_)) / D
    dni = (dn * cn_ * dn_ - 1j * m * (sn * cn * sn_)) / D

    # Evaluating Jacobi elliptic functions in the complex domain
    # <http://www.peliti.org/Notes/elliptic.pdf>
    X0 = sn * dn_
    X1 = cn * cn_
    Y = sn_ * dn
    K = ellipk(m)
    nx = np.floor((np.real(u) + 2 * K) / (4 * K))
    phi = np.arctan2(X0, X1) + 1j * np.arctanh(Y) + 2 * np.pi * nx

    return sni, cni, dni, phi

def SchwarzChristoffel(X,disc2square=True):
    r"""Schwarz Christoffel Mapping or Conformal Mapping
    """
    Ke = 1.854
    m = 1/np.sqrt(2)
    if disc2square:
        #K = 1.854
        X = forceUnitCircle(X)
        #ellipkinc(phi, m)
        u,v = X[:,0], X[:,1]
        ru,rv = (u-v)*m, (u+v)*m

        A = ru**2 + rv**2
        B = ru**2 - rv**2
        U = 1 + 2*B - (A**2)
        T = np.sqrt((1+A**2)**2 - 4*(B**2))
        cos_a = (2*A - T)/U
        cos_b = U/(2*A + T)
        al = np.arccos(np.clip(cos_a,-1,1))
        bt = np.arccos(np.clip(cos_b,-1,1))

        rx = np.sign(ru)*(1-ellipkinc(al,m**2)/(2*Ke))
        ry = np.sign(rv)*(ellipkinc(bt,m**2)/(2*Ke))

        Xi = rx+ry
        Yi = ry-rx

    else:
        X = np.clip(X,-1,1)
        u,v = X[:,0], X[:,1]

        zr = u/2 - v/2
        zi = u/2 + v/2

        #complex_ellipj(x,m)
        r = ellipj_clx(Ke*(1-zr)-1j*Ke*zi,m**2)[1]

        Xi = (r.real + r.imag)*m
        Yi = (r.imag - r.real)*m

    Y = np.c_[Xi,Yi]
    return Y#,np.c_[rx,ry],(al,bt)

def forceUnitCircle(X):
    r"""Force points to be in Unit Circle"""
    xi,yi = X[:,0],X[:,1]
    r = np.sqrt(xi**2+yi**2)
    r = np.clip(r,0,1)
    t = np.arctan2(yi,xi)
    xj = r*np.cos(t)
    yj = r*np.sin(t)
    return np.c_[xj,yj]

def RadialMapping(X,disc2square=True,esp=1e-3):
    r"""Radial Mapping
    
    

    # disc to square
   
    .. math::

        r = sqrt(u^2 + v^2)

                |  (0,0)           r=0
        (x,y) = |  (sign(u)*r, sign(v)*rv/u)   u^2 >= v^2
                |  (sign(u)*ru/v, sign(v)*r)   u^2 < v^2


    # square to disc
    
    .. math::
        
        r = sqrt(x^2 + y^2)

                |  (0,0)           r=0
        (u,v) = |  (sign(x)*x^2/r, sign(y)*xy/r)    x^2 >= y^2
                |  (sign(x)*xy/r,  sign(y)*y^2/r)   x^2 <  y^2


    """
    Xi = np.zeros_like(X[:,0])
    Yi = np.zeros_like(X[:,0])
    if disc2square:
        u,v = X[:,0], X[:,1]
        r =  np.sqrt(u**2+v**2)
        Xi = np.zeros_like(u)
        Yi = np.zeros_like(v)

        idx_u0 = np.where(np.isclose(u,0))[0]
        idx_v0 = np.where(np.isclose(v,0))[0]
        u[idx_u0] = esp
        v[idx_u0] = esp


        idx1 = np.where(u**2>=v**2)[0]
        idx2 = np.where(u**2<v**2)[0]

        Xi[idx1] = np.sign(u[idx1])*r[idx1]
        Yi[idx1] = np.sign(u[idx1])*r[idx1]*(v[idx1]/u[idx1])
        Yi[idx_u0] = 0*Yi[idx_u0]

        Xi[idx2] = np.sign(v[idx2])*r[idx2]*(u[idx2]/v[idx2])
        Xi[idx_v0] = 0*Xi[idx_v0]

        Yi[idx2] = np.sign(v[idx2])*r[idx2]

    else:
        x,y = X[:,0], X[:,1]
        r =  np.sqrt(x**2+y**2)

        idx0 = np.where(np.isclose(r,0))[0]
        r[idx0] = esp
        idx = (x**2)>=(y**2)
        idx1 = np.where(idx)[0]
        idx2 = np.where(~idx)[0]

        x1,x2 = x[idx1],x[idx2]
        y1,y2 = y[idx1],y[idx2]
        r1,r2 = r[idx1],r[idx2]

        Xi[idx1] = np.sign(x1)*(x1**2)
        Yi[idx1] = np.sign(x1)*(x1*y1)

        Xi[idx2] = np.sign(y2)*(x2*y2)
        Yi[idx2] = np.sign(y2)*(y2**2)

        Xi, Yi = Xi/r, Yi/r

        Xi[idx0], Yi[idx0] = 0*Xi[idx0], 0*Yi[idx0]

    Y = np.c_[Xi,Yi]
    return Y

def FGSquircular(X,disc2square=True):
    r"""FGSquircular Mapping
    """
    X[np.isclose(X,0)]=0
    if disc2square:
        X = forceUnitCircle(X)
        X[np.isclose(X,0)]=0
        u,v = X[:,0], X[:,1]
        r = u**2 + v**2 - np.sqrt( (u**2 + v**2)*(u**2 + v**2 -4*(u**2)*(v**2)) )
        idx=u*v==0
        esp = (u*v==0)*np.ones_like(r)*1e-5
        Xi = np.sign(u*v)*np.sqrt(r)/(v*np.sqrt(2)+ esp)
        Yi = np.sign(u*v)*np.sqrt(r)/(u*np.sqrt(2)+ esp)
        Xi[u*v==0]=u[u*v==0]
        Yi[u*v==0]=v[u*v==0]
    else:
        x,y = X[:,0], X[:,1]
        r = np.sqrt(x**2 + y**2 - (x**2)*(y**2))/np.sqrt(x**2 + y**2)
        Xi = x*r
        Yi = y*r
        Xi[x*y==0]=x[x*y==0]
        Yi[x*y==0]=y[x*y==0]
    Y = np.c_[Xi,Yi]
    return Y

def Elliptical(X,disc2square=True):
    r"""Elliptical Mapping
    """
    if disc2square:
        X = forceUnitCircle(X)
        u,v = X[:,0], X[:,1]
        r = u**2 - v**2
        Xi = 0.5*np.sqrt(2 + r + 2*np.sqrt(2)*u)  -  0.5*np.sqrt(2 + r - 2*np.sqrt(2)*u)
        Yi = 0.5*np.sqrt(2 - r + 2*np.sqrt(2)*v)  -  0.5*np.sqrt(2 - r - 2*np.sqrt(2)*v)
    else:
        x,y = X[:,0], X[:,1]

        Xi = x*np.sqrt(1 - y**2/2)
        Yi = y*np.sqrt(1 - x**2/2)

    Y = np.c_[Xi,Yi]
    return Y

def ShirleyEA(X,disc2square=True,verbose=0):
    r"""Shirley Mapping
    
    # disc to square
    .. math::

        r = sqrt(u^2 + v^2)

        phi = |   atan2(v,u)        if atan2(v,u) >= - pi/4
            |   atan2(v,u) + 2pi  else


                |  (0,0)                      if r=0
        (x,y) = |  (sign(u)*r, sign(v)*rv/u)  if u^2 >= v^2
                |  (sign(u)*ru/v, sign(v)*r)  if u^2 < v^2


    # square to disc
    .. math::

        r = sqrt(x^2 + y^2)

                |  (0,0)           r=0
        (u,v) = |  (sign(x)*x^2/r, sign(y)*xy/r)    x^2 >= y^2
                |  (sign(x)*xy/r,  sign(y)*y^2/r)   x^2 <  y^2


    """
    X[np.isclose(X,0)]=0
    Xi = np.zeros_like(X[:,0])
    Yi = np.zeros_like(X[:,1])

    PI4 = np.pi/4.0

    if disc2square:
        X = forceUnitCircle(X)
        X[np.isclose(X,0)]=0

        u,v = X[:,0], X[:,1]
        r = np.sqrt(u**2 + v**2)

        phi = np.arctan2(v,u)

        if verbose: print(phi)

        phi[phi<-PI4] += 2*np.pi

        if verbose: print([phi.min(), phi.max()],'-->',[-np.pi/4, 7*np.pi/4])

        idx1 = np.where(phi<PI4)[0]
        idx2 = np.where((phi>=PI4)*(phi<3*PI4))[0]
        idx3 = np.where((phi>=3*PI4)*(phi<5*PI4))[0]
        idx4 = np.where(phi>=5*PI4)[0]

        if verbose: print(idx1,idx2,idx3,idx4)


        Xi[idx1],Yi[idx1] =  r[idx1],                             r[idx1]*phi[idx1]/PI4
        Xi[idx2],Yi[idx2] = -r[idx2]*(phi[idx2] - np.pi/2)/PI4,   r[idx2]
        Xi[idx3],Yi[idx3] = -r[idx3],                            -r[idx3]*(phi[idx3]-np.pi)/PI4
        Xi[idx4],Yi[idx4] =  r[idx4]*(phi[idx4]-3*np.pi/2)/PI4,  -r[idx4]

        #Xi,Yi = (4/np.pi)*r*(phi-3*np.pi/2), -r
        #idx = phi < 5*np.pi/4
        #Xi[idx],Yi[idx] = (4/np.pi)*r*(phi-3*np.pi/2), -r

        #Xi,Yi = (Xi+1)/2, (Yi+1)/2
    else:
        u,v = X[:,0], X[:,1]
        r = np.sqrt(u**2 + v**2)
        idx1 = np.where((u>-v)*(u>v))[0]
        idx2 = np.where((u>-v)*(u<=v))[0]
        idx3 = np.where((u<=-v)*(u<v))[0]
        idx4 = np.where((u<=-v)*(u>=v))[0]

        if verbose: print(idx1,idx2,idx3,idx4)

        Xi[idx1],Yi[idx1] =  u[idx1], (np.pi/4)*(v[idx1]/u[idx1])
        Xi[idx2],Yi[idx2] =  v[idx2], (np.pi/4)*(2 - (u[idx2]/v[idx2]))
        Xi[idx3],Yi[idx3] = -u[idx3], (np.pi/4)*(4 + (v[idx3]/u[idx3]))
        Xi[idx4],Yi[idx4] = -v[idx4], (np.pi/4)*(6 - u[idx4]/v[idx4])
        Yi[idx4][v[idx4]==0] = 0

        Xi,Yi = Xi*np.cos(Yi), Xi*np.sin(Yi)
    Y = np.c_[Xi,Yi]
    return Y

def ds_mapping(X,disc2square=True,method='Shirley'):
    r"""Disc-Square Mapping (Transformation)
   
    Disc-Square Mapping (Transformation)
    
    Mapping Disc to Square or Square to Disc

    Parameters
    ----------
    X: 2D grid with |x|<=1, |y|<=1

    disc2square: boot, default=True
          if True, map disc to square, else square to disc
    methods {Radial, FGSquircular, Elliptical, ShirleyEA, SchwarzChristoffel, Conformal}

    Returns
    -------
    Xm : of same size as X
    """
    if method.lower()=='radial' or method.lower()=='radialmapping':
        return RadialMapping(X,disc2square=disc2square)
    elif method.lower()=='fgsquircular':
        return FGSquircular(X,disc2square=disc2square)
    elif method.lower()=='elliptical':
        return Elliptical(X,disc2square=disc2square)
    elif method.lower()=='shirley' or method.lower()=='shirleyea':
        return ShirleyEA(X,disc2square=disc2square)
    elif method.lower()=='schwarzchristoffel' or method.lower()=='conformal':
        return SchwarzChristoffel(X,disc2square=disc2square)

def im2vec(file_name,gray=True, res =101):
    r"""Image to Vector

    
    Parameters
    ----------
    file name or Image Matrix I

    res, resolution to change as square matrix

    Returns
    -------
    X = [x-corrd, y-corrd, pixle value]

    """
    #from skimage.transform import resize
    if isinstance(file_name,str):
        I = plt.imread(file_name)
    else:
        I = file_name

    if res is None: res = max(I.shape[0], I.shape[1])
    if res%2==1: res+=1
    I = resize(I,[res,res])
    if gray and len(I.shape)==3:
        I = 0.2989*I[:,:,0]+0.5870*I[:,:,1] +0.1140*I[:,:,2]

    ii = np.arange(res)
    xi = 2*(ii/(res-1)-0.5)
    yi = 2*(ii/(res-1)-0.5)

    xi,yi = np.meshgrid(xi,yi)

    if len(I.shape)<3:
        X = np.c_[xi.reshape(-1),yi.reshape(-1),I.reshape(-1)]
    else:
        X = np.c_[xi.reshape(-1),yi.reshape(-1),I[:,:,0].reshape(-1),I[:,:,1].reshape(-1),I[:,:,2].reshape(-1)]

    return X

def vec2im(X,res=101,bg=1,fillgaps=False,smooth=True):
    r"""Vector to image
    
    """
    ch = 1 if len(X.shape)==3 else 3
    I = np.zeros([res,res,3]) if ch==3 else np.zeros([res,res])
    I = I*0 + bg

    #xj = X[:,0]
    #yj = X[:,1]
    #Iv = X[:,2:]

    ii = np.arange(res)
    xi = 2*(ii/(res-1)-0.5)
    yi = 2*(ii/(res-1)-0.5)
    #xi,yi = np.meshgrid(xi,yi)

    for k in range(len(X)):
        px,py = X[k,0],X[k,1]
        i,j = np.argmin(np.abs(xi-px)), np.argmin(np.abs(yi-py))
        I[j,i] = X[k,2] if ch==1 else X[k,2:]

    if fillgaps:
        #TODO
        Ic = I.copy()
        for i in range(1,I.shape[0]-1):
            if ch==1:
                idx = np.where(I[i]==bg)[0]
                idx = idx[np.where(idx>0 and idx<I.shape[0]-1)]


    if smooth:
        import scipy
        Ic = I.copy()
        kernel = np.array([[1, 1, 1], [1, 1, 1],[1, 1, 1]])/9.0
        if ch==1:
            I = scipy.signal.convolve2d(Ic, kernel, mode='same')
        else:
            I[:,:,0] = scipy.signal.convolve2d(Ic[:,:,0], kernel, mode='same')
            I[:,:,1] = scipy.signal.convolve2d(Ic[:,:,1], kernel, mode='same')
            I[:,:,2] = scipy.signal.convolve2d(Ic[:,:,2], kernel, mode='same')
        I = np.clip(I,0,1)
    return I

def transform_image(X,file_name=None,disc2square=True,res=100,method='Radial',bg=1,smooth=False,shrink_factor=1):
    r"""Transform Image
    """
    if file_name is not None:
        X = im2vec(file_name,gray=False, res=2*res)
        if shrink_factor!=1:
            X[:,0] *=shrink_factor
            X[:,1] *=shrink_factor

    if disc2square:
        X0 = X.copy()
        Xd = X.copy()
        idx = np.sqrt(Xd[:,0]**2 + Xd[:,1]**2)<=1
        Xd = Xd[idx]
        X0[~idx,2:] = bg

        XT = ds_mapping(Xd[:,:2],disc2square=disc2square,method=method)
        IT = vec2im(np.c_[XT,Xd[:,2:]],res=res,bg=bg,smooth=smooth)
        I = X0[:,2].reshape([2*res,2*res]) if X0.shape[1]==3 else X0[:,2:].reshape([2*res,2*res,3])
    else:
        Xs = X.copy()
        XT = ds_mapping(Xs[:,:2],disc2square=disc2square,method=method)
        IT = vec2im(np.c_[XT,Xs[:,2:]],res=res,bg=bg,smooth=smooth)
        I = Xs[:,2].reshape([2*res,2*res]) if X.shape[1]==3 else Xs[:,2:].reshape([2*res,2*res,3])
    return IT, X, I

def transform_image_all(file,disc2square=True,res=200,bg=1,smooth=False,shrink_factor=1, show=True):
    r"""Transform Image to all
    """
    methods = ['RadialMapping', 'FGSquircular', 'Elliptical', 'Shirley', 'SchwarzChristoffel']
    XT = []
    for k,method in enumerate(methods):
        #I, Id, Is = TranImage(file,res=res,method=method,bg=bg,smooth=smooth)
        if k==0:
            IT, X, I = transform_image(X=None,file_name=file,disc2square=disc2square,res=res,method=method,bg=bg,smooth=smooth,shrink_factor=shrink_factor)
        else:
            IT, X, I = transform_image(X=X,file_name=None,disc2square=disc2square,res=res,method=method,bg=bg,smooth=smooth,shrink_factor=shrink_factor)
        XT.append(IT)

    if show:
        plt.figure(figsize=(15,5))
        plt.subplot(1,len(XT)+1,1)
        plt.imshow(I)
        plt.axis('off')
        plt.title('input')
        for k in range(len(XT)):
            plt.subplot(1,len(XT)+1,k+2)
            plt.imshow(XT[k])
            plt.title(methods[k])
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    return XT, I

def multi_plots(X,grid=(1,None),figsize=(15,5),titles=[], show=True):
    r"""Multiple Plots
    """
    N = len(X)
    plt.figure(figsize=figsize)
    r = int(grid[0])
    c = int(np.ceil(N/r))
    for k in range(N):
        Xi = X[k]
        plt.subplot(r,c,k+1)
        plt.plot(Xi[:,0], Xi[:,1],'.')
        plt.axis('square')
        plt.yticks([])
        plt.xticks([])
        if len(titles)==N: plt.title(titles[k])

    if show: plt.show()

def demo_1(n=100):
    r"""demo1"""
    X = get_circle(r=1,n=n)
    X = np.vstack([X,X/1.2,X/1.5,X/2, X/3, X/4])
    #print(X.shape)
    plt.figure(figsize=(15,5))
    plt.subplot(141)
    plt.plot(X[:,0], X[:,1],'.')
    plt.axis('square')
    #plt.yticks([])
    plt.title('input')
    plt.subplot(142)
    Y = RadialMapping(X,disc2square=True)
    plt.plot(Y[:,0], Y[:,1],'.')
    plt.axis('square')
    plt.yticks([])
    plt.title('RadialMapping')
    plt.subplot(143)
    Y = FGSquircular(X,disc2square=True)
    plt.plot(Y[:,0], Y[:,1],'.')
    plt.axis('square')
    plt.yticks([])
    plt.title('FGSquircular')
    plt.subplot(144)
    Y = Elliptical(X,disc2square=True)
    plt.plot(Y[:,0], Y[:,1],'.')
    plt.axis('square')
    plt.yticks([])
    plt.title('Elliptical')
    plt.show()

def demo_2():
    r"""demo2"""
    Xd = get_circular_grid(n=20,al=10)
    Xs = get_square_grid(n=20,vl=10)

    methods = ['RadialMapping','FGSquircular','Elliptical','Shirley','SchwarzChristoffel']

    Xds = [ds_mapping(Xd,disc2square=True,method=method) for method in methods]
    Xsd = [ds_mapping(Xs,disc2square=False,method=method) for method in methods]


    #titles_S = ['input:S','RadialMapping','FGSquircular','Elliptical','Shirley']

    #print('T:X->S')
    multi_plots([Xd]+Xds,titles=['input']+methods)
    #Xss  = [fun1(Xs/np.sqrt(2),disc2square=True) for fun1 in FUN]
    multi_plots([Xs]+Xsd,titles=['input']+methods)


# #========================================
# ## Simulation
# #======================================

def get_circle(r=1,n=100):
    r"""Get a circle of radius r with n points"""
    t = 2*np.pi*np.linspace(0,1,n)
    x = r*np.cos(t)
    y = r*np.sin(t)
    return np.c_[x,y]

def get_circular_grid(n=10,al=10,rl=None,rmax=1):
    r"""Get a circular grid"""

    if rl is None: rl=al

    PHI = np.linspace(0,2*np.pi,al,endpoint=False)
    X = []
    R = np.linspace(0,rmax,n)
    for phi in PHI:
        Xi = np.c_[R*np.cos(R*0+phi),R*np.sin(R*0+phi)]
        X.append(Xi)

    R = np.linspace(0,rmax,rl)[1:]
    nn=0
    for r in R:
        nn += al
        PHI = np.linspace(0,2*np.pi,nn,endpoint=False)
        Xi = np.c_[r*np.cos(PHI),r*np.sin(PHI)]
        X.append(Xi)
    X = np.vstack(X)
    return X

def get_square(n=5,r=1):
    r"""Get a uniform square grid of n by n points between -r to r"""
    #dl = 2*r/n
    xi = np.linspace(-r,r,n)
    yi = np.linspace(-r,r,n)
    x,y = np.meshgrid(xi,yi)
    return np.c_[x.reshape(-1),y.reshape(-1)]

def get_square_grid(n=10,vl=5,hl=None,r=1):
    r"""Get a square grid of n by n points between -r to r"""
    xi = np.linspace(-r,r,n)
    yi = np.linspace(-r,r,n)

    if hl is None: hl=vl
    X = []
    ld = np.linspace(-r,r,vl)
    for l in ld:
        Xi = np.c_[xi,yi*0+l]
        X.append(Xi)
    ld = np.linspace(-r,r,hl)
    for l in ld:
        Yi = np.c_[xi*0+l,yi]
        X.append(Yi)
    X = np.vstack(X)
    return X

def get_sphare_v0(n1=100,n2=100,r=1,prnag=[0,2*np.pi],trang=[0,np.pi]):
    phi = np.linspace(prnag[0],prnag[1],n1)
    tht = np.linspace(trang[0],trang[1],n2)
    thT,phI = np.meshgrid(tht,phi)

    thT = thT.reshape(-1)
    phI = phI.reshape(-1)

    x = r*np.cos(phI)*np.sin(thT)
    y = r*np.sin(phI)*np.sin(thT)
    z = r*np.cos(thT)
    return np.c_[x,y,z]

def get_sphare(n1=100,n2=100,r=1,r2=1,r3=1,phi_rang=[0,2*np.pi],theta_rang=[0,np.pi]):
    r"""Get a sphare"""
    phi = np.linspace(phi_rang[0],phi_rang[1],n1)
    tht = np.linspace(theta_rang[0],theta_rang[1],n2)
    thT,phI = np.meshgrid(tht,phi)

    thT = thT.reshape(-1)
    phI = phI.reshape(-1)

    x = r*np.cos(phI)*np.sin(thT)
    y = r2*np.sin(phI)*np.sin(thT)
    z = r3*np.cos(thT)
    return np.c_[x,y,z]

def get_ellipsoid(n1=100,n2=100,rx=1,ry=2,rz=1,phi_rang=[0,2*np.pi],theta_rang=[0,np.pi]):
    r"""Get a ellipsoid
    
    Parameters
    ----------
    n1,n2= (int, int)

    rx: radius-x
    ry: radius-y
    rz: radius-z

    phi_rang: phi range
    theta_rang: theta range

    Returns
    -------
    V: (n,3)
     -  3D points

    See Also
    --------
    get_sphare, get_circle, get_circular_grid, get_square, get_square_grid

    Examples
    --------
    #sp.geometry.get_ellipsoid
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp

    V = sp.geometry.get_ellipsoid(n1=50, n2=50, rx=1, ry=2, rz=1,)
    V += 0.01*np.random.randn(V.shape[0],V.shape[1])

    X = sp.create_signal_1d(V.shape[0],bipolar=False,sg_winlen=21,sg_polyorder=2,seed=1)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(5,6))
    ax.scatter3D(V[:,0], V[:,1], V[:,2], c=X, cmap='jet',s=10)
    ax.axis('off')
    ax.view_init(elev=60, azim=45, roll=15)
    ax.set_xlim([-1,1])
    ax.set_ylim([-2,2])
    ax.set_zlim([-1,1])
    ax.set_title('ellipsoid')
    """
    phi = np.linspace(phi_rang[0],phi_rang[1],n1)
    tht = np.linspace(theta_rang[0],theta_rang[1],n2)
    thT,phI = np.meshgrid(tht,phi)

    thT = thT.reshape(-1)
    phI = phI.reshape(-1)

    x = rx*np.cos(phI)*np.sin(thT)
    y = ry*np.sin(phI)*np.sin(thT)
    z = rz*np.cos(thT)
    return np.c_[x,y,z]

def project_on_hemisphere(X,origin_shift =[0,0,0],r_ratio=[None,None,None]):
    r"""Project on Hemisphare"""
    Xc = X.copy()
    Xc[:,2] -= Xc[:,2].min()
    Xc[:,2] -= origin_shift[2]
    Xc[:,2]  = np.clip(Xc[:,2],0,None)

    Xc[:,0] -= Xc[:,0].mean()
    Xc[:,1] -= Xc[:,1].mean()

    Xc[:,0] +=origin_shift[0]
    Xc[:,1] +=origin_shift[1]

    rad = np.linalg.norm(Xc, axis=1)
    zen = np.arccos(Xc[:,-1] / rad)
    azi = np.arctan2(Xc[:,1], Xc[:,0])

    #x0 = Xab[:,2]*np.cos(Xab[:,0])*np.sin(Xab[:,1])
    #y0 = Xab[:,2]*np.sin(Xab[:,0])*np.sin(Xab[:,1])
    #z0 = Xab[:,2]*np.cos(Xab[:,1])

    if r_ratio[0] is None:
        d = X.max(0)-X.min(0)
        r_ratio = [1,d[1]/d[0],1]

    r = np.max(rad)
    xi = r*np.cos(azi)*np.sin(zen)*r_ratio[0]
    yi = r*np.sin(azi)*np.sin(zen)*r_ratio[1]
    zi = r*np.cos(zen)*r_ratio[2]

    Xp = np.c_[xi,yi,zi]
    return Xp, Xc

def points_insideCurve(curvXY,gridXY,plot=False):
    r"""Extract points of gridXY, that belongs to insideCurve"""
    gtXY = np.arctan2(gridXY[:,1],gridXY[:,0])
    grXY = np.sqrt(gridXY[:,1]**2 + gridXY[:,0]**2)
    ctXY = np.arctan2(curvXY[:,1],curvXY[:,0])
    crXY = np.sqrt(curvXY[:,1]**2 + curvXY[:,0]**2)


    points = []
    indx =[]

    for i in range(len(gtXY)):
        th,r = gtXY[i], grXY[i]
        idx = np.argmin(np.abs(ctXY-th))
        if crXY[idx]>=r:
            points.append(gridXY[i,:])
            indx.append(i)

    points = np.array(points)
    indx = np.array(indx)

    if plot:
        plt.figure(figsize=(5,4))
        #plt.imshow(X,cmap='jet',extent=[-1,1,-1,1])
        plt.scatter(gridXY[:,0],gridXY[:,1],s=1, facecolors='none', edgecolors='k')
        plt.scatter(curvXY[:,0],curvXY[:,1])
        plt.scatter(points[:,0],points[:,1],s=3, facecolors='none', edgecolors='r')
        plt.axvline(x=0,ls='--',lw=0.5,color='k')
        plt.axhline(y=0,ls='--',lw=0.5,color='k')
        plt.axis('square')
        plt.show()
    return points, indx

def create_random_map_2d(n_samples=500,p=0.3,n1=100,n2=300,trim=True,plot=False,seed=None,seed2=None,
                        param_1d=dict(sg_nwin=10,sg_polyorder=1,iterations=3,max_dxdt=None),
                        param_2d=dict(sg_winlen=31,sg_polyorder=0,iterations=3,max_dxdt=0.02,max_itr=10)):
    r"""Create a random 2d Patch of arbitary shape"""

    xb = create_signal_1d(n=n1,seed=seed,circular=True,**param_1d)
    xb = xb*(1-p) + p

    th = 2*np.pi*np.arange(len(xb))/(len(xb)-1)
    xi = xb*np.cos(th)
    yi = xb*np.sin(th)
    curvXY = np.c_[xi,yi]

    Z = create_signal_2d(n=n2,seed=seed2,**param_2d)

    idx = 2*(np.arange(n2)/(n2-1) -0.5)
    Xi,Yi =np.meshgrid(idx,idx)

    gridXYZ = np.c_[Xi.reshape(-1),Yi.reshape(-1),Z.reshape(-1)]
    #Xi.shape, Yi.shape
    points,indx = points_insideCurve(curvXY=curvXY, gridXY=gridXYZ)

    X = np.zeros_like(Z).reshape(-1) + np.nan
    X[indx]  = Z.reshape(-1)[indx]
    X = X.reshape(n2,n2)
    Xc = X.copy()
    if trim:
        id_x = np.where(np.mean(np.isnan(X),0)<1)[0]
        id_y = np.where(np.mean(np.isnan(X),1)<1)[0]

        Xc = X[:,id_x][id_y,:]

    xk,yk = np.where(~np.isnan(Xc))[0], np.where(~np.isnan(Xc))[1]
    idx = np.arange(len(xk))
    np.random.seed(seed)
    np.random.shuffle(idx)
    np.random.seed(None)
    idx_r = idx[:n_samples]
    Xs = np.c_[xk[idx_r],yk[idx_r],Xc[xk[idx_r],yk[idx_r]]]
    if plot:
        plt.figure(figsize=(15,4))
        plt.subplot(141)
        plt.plot(xi,yi,'-')
        plt.scatter(xi, yi, s=80, facecolors='none', edgecolors='r')
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        plt.axvline(x=0,ls='--',lw=0.5,color='k')
        plt.axhline(y=0,ls='--',lw=0.5,color='k')
        plt.axis('square')

        plt.subplot(142)
        plt.imshow(Z,cmap='jet',extent=[-1,1,-1,1],origin='lower')
        plt.plot(xi,yi,'-')
        plt.axis('square')
        plt.axvline(x=0,ls='--',lw=0.5,color='k')
        plt.axhline(y=0,ls='--',lw=0.5,color='k')

        plt.subplot(143)
        plt.imshow(X,cmap='jet',extent=[-1,1,-1,1],origin='lower')
        plt.plot(xi,yi,'-')
        plt.axis('square')
        plt.axvline(x=0,ls='--',lw=0.5,color='k')
        plt.axhline(y=0,ls='--',lw=0.5,color='k')
        plt.subplot(144)
        plt.scatter(Xs[:,1],Xs[:,0],s=30,alpha=0.5,c=Xs[:,2], cmap='jet')
        plt.axis('square')
        plt.show()
    return Xc,Xs

def get_boundary_points(XY):
    r"""Get points of boundary"""
    from scipy.spatial.qhull import Delaunay
    tri = Delaunay(XY)
    tri.simplices
    Tobj = G.TriAng(F = tri.simplices)
    Eb = Tobj.getBorder_edges()
    Pb = np.unique(np.hstack([list(e) for e in Eb]))
    return Pb, XY[Pb]

def get_boundary_points_r(XY,K=3,n=360,shift=True,verbose=0):
    r"""Get points of boundary"""
    if shift: XY = XY - XY.mean(0)
    gtXY = np.arctan2(XY[:,1],XY[:,0])# + np.pi
    grXY = np.sqrt(XY[:,1]**2 + XY[:,0]**2)

    #tht = np.arange(0,2*np.pi+delt,delt)
    theta = np.linspace(-np.pi,np.pi+2*np.pi/n,n)
    rad   = np.zeros_like(theta)
    points = []

    Idx =[]
    for i in range(len(theta)):
        idx_t = np.argsort(np.abs(gtXY-theta[i]))[:K]

        if verbose:
            print('t_i',180*theta[i]/np.pi)
            print('t_n',180*gtXY[idx_t]/np.pi)

        rad[i] = np.max(grXY[idx_t])
        idx_r = np.argmax(grXY[idx_t])
        ki = idx_t[idx_r]
        if ki not in Idx:
            Idx.append(ki)
            points.append(XY[ki])
        #---
        #idx_t = np.argmin(np.abs(gtXY-tht[i]))
        #print(gtXY[idx_t])
        #rad[i] = grXY[idx_t]
        #points.append(XY[idx_t])
    Idx = np.array(Idx)
    points = np.array(points)
    return points,Idx,(theta,rad)

def lin_inter(x1,x2,y1,y2,xt):
    r"""linear Inpeterpolation
    """
    yt = y1 + (y2-y1)*(xt-x1)/(x2-x1)
    return yt

def map_to_circle_v1(X,method ='radial',bpoints=200,K=20,origin=[None,None],verbose=0,force=True,shift_back=False):
    r"""map to circle
    """
    XYi = X[:,:2]
    XYj = XYi - XYi.mean(0)
    if origin[0] is not None:
        XYj += np.array(origin)

    Bpoints,Idx,(tht,rad) = get_boundary_points_r(XY=XYj,K=K,n=bpoints,shift=False)

    #Idx = get_boundary(X,method='delaunay',shift_to0=True,K=5,n=360)[0]

    gtXY = np.arctan2(XYj[:,1],XYj[:,0])
    grXY = np.sqrt(XYj[:,1]**2 + XYj[:,0]**2)

    BtXY = gtXY[Idx]
    BrXY = grXY[Idx]
    XY = np.zeros_like(XYj)
    for i in range(len(XYj)):
        tht_i = gtXY[i]
        rad_i = grXY[i]
        if i in Idx:
            XY[i] = [np.cos(tht_i),np.sin(tht_i)]
        elif np.sum((BtXY-tht_i)==0):
            XY[i] = [np.cos(tht_i),np.sin(tht_i)]
        else:
            t_diff = BtXY-tht_i

            close_tp = np.where(t_diff>0,t_diff,np.inf)
            close_tn = np.where(t_diff<0,t_diff,np.inf)

            if np.all(np.isinf(close_tp)):
                close_tp = close_tn+2*np.pi
            if np.all(np.isinf(close_tn)):
                close_tn = close_tp -2*np.pi

            t1 = BtXY[np.argmin(close_tp)]
            r1 = BrXY[np.argmin(close_tp)]

            t2 = BtXY[np.argmin(np.abs(close_tn))]
            r2 = BrXY[np.argmin(np.abs(close_tn))]

            if t1==t2:
                print(t1,t2,r1,r2)
                print(t_diff)
                print('')
                #print(close_tp)
                print('')
                #print(close_tn)
                print('')

            rad_k = lin_inter(t1,t2,r1,r2,tht_i)
            rad_j = rad_i/rad_k
            rad_j = np.clip(rad_j,0,1)
            XY[i] = [rad_j*np.cos(tht_i),rad_j*np.sin(tht_i)]
            if verbose:
                print(tht_i,rad_i)
                print(t1,t2,r1,r2)
                print(rad_k, rad_j)
                print('--')
    if force: XY = forceUnitCircle(XY)
    if shift_back:
        return XY+XYi.mean(0)
    return XY

def map_to_circle_v2(X,method ='radial_scan',bpoints=200,K=20,origin=[None,None],verbose=0,force=True,shift_back=False):
    r"""map to circle
    """
    XYi = X[:,:2]
    XYj = XYi - XYi.mean(0)
    if origin[0] is not None:
        XYj += np.array(origin)

    Idx,points,_ = get_boundary_v2(X[:,:2],method=method,shift_to0=True,K=K,n=360,verbose=0)

    #Bpoints,Idx,(tht,rad) = getBpoints_r(XY=XYj,K=K,n=bpoints,shift=False)
    #Idx = get_boundary(X,method='delaunay',shift_to0=True,K=5,n=360)[0]

    gtXY = np.arctan2(XYj[:,1],XYj[:,0])
    grXY = np.sqrt(XYj[:,1]**2 + XYj[:,0]**2)

    BtXY = gtXY[Idx]
    BrXY = grXY[Idx]
    XY = np.zeros_like(XYj)
    for i in range(len(XYj)):
        tht_i = gtXY[i]
        rad_i = grXY[i]
        if i in Idx:
            XY[i] = [np.cos(tht_i),np.sin(tht_i)]
        elif np.sum((BtXY-tht_i)==0):
            XY[i] = [np.cos(tht_i),np.sin(tht_i)]
        else:
            t_diff = BtXY-tht_i

            close_tp = np.where(t_diff>0,t_diff,np.inf)
            close_tn = np.where(t_diff<0,t_diff,np.inf)

            if np.all(np.isinf(close_tp)):
                close_tp = close_tn+2*np.pi
            if np.all(np.isinf(close_tn)):
                close_tn = close_tp -2*np.pi

            t1 = BtXY[np.argmin(close_tp)]
            r1 = BrXY[np.argmin(close_tp)]

            t2 = BtXY[np.argmin(np.abs(close_tn))]
            r2 = BrXY[np.argmin(np.abs(close_tn))]

            if t1==t2:
                print(t1,t2,r1,r2)
                print(t_diff)
                print('')
                #print(close_tp)
                print('')
                #print(close_tn)
                print('')

            rad_k = lin_inter(t1,t2,r1,r2,tht_i)
            rad_j = rad_i/rad_k
            rad_j = np.clip(rad_j,0,1)
            XY[i] = [rad_j*np.cos(tht_i),rad_j*np.sin(tht_i)]
            if verbose:
                print(tht_i,rad_i)
                print(t1,t2,r1,r2)
                print(rad_k, rad_j)
                print('--')
    if force: XY = forceUnitCircle(XY)
    if shift_back:
        return XY+XYi.mean(0)
    return XY

def get_boundary_v1(X,method='delaunay',shift_to0=True,K=5,n=360,verbose=0):
    r"""Get Boundaries
    """
    if method=='delaunay':
        from scipy.spatial.qhull import Delaunay
        tri = Delaunay(X)
        tri.simplices
        Tobj = G.TriAng(F=tri.simplices)
        Eb = Tobj.getBorder_edges()

        index = np.unique(np.hstack([list(e) for e in Eb]))

        XY = X - X.mean(0)
        gtXY = np.arctan2(XY[index,1],XY[index,0])# + np.pi

        idx = np.argsort(gtXY)

        index = index[idx]

        return index, X[index]

    elif method=='radial_scan':

        XY = X - X.mean(0)
        gtXY = np.arctan2(XY[:,1],XY[:,0])# + np.pi
        grXY = np.sqrt(XY[:,1]**2 + XY[:,0]**2)

        #tht = np.arange(0,2*np.pi+delt,delt)
        theta = np.linspace(-np.pi,np.pi+2*np.pi/n,n)
        rad   = np.zeros_like(theta)
        points = []

        Idx =[]
        for i in range(len(theta)):
            idx_t = np.argsort(np.abs(gtXY-theta[i]))[:K]
            if verbose:
                print('t_i',180*theta[i]/np.pi)
                print('t_n',180*gtXY[idx_t]/np.pi)

            rad[i] = np.max(grXY[idx_t])
            idx_r  = np.argmax(grXY[idx_t])
            ki = idx_t[idx_r]
            if ki not in Idx:
                Idx.append(ki)
                points.append(XY[ki])

        Idx = np.array(Idx)
        points = np.array(points)
        return Idx,points,(theta,rad)

    elif method=='peak_detection':
        XY = X - X.mean(0)
        gtXY = np.arctan2(XY[:,1],XY[:,0])# + np.pi
        grXY = np.sqrt(XY[:,1]**2 + XY[:,0]**2)

        ij = np.argsort(gtXY)

        peaks, _ = signal.find_peaks(grXY[ij])

        if verbose:
            plt.figure(figsize=(15,3))
            plt.plot(grXY[ij])
            plt.plot(peaks,grXY[ij][peaks],'*')

        return peaks,XY[ij][peaks],ij

def get_boundary_v2(X,method='delaunay',shift_to0=True,K=5,n=360,verbose=0):
    r"""Get Boundaries
    """
    if method=='delaunay':
        from scipy.spatial.qhull import Delaunay
        tri = Delaunay(X)
        tri.simplices
        Tobj = G.TriAng(F=tri.simplices)
        Eb = Tobj.getBorder_edges()

        index = np.unique(np.hstack([list(e) for e in Eb]))

        XY = X - X.mean(0)
        gtXY = np.arctan2(XY[index,1],XY[index,0])# + np.pi

        idx = np.argsort(gtXY)

        index = index[idx]

        return index, X[index]

    elif method=='radial_scan':

        XY = X - X.mean(0)
        gtXY = np.arctan2(XY[:,1],XY[:,0])# + np.pi
        grXY = np.sqrt(XY[:,1]**2 + XY[:,0]**2)

        #tht = np.arange(0,2*np.pi+delt,delt)
        theta = np.linspace(-np.pi,np.pi+2*np.pi/n,n)
        rad   = np.zeros_like(theta)
        points = []

        Idx =[]
        for i in range(len(theta)):
            #idx_t = np.argsort(np.abs(gtXY-theta[i]))[:K]
            ix = np.arange(len(gtXY))
            diff   = np.abs(np.r_[gtXY-theta[i],gtXY-2*np.pi-theta[i]])
            difIdx = np.r_[ix,ix]
            idx_t = difIdx[np.argsort(diff)][:K]

            rad[i] = np.max(grXY[idx_t])
            idx_r  = np.argmax(grXY[idx_t])
            ki = idx_t[idx_r]

            if verbose:
                print('t_i',np.around(180*theta[i]/np.pi,2))
                print('t_n',np.around(180*gtXY[idx_t]/np.pi,2))
                print('t_m',np.around(180*gtXY[idx_t[idx_r]]/np.pi,2))
                print('')

            if ki not in Idx:
                Idx.append(ki)
                points.append(XY[ki])

        Idx = np.array(Idx)
        points = np.array(points)
        return Idx,points,(theta,rad)

    elif method=='peak_detection':
        XY = X - X.mean(0)
        gtXY = np.arctan2(XY[:,1],XY[:,0])# + np.pi
        grXY = np.sqrt(XY[:,1]**2 + XY[:,0]**2)
        ij = np.argsort(gtXY)

        peaks, _ = signal.find_peaks(grXY[ij])

        if verbose:
            plt.figure(figsize=(15,3))
            plt.plot(grXY[ij])
            plt.plot(peaks,grXY[ij][peaks],'*')

        return peaks,XY[ij][peaks],ij


def plot_map_2d(X,V,res=[128,128],vminmax=[None,None],fmt='%.2f',method=1,ax=None,fig=None,colorbar=True,lines=True,
                title=None,show_points=True,alpha=None,origin='lower',
                vlines =[0,np.pi,2*np.pi], hlines=[0,np.pi/2,np.pi]):
    r"""plot 2d map
    """
    V2I = G.Inter2DPlane(V[:,:2], res=res)
    if method==2:
        Vmap = V2I.get_image2(X)
    else:
        Vmap = V2I.get_image(X)
    vmin, vmax = vminmax
    if vmin is None: vmin = np.nanmin(X)
    if vmax is None: vmax = np.nanmax(X)


    if ax is None:
        fig = plt.figure(figsize = [8,4])
        ax = fig.add_subplot()

    im = ax.imshow(Vmap,cmap='jet',aspect='auto',origin=origin,alpha=alpha,
              extent=[V[:,0].min(),V[:,0].max(),V[:,1].min(),V[:,1].max()],
             vmin=vmin, vmax=vmax)

    if colorbar:
        cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
        fig.colorbar(im, cax=cbar_ax,format=fmt)
    if show_points: ax.plot(V[:,0],V[:,1],'.k',alpha=0.9,ms=0.7)
    if lines:
        #v0,v1,v2 = [0,np.pi,2*np.pi]
        #h0,h1,h2 = [0,np.pi/2,np.pi]
        ax.vlines(vlines,V[:,1].min(),V[:,1].max(),color='k',ls='-',lw=1)
        #ax.vlines([v1],V[:,1].min(),V[:,1].max(),color='k',ls='--',lw=1)
        ax.hlines(hlines,V[:,0].min(),V[:,0].max(),color='k',ls='-',lw=1)
        #ax.vlines([h1],V[:,1].min(),V[:,1].max(),color='k',ls='--',lw=1)
        ax.set_xticks(vlines)
        #ax.set_xticklabels([r'$0$',r'$\pi$',r'$2\pi$'])
        ax.set_yticks(hlines)
        #ax.set_yticklabels([r'$0$',r'$\pi$'])
    if title is not None: ax.set_title(title)
    return ax


def demo_3(seed=1):
    r"""demo3
    """
    X,Xs =  create_random_map_2d(n1=100,n2=100,p=0.3,seed=seed,plot=True,trim=True,n_samples=500)
    np.random.seed(None)
    Xa = map_to_circle_v2(Xs,method ='radial_scan',bpoints=200,K=10,origin=[None,None],verbose=0,force=True,shift_back=False)
    Y1 = RadialMapping(Xa*1,disc2square=True)
    Y2 = FGSquircular(Xa*1,disc2square=True)
    Y3 = Elliptical(Xa*1,disc2square=True)
    Y4 = SchwarzChristoffel(Xa*1,disc2square=True)

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.imshow(X,aspect='auto',origin='lower',cmap='jet')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('Input-map: X')
    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot(121)
    ax.scatter(Xs[:,1],Xs[:,0],s=50,alpha=0.5,c=Xs[:,2], cmap='jet')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('Sampled points: S')
    ax = fig.add_subplot(122)
    plot_map_2d(Xs[:,2],np.c_[Xs[:,1],Xs[:,0]],res=[128,128],fig=fig,ax=ax,lines=False,colorbar=False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('Interpolated S')
    plt.show()


    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot(121)
    ax.scatter(Xa[:,1],Xa[:,0],s=50,alpha=0.5,c=Xs[:,2], cmap='jet')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('Circular mapping of Sc')
    ax = fig.add_subplot(122)
    plot_map_2d(Xs[:,2],np.c_[Xa[:,1],Xa[:,0]],res=[128,128],fig=fig,ax=ax,lines=False,colorbar=False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('Interpolated Sc')
    plt.show()

    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot(141)
    plot_map_2d(Xs[:,2],np.c_[Y1[:,1],Y1[:,0]],res=[128,128],fig=fig,ax=ax,lines=False,colorbar=False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('RadialMapping of Sc')
    ax = fig.add_subplot(142)
    plot_map_2d(Xs[:,2],np.c_[Y2[:,1],Y2[:,0]],res=[128,128],fig=fig,ax=ax,lines=False,colorbar=False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('FGSquircular of Sc')
    ax = fig.add_subplot(143)
    plot_map_2d(Xs[:,2],np.c_[Y3[:,1],Y3[:,0]],res=[128,128],fig=fig,ax=ax,lines=False,colorbar=False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('Elliptical of Sc')
    ax = fig.add_subplot(144)
    plot_map_2d(Xs[:,2],np.c_[Y4[:,1],Y4[:,0]],res=[128,128],fig=fig,ax=ax,lines=False,colorbar=False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('SchwarzChristoffel of Sc')
    plt.tight_layout()
    plt.show()



#
# def ShirleyEA_v0(X,disc2square=True,verbose=0):
#     '''
#     # disc to square
#     ----------------
#     r = sqrt(u^2 + v^2)
#
#     phi = |   atan2(v,u)        if atan2(v,u) >= - pi/4
#           |   atan2(v,u) + 2pi  else
#
#
#             |  (0,0)                      if r=0
#     (x,y) = |  (sign(u)*r, sign(v)*rv/u)  if u^2 >= v^2
#             |  (sign(u)*ru/v, sign(v)*r)  if u^2 < v^2
#
#
#     # square to disc
#     -----------------
#     r = sqrt(x^2 + y^2)
#
#             |  (0,0)           r=0
#     (u,v) = |  (sign(x)*x^2/r, sign(y)*xy/r)    x^2 >= y^2
#             |  (sign(x)*xy/r,  sign(y)*y^2/r)   x^2 <  y^2
#
#
#     '''
#     X[np.isclose(X,0)]=0
#     Xi = np.zeros_like(X[:,0])
#     Yi = np.zeros_like(X[:,1])
#
#     if disc2square:
#         X = forceUnitCircle(X)
#         X[np.isclose(X,0)]=0
#         u,v = X[:,0], X[:,1]
#         r = np.sqrt(u**2 + v**2)
#
#         phi = np.arctan2(v,u)
#
#         if verbose: print(phi)
#
#         phi[phi<-np.pi/4] = phi[phi<-np.pi/4] + 2*np.pi
#
#         if verbose: print(phi)
#
# #         idx1 = np.where(phi<np.pi/4)[0]
# #         idx2 = np.where((phi>=np.pi/4) and (phi<3*np.pi/4))[0]
# #         idx3 = np.where((phi>=3*np.pi/4) and (phi<5*np.pi/4))[0]
# #         idx4 = np.where(phi>=5*np.pi/4)[0]
#
# #         if verbose: print(idx1,idx2,idx3,idx4)
#
# #         Xi[idx1],Yi[idx1] = r[idx1], (4/np.pi)*r[idx1]*phi[idx1]
# #         Xi[idx2],Yi[idx2] = (-4/np.pi)*r[idx2]*(phi[idx2]-np.pi/2), r[idx2]
# #         Xi[idx3],Yi[idx3] = -r[idx3],(-4/np.pi)*r[idx3]*(phi[idx3]-np.pi)
# #         Xi[idx4],Yi[idx4] = (4/np.pi)*r[idx4]*(phi[idx4]-3*np.pi/2), -r[idx4]
#
#         Xi,Yi = (4/np.pi)*r*(phi-3*np.pi/2), -r
#         idx = phi < 5*np.pi/4
#         Xi[idx],Yi[idx] = (4/np.pi)*r*(phi-3*np.pi/2), -r
#
#
#
#     else:
#         u,v = X[:,0], X[:,1]
#         idx1 = np.where(u**2>v**2)[0]
#         idx2 = np.where( (u**2<=v**2)*(y**2>0))[0]
#         Xi[idx1],Yi[idx1] = u[idx1], (np.pi/4)*v[idx1]/u[idx1]
#         Xi[idx2],Yi[idx2] = v[idx2], np.pi/2 - (np.pi/4)*u[idx1]/v[idx1]
#     Y = np.c_[Xi,Yi]
#     return Y
# def SchwarzChristoffel_v0(X,disc2square=True):
#     Ke = 1.854
#     m  = 1/np.sqrt(2)
#     if disc2square:
#         X = forceUnitCircle(X)
#         #ellipkinc(phi, m)
#         u,v = X[:,0], X[:,1]
#         tht = (1+1j)*(u + v*1j)*m
#         phi = np.arccos(tht)
#         #r = (1-1j)*ellipkinc(phi, m)/(-Ke)
#         r = ((1-1j)/-Ke)*ellipkinc_clx(phi, m)
#         Xi = r.real + 1
#         Yi = r.imag - 1
#     else:
#         #ellipj(u, m)
#         x,y = X[:,0], X[:,1]
#         tht = (1+1j)*(x + y*1j)/2
#         r = (1-1j)*m*ellipj_clx(Ke*tht - Ke,m)[1]
#         Xi = r.real
#         Yi = r.imag
#     Y = np.c_[Xi,Yi]
#     return Y

# def Conformal_v0(X,disc2square=True):
#     K = 1.854
#     if disc2square:
#         X = forceUnitCircle(X)
#         #ellipkinc(phi, m)
#         u,v = X[:,0], X[:,1]
#         u1,v1 = (u-v)/np.sqrt(2), (u+v)/np.sqrt(2)
#         A = u1**2 + v1**2
#         B = u1**2 - v1**2
#         T = np.sqrt((1+A**2)**2 - 4*B**2)
#         U = 1+2*B - A**2
#         alp = np.arccos((2*A-T)/U)
#         bet = np.arccos(U/(2*A+T))
#
# def GeoMap(X,disc2square=True,method='Shirley'):
#     '''
#     methods = Radial, FGSquircular, Elliptical, ShirleyEA, SchwarzChristoffel, Conformal
#     '''
#     if method.lower()=='radial' or method.lower()=='radialmapping':
#         return RadialMapping(X,disc2square=disc2square)
#     elif method.lower()=='fgsquircular':
#         return FGSquircular(X,disc2square=disc2square)
#     elif method.lower()=='elliptical':
#         return Elliptical(X,disc2square=disc2square)
#     elif method.lower()=='shirley' or method.lower()=='shirleyea':
#         return ShirleyEA(X,disc2square=disc2square)
#     elif method.lower()=='schwarzchristoffel' or method.lower()=='conformal':
#         return SchwarzChristoffel(X,disc2square=disc2square)
#
