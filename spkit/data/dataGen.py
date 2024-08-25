##------------------------------Dataset Generators---------------------------------------------
from __future__ import absolute_import, division, print_function

import sys, os
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
from ..utils import deprecated


@deprecated("due to naming convension, please use 'mclassGaus' for updated/improved functionality")
def mclassGaus(N=100, nClasses=2,var =0.1,ShowPlot=False):
    y = []
    X =np.zeros([2,1])

    bi = list(range(-1,2))
    bx1,bx2 =np.meshgrid(bi,bi)
    bx1 = np.reshape(bx1,-1)
    bx2 = np.reshape(bx2,-1)

    ind = list(range(bx1.shape[0]))
    np.random.shuffle(ind)

    for i in range(nClasses):
        x1 = var*np.random.randn(1,N) + bx1[ind[i]]
        x2 = var*np.random.randn(1,N) + bx2[ind[i]]
        xi = np.vstack([x1,x2])
        X = np.hstack([X,xi])
        y  = y + [i]*N

    X = np.delete(X,0,1)
    y = np.array(y)
    y = np.reshape(y,[1,y.shape[0]])
    print(X.shape,y.shape)
    if ShowPlot:
        for i in range(nClasses):
            ii = np.where(y==i)[1]
            plt.plot(X[0,ii],X[1,ii],'*')
        plt.show()
    return X,y

def mclass_gauss(N=100,nClasses=2,var=0.1,ShowPlot=False,return_para=False):
    r"""Generate Multi-class gaussian samples


    Parameters
    ----------
    N: int, deafult=100
      - number of samples from each class
      - example N = 100, 100 samples for each class

    nClasses: int,  default=0.5
      - number of classes

    var: scalar, str, default=0.1
      - variance  -  noise
    
    ShowPlot: bool, default=False
      - Plot the data, 
        
        .. versionadded:: 0.0.9.7
            Added to return parameters

    return_para: bool, default=False
      - if True, return the parameters

    Returns
    -------
    X: 2d-array
      - data matrix with a sample for each row
      - shape (n, 2)

        .. versionchanged:: 0.0.9.7
            shape is changed to (n, 2)
           
    y: 1d-array
      - vector with the labels

        .. versionchanged:: 0.0.9.7
            shape is changed to (n, )

    See Also
    --------
    gaussian, linear, moons, sinusoidal, spiral, create_dataset

    Examples
    --------
    #sp.data.mclass_gauss
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    np.random.seed(4)
    X, y =  sp.data.mclass_gauss(N=100,nClasses=3,var=0.3)
    np.random.seed(None)
    plt.figure()
    plt.plot(X[y==0,0],X[y==0,1],'o')
    plt.plot(X[y==1,0],X[y==1,1],'o')
    plt.plot(X[y==2,0],X[y==2,1],'o')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Multi-Class Gaussian Data')
    plt.show()
    """
    y = []
    X =np.zeros([2,1])

    bi = list(range(-1,2))
    bx1,bx2 =np.meshgrid(bi,bi)
    bx1 = np.reshape(bx1,-1)
    bx2 = np.reshape(bx2,-1)

    ind = list(range(bx1.shape[0]))
    np.random.shuffle(ind)

    for i in range(nClasses):
        x1 = var*np.random.randn(1,N) + bx1[ind[i]]
        x2 = var*np.random.randn(1,N) + bx2[ind[i]]
        xi = np.vstack([x1,x2])
        X = np.hstack([X,xi])
        y  = y + [i]*N

    X = np.delete(X,0,1)
    y = np.array(y)
    y = np.reshape(y,[1,y.shape[0]])
    #print(X.shape,y.shape)
    if ShowPlot:
        for i in range(nClasses):
            ii = np.where(y==i)[1]
            plt.plot(X[0,ii],X[1,ii],'*')
        plt.show()

    y = np.squeeze(y)
    X = np.squeeze(X.T)

    return X,y

def spiral(N=[100,100],s=0.5, wrappings='random',m='random',return_para=False,**kwargs):
    r"""Generate a 2-class dataset of spirals

    Generating 2-classes of spirals

    Parameters
    ----------
    N: list or two int, default =[100,100]
      - vector that fix the number of samples from each class
      - example N = [100,100], 100 samples for each class

    s: scalar, default=0.5
      - standard deviation of the gaussian noise.

    wrappings: scalar, str, default='random'
      - number of wrappings of each spiral.
    
    m: scalar, str, default='random'
      - multiplier m of x * sin(m * x) for the second spiral.

        
        .. versionadded:: 0.0.9.7
            Added to return parameters

    return_para: bool, default=False
      - if True, return the parameters

    Returns
    -------
    X: 2d-array
      - data matrix with a sample for each row
      - shape (n, 2)

        .. versionchanged:: 0.0.9.7
            shape is changed to (n, 2)
           
    y: 1d-array
      - vector with the labels

        .. versionchanged:: 0.0.9.7
            shape is changed to (n, )

    (s, wrappings, m): parameters
       -  if return_para=True
    

    See Also
    --------
    gaussian, linear, moons, sinusoidal, mclass_gauss, create_dataset

    Examples
    --------
    #sp.data.spiral
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    X, y =  sp.data.spiral(N =[100, 100],s=0.1,wrappings=2,m=3)
    plt.figure()
    plt.plot(X[y==0,0],X[y==0,1],'o')
    plt.plot(X[y==1,0],X[y==1,1],'o')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Spiral Data')
    plt.show()
    """
    valid_args = ['s', 'wrappings','m','warn']
    warn_str = ' To turn of this warning, set `warn=False`. [0.0.9.7]'
    WARN = True
    if 'warn' in kwargs:
        WARN = kwargs['warn']

    if WARN:
        for key in kwargs:
            if key not in valid_args:
                warnings.warn(f'Argument {key} is not valid for spiral, will be ignored.' + warn_str)

    if type(m)==str and m == 'random':
        m = 1 + np.random.rand()

    if type(wrappings)==str and wrappings =='random':
        wrappings = 1 + np.random.rand() * 8


    oneDSampling = np.random.rand(N[0], 1)*wrappings*np.pi

    x1 = np.hstack([np.multiply(oneDSampling,np.cos(oneDSampling)), np.multiply(oneDSampling,np.sin(oneDSampling))])

    x1 = x1 + np.random.randn(N[0], 2)*s


    oneDSampling = np.random.rand(N[1], 1)*wrappings*np.pi

    x2 = np.hstack([np.multiply(oneDSampling,np.cos(m*oneDSampling)), np.multiply(oneDSampling,np.sin(m*oneDSampling))])

    x2 = x2 + np.random.randn(N[1], 2)*s

    X = np.vstack([x1,x2])


    Y = np.ones([sum(N),1])
    Y[:N[0],0] = 0
    Y = np.squeeze(Y)

    if return_para:
        return X, Y, (s, wrappings, m)

    return X, Y

def sinusoidal(N=[100,100],s=0.1,return_para=False,**kwargs):
    r"""Generate a 2-class  dataset separated by a sinusoidal line

    Sample a dataset from a dataset separated by a sinusoidal line
    

    Parameters
    ----------
    N: list or two int, default =[100,100]
      - vector that fix the number of samples from each class
      - example N = [100,100], 100 samples for each class

    s: scalar, default=0.1
      - standard deviation of the gaussian noise.


        
        .. versionadded:: 0.0.9.7
            Added to return parameters

    return_para: bool, default=False
      - if True, return the parameters

    Returns
    -------
    X: 2d-array
      - data matrix with a sample for each row
      - shape (n, 2)

        .. versionchanged:: 0.0.9.7
            shape is changed to (n, 2)
           
    y: 1d-array
      - vector with the labels

        .. versionchanged:: 0.0.9.7
            shape is changed to (n, )

    s: scalar
      - parameter, if return_para=True

    See Also
    --------
    gaussian, linear, moons, mclass_gauss, spiral, create_dataset

    Examples
    --------
    #sp.data.sinusoidal
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    np.random.seed(2)
    X, y =  sp.data.sinusoidal(N =[100, 100],s=0.1)
    np.random.seed(None)
    plt.figure()
    plt.plot(X[y==0,0],X[y==0,1],'o')
    plt.plot(X[y==1,0],X[y==1,1],'o')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Sinusodal Data')
    plt.show()
    """
    valid_args = ['s','warn']
    warn_str = ' To turn of this warning, set `warn=False`. [0.0.9.7]'
    WARN = True
    if 'warn' in kwargs:
        WARN = kwargs['warn']

    if WARN:
        for key in kwargs:
            if key not in valid_args:
                warnings.warn(f'Argument {key} is not valid for spiral, will be ignored.' + warn_str)
                
    X = np.array([0,0])
    while(X.shape[0]<=N[0]):
        xx = np.random.rand()
        yy = np.random.rand()
        fy = 0.7 * 0.5 * np.sin(2 * np.pi * xx) + 0.5
        if(yy <= fy):
            xi = np.array([xx + s*np.random.rand(), yy + s*np.random.rand()])
            X = np.vstack([X, xi])

    X = np.delete(X,0,0)

    while(X.shape[0] < sum(N)):
        xx = np.random.rand()
        yy = np.random.rand()
        fy = 0.7 * 0.5 * np.sin(2 * np.pi * xx) + 0.5
        if(yy > fy):
            xi = np.array([xx + s*np.random.rand(), yy + s*np.random.rand()])
            X = np.vstack([X, xi])

    Y = np.ones([sum(N),1])
    Y[:N[0],0] = 0
    Y = np.squeeze(Y)

    if return_para:
        return X, Y, s

    return X, Y

def moons(N=[100,100], s =0.1, d='random', angle = 'random',return_para=False, **kwargs):
    r"""Generate a 2-class dataset from two "moon" distributions

    Sample a dataset from two "moon" distributions


    Parameters
    ----------
    N: list or two int, default =[100,100]
      - vector that fix the number of samples from each class
      - example N = [100,100], 100 samples for each class

    s: scalar, default=0.1
      - standard deviation of the gaussian noise.

    d: scalar, str, default='random'
      - 1x2 translation vector between the two classes. 
      - With d = 0 the classes are placed on a circle.
    
    angle: scalar , default='random'
      - rotation angle of the moons (radians)

        
        .. versionadded:: 0.0.9.7
            Added to return parameters

    return_para: bool, default=False
      - if True, return the parameters

    Returns
    -------
    X: 2d-array
      - data matrix with a sample for each row
      - shape (n, 2)

        .. versionchanged:: 0.0.9.7
            shape is changed to (n, 2)
           
    y: 1d-array
      - vector with the labels

        .. versionchanged:: 0.0.9.7
            shape is changed to (n, )

    (s, d, angle): parameters
       - if return_para=True

    
    See Also
    --------
    gaussian, linear, sinusoidal, mclass_gauss, spiral, create_dataset

    Examples
    --------
    #sp.data.moons
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    np.random.seed(7)
    X, y =  sp.data.moons(N =[100, 100],s=0.2,d='random', angle='random')
    np.random.seed(None)
    plt.figure()
    plt.plot(X[y==0,0],X[y==0,1],'o')
    plt.plot(X[y==1,0],X[y==1,1],'o')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Moons Data')
    plt.show()
    """
    valid_args = ['s','d','angle','warn']
    warn_str = ' To turn of this warning, set `warn=False`. [0.0.9.7]'
    WARN = True
    if 'warn' in kwargs:
        WARN = kwargs['warn']

    if WARN:
        for key in kwargs:
            if key not in valid_args:
                warnings.warn(f'Argument {key} is not valid for spiral, will be ignored.' + warn_str)

    if type(angle)==str and angle =='random':
        angle = np.random.rand() * np.pi

    if type(d)==str and d == 'random':
        d = (-0.6 * np.random.rand(1, 2) + np.array([-0.2, -0.2]))

    d1 = np.array([[np.cos(-angle),  -np.sin(-angle)],[np.sin(-angle),   np.cos(-angle)]])

    d = np.dot(d1,d.T).T[0]

    oneDSampling =  (np.pi + np.random.rand(1, N[0]) * 1.3 * np.pi + angle)[0]

    X = np.hstack([ np.array([np.sin(oneDSampling)]).T,   np.array([np.cos(oneDSampling)]).T])

    X = X + np.random.randn(N[0],2)*s


    oneDSampling =  (np.random.rand(1, N[1]) * 1.3 * np.pi + angle)[0]

    X1 = np.hstack([ np.array([np.sin(oneDSampling)]).T,   np.array([np.cos(oneDSampling)]).T])

    X1 = X1 + np.random.randn(N[1],2)*s + np.tile(d,(N[1],1))

    #[sin(oneDSampling.T) cos(oneDSampling')] + randn(N(2),2)*s + repmat(d, N(2), 1)
    #np.tile(d,(10,1))

    X = np.vstack([X,X1])

    Y = np.ones([sum(N),1])
    Y[:N[0],0] = 0
    Y = np.squeeze(Y)

    if return_para:
        return X, Y, (s, d, angle)

    return X, Y

def gaussian(N=[100,100], ndist=3, means='random', sigmas='random',return_para=False,**kwargs):
    r"""Generate a 2-class dataset from a mixture of gaussians

    Sample a dataset from a mixture of gaussians

    Parameters
    ----------
    N: list or two int, default =[100,100]
      - vector that fix the number of samples from each class
      - example N = [100,100], 100 samples for each class

    ndist: scalar, default=3
      - number of gaussian for each class. Default is 3

    means:  array, shape (2*ndist X 2), default='random'
      - vector of size(2*ndist X 2) with the means of each gaussian.
    
    sigmas: array , default='random'
      - A sequence of covariance matrices of size (2*ndist, 2)

        .. versionadded:: 0.0.9.7
            Added to return parameters

    return_para: bool, default=False
      - if True, return the parameters

    Returns
    -------
    X: 2d-array
      - data matrix with a sample for each row
      - shape (n, 2)

        .. versionchanged:: 0.0.9.7
            shape is changed to (n, 2)
           
    y: 1d-array
      - vector with the labels

        .. versionchanged:: 0.0.9.7
            shape is changed to (n, )

    (ndist, means, sigmas): parameters
       - if return_para is True
       
    See Also
    --------
    linear, moons, sinusoidal, mclass_gauss, spiral, create_dataset

    Examples
    --------
    #sp.data.gaussian
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    np.random.seed(3)
    X, y =  sp.data.gaussian(N =[100, 100],ndist=3, means='random', sigmas='random')
    np.random.seed(None)
    plt.figure()
    plt.plot(X[y==0,0],X[y==0,1],'o')
    plt.plot(X[y==1,0],X[y==1,1],'o')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Gaussian Data')
    plt.show()
    """
    valid_args = ['ndist','means','sigmas','warn']
    warn_str = ' To turn of this warning, set `warn=False`. [0.0.9.7]'
    WARN = True
    if 'warn' in kwargs:
        WARN = kwargs['warn']

    if WARN:
        for key in kwargs:
            if key not in valid_args:
                warnings.warn(f'Argument {key} is not valid for spiral, will be ignored.' + warn_str)

    if type(sigmas)==str and sigmas == 'random':
        sigmas =[0,0]
        for i in range(ndist*2):
            sigma = np.random.rand(2, 2) + np.eye(2) * 2
            sigma[0,1] =sigma[1,0]
            sigmas = np.vstack([sigmas, sigma])

        sigmas = np.delete(sigmas,0,0)

    if type(means)==str and means == 'random':
        means = np.random.rand(ndist * 2, 2) * 20 - 10

    X = [0 , 0]

    for i in range(N[0]):
        dd = np.floor(np.random.rand() * ndist)
        dd = int(dd)
        xi = np.dot(np.random.randn(1,2),sigmas[dd*2:dd*2+2,:]) + means[dd, :]
        X = np.vstack([X,xi])

    X = np.delete(X,0,0)

    for i in range(N[1]):
        dd = np.floor(np.random.rand() * ndist + ndist)
        dd = int(dd)
        xi = np.dot(np.random.randn(1,2),sigmas[dd*2:dd*2+2,:]) + means[dd, :]
        X = np.vstack([X,xi])

    Y = np.ones([sum(N),1])
    Y[:N[0],0] = 0
    
    Y = np.squeeze(Y)

    if return_para:
        return X, Y, (ndist, means, sigmas)

    return X, Y

def linear(N=[100,100], m ='random', b ='random', s =0.1,return_para=False,**kwargs):
    r"""Generate a 2-class dataset separated by a linear boundary

    Generating samples using:

    .. math ::

        y = m*x = b    

    Parameters
    ----------
    N: list or two int, default =[100,100]
      - vector that fix the number of samples from each class
      - example N = [100,100], 100 samples for each class

    m: scalar, str, default='random'
      - slope of the separating line. 

    
    b: scalar, str, default='random'
      - bias of the line. Default is random.
    
    s: float,default= 0.1
      - standard deviation of the gaussian noise. Default is 0.1

        .. versionadded:: 0.0.9.7
            Added to return parameters

    return_para: bool, default=False
      - if True, return the parameters

    Returns
    -------
    X: 2d-array
      - data matrix with a sample for each row
      - shape (n, 2)

        .. versionchanged:: 0.0.9.7
            shape is changed to (n, 2)
           
    y: 1d-array
      - vector with the labels

        .. versionchanged:: 0.0.9.7
            shape is changed to (n, )

    (m, b, s): parameters
       -  if return_para is True

    
    See Also
    --------
    gaussian, moons, sinusoidal, mclass_gauss, spiral, create_dataset

    Examples
    --------
    #sp.data.linear
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    np.random.seed(3)
    X, y =  sp.data.linear(N =[100, 100],s=0.1)
    np.random.seed(None)
    plt.figure()
    plt.plot(X[y==0,0],X[y==0,1],'o')
    plt.plot(X[y==1,0],X[y==1,1],'o')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Linear Class Data')
    plt.show()
    """
    valid_args = ['m','b','s','warn']
    warn_str = ' To turn of this warning, set `warn=False`. [0.0.9.7]'
    WARN = True
    if 'warn' in kwargs:
        WARN = kwargs['warn']

    if WARN:
        for key in kwargs:
            if key not in valid_args:
                warnings.warn(f'Argument {key} is not valid for spiral, will be ignored.' + warn_str)

    if type(b) ==str and b == 'random':
        b = np.random.rand()*0.5

    if type(m) ==str and m == 'random':
        m = np.random.rand() * 2 + 0.01


    X =np.array([0,0])

    while(X.shape[0]<=N[0]):
        xx = np.random.rand()
        yy = np.random.rand()
        fy = xx * m + b
        if (yy<= fy):
            xi = [xx + np.random.randn()*s, yy + np.random.randn()*s]
            X = np.vstack([X,xi])

    X = np.delete(X,0,0)

    while(X.shape[0]<sum(N)):
        xx = np.random.rand()
        yy = np.random.rand()
        fy = xx * m + b
        if (yy > fy):
            xi = [xx + np.random.randn()*s, yy + np.random.randn()*s]
            X = np.vstack([X,xi])

    Y = np.ones([sum(N),1])
    Y[:N[0],0] = 0
    Y = np.squeeze(Y)

    if return_para:
        return X, Y, (m, b, s)

    return X, Y

@deprecated("due to naming convension, please use 'linear_data' for updated/improved functionality")
def linear_data(*args, **kwargs):
    r"""Linear Data
    """
    return linear(*args, **kwargs)

def create_dataset(N=100, Dtype='GAUSSIANS',noise=0,use_preset=False,return_para=False,**kwargs):
    r"""Sample a 2D dataset from different distributions

    Create 2D dataset for 2-class from different distributions


    Parameters
    ----------
    N: int, default=100
       -  Number of total samples, equally divided into two classes
       - for N=100, there will be 50 in class 0 and 50 in class 1
          
    Dtype: str, default='GAUSSIANS'
       -  Type of distribution used. 
       -  It must be one from  {'MOONS' 'GAUSSIANS' 'LINEAR' 'SINUSOIDAL' 'SPIRAL'}
       -  Or                   {'moons' 'gaussians' 'linear' 'sinusoidal' 'spiral'}
    
    noise: scalar [0,1], default=0
      -  probability to have a wrong label in the dataset
      -  noise=0 mean no wrong label

    return_para: bool, default=False
      - if True, parameters are returned

    Other parameters: **kwargs
        - Other parameters can be passed, depending on the selected distibution
        - if not passed, default setting of those parameters are used.

    warn: bool, default=True
      -  To turn off the warning of supplying irrelevent arguments, pass `warn=False`.


    1. 'GAUSSIANS' parameters: :func:`gaussian`
          * ndist: scalar, default=3
             - number of gaussian for each class. 
          * means:  array, shape (2*ndist X 2), default='random'
             - vector of size(2*ndist X 2) with the means of each gaussian.
          * sigmas: array , default='random'
             - A sequence of covariance matrices of size (2*ndist, 2)

    2.  MOONS' parameters: :func:`moons`
        * s: scalar, default=0.1
            - standard deviation of the gaussian noise.

        * d: scalar, str, default='random'
           - 1x2 translation vector between the two classes. 
           - With d = 0 the classes are placed on a circle.
            
        * angle: scalar , default='random'
           - rotation angle of the moons (radians)

    3. 'LINEAR' parameters: :func:`linear`
        * m: scalar, str, default='random'
            - slope of the separating line. 

        * b: scalar, str, default='random'
            - bias of the line. Default is random.

        * s: float,default= 0.1
            - standard deviation of the gaussian noise. Default is 0.1

    4. 'SINUSOIDAL' parameters: :func:`sinusoidal`
        * s: scalar, default=0.1
            - standard deviation of the gaussian noise.

    5. 'SPIRAL' parameters: :func:`spiral`
        * s: scalar, default=0.5
            - standard deviation of the gaussian noise.

        * wrappings: scalar, str, default='random'
            - number of wrappings of each spiral.

        * m: scalar, str, default='random'
            - multiplier m of x * sin(m * x) for the second spiral.
    

    Returns
    -------

    X: 2d-array
      - data matrix with a sample for each row
      - shape (n, 2)
           
    y: 1d-array
      - vector with the labels


    See Also
    --------
    gaussian, linear, moons, sinusoidal, spiral, mclass_gauss

    Examples
    --------
    #sp.data.create_dataset
    import numpy as np
    import matplotlib.pyplot as plt
    import spkit as sp
    DTypes = ['moons','gaussian','linear','sinusoidal','spiral']
    plt.figure(figsize=(15,3))
    for i, dtype in enumerate(DTypes):
        #print(dtype)
        X,y = sp.data.create_dataset(N=200, Dtype=dtype,use_preset=True)
        plt.subplot(1,5,i+1)
        plt.plot(X[y==0,0],X[y==0,1],'o')
        plt.plot(X[y==1,0],X[y==1,1],'o')
        plt.title(f'{dtype}')
    plt.tight_layout()
    plt.show()
    """

    #Dtype = Dtype.upper()
    means1 = np.array([[-5, -7],[2, -9],[10, 5],[12,-6]])
    sigma1 = np.tile(np.eye(2)* 3, (4, 1))

    class presets:
        moons_kw = dict(s=0.1, d=np.array([-0.5, -0.5]), angle=0)
        gaussian_kw = dict(ndist=2, means= means1, sigmas = sigma1)
        linear_kw  = dict(m = 1, b =0, s =0.1)
        sinusodal_kw = dict(s=0.01)
        spiral_kw = dict(s = 0.5, wrappings = 2, m = 2)

    NN = [int(np.floor(N / 2.0)), int(np.ceil(N / 2.0))]

    if Dtype in ['MOONS','moons']:
        if use_preset:
            X, Y, param = moons(NN,return_para=True, **presets.moons_kw)
        else:
            X, Y, param = moons(NN,return_para=True,**kwargs)

    elif Dtype in ['GAUSSIANS', 'gaussians','gaussian']:
        if use_preset:
            X, Y, param = gaussian(NN, return_para=True, **presets.gaussian_kw)
        else:
            X, Y, param = gaussian(NN, return_para=True,**kwargs)

    elif Dtype in ['LINEAR','linear']:
        if use_preset:
            X, Y, param = linear(NN,return_para=True,**presets.linear_kw )
        else:
            X, Y, param = linear(NN,return_para=True,**kwargs)

    elif Dtype in ['SINUSOIDAL','sinusoidal']:
        if use_preset:
            X, Y, param = sinusoidal(NN, return_para=True,**presets.sinusodal_kw )
        else:
            X, Y, param = sinusoidal(NN, return_para=True,**kwargs)

    elif Dtype in ['SPIRAL','spiral']:
        if use_preset:
            X, Y, param = spiral(NN,return_para=True,**presets.spiral_kw)
        else:
            X, Y, param = spiral(NN,return_para=True,**kwargs)
    else:

        raise ValueError('Specified dataset type is not correct. It must be one of {MOONS, GAUSSIANS, LINEAR, SINUSOIDAL, SPIRAL}')

    idx  = np.arange(Y.shape[0])
    np.random.shuffle(idx)

    idx = idx[:int(noise*len(idx))]
    Y[idx] = abs(Y[idx]-1)

    if return_para:
         return X, Y, param

    return X, Y
