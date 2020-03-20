##------------------------------Dataset Generators---------------------------------------------
from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt

def mclassGaus(N=100, nClasses = 2,var =0.1,ShowPlot=False):
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

def spiral(N=[100,100], s = 0.5, wrappings = 'random', m = 'random'):
    '''
    %Sample a dataset from a dataset separated by a sinusoidal line
    %   X, Y, s, wrappings, m = spiral(N, s, wrappings, m)
    %    INPUT
    %	N         1x2 vector that fix the numberof samples from each class         N =[n1, n0]
    %	s         standard deviation of the gaussian noise. Default is 0.5.
    %	wrappings number of wrappings of each spiral. Default is random.
    %	m 	  multiplier m of x * sin(m * x) for the second spiral. Default is random.
    %    OUTPUT
    %	X data matrix with a sample for each row
    %   	Y vector with the labels
    %
    %   EXAMPLE:
    %       X, Y,_ ,_ ,_ = spiral([10, 10])
            X, Y, s, w, m = spiral([10, 10])
            X, Y, s, wrappings, m = spiral(N=[100,100], s = 0.5)
            X, y, s, wrappings, m = spiral(N=[100,100], s = 0.5,  wrappings =4.5, m = 3.2)
    '''
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

    return X.T, Y.T, s, wrappings, m

def sinusoidal(N=[100,100], s = 0.1):
    '''
    %Sample a dataset from a dataset separated by a sinusoidal line
    %   X, Y, s = sinusoidal(N, s)
    %    INPUT
    %	N      1x2 vector that fix the numberof samples from each class
    % 	s      standard deviation of the gaussian noise. Default is 0.1
    %    OUTPUT
    %	X data matrix with a sample for each row
    %   	Y vector with the labels
    %
    %   EXAMPLE:
    %       X, Y,_ = sinusoidal([10, 10])
    %       X, Y,_ = sinusoidal(N = [10, 10],s=0.5)
    '''
    X = np.array([0,0])
    while(X.shape[0]<=N[0]):
        xx = np.random.rand();
        yy = np.random.rand();
        fy = 0.7 * 0.5 * np.sin(2 * np.pi * xx) + 0.5;
        if(yy <= fy):
            xi = np.array([xx + s*np.random.rand(), yy + s*np.random.rand()])
            X = np.vstack([X, xi])

    X = np.delete(X,0,0)

    while(X.shape[0] < sum(N)):
        xx = np.random.rand();
        yy = np.random.rand();
        fy = 0.7 * 0.5 * np.sin(2 * np.pi * xx) + 0.5;
        if(yy > fy):
            xi = np.array([xx + s*np.random.rand(), yy + s*np.random.rand()])
            X = np.vstack([X, xi])

    Y = np.ones([sum(N),1])
    Y[:N[0],0] = 0

    return X.T, Y.T, s

def moons(N=[100,100], s =0.1, d='random', angle = 'random'):
    '''
    % Sample a dataset from two "moon" distributions
    %   X, Y, s, d, angle = moons(N, s, d, angle)
    %    INPUT
    %	N     1x2 vector that fix the numberof samples from each class
    %	s     standard deviation of the gaussian noise. Default is 0.1
    %	d     translation vector between the two classes. With d = 0
    %	      the classes are placed on a circle. Default is random.
    %	angle rotation angle of the moons. Default is random.
    %    OUTPUT
    %	X data matrix with a sample for each row
    %   	Y vector with the labels
    %
    %   EXAMPLE:
    %       X, Y,s,d,a = moons([10, 10])
            X, Y,_,_,_ = moons(N =[10, 10],s=0.5)
    '''
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

    return X.T, Y.T, s, d, angle

def gaussian(N=[100,100], ndist = 3, means ='random', sigmas='random'):
    '''
    %Sample a dataset from a mixture of gaussians
    %   X, Y, ndist, means, sigmas = gaussian(N, ndist, means, sigmas)
    %    INPUT
    %	N      1x2 vector that fix the numberof samples from each class
    %	ndist  number of gaussian for each class. Default is 3.
    %	means  vector of size(2*ndist X 2) with the means of each gaussian.
    %	       Default is random.
    %	sigmas A sequence of covariance matrices of size (2*ndist, 2).
    %	       Default is random.
    %    OUTPUT
    %	X data matrix with a sample for each row
    %   	Y vector with the labels
    %
    %   EXAMPLE:
    %       X, Y, ndist, means, sigmas = gaussian([10, 10])
            X, Y,_,_,_ = gaussian(N =[10, 10], ndist = 2)
    '''

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

    return X.T, Y.T, ndist, means, sigmas

def linear_data(N=[100,100], m ='random', b ='random', s =0.1):
    '''
    %Sample a dataset from a linear separable dataset
    %   X, Y, m, b, s = linear(N, m, b)
    %    INPUT
    %	N      1x2 vector that fix the numberof samples from each class
    %	m      slope of the separating line. Default is random.
    %	b      bias of the line. Default is random.
    % 	s      standard deviation of the gaussian noise. Default is 0.1
    %    OUTPUT
    %	X data matrix with a sample for each row
    %   	Y vector with the labels
    %
    %   EXAMPLE:
    %       X, Y, m, b, s = linearData([10, 10])
    %       X, Y, _, _,_ = linearData(N =[10, 10],s=0.5)
    '''

    if type(b) ==str and b == 'random':
        b = np.random.rand()*0.5;

    if type(m) ==str and m == 'random':
        m = np.random.rand() * 2 +0.01;


    X =np.array([0,0])

    while(X.shape[0]<=N[0]):
        xx = np.random.rand()
        yy = np.random.rand()
        fy = xx * m + b;
        if (yy<= fy):
            xi = [xx + np.random.randn()*s, yy + np.random.randn()*s]
            X = np.vstack([X,xi])

    X = np.delete(X,0,0)

    while(X.shape[0]<sum(N)):
        xx = np.random.rand()
        yy = np.random.rand()
        fy = xx * m + b;
        if (yy > fy):
            xi = [xx + np.random.randn()*s, yy + np.random.randn()*s]
            X = np.vstack([X,xi])

    Y = np.ones([sum(N),1])
    Y[:N[0],0] = 0

    return X.T, Y.T, m, b, s

def create_dataset(N=100, Dtype='GAUSSIANS', noise=0, varargin = 'PRESET',**Options):
    '''
    %Sample a dataset from different distributions
    %   [X, Y, varargout] = create_dataset(N, type, noise, varargin)
    %
    %   INPUT
    %       N     Number of samples
    %       Dtype  Type of distribution used. It must be one from
    %            'MOONS' 'GAUSSIANS' 'LINEAR' 'SINUSOIDAL' 'SPIRAL'
    %       noise probability to have a wrong label in the dataset
    %
    %       The meaning of the optional parameters depend on the type of the
    %       dataset, if is set to 'PRESET'a fixed set of parameters is used:
    %       'MOONS' parameters:
    %           1- s: standard deviation of the gaussian noise. Default is 0.1
    %           2- d: 1X2 translation vector between the two classes. With d = 0
    %                 the classes are placed on a circle. Default is random.
    %           3- angle: rotation angle of the moons in (radians). Default is random.
    %
    %       'GAUSSIANS' parameters:
    %           1- ndist: number of gaussians for each class. Default is 3.
    %           2- means: vector of size(2*ndist X 2) with the means of each gaussian.
    %              Default is random.
    %           3- sigmas: A sequence of covariance matrices of size (2*ndist, 2).
    %              Default is random.
    %
    %       'LINEAR' parameters:
    %           1- m: slope of the separating line. Default is random.
    %           2- b: bias of the line. Default is random.
    %           3- s: standard deviation of the gaussian noise. Default is 0.1
    %
    %       'SINUSOIDAL' parameters:
    %           1- s: standard deviation of the gaussian noise. Default is 0.1
    %
    %       'SPIRAL' parameters:
    %           1- s: standard deviation of the gaussian noise. Default is 0.5.
    %           2- wrappings: wrappings number of wrappings of each spiral. Default is random.
    %           3- m: multiplier m of x * sin(m * x) for the second spiral. Default is
    %                 random.
    %
    %  OUTPUT
    %   X data matrix with a sample for each row
    %   Y vector with the labels
    %   varargout parameters used to sample data
    %   EXAMPLE:
    %       [X, Y] = create_dataset(100, 'SPIRAL', 0.01);
    %       [X, Y] = create_dataset(100, 'SPIRAL', 0.01, 'PRESET');
    %       [X, Y] = create_dataset(100, 'SPIRAL', 0, 0.1, 2, 2);
    %	[X, Y] = gaussian(NN, 2, [-5, -7; 2, -9; 10, 5; 12,-6], repmat(eye(2)* 3, 4, 1));
    '''
    Dtype = Dtype.upper()

    NN = [int(np.floor(N / 2.0)), int(np.ceil(N / 2.0))];

    usepreset = 0
    if varargin =='PRESET':
        usepreset = 1

    if Dtype =='MOONS':

        if usepreset == 1:
            X, Y, s, d, angle = moons(NN, s = 0.1, d = np.array([-0.5, -0.5]), angle = 0)  #s =0.1, d='random', angle = 'random'
        else:
            #Default Setting : moons(NN,s =0.1, d='random', angle = 'random')
            s = 0.1
            d = angle ='random'

            if Options.has_key('s'):
                s = Options['s']
            if Options.has_key('d'):
                d = Options['d']
            if Options.has_key('angle'):
                angle = Options['angle']

            X, Y, s, d, angle = moons(NN, s=s, d=d, angle=angle)

        varargout= [s, d, angle]

    elif Dtype=='GAUSSIANS':

        if usepreset == 1:
            # gaussian(N, ndist = 3, means ='random', sigmas='random')

            means1 = np.array([[-5, -7],[2, -9],[10, 5],[12,-6]])
            sigma1 = np.tile(np.eye(2)* 3, (4, 1))

            X, Y, ndist, means, sigmas = gaussian(NN, ndist =2, means = means1, sigmas = sigma1)
        else:
            #Default Setting : gaussian(N, ndist = 3, means ='random', sigmas='random')

            ndist = 3
            means = sigmas = 'random'

            if Options.has_key('ndist'):
                ndist = Options['ndist']

            if Options.has_key('means'):
                means = Options['means']

            if Options.has_key('sigmas'):
                sigmas = Options['sigmas']

            X, Y, ndist, means, sigmas = gaussian(NN, ndist = ndist, means = means, sigmas = sigma)

        varargout = [ndist, means, sigmas ]

    elif Dtype =='LINEAR':

        if usepreset == 1:
            # linear_data(N, m ='random', b ='random', s =0.1)
            X, Y, m, b, s = linear_data(NN, m = 1, b =0, s =0.1)
        else:
            #Default Setting : linear_data(N, m ='random', b ='random', s =0.1)

            s, m, b = 0.1, 'random','random'

            if Options.has_key('m'):
                m = Options['m']

            if Options.has_key('b'):
                b = Options['b']

            if Options.has_key('s'):
                s = Options['s']

            X, Y, m, b, s = linear_data(NN, m = m, b =b, s =s)

        varargout = [m, b, s]

    elif Dtype=='SINUSOIDAL':

        if usepreset == 1:
            # sinusoidal(N, s = 0.1)
            X, Y, s = sinusoidal(NN, s = 0.01)
        else:
            #Default Setting : sinusoidal(N, s = 0.1)
            s = 0.1
            if Options.has_key('s'):
                s = Options['s']

            X, Y, s = sinusoidal(NN, s = s)

        varargout = [s]

    elif Dtype =='SPIRAL':

        if usepreset == 1:
            # spiral(N, s = 0.5, wrappings = 'random', m = 'random')

            X, Y, s, wrappings, m = spiral(NN, s = 0.5, wrappings = 2, m = 2)
        else:
            # Default Setting :  spiral(N, s = 0.5, wrappings = 'random', m = 'random')

            s, wrappings, m = 0.5, 'random', 'random'

            if Options.has_key('s'):
                s = Options['s']

            if Options.has_key('wrappings'):
                wrappings = Options['wrappings']

            if Options.has_key('m'):
                m = Options['m']

            X, Y, s, wrappings, m = spiral(NN, s = s, wrappings = wrappings, m = m)

        varargout = [s, wrappings, m]

    else:

        #tkMessageBox.showerror("Tips and tricks",'Specified dataset type is not correct. It must be one of MOONS, GAUSSIANS, LINEAR, SINUSOIDAL, SPIRAL')
        raise ValueError('Specified dataset type is not correct. It must be one of MOONS, GAUSSIANS, LINEAR, SINUSOIDAL, SPIRAL')

    ind  = np.arange(Y.shape[1])
    np.random.shuffle(ind)

    ind1 = ind[:int(noise*len(ind))]
    Y[0,ind1] = abs(Y[0,ind1]-1)
    #swap = np.random.rand(Y.shape[1])<=noise
    #Y[swap] = Y[swap] + 1
    #Y[np.where(Y==2)]=0

    return X.T, Y[0,:], varargout
