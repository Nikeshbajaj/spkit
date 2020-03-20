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
from matplotlib.gridspec import GridSpec
import warnings


class LR():
    def  __init__(self,X,y,alpha=0.01,lambd=0,polyfit=True,degree=3,FeatureNormalize=False):
        warnings.warn("Note: In current version, shape of X is (nf, n) and shape of y is (n,1), where nf is number of features and n is number of samples. These will be changed to conventional shapes (e.g. (n,nf), (n,)) of matrixs, in the later version.")
        self.alpha=alpha
        self.cost=[]
        self.itr=0
        self.lambd =lambd
        self.polyfit = polyfit
        self.degree = degree
        self.FeatureNormalize=FeatureNormalize
        if self.polyfit:
            self.X = self.PolyFeature(X)
        else:
            self.X = X

        self.means = np.mean(self.X,1).reshape(self.X.shape[0],-1)
        self.stds  = np.std(self.X,1).reshape(self.X.shape[0],-1)

        if FeatureNormalize:
            self.X = self.Normalize(self.X)

        self.y = y
        self.W =np.random.randn(1,self.X.shape[0])*np.sqrt(2.0/self.X.shape[0])
        self.b = 0


    def fit(self,X,y,itr=100,verbose=True):
        nf,N = X.shape
        if self.polyfit:
            X = self.PolyFeature(X)
        if self.FeatureNormalize:
            X = self.Normalize(X)
        #print(X.shape)
        for i in range(itr):
            self.itr +=1
            Z = np.dot(self.W,X) + self.b
            A = self.sigmoid(Z)
            err = A-y
            #J = np.sum(err**2)+self.lambd*np.sum(self.W**2)
            #J/=(2.0*A.shape[0])
            J = self.CostF(A,y)
            self.cost.append(J)
            if verbose:
                print('Epoc @ ',self.itr,' Cost ',J)
            dW = np.dot(X,err.T).T + self.lambd*self.W
            db = np.sum(err,axis=1,keepdims=True)+ self.lambd*self.b
            self.W -= self.alpha*dW
            self.b -= self.alpha*db
            #print(self.W,self.b)

    def PolyFeature(self,X):
        nf,N = X.shape
        Xnew = np.vstack([X,X**2,X[0,:]*X[1,:]])
        if self.degree>2:
            for d in range(3,self.degree+1):
                Xnew = np.vstack([Xnew,X**d])
        return Xnew

    def Normalize(self,X):
        #self.means = np.mean(X,1).reshape(X.shape[0],-1)
        #self.stds  = np.std(X,1).reshape(X.shape[0],-1)
        X1=(X-self.means)/self.stds
        return X1

    def CostF(self,A,y):
        J = -np.sum(y*np.log(A+1e-20)+(1-y)*np.log(1-A+1e-20))
        J = J/(2.0*A.shape[1])
        return J

    def sigmoid(self,z):
        return 1.0/(1+np.exp(-z))

    def predict(self,X):
        if self.polyfit:
            X = self.PolyFeature(X)
        if self.FeatureNormalize:
            X = self.Normalize(X)
        Z = np.dot(self.W,X) + self.b
        A = self.sigmoid(Z)
        return A,A>0.5

    def getWeights(self):
        return self.W,self.b

    def Bplot(self,ax=None,density =500,hardbound=False):
        if ax is None: fig, ax = plt.subplots(111)
        x1mn = np.min(self.X[0,:])
        x2mn = np.min(self.X[1,:])
        x1mx = np.max(self.X[0,:])
        x2mx = np.max(self.X[1,:])
        x1 = np.linspace(x1mn,x1mx,density)
        x2 = np.linspace(x2mn,x2mx,density)
        xv, yv = np.meshgrid(x1, x2)
        Xall = np.vstack([xv.flatten(),yv.flatten()])

        yp,yi = self.predict(Xall)
        yp = yp.reshape(density,density)
        self.yp=yp
        if hardbound:
            ax.imshow(yp>0.5, origin='lower', interpolation='bicubic',extent=(xv.min(), xv.max(), yv.min(), yv.max()))
        else:
            ax.imshow(yp, origin='lower', interpolation='bicubic',extent=(xv.min(), xv.max(), yv.min(), yv.max()))

        in0 = np.where(self.y==0)[1]
        in1 = np.where(self.y==1)[1]
        #print(in0,in1)
        ax.plot(self.X[0,in0],self.X[1,in0],'.b')
        ax.plot(self.X[0,in1],self.X[1,in1],'.r')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title('Decision Boundry')
        #plt.show()

    def LCurvePlot(self,ax=None):
        if ax is None: fig, ax = plt.subplots(111)
        ax.plot(range(self.itr),self.cost)
        ax.set_xlabel('iteration')
        ax.set_ylabel('Cost')
        ax.set_title('Learning Curve')
        ax.grid(True)
        #plt.show()

    def Wplot(self,ax=None):
        if ax is None: fig, ax = plt.subplots(111)
        W,b = self.getWeights()
        bW = np.hstack([b[0],W[0]])
        Wl = ['b']+['W'+str(i) for i in range(1,len(W[0])+1)]
        #ax.bar(np.arange(len(bW))*2,bW)
        ax.stem(np.arange(len(bW)),bW)
        ax.set_xticks(np.arange(len(bW)))
        ax.set_xticklabels(Wl)#np.arange(len(bW))
        ax.text(len(bW)*0.4,np.max(bW)*0.5,r'$\hat y = \sigma(Wx+b)$',fontsize=16,alpha=0.5)
        ax.grid(alpha=0.5)
        #ax.set_ylim([None,np.max(bW)*1.5])
        ax.set_ylabel('Weights')
        #plt.show()
    def Wplot2(self,ax=None):
        if ax is None: fig, ax = plt.subplots(111)
        W,b = self.getWeights()
        bW = np.hstack([b[0],W[0]])
        Wl = ['b']+['W'+str(i) for i in range(1,len(W[0])+1)]
        ax.matshow([bW],cmap="YlGn",origin='lower')
        ax.set_xticks(np.arange(len(bW)))
        ax.set_xticklabels(Wl)#np.arange(len(bW))
        ax.set_yticklabels([])
        ax.text(len(bW)*0.4,-1,r'$\hat y = \sigma(Wx+b)$',fontsize=16)
        for i in range(len(bW)):
            ax.text(i-0.2,0,str(np.around(bW[i],3)))


class LogisticRegression():
    def  __init__(self,alpha=0.01,lambd=0,polyfit=False,degree=3,FeatureNormalize=False,penalty='l2',
                  tol=0.01,rho=0.9,C=1.0,fit_intercept=True):

        '''
        fitting a linear model
        for binary class::  yp = sigmoid(XW + b)
        for multiclass  ::  yp = softmax(XW + b)

        Optimize W and b, using gradient decent and regularize with 'l1','l2' or 'elastic-net' penalty

        Parameters
        ----------
        fit_intercept: bool(default - True), if use intercept (b) to fit, if false, b = 0
        alpha : Learning rate, the rate to update the weights at each iteration
        penalty : {'l1', 'l2, 'elastic-net', 'none'}, default='l2'
                regularization 'l1' --> lambd*|W|, 'l2' --> lambd*|W|**2, 'elastic-net' - lambd*(rho*|W|^2 + (1-rho)*|W|))
        lambd : penalty value (default =0)
        rho   : used for elastic-net penalty
        tol : float, default=1e-4 - Tolerance for stopping criteria.
        polyfit: if polynomial features to be used, default = False
        degree : degree of polynomial features, used if polyfit is true, default =2
        FeatureNormalize: default =False, if true, before fitting and after polyfit, features are normalized
                and computed mean and std is saved to be used while prediction

        Methods
        -------
        fit(X,y,max_itr = -1,verbose=0):
            optimize W and b, for given X and y until cost J<tol or max_itr are over. if max_itr =-1, it is set to 10000,
           verbose: 0 silent, 1, print cost, 11, debug mode

        predict(X): Estimate class label, return - (n,) for input of X (n,nf)
        predict_proba(X): Estimate probability of class labels, return  -(n,nclass)
        getWeights(): return weights, W,b
        plot_lcurve():  plot learning curve, cost vs iteration, used to diagnosis to tune learning rate alpha
        plot_weights(): plot weights as line,
        plot_weights2(): plot weights as heatmap
        plot_boundries(X,y), plot decision boundries, only for 2D data, (nf=2), before polyfit.


        Attributes
        ----------
        nclass: ndarray of shape (n_classes, )
            A list of class labels known to the classifier.
        W: ndarray of shape (1, n_features) or (n_classes, n_features)
            Coefficient of the features in the decision function.
        b: intercept weight


        Examples
        --------
        # Binary Class ------
        N = 300
        np.random.seed(1)
        X = np.random.randn(N,2)
        y = np.random.randint(0,2,N)
        y.sort()

        X[y==0,:]+=2 # just creating classes a little far
        print(X.shape, y.shape)

        clf = LogisticRegression(alpha=0.1)
        clf.fit(X,y,max_itr=1000)
        yp  = clf.predict(X)
        ypr = clf.predict_proba(X)
        print('Accuracy : ',np.mean(yp==y))
        print('Loss     : ',clf.Loss(y,ypr))

        #----
        Accuracy :  0.96
        Loss     :  0.07046678918015
        #---

        clf.plot_Lcurve()
        clf.plot_weights()
        clf.plot_weights2()
        clf.plot_boundries(X,y,alphaP=1)


        # Multi Class ------

        N =300
        X = np.random.randn(N,2)
        y = np.random.randint(0,3,N)
        y.sort()

        X[y==0,1]+=3
        X[y==2,0]-=3
        print(X.shape, y.shape)

        plt.plot(X[y==0,0],X[y==0,1],'.b')
        plt.plot(X[y==1,0],X[y==1,1],'.r')
        plt.plot(X[y==2,0],X[y==2,1],'.g')
        plt.show()

        clf = LogisticRegression(alpha=0.1,polyfit=True,degree=3,lambd=0,FeatureNormalize=True)
        clf.fit(X,y,max_itr=1000)
        yp  = clf.predict(X)
        ypr = clf.predict_proba(X)
        print('Accuracy : ',np.mean(yp==y))
        print('Loss     : ',clf.Loss(clf.oneHot(y),ypr))

        clf.plot_Lcurve()
        clf.plot_weights()
        clf.plot_weights2()
        clf.plot_boundries(X,y,alphaP=1)

        '''

        self.fit_intercept = fit_intercept

        self.penalty = penalty
        self.rho   = rho
        self.tol   = tol
        self.alpha = alpha
        self.lambd = lambd

        self.cost  = []
        self.itr   = 0

        self.polyfit = polyfit
        self.degree  = degree

        self.FeatureNormalize = FeatureNormalize
        self.mu,self.sig = None,None

        self.trainded = False
        self.C = C # for future use

    def __repr__(self):
        #info = 'LogisticRegression(' +\
        #       f"alpha={self.alpha},lambd={self.lambd},polyfit={self.polyfit}," +\
        #       f"degree={self.degree},FeatureNormalize={self.FeatureNormalize},\n\t penalty={self.penalty},"+\
        #       f"tol={self.tol},rho={self.rho},C={self.C},fit_intercept={self.fit_intercept})"

        info = "LogisticRegression(" +\
               "alpha={},lambd={},polyfit={},".format(self.alpha,self.lambd,self.polyfit) +\
               "degree={},FeatureNormalize={},\n\t penalty={},".format(self.degree,self.FeatureNormalize,self.penalty) +\
               "tol={},rho={},C={},fit_intercept={})".format(self.tol,self.rho,self.C,self.fit_intercept)
        return info

    def fit(self,X,y,max_itr=-1,verbose=0,printAt=100,warm=True):
        '''
        X : ndarray : (N,nf),
        y : ndarray: int (N,)  y \in [0,1,...]
        max_itr: int, number of iteration, if -1, set to 10000
        verbose: int,0, silent, 1, minimum verbosity, 11, debug mode
        warm: bool, if false, weights will be reinitialize, else, last trained weights will be used
        '''


        if verbose>10: print('Verbosity - Debug mode')

        self.nclass = len(set(y))

        if self.nclass>2: y = self.oneHot(y)

        if y.ndim==1: y = y[:,None]

        self.printAt = printAt
        if self.polyfit: X = self.PolyFeature(X,degree = self.degree)
        if self.FeatureNormalize: X = self.Normalize(X)

        N,nf = X.shape
        if verbose>10: print('X',N,nf)
        if warm and not(self.trainded):
            nW = self.nclass if self.nclass>2 else 1
            self.W = np.random.randn(nW,nf)*np.sqrt(2.0/N)
            self.b = 0 if self.nclass<3 else np.zeros(self.nclass)

        if verbose>10: print(X.shape,self.W.shape)
        if max_itr==-1: max_itr=100000
        for i in range(max_itr):
            self.itr += 1
            if verbose>10: print('1')
            # Compute output
            Z  = np.dot(X,self.W.T) + self.b

            if verbose>10: print('Z',Z.shape)

            yp = self.softmax(Z) if Z.shape[1]>1 else self.sigmoid(Z)

            if verbose>10: print('yp',yp.shape)
            # Error
            err = yp - y
            if verbose>10: print('error',err.shape,'#None',np.sum(np.isnan(err)))
            # Regularization
            regW = self.regularization(self.W)

            if verbose>10: print('reg',regW)

            # Derivatives
            dW = np.dot(err.T,X) + regW
            db = np.sum(err) if self.nclass<2 else np.sum(err,axis=0)

            if verbose>10:print('dW',dW,db)
            # Update weights
            self.W -= self.alpha*dW/N
            if self.fit_intercept:
                self.b -= self.alpha*db/N

            #print(self.W,self.b)

            # Cost
            J = self.Loss(y,yp) + np.sum(np.abs(regW))/N
            if verbose>10:print('Cost J',J)
            self.cost.append(J)
            if verbose and i%self.printAt==0:
                print('Epoc @ ',self.itr,' Cost ',J)

            if J<self.tol: break

        if J>self.tol:
            if verbose: print('Cost did not reduce to ' +st(self.tol)+', try incearsing number of iteration or tolarance threshold')
        self.trainded = True

    def PolyFeature(self,X,degree=2):
        if degree>1:
            Xp = np.c_[X,X**2,np.prod(X,axis=1)]
            if degree>2:
                for d in range(3,degree+1):
                    Xp = np.c_[Xp,X**d,np.prod(Xp,axis=1)]
        else:
            return X
        return Xp

    def regularization(self,W):
        regW =0*W
        if (self.lambd>0) and self.penalty is not None:
            if self.penalty == 'l2':
                regW = self.lambd*W
            elif self.penalty=='l1':
                dw = 2*(W>0)-1
                regW = self.lambd*dw
            elif self.penalty=='elastic-net':
                dw = 2*(W>0)-1
                regW = self.lambd*((1.0-self.rho)*W+self.rho*dw)
        return regW

    def Normalize(self,X):
        if self.mu is None:
            self.mu  = np.mean(X,axis=0)#[None,:]
            self.sig = np.std(X,axis=0)#[None,:]
        Xn = (X-self.mu)/self.sig
        return Xn

    def Loss(self,y,yp):
        J = -np.mean(y*np.log(yp+1e-20)+(1-y)*np.log(1-yp+1e-20))/2
        #J = J/(2.0*yp.shape[1])
        return J

    def sigmoid(self,z):
        return 1.0/(1+np.exp(-z))

    def softmax(self,z):
        ze = np.exp(z - np.max(z,axis=1)[:,None])
        zs = ze/(ze.sum(1)[:,None] + 1e-10)
        assert np.sum(np.isnan(zs))==0
        return zs

    def predict(self,X):
        if self.polyfit: X = self.PolyFeature(X,degree=self.degree)
        if self.FeatureNormalize: X = self.Normalize(X)
        Z = np.dot(X,self.W.T) + self.b
        yp = self.softmax(Z) if Z.shape[1]>1 else self.sigmoid(Z)
        yp = np.argmax(yp,axis=1) if yp.shape[1]>1 else np.squeeze(1*(yp>0.5))
        #yp = np.squeeze(yp)
        return yp

    def predict_proba(self,X):
        if self.polyfit: X = self.PolyFeature(X,degree=self.degree)
        if self.FeatureNormalize: X = self.Normalize(X)
        Z = np.dot(X,self.W.T) + self.b
        yp = self.softmax(Z) if Z.shape[1]>1 else np.squeeze(self.sigmoid(Z))
        #yp = np.squeeze(yp)
        return yp

    def getWeights(self):
        return self.W,self.b

    def getWeightsAsList(self):
        W,b = self.getWeights()
        if W.shape[1]>1:
            bW = np.c_[b,W].reshape(-1)
            Wl =[]
            for j in range(W.shape[0]):
                Wl = Wl + [r'$b_'+str(j)+'$']+[r'$W_{'+str(j)+str(i)+'}$'  for i in range(1,len(W[0])+1)]
        else:
            #bW = np.c_[b,W][0]
            bW = np.hstack([b,W[0]])
            Wl = ['b']+['W'+str(i) for i in range(1,len(W[0])+1)]
        return bW, Wl

    def plot_boundries(self,X,y,ax=None,density =500,hardbound=False,alphaP=1,alphaB=1):
        assert X.shape[1]==2
        # Only can be plotted for 2D data

        if ax is None: fig, ax = plt.subplots()
        x1mn = np.min(X[:,0])
        x2mn = np.min(X[:,1])
        x1mx = np.max(X[:,0])
        x2mx = np.max(X[:,1])
        x1 = np.linspace(x1mn,x1mx,density)
        x2 = np.linspace(x2mn,x2mx,density)
        xv, yv = np.meshgrid(x1, x2)
        Xall = np.c_[xv.flatten(),yv.flatten()]
        #print(Xall.shape)
        yp = self.predict_proba(Xall)
        if yp.ndim>1 and yp.shape[1]>1:
            yp = np.argmax(yp,axis=1)
            hardbound = False

        yp = yp.reshape(density,density)

        self.yp=yp

        if hardbound:
            ax.imshow(yp>0.5, origin='lower', interpolation='bicubic',alpha=alphaB,
                      extent=(xv.min(), xv.max(), yv.min(), yv.max()))
        else:
            ax.imshow(yp, origin='lower', interpolation='bicubic',alpha=alphaB,
                      extent=(xv.min(), xv.max(), yv.min(), yv.max()))


        for c in list(set(y)):
            in0 = np.where(y==c)[0]
            ax.plot(X[in0,0],X[in0,1],'.',alpha=alphaP)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title('Decision Boundry')
        #plt.show()

    def plot_Lcurve(self,ax=None):
        if ax is None: fig, ax = plt.subplots()
        ax.plot(range(self.itr),self.cost)
        ax.set_xlabel('iteration')
        ax.set_ylabel('Cost')
        ax.set_title('Learning Curve')
        ax.grid(True)
        #plt.show()

    def plot_weights(self,ax=None):
        if ax is None: fig, ax = plt.subplots()
        bW, Wl = self.getWeightsAsList()
        #ax.bar(np.arange(len(bW))*2,bW)
        if 'use_line_collection' in self.getArg_(ax.stem):
            ax.stem(np.arange(len(bW)),bW,use_line_collection=True)
        else:
            ax.stem(np.arange(len(bW)),bW)

        ax.set_xticks(np.arange(len(bW)))
        ax.set_xticklabels(Wl)#np.arange(len(bW))
        ax.text(len(bW)*0.4,np.max(bW)*0.5,r'$\hat y = \sigma(XW+b)$',fontsize=16,alpha=0.5)
        ax.grid(alpha=0.5)
        #ax.set_ylim([None,np.max(bW)*1.5])
        ax.set_ylabel('Weights')
        #plt.show()
    def plot_weights2(self,ax=None,fontsize=10,grid=True):
        if grid and self.W.shape[1]>1:
            self.plot_weightMatrix(ax=ax,fontsize=fontsize)
        else:
            if ax is None: fig, ax = plt.subplots()
            bW, Wl = self.getWeightsAsList()
            ax.matshow([bW],cmap="YlGn",origin='lower')
            ax.set_xticks(np.arange(len(bW)))
            ax.set_xticklabels(Wl,fontsize=fontsize)#np.arange(len(bW))
            ax.set_yticklabels([])
            ax.text(len(bW)*0.4,-1,r'$\hat y = \sigma(XW+b)$',fontsize=fontsize)
            for i in range(len(bW)):
                ax.text(i-0.2,0,str(np.around(bW[i],3)),fontsize=fontsize)

    def plot_weightMatrix(self,ax=None,fontsize=10):
        W,b = self.getWeights()
        bW = np.c_[b,W].T
        if ax is None: fig, ax = plt.subplots()

        ax.matshow(bW, cmap="YlGn")
        for (i, j), z in np.ndenumerate(bW):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

        ax.set_xticks(np.arange(bW.shape[1]))
        ax.set_yticks(np.arange(W.shape[1]+1))
        ax.set_yticklabels(['$b$']+[r'$W_'+str(j)+'$' for j in range(W.shape[1])])

    def getArg_(self,fun):
        import inspect
        args = list(inspect.getfullargspec(fun))
        argsi = []
        for arg in args:
            if arg is not None:
                if type(arg)==list:
                    argsi.extend(arg)
                elif type(arg)==dict:
                    argsi.extend(list(arg.keys()))
                elif type(arg)==str:
                    argsi.append(arg)
        return list(set(argsi))

    def oneHot(self,y):
        assert y.ndim==1
        assert set(y)==set(range(len(set(y))))
        class_labels = list(set(y))
        nclass = len(class_labels)
        class_labels.sort()
        if nclass>2:
            yb = np.zeros([y.shape[0],nclass])
            for c in class_labels:
                yb[y==c,c]=1
        else:
            yb = y
        return yb
