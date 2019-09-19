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
