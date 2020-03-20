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

class NaiveBayes():
    """
    The Gaussian Naive Bayes classifier.
    Based on the bayes rule

    X::shape (n, nf), n samples with nf features
    y::shape (n,) or (n,1)

    """
    def __init__(self):
        self.classes = None
        self.X = None
        self.y = None
        self.class_labels  = None
        self.feature_names = None
        # Parameters :: mu, sigma (mean and variance) and prior of each class
        self.parameters = dict()

    def __repr__(self):
        info = "NaiveBayes()"
        return info

    def fit(self, X, y):
        assert X.shape[0]==y.shape[0]
        assert len(y.shape)==1 or y.shape[1]==1
        self.n, self.nf = X.shape
        self.X = X
        self.y = y
        self.classes = np.unique(y)

        # Calculate the mean and variance of each feature for each class
        # Calculate the prior for each class c
        # P(Y) = number of samples in class c / total numberof samples
        for c in self.classes:
            param = {}
            param["mu"]    = X[np.where(y==c)].mean(0)
            param["sig"]   = X[np.where(y==c)].var(0) + 1e-10
            param["prior"] = X[np.where(y==c)].shape[0]/self.n
            self.parameters[c]  = param

    # Gaussian probability distribution
    # P(xi) ~  N(mu, sig)
    def _Pxy(self, mu, sig, x):
        a = (1.0 / (np.sqrt((2.0 * np.pi) * sig)))
        pxy = a*np.exp(-(((x - mu)**2) / (2 * sig)))
        return pxy

    # Classify using Bayes Rule, P(Y|X) = P(X|Y)*P(Y)/P(X)
    # P(X|Y) - Probability. Gaussian distribution (fun _Pxy)
    # P(Y)   - Prior
    # P(X)   - Scales the posterior to the range 0 - 1
    # P(Y|X) - (posterior)
    # Classify the sample as the class that results in the largest
    def _Pyx(self, xi):
        Pyx = []
        # Go through list of classes
        for c in self.classes:
            Pyc  = self.parameters[c]["prior"]
            mu   = self.parameters[c]["mu"]
            sig  = self.parameters[c]["sig"]
            Pxyc = self._Pxy(mu, sig, xi).prod(axis=-1)
            Pyxc = Pyc*Pxyc + 1e-10   # adding small value to avoid all zero
            Pyx  = Pyxc if c==self.classes[0] else np.c_[Pyx,Pyxc]
        #Normalizing
        Pyx /= Pyx.sum(-1)[None].T
        return Pyx

    # Predict the class labels corresponding to the
    # samples in X
    def predict(self, X):
        Pyx = self._Pyx(X)
        return self.classes[Pyx.argmax(-1)]
    def predict_prob(self,X):
        return self._Pyx(X)

    def set_class_labels(self,labels):
        assert len(labels)==len(self.classes)
        self.class_labels = labels

    def set_feature_names(self,fnames):
        assert len(fnames)==self.nf
        self.feature_names = fnames

    def _getPDF(self,mean,var,imin,imax,points=1000):
        xi  = np.linspace(imin,imax,points)
        a  = (1.0 / (np.sqrt((2.0 * np.pi) * var)))
        px = a*np.exp(-(((xi - mean)**2) / (2 * var)))
        return px,xi
    def VizPx(self,nfeatures = None):
        if self.class_labels is None:
            self.class_labels  = ['C'+str(c) for c in self.classes]
        if self.feature_names is None:
            self.feature_names = ['f'+str(i+1) for i in range(self.nf)]

        if nfeatures is None:
            ngrid = int(np.ceil(np.sqrt(self.nf)))
            NF = list(range(self.nf))
        else:
            ngrid = int(np.ceil(np.sqrt(len(nfeatures))))
            NF = list(nfeatures)

        mn = self.X.min(0)
        mx = self.X.max(0)

        for j in NF:
            plt.subplot(ngrid,ngrid,j+1-NF[0])
            for i in range(len(self.classes)):
                c = self.classes[i]
                imin, imax = mn[j],mx[j]
                imin -= 0.2*imin
                imax += 0.2*imax
                imean = self.parameters[c]['mu'][j]
                ivar  = self.parameters[c]['sig'][j]
                Px,xi  = self._getPDF(imean,ivar,imin,imax)
                plt.plot(xi,Px/Px.sum(), label=self.class_labels[i])
                plt.xlabel(self.feature_names[j])
                plt.ylabel(r'P(x)')
                plt.xlim([xi[0],xi[-1]])
                if j+1-NF[0]==ngrid: plt.legend(bbox_to_anchor=(1.05,1),loc = 2)
                plt.grid(alpha=0.5)
                plt.tight_layout()
                plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))

        plt.show()
