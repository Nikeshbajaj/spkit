'''
@Author: Nikesh Bajaj
@Homepage: http://nikeshbajaj.in
@Email: n.bajaj@qmul.ac.uk, nikkeshbajaj@gmail.com
---
Fork it, use it, share it, cite it
and if you add extra functionalities, let me know. I would update it.

https://doi.org/10.6084/m9.figshare.7797095.v1
Bajaj, Nikesh (2019): Decision Tree with Visualsation. figshare. Code.

'''
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



#Libraries
import numpy as np
import matplotlib.pyplot as plt
import copy
#import time


# Super class for Classification and Regression
class DecisionTree(object):
    '''Super class of RegressionTree and ClassificationTree.

    '''
    def __init__(self, min_samples_split=2, min_impurity=1e-7, max_depth=float("inf"), thresholdFromMean=False):
        '''
        Optimizing depth:: In this version, a very large value can be used to build a tree and later can be shrink to lower depth (d),
                    using ".updateTree(shrink=True, max_depth=d)". The optimal value of depth for given data can be choosing by analysing
                    learning curve using ".getLcurve" method and/or "plotTree(Measures=True)"
        Parameters:
        -----------
        max_depth ::int:>0, maximum depth to go for tree, default is Inf, which leads to overfit
                            decrease the max depth to reduce the overfitting.
        min_samples_split::int: minimum number of samples to split further
        min_impurity     ::float: minimum impurity (or gain) to split

        thresholdFromMean::bool, if threshold is selcted from mean of two
                           concecutive unique values of selected a an unique value of feaure.
                           Only applicable to float or int type features, not to catogorical type.
                           default is False.
        '''

        self.tree = None  # Root node in dec. tree

        # Minimum n of samples to split
        self.min_samples_split = min_samples_split

        # The minimum impurity to split
        self.min_impurity = min_impurity

        # The maximum depth to grow the tree to

        if max_depth<1:
            print('Maximimum depth should start with 1, in version 0.0.6, max_depth could be 0, which is equivalent to 1 in this version')
            assert max_depth>=1
        self.max_depth = max_depth
        self.Lcurve ={}
        # if threshold is consider from unique values of middle of two unique values
        # Not applicable to catogorical feature
        self.thresholdFromMean =thresholdFromMean

        self.trained = False

        # Variables that comes from SubClass
        self.verbose = None
        self.feature_names = None
        self.randomBranch=None
        self.__author__ = 'Nikesh Bajaj'
        # Function to calculate impurity (classif.=>info gain, regr=>variance reduct.)
        self._impurity_calculation = None

        # Function to determine prediction of y at leaf
        self._leaf_value_calculation = None

    def fit(self, X, y):
        '''
        Building a tree and saving in an dictionary at self.tree
        Parameters:
        -----------
              X:: ndarray (number of sample, number of features)
              y:: list of 1D array
        verbose::0(default)-no progress or tree
               ::1 - show progress
               ::2 - show tree
        feature_names:: (optinal) list, Provide for better look at tree while plotting or shwoing the progress,
                       default to None, if not provided, features are named as f1,...fn.

        '''
        self.nfeatures = X.shape[1]
        self.one_dim = len(np.shape(y)) == 1
        #self.labels = list(set(y))
        #self.labels.sort()

        #self.verbose = verbose
        self.branch = None
        #self.randomBranch = randomBranch
        if self.verbose ==4: self.fig, self.ax =plt.subplots(1,1)
        self.cpath = ''
        self.space='->'
        self.FastPlot =True
        if self.verbose>0:
            print('Number of features::',self.nfeatures)
            print('Number of samples ::',X.shape[0])
        self.set_featureNames(feature_names=self.feature_names)
        if self.verbose>0:
            print('---------------------------------------')
            print('|Building the tree.....................')
        self.tree = self._build_tree(X, y)
        self.tree = self.pruneTree(self.tree)
        self.tree = self.pruneTree(self.tree)
        self.tree = self.pruneTree(self.tree)
        if self.verbose:
            print('|\n|.........................tree is buit!')
            print('---------------------------------------')
        self.trained = True
    def _build_tree(self, X, y, current_depth=1):
        ''' Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data'''


        # if not spliting
        y_temp = copy.copy(y)
        est_value_temp = self._leaf_value_calculation(y_temp)
        label_prob_temp, label_counts_temp = self._leaf_prob_calculation(y_temp)

        largest_impurity = 0
        best_criteria = None    # Feature index and threshold
        best_sets = None        # Subsets of the data

        # Check if expansion of y is needed
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        # Add y as last column of X
        Xy = np.c_[X, y]

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # Calculate the impurity for each feature
            for feature_i in range(n_features):
                # iterate over all values of feature_i
                pf1 = (feature_i+1)/n_features

                if self.verbose==2: status1 = '|'+self.space[:-1]+'Feature::'+str(feature_i+1)+\
                    '_'+self.feature_names[feature_i]

                if self.verbose==1:
                    print('|subtrees::|'+str(int(pf1*100))+'%|'+'-'*int(pf1*20)+'>'+\
                               '.'*(20-int(pf1*20))+['|\\','|-','|/','||'][feature_i%4],end='\r',flush=True)

                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.sort(np.unique(feature_values))
                if self.thresholdFromMean and (isinstance(unique_values[0], int) or isinstance(unique_values[0], float)):
                    unique_values = (unique_values[:-1]+unique_values[1:])/2


                # Iterate through all unique values of feature column i and
                # calculate the impurity
                for threshold in unique_values:
                    # Divide X and y depending on if the feature value of X at index feature_i
                    # meets the threshold
                    tXy, fXy = self.splitX(Xy, feature_i, threshold)

                    if len(tXy) > 0 and len(fXy) > 0:
                        # Select the y-values of the two sets
                        ty = tXy[:, n_features:]
                        fy = fXy[:, n_features:]

                        # Calculate impurity
                        impurity = self._impurity_calculation(y, ty, fy)

                        # If this threshold resulted in a higher information gain than previously
                        # recorded save the threshold value and the feature
                        # index
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            thr_str = np.around(threshold,2) if isinstance(threshold, int) or isinstance(threshold, float) else str(threshold)
                            if self.verbose: status3 = ' Gain::'+str(np.around(impurity,2))+\
                            ' thr::'+str() + '_Depth = '+str(current_depth)+'   '

                            #est_value_temp,label_prob_temp,label_counts_temp
                            node ={'feature_index':feature_i,"threshold": threshold,
                                   'feature_name':self.feature_names[feature_i],
                                   'impurity':impurity,
                                   'value':None,"leaf": False,
                                   'label_counts':label_counts_temp,'proba':label_prob_temp,
                                   'est_value':est_value_temp,'depth':current_depth-1}

                            sets = {"tX": tXy[:, :n_features],
                                    "ty": tXy[:, n_features:],
                                    "fX": fXy[:, :n_features],
                                    "fy": fXy[:, n_features:]}
                            if self.verbose==2:
                                print(status1+status3,end='\r',flush=True)

        if largest_impurity > self.min_impurity:
            # Build subtrees for the right and left branches
            #self.cpath = self.cpath[:current_depth]
            if self.branch:
                #self.cpath = self.cpath +'T'
                S = list(self.cpath) +['']*(current_depth+1)
                S[current_depth-1] ='T'
                self.cpath = ''.join(S)

            elif self.branch==False:
                #self.cpath = self.cpath +'F'
                S = list(self.cpath) +['']*(current_depth+1)
                S[current_depth-1] ='F'
                self.cpath = ''.join(S)


            self.cpath =self.cpath[:current_depth]

            if self.verbose==3: print(self.branch,current_depth,'|',self.cpath)

            rd = np.random.random(1)[0] if self.randomBranch or self.verbose==2 else 1



            if rd>0.5:

                if self.verbose==2: print('|\n|'+self.space+'True branch (>>>)..')

                self.branch = True
                self.space = '-'+self.space
                tBranch = self._build_tree(sets["tX"], sets["ty"], current_depth + 1)

                if len(self.space)>1: self.space  = self.space[1:]

                if self.verbose==2: print('|\n|'+self.space+'False branch (<<<)..')

                self.branch = False
                fBranch = self._build_tree(sets["fX"], sets["fy"], current_depth + 1)

            else:
                if self.verbose==2: print('|\n|'+self.space+'False branch (<<<)..')

                self.branch = False
                fBranch = self._build_tree(sets["fX"], sets["fy"], current_depth + 1)

                if self.verbose==2: print('|\n|'+self.space+'True branch (>>>)..')

                self.branch = True
                self.space = '-'+self.space
                tBranch = self._build_tree(sets["tX"], sets["ty"], current_depth + 1)

                if len(self.space)>1: self.space  = self.space[1:]



            node['T'] =tBranch
            node['F'] =fBranch

            if self.verbose==4: self.plotTreePath(self.cpath,ax=self.ax,fig=self.fig)
            return node

        #Leaf Node
        leaf_value = self._leaf_value_calculation(y)
        #self.task = 'classification'
        leaf_prob, leaf_label_counts = self._leaf_prob_calculation(y)
        #node ={'feature_index':None,"threshold": None,
        #       'feature_name' :None,'impurity':None,
        #       'value':leaf_value,"leaf": True, 'T':None,'F':None}

        node ={'value':leaf_value,"leaf": True,'proba':leaf_prob,'label_counts':leaf_label_counts,'est_value':leaf_value,'depth':current_depth-1}


        if self.branch:
            S = list(self.cpath) +['']*(current_depth+1)
            S[current_depth-1] ='T'
            self.cpath = ''.join(S)

        elif self.branch==False:
            S = list(self.cpath) +['']*(current_depth+1)
            S[current_depth-1] ='F'
            self.cpath = ''.join(S)

        self.cpath =self.cpath[:current_depth]

        if self.verbose==2: print('|'+self.space+'{Leaf Node:: value:',leaf_value,'}_Depth ='+\
                                  str(current_depth)+'  \n')

        elif self.verbose==3: print(self.branch,current_depth,'|',self.cpath)

        elif self.verbose==4: self.plotTreePath(self.cpath,ax=self.ax,fig=self.fig)

        return node
    def _predict_value(self, x, tree=None,path=''):
        ''' Do a recursive search down the tree and make a prediction of the data sample by the
            value of the leaf that we end up at '''

        # check if sample has same number of features
        assert len(x)==self.nfeatures

        if tree is None:
            tree = self.tree

        # If it is a leaf node, return value as the prediction
        if tree['leaf']:
            return tree['value'],path

        else:
            # Choose the feature that we will test
            fvalue = x[tree['feature_index']]


        # Determine if we will follow left or right branch
        branch = tree['F']
        ipath = 'F'
        if isinstance(fvalue, int) or isinstance(fvalue, float):
            if fvalue >= tree['threshold']:
                branch = tree['T']
                ipath = 'T'
        elif fvalue == tree['threshold']:
            branch = tree['T']
            ipath = 'T'

        path = path +ipath
        # Test subtree
        return self._predict_value(x, branch,path)
    def _predict_value_depth(self, x, tree=None,path='',depth=0,max_depth=np.inf):
        ''' Do a recursive search down the tree and make a prediction of the data sample by the
            value of the leaf that we end up at '''

        # check if sample has same number of features
        assert len(x)==self.nfeatures

        if tree is None:
            tree = self.tree

        # If it is a leaf node, return value as the prediction
        if tree['leaf'] or depth>=max_depth:
            return tree['est_value'],path

        else:
            # Choose the feature that we will test
            fvalue = x[tree['feature_index']]
        # Determine if we will follow left or right branch
        branch = tree['F']
        ipath = 'F'
        if isinstance(fvalue, int) or isinstance(fvalue, float):
            if fvalue >= tree['threshold']:
                branch = tree['T']
                ipath = 'T'
        elif fvalue == tree['threshold']:
            branch = tree['T']
            ipath = 'T'

        path = path +ipath
        # Test subtree
        return self._predict_value_depth(x, branch,path,depth=depth+1,max_depth=max_depth)
    def _predict_proba(self, x, tree=None,path=''):
        ''' Do a recursive search down the tree and make a prediction of the data sample by the
            value of the leaf that we end up at '''

        # check if sample has same number of features
        assert len(x)==self.nfeatures

        if tree is None:
            tree = self.tree

        # If it is a leaf node, return value as the prediction
        if tree['leaf']:
            return tree['proba'],tree['label_counts'],path

        else:
            # Choose the feature that we will test
            fvalue = x[tree['feature_index']]


        # Determine if we will follow left or right branch
        branch = tree['F']
        ipath = 'F'
        if isinstance(fvalue, int) or isinstance(fvalue, float):
            if fvalue >= tree['threshold']:
                branch = tree['T']
                ipath = 'T'
        elif fvalue == tree['threshold']:
            branch = tree['T']
            ipath = 'T'

        path = path +ipath
        # Test subtree
        return self._predict_proba(x, branch, path)
    def _predict_proba_depth(self, x, tree=None,path='',depth=0,max_depth=np.inf):
        ''' Do a recursive search down the tree and make a prediction of the data sample by the
            value of the leaf that we end up at '''

        # check if sample has same number of features
        assert len(x)==self.nfeatures

        if tree is None:
            tree = self.tree

        # If it is a leaf node, return value as the prediction
        if tree['leaf'] or depth>=max_depth:
            return tree['proba'],tree['label_counts'],path

        else:
            # Choose the feature that we will test
            fvalue = x[tree['feature_index']]


        # Determine if we will follow left or right branch
        branch = tree['F']
        ipath = 'F'
        if isinstance(fvalue, int) or isinstance(fvalue, float):
            if fvalue >= tree['threshold']:
                branch = tree['T']
                ipath = 'T'
        elif fvalue == tree['threshold']:
            branch = tree['T']
            ipath = 'T'

        path = path +ipath
        # Test subtree
        return self._predict_proba_depth(x, branch, path,depth=depth+1,max_depth=max_depth)
    def predict_(self, X,treePath=False):
        '''Classify samples one by one and return the set of labels '''

        if treePath:
            y_pred = np.array([list(self._predict_value(x)) for x in X])
        else:
            y_pred = np.array([self._predict_value(x)[0] for x in X])

        return y_pred
    def predict(self, X,treePath=False,max_depth=np.inf):
        '''
        USE "max_depth" to limit the depth of tree for expected values

        Compute expected value for each sample in X, up till depth=max_depth of the tree

        For classification:
        Expected value is a label with maximum probabilty among the samples at leaf nodes.
        For probability and count of each labels at leaf node, use ".predict_proba" method

        For Regression:
        Expected value is a mean value of smaples at the leaf node.
        For Standard Deviation and number of samples at leaf node, use ".predict_proba" method
        '''

        if treePath:
            y_pred = np.array([list(self._predict_value_depth(x,max_depth=max_depth)) for x in X])
        else:
            y_pred = np.array([self._predict_value_depth(x,max_depth=max_depth)[0] for x in X])
        return y_pred
    def predict_proba_(self, X,label_counts=False,treePath=False):
        '''Compute probability of samples one by one and return the set of labels'''

        y_prob   = np.array([self._predict_proba(x)[0] for x in X]).astype(float)

        if label_counts and treePath:
            y_counts = [self._predict_proba(x)[1] for x in X]
            y_paths  = [self._predict_proba(x)[2] for x in X]
            return y_prob,y_counts,y_paths
        elif label_counts:
            y_counts = [self._predict_proba(x)[1] for x in X]
            return y_prob,y_counts
        elif treePath:
            y_paths  = [self._predict_proba(x)[2] for x in X]
            return y_prob,y_paths
        else:
            return y_prob
    def predict_proba(self, X,label_counts=False,treePath=False,max_depth=np.inf):
        '''
        USE "max_depth" to limit the depth of tree for expected values
        Compute probabilty/SD for labeles at the leaf till max_depth level, for each sample in X

        For classification:
        Returns the Probability of samples one by one and return the set of labels
        label_counts=True: Includes in the return, the counts of labels at the leaf

        For Regression:
        Returns the standard deviation of values at the leaf node. Mean value is returened with ".predice()" method
        label_counts=True: Includes in the return, the number of samples at the leaf

        treePath=True: Includes the path of tree for each sample as string
        '''

        y_prob   = np.array([self._predict_proba_depth(x,max_depth=max_depth)[0] for x in X]).astype(float)

        if label_counts and treePath:
            y_counts = [self._predict_proba_depth(x,max_depth=max_depth)[1] for x in X]
            y_paths  = [self._predict_proba_depth(x,max_depth=max_depth)[2] for x in X]
            return y_prob,y_counts,y_paths
        elif label_counts:
            y_counts = [self._predict_proba_depth(x,max_depth=max_depth)[1] for x in X]
            return y_prob,y_counts
        elif treePath:
            y_paths  = [self._predict_proba_depth(x,max_depth=max_depth)[2] for x in X]
            return y_prob,y_paths
        else:
            return y_prob
    def get_tree(self):
        if self.trained:
            return self.tree
        else:
            print("No tree found, haven't trained yet!!")
    def set_featureNames(self,feature_names=None):
        if feature_names is None or len(feature_names)!=self.nfeatures:
            if self.verbose: print('setting feature names to default..f1, f2....fn')
            self.feature_names = ['f'+str(i) for i in range(1,self.nfeatures+1)]
        else:
            self.feature_names = feature_names
    def splitX(self,X,ind,threshold):
        if isinstance(threshold, int) or isinstance(threshold, float):
            X1 = X[X[:,ind]>=threshold,:]
            X2 = X[X[:,ind]<threshold,:]
        else:
            X1 = X[X[:,ind]==threshold,:]
            X2 = X[X[:,ind]!=threshold,:]
        return X1,X2
    def pruneTree(self,DT):
        if not DT['leaf']:
            if DT['T']['leaf'] and DT['F']['leaf']:
                if DT['T']['value']==DT['F']['value']:
                    c_depth = DT['depth']
                    #print('same valued leafs')
                    if self.task =='classification':
                        T_label_counts = DT['T']['label_counts']
                        F_label_counts = DT['F']['label_counts']
                        assert set(T_label_counts.keys())==set(F_label_counts.keys())==set(self.targets)
                        leaf_label_counts = {lbl:T_label_counts[lbl]+F_label_counts[lbl] for lbl in self.targets}
                        frq = [leaf_label_counts[lbl] for lbl in self.targets]
                        leaf_prob   = list(np.array(frq)/np.sum(frq))

                        leaf_value = self.targets[np.argmax(leaf_prob)]

                        # just checking !!!!
                        if leaf_value!=DT['est_value']:
                            print('New estimated value is not same as old !!!\n',leaf_value, DT['est_value'])


                    else:
                        n1 = DT['T']['label_counts']
                        n2 = DT['F']['label_counts']

                        m1 = DT['T']['value']
                        m2 = DT['F']['value']

                        s1 =  DT['T']['proba']
                        s2 =  DT['F']['proba']

                        leaf_label_counts = n1+n2

                        leaf_value = (m1*n1 + m2*n2)/(n1+n2)
                        d1 = leaf_value - m1
                        d2 = leaf_value - m2

                        s = ( n1*(s1**2+d1**2) + n2*(s2**2 + d2**2) ) / (n1+n2)

                        leaf_prob = np.sqrt(s)

                    return {'value':leaf_value,"leaf": True,'proba':leaf_prob,'label_counts':leaf_label_counts,'est_value':leaf_value,'depth':c_depth-1}
            else:
                DT['T'] = self.pruneTree(DT['T'])
                DT['F'] = self.pruneTree(DT['F'])
        return DT
    def shrinkTree(self,DT,max_depth=np.inf):
        DT0 = copy.deepcopy(DT)
        if not DT0['leaf']:
            if DT0['T']['depth']>max_depth or DT0['F']['depth']>max_depth:
                leaf_value = DT0['est_value']
                leaf_prob  = DT0['proba']
                leaf_label_counts  = DT0['label_counts']
                depth = DT0['depth']
                return {'value':leaf_value,"leaf": True,'proba':leaf_prob,'label_counts':leaf_label_counts,'est_value':leaf_value,'depth':depth}
            else:
                DT0['T'] = self.shrinkTree(DT0['T'],max_depth=max_depth)
                DT0['F'] = self.shrinkTree(DT0['F'],max_depth=max_depth)
        return DT0
    def updateTree(self,DT=None,shrink=False,max_depth=np.inf):
        if DT is None: DT = self.get_tree()
        if shrink:
            DT0 = self.shrinkTree(DT,max_depth=max_depth)
            self.tree = DT0
    def plotTree_(self,scale=True,show=True, showtitle =True, showDirection=False,DiffBranchColor=True,legend=True):
        if not(self.trained):
            print('Tree has not been built yet.... Do training first!!')
            assert self.trained

        import copy
        self.DT = copy.deepcopy(self.tree)
        if not(self.DT['leaf']):
            if self.DictDepth(self.DT)>1 and scale:
                r = self.DictDepth(self.DT['T'])
                l = self.DictDepth(self.DT['F'])
            else:
                r,l=1,1
            self.set_xyNode(self.DT,lxy=[1-l,1],xy=[1,1],rxy=[1+r,1],ldiff=1)

            self.showTree(self.DT,DiffBranchColor=DiffBranchColor)
            x,y = self.DT['xy']
            if r>=l: x1,_= self.DT['T']['xy']
            else: x1,_= self.DT['F']['xy']
            if showtitle: plt.title('Decision Tree\n\n',color='k',horizontalalignment='center')
            #if showDirection: plt.xlabel(r'$<-False$ | $True->$',color='r',horizontalalignment='center')
            if DiffBranchColor:
                plt.plot(x,y,'-b', label = 'True branch')
                plt.plot(x,y,'--r', label = 'False branch')
                if legend: plt.legend()
            elif showDirection:
                plt.text(x1,y,r'$<-False$ | $True->$',color='r',horizontalalignment='center')
            if show: plt.show()
        else:
            print('Error:: Tree has no branches!!\nAll target values must be same')
    def plotTree(self,scale=True,show=True, showtitle =True,showDirection=False,DiffBranchColor=True,legend=True,showNodevalues=True,
                  showThreshold=True,hlines=False,Measures = False,dcml=0):
        if Measures: hlines=True

        if not(self.trained):
            print('Tree has not been built yet.... Do training first!!')
            assert self.trained

        import copy
        self.DT = copy.deepcopy(self.tree)
        if not(self.DT['leaf']):
            if self.DictDepth(self.DT)>1 and scale:
                r = self.DictDepth(self.DT['T'])
                l = self.DictDepth(self.DT['F'])
            else:
                r,l=1,1
            self.set_xyNode(self.DT,lxy=[1-l,1],xy=[1,1],rxy=[1+r,1],ldiff=1)

            self.showTree(self.DT,DiffBranchColor=DiffBranchColor,showNodevalues=showNodevalues,showThreshold=showThreshold)
            x,y = self.DT['xy']
            if r>=l: x1,_= self.DT['T']['xy']
            else: x1,_= self.DT['F']['xy']
            if showtitle: plt.title('Decision Tree\n\n',color='k',horizontalalignment='center')
            #if showDirection: plt.xlabel(r'$<-False$ | $True->$',color='r',horizontalalignment='center')
            if DiffBranchColor:
                plt.plot(x,y,'-b', label = 'True branch')
                plt.plot(x,y,'--r', label = 'False branch')
                if legend: plt.legend()
            elif showDirection:
                plt.text(x1,y,r'$<-False$ | $True->$',color='r',horizontalalignment='center')
            if hlines: self._hlines()
            if Measures and len(self.Lcurve)>0:
                self._showMeasures(dcml=dcml)
            if show: plt.show()
        else:
            print('Error:: Tree has no branches!!\nAll target values must be same')
    def DictDepth(self,DT,n = 0):
        if type(DT)==dict:
            n+=1
            for key in DT.keys():
                n = self.DictDepth(DT[key],n=n)
        return n
    def getTreeDepth(self,DT=None):
        if DT is None:
            DT = copy.deepcopy(self.tree)
        return self.treeDepth(DT)
    def treeDepth(self,DT,mx=0):
        if DT['leaf']:
            if DT['depth']>mx:
                mx = DT['depth']
            return mx
        else:
            mx1 = self.treeDepth(DT['T'],mx=mx)
            mx2 = self.treeDepth(DT['F'],mx=mx)
            return max(mx1,mx2)
    def set_xyNode(self,DT,lxy=[0,1],xy=[1,1],rxy=[2,1],ldiff=1):
        DT['xy'] = xy
        if not(DT['leaf']):
            ixy =xy[:] #ixy =xy.copy()
            ixy[0] = (xy[0]+rxy[0])/2.0
            ixy[1]-=ldiff
            ilxy =xy[:]  #ilxy =xy.copy()
            irxy =rxy[:] #irxy =rxy.copy()
            self.set_xyNode(DT['T'],xy=ixy,lxy=ilxy,rxy=irxy)

            ixy =xy[:] # ixy =xy.copy()
            ixy[0] = (xy[0]+lxy[0])/2.0
            ixy[1]-=ldiff
            ilxy =lxy[:] #ilxy =lxy.copy()
            irxy =xy[:] #irxy =xy.copy()
            self.set_xyNode(DT['F'],xy=ixy,lxy=ilxy,rxy=irxy)
    def showTree(self,DT,DiffBranchColor=False,showNodevalues=True,showThreshold=True):
        d = 0.0
        x,y = DT['xy']

        if not(DT['leaf']):
            fn  = DT['feature_name']
            thr = DT['threshold']
            #st =fn+'\n(=>'+str(np.around(thr,2))+'?)\n'
            #plt.text(x,y-d*y,st,horizontalalignment='center',
            #         verticalalignment='bottom')
            plt.plot(x,y,'ok',alpha=0.8)
            x1,y1 =DT['T']['xy']
            x2,y2 =DT['F']['xy']
            plt.plot([x,x1],[y,y1],'-b',alpha=1)
            if DiffBranchColor:
                plt.plot([x,x2],[y,y2],'--r',alpha=1)
            else:
                plt.plot([x,x2],[y,y2],'-b',alpha=1)
            if isinstance(thr, int) or isinstance(thr, float):
                st =fn+'\n('+r'$\geq$'+str(np.around(thr,2))+'?)\n'
            else:
                st =fn+'\n(='+thr+'?)\n'
            if not(showThreshold): st = fn+'\n'
            if showNodevalues: plt.text(x,y-d*y,st,horizontalalignment='center',verticalalignment='bottom')
            self.showTree(DT['T'],DiffBranchColor=DiffBranchColor,showNodevalues=showNodevalues,showThreshold=showThreshold)
            self.showTree(DT['F'],DiffBranchColor=DiffBranchColor,showNodevalues=showNodevalues,showThreshold=showThreshold)
        else:
            val = DT['value']
            if isinstance(val, float): val = np.around(val,2)
            plt.text(x,y+d*y,'\nv:'+str(val),horizontalalignment='center',
                     verticalalignment='top')
            plt.plot(x,y,'og')
        plt.axis('off')
    def plotTreePath(self,path,ax=None,fig=None):
        if ax is None:
            fig,ax = plt.subplots(1,1)
        lx,x,rx,y =0,1,2,0
        ix,iy =x,y
        for b in list(path):
            jx,jy =ix,iy
            if b=='T':
                lx = ix
            else:
                rx = ix

            ix  = (lx +rx)/2.
            iy -=1
            if b=='T':
                plt.plot([jx,ix],[jy,iy],'bo-')
            else:
                plt.plot([jx,ix],[jy,iy],'ro-')
            if not(self.FastPlot): fig.canvas.draw()

        ax.plot(ix,iy,'og')
        ax.axis('off')
        fig.canvas.draw()
    def _hlines(self,diff=1,ls='-.',lw=0.5,color='k'):
        for i in range(self.getTreeDepth()):
            plt.axhline(y=-i*diff,ls=ls,color=color,lw=lw)
    def _showMeasures(self,color='b',dcml=0):
        if len(self.Lcurve)>0:
            xmn,xmx = plt.gca().get_xlim()
            msr = self.Lcurve['measure']
            sfx = '%' if self.Lcurve['measure']=='acc' else ''
            lst = [ l1 for l1 in list(self.Lcurve[1].keys()) if self.Lcurve[1][l1] is not None ]
            #st ='train% | test%'
            st = ' | '.join([l+sfx for l in lst])
            plt.text(xmx,0.5,st)
            for d in range(self.getTreeDepth()):
                pf = [self.Lcurve[d+1][l1] for l1 in lst]
                if msr=='acc': pf = [p*100 for p in pf]
                pf = [np.around(p,dcml) for p in pf]
                if dcml==0: pf = [int(p) for p in pf]
                st = ' | '.join([str(p)+sfx for p in pf])
                plt.text(xmx,-d,st,color=color,verticalalignment='center')
        else:
            print('No Learning curve values found!!!\n Please use method "getLcurve(Xt,yt,Xs,ys)" to create Lcurve')
    def getLcurve(self,Xt=None,yt=None,Xs=None,ys=None,measure='acc'):
        Lcurve ={}
        assert measure in ['acc','mse','mae']
        if self.trained and (Xt is not None or Xs is not None):
            Lcurve['measure']=measure
            for d in range(self.getTreeDepth()):
                Tr,Ts = None,None
                if Xt is not None:
                    ytp = self.predict(Xt,max_depth=d+1)
                    Tr  = self. _loss(yt,ytp,measure=measure)
                if Xs is not None:
                    ysp = self.predict(Xs,max_depth=d+1)
                    Ts  = self. _loss(ys,ysp,measure=measure)
                Lcurve[d+1] = {'train':Tr,'test':Ts}
        self.Lcurve = Lcurve
        return Lcurve
    def plotLcurve(self,ax=None,title=True):
        if len(self.Lcurve)>0:
            if ax is None: fig, ax = plt.subplots()
            depth = np.arange(1,self.getTreeDepth()+1).astype(int)
            Tr = np.array([self.Lcurve[d]['train'] for d in depth]).astype(float)
            Ts = np.array([self.Lcurve[d]['test'] for d in depth]).astype(float)
            if not(np.isnan(Tr[0])): ax.plot(depth,Tr,label='Training')
            if not(np.isnan(Ts[0])): ax.plot(depth,Ts,label='Testing')
            ax.legend()
            ax.set_xlabel('Depth of Tree')
            ax.set_ylabel(self.Lcurve['measure'].upper())
            ax.grid(alpha=0.3)
            if title: ax.set_title('Learning Curve')
        else:
            print('No Learning curve values found!!!\n Please use method "getLcurve(Xt,yt,Xs,ys)" to create Lcurve')
    def _loss(self,y,yp,measure='acc'):
        assert y.shape==yp.shape
        if measure=='mse':
            return np.mean((yp-y)**2)
        elif measure=='mae':
            return np.mean(np.abs(yp-y))
        elif measure=='logloss':
            loss = -np.mean(y*np.log(yp+1e-10)+(1-y)*np.log(1-yp+1e-10))
            return loss
        else:
            return np.mean(yp==y)

class ClassificationTree(DecisionTree):

    def __repr__(self):
        info = 'ClassificationTree(' +\
               'max_depth={}, min_samples_split={},min_impurity={},'.format(self.max_depth,self.min_samples_split,self.min_impurity) +\
               'thresholdFromMean='.format(self.thresholdFromMean)
        return info
    def _entropy(self,y):
        '''
		Calculate the entropy of array y
        H(y) = - sum(p(y)*log2(p(y)))
        '''
        yi = y
        if len(y.shape)>1 and y.shape[1]>1:
            yi = np.argmax(y,axis=1)

        _,counts =np.unique(yi,return_counts=True)
        py = counts/counts.sum()
        Hy = -sum(py*np.log2(py))
        return Hy
    def _infoGain(self, y, y1, y2):
        '''
        Calculate the information Gain with Entropy
        I_gain = H(y) - P(y1) * H(y1) - (1 - P(y1)) * H(y2)
        '''
        p = len(y1) / len(y)
        info_gain = self._entropy(y) - p * self._entropy(y1) - (1 - p) * self._entropy(y2)
        return info_gain
    def _majority_vote(self, y):
        #mojority of label
        y = y.astype(int)
        labels, counts = np.unique(y,return_counts=True)
        return labels[np.argmax(counts)]
    def _probability(self, y):
        ''' Computing Probability'''
        y = y.astype(int)
        labels,counts = np.unique(y,return_counts=True)
        labels = list(labels)
        counts = list(counts)
        assert set(labels).issubset(self.targets)

        label_counts = {lbl:0 for lbl in self.targets}
        for lbl in labels:
            label_counts[lbl]+=counts[labels.index(lbl)]

        frq = [label_counts[lbl] for lbl in self.targets]
        px = np.array(frq)/np.sum(frq)
        return list(px), label_counts
    def fit(self, X, y,verbose=0,feature_names=None,randomBranch=False):
        '''
        Parameters:
        -----------
              X	:: ndarray (number of sample, number of features)
              y	:: list of 1D array
        verbose	::0 - no progress or tree (silent)
				::1 - show progress in short
				::2 - show progress with details with branches
				::3 - show progress with branches True/False
				::4 - show progress in short with plot tree

        feature_names:: (optinal) list, Provide for better look at tree while plotting or shwoing the progress,
                       default to None, if not provided, features are named as f1,...fn
        '''
        self._impurity_calculation = self._infoGain
        self._leaf_value_calculation = self._majority_vote
        self._leaf_prob_calculation = self._probability
        self.verbose =verbose
        self.feature_names = feature_names
        self.targets = list(set(y))
        self.targets.sort()
        self.task = 'classification'
        self.randomBranch=randomBranch
        super(ClassificationTree, self).fit(X, y)

class RegressionTree(DecisionTree):

    def __repr__(self):
        info = 'RegressionTree(' +\
               'max_depth={}, min_samples_split={},min_impurity={},'.format(self.max_depth,self.min_samples_split,self.min_impurity) +\
               'thresholdFromMean='.format(self.thresholdFromMean)
        return info
    def _varReduction(self, y, y1, y2):
        '''
        Calculate the variance reduction
        VarRed = Var(y) - P(y1) * Var(y1) - P(p2) * Var(y2)
        '''
        assert len(y.shape)==1 or y.shape[1]==1

        p1 = len(y1) / len(y)
        p2 = len(y2) / len(y)

        variance_reduction = np.var(y) - p1 * np.var(y1) - p2 * np.var(y2)

        return np.sum(variance_reduction)
    def _mean_of_y(self, y):
        value = np.mean(y, axis=0)
        if isinstance(value,float) or isinstance(value,int):
            return value
        else:
            return value if len(value) > 1 else value[0]
    def _std_of_y(self, y):
        std = np.std(y, axis=0)
        if not(isinstance(std,float) or isinstance(std,int)):
            std if len(std) > 1 else std[0]
        return std,len(y)
    def fit(self, X, y,verbose=0,feature_names=None,randomBranch=False):
        '''
        Parameters:
        -----------
              X:: ndarray (number of sample, number of features)
              y:: list of 1D array
        verbose::0 - no progress or tree (silent)
               ::1 - show progress in short
               ::2 - show progress with details with branches
			   ::3 - show progress with branches True/False
			   ::4 - show progress in short with plot tree

        feature_names:: (optinal) list, Provide for better look at tree while plotting or shwoing the progress,
                       default to None, if not provided, features are named as f1,...fn
        '''
        self._impurity_calculation = self._varReduction
        self._leaf_value_calculation = self._mean_of_y
        self._leaf_prob_calculation = self._std_of_y
        self.verbose =verbose
        self.feature_names = feature_names
        self.targets = (min(y),max(y))
        self.task = 'regression'
        self.randomBranch=randomBranch
        super(RegressionTree, self).fit(X, y)
