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


#Libraries
import numpy as np
import matplotlib.pyplot as plt
#import time


# Super class for Classification and Regression
class DecisionTree(object):
    """Super class of RegressionTree and ClassificationTree.

    """
    def __init__(self, min_samples_split=2, min_impurity=1e-7, max_depth=float("inf"), thresholdFromMean=False):

        '''
        Parameters:
        -----------
        min_samples_split::int: minimum number of samples to split further
        min_impurity     ::float: minimum impurity (or gain) to split
        max_depth        ::int:0<, maximum depth to go for tree, default is Inf, which leads to overfit
                            decrease the max depth to reduce the overfitting
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
        self.max_depth = max_depth

        # if threshold is consider from unique values of middle of two unique values
        # Not applicable to catogorical feature
        self.thresholdFromMean =thresholdFromMean

        self.trained = False

        # Variables that comes from SubClass
        self.verbose = None
        self.feature_names = None
        self.randomBranch=None

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
        if self.verbose>0:
            print('|\n|.........................tree is buit!')
            print('---------------------------------------')
        self.trained = True
    def _build_tree(self, X, y, current_depth=0):
        """ Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data"""

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

                            if self.verbose: status3 = ' Gain::'+str(np.around(impurity,2))+\
                            ' thr::'+str(np.around(threshold,2)) + '_Depth = '+str(current_depth)+'   '

                            node ={'feature_index':feature_i,"threshold": threshold,
                                   'feature_name':self.feature_names[feature_i],
                                   'impurity':impurity,
                                   'value':None,"leaf": False}

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

        node ={'feature_index':None,"threshold": None,
               'feature_name' :None,'impurity':None,
               'value':leaf_value,"leaf": True, 'T':None,'F':None}

        node ={'value':leaf_value,"leaf": True}


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
    def predict_value(self, x, tree=None,path=''):
        """ Do a recursive search down the tree and make a prediction of the data sample by the
            value of the leaf that we end up at """

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
        return self.predict_value(x, branch,path)
    def predict(self, X,treePath=False):
        """ Classify samples one by one and return the set of labels """

        if treePath:
            y_pred = np.array([list(self.predict_value(x)) for x in X])
        else:
            y_pred = np.array([self.predict_value(x)[0] for x in X])

        return y_pred
    def get_tree(self):
        if self.trained:
            return self.tree
        else:
            print("No tree found, haven't trained yet!!")
    def set_featureNames(self,feature_names=None):
        if feature_names is None or len(feature_names)!=self.nfeatures:
            print('setting feature names to default..f1, f2....fn')
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
                    #print('same valued leafs')
                    return {'value': DT['T']['value'], 'leaf': True}
            else:
                DT['T'] = self.pruneTree(DT['T'])
                DT['F'] = self.pruneTree(DT['F'])
        return DT
    def plotTree(self,scale=True,show=True, showtitle =True, showDirection=True,DiffBranchColor=False,legend=True):
        import copy
        self.DT = copy.deepcopy(self.tree)
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
    def DictDepth(self,DT,n = 0):
        if type(DT)==dict:
            n+=1
            for key in DT.keys():
                n = self.DictDepth(DT[key],n=n)
        return n
    def set_xyNode(self,DT,lxy=[0,1],xy=[1,1],rxy=[2,1],ldiff=1):
        DT['xy'] = xy
        if not(DT['leaf']):
            ixy =xy.copy()
            ixy[0] = (xy[0]+rxy[0])/2.0
            ixy[1]-=ldiff
            ilxy =xy.copy()
            irxy =rxy.copy()
            self.set_xyNode(DT['T'],xy=ixy,lxy=ilxy,rxy=irxy)

            ixy =xy.copy()
            ixy[0] = (xy[0]+lxy[0])/2.0
            ixy[1]-=ldiff
            ilxy =lxy.copy()
            irxy =xy.copy()
            self.set_xyNode(DT['F'],xy=ixy,lxy=ilxy,rxy=irxy)
    def showTree(self,DT,DiffBranchColor=False):
        d =0.0
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
            plt.text(x,y-d*y,st,horizontalalignment='center',verticalalignment='bottom')
            self.showTree(DT['T'],DiffBranchColor=DiffBranchColor)
            self.showTree(DT['F'],DiffBranchColor=DiffBranchColor)
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

class ClassificationTree(DecisionTree):
    def entropy(self,y):
        """ Calculate the entropy of array y
        H(y) = - sum(p(y)*log2(p(y)))
        """
        yi = y
        if len(y.shape)>1 and y.shape[1]>1:
            yi = np.argmax(y,axis=1)

        _,counts =np.unique(yi,return_counts=True)
        py = counts/counts.sum()
        Hy = -sum(py*np.log2(py))
        return Hy

    def _infoGain(self, y, y1, y2):
        # Calculate information gain
        p = len(y1) / len(y)
        info_gain = self.entropy(y) - p * self.entropy(y1) - (1 - p) * self.entropy(y2)
        return info_gain

    def _majority_vote(self, y):
        #mojority of label
        y = y.astype(int)
        label,count = np.unique(y,return_counts=True)
        return label[np.argmax(count)]

    def fit(self, X, y,verbose=0,feature_names=None,randomBranch=False):
        '''
        Parameters:
        -----------
              X:: ndarray (number of sample, number of features)
              y:: list of 1D array
        verbose::0(default)-no progress or tree
               ::1 - show progress
               ::2 - show tree
        feature_names:: (optinal) list, Provide for better look at tree while plotting or shwoing the progress,
                       default to None, if not provided, features are named as f1,...fn
        '''
        self._impurity_calculation = self._infoGain
        self._leaf_value_calculation = self._majority_vote
        self.verbose =verbose
        self.feature_names = feature_names
        self.randomBranch=randomBranch
        super(ClassificationTree, self).fit(X, y)

class RegressionTree(DecisionTree):
    def _varReduction(self, y, y1, y2):
        assert len(y.shape)==1 or y.shape[1]==1
        # Calculate the variance reduction
        p1 = len(y1) / len(y)
        p2 = len(y2) / len(y)

        variance_reduction = np.var(y) - p1 * np.var(y1) - p2 * np.var(y2)

        return np.sum(variance_reduction)

    def _mean_of_y(self, y):
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self, X, y,verbose=0,feature_names=None,randomBranch=False):
        '''
        Parameters:
        -----------
              X:: ndarray (number of sample, number of features)
              y:: list of 1D array
        verbose::0(default)-no progress or tree
               ::1 - show progress
               ::2 - show tree
        feature_names:: (optinal) list, Provide for better look at tree while plotting or shwoing the progress,
                       default to None, if not provided, features are named as f1,...fn
        '''
        self._impurity_calculation = self._varReduction
        self._leaf_value_calculation = self._mean_of_y
        self.verbose =verbose
        self.feature_names = feature_names
        self.randomBranch=randomBranch
        super(RegressionTree, self).fit(X, y)
