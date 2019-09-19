'''
Signal Processing Toolkit ->ML
Machine Learning from scrach
Naive Bayes - Example 2: Breast Cancer

@Author _ Nikesh Bajaj
PhD Student at Queen Mary University of London 
Conact _ http://nikeshbajaj.in 
Github:: https://github.com/Nikeshbajaj
n[dot]bajaj@qmul.ac.uk
bajaj[dot]nikkey@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt

#for dataset and splitting
from sklearn import datasets
from sklearn.model_selection import train_test_split


from spkit.ml import NaiveBayes


#Data
data = datasets.load_breast_cancer()
X = data.data
y = data.target

Xt,Xs,yt,ys = train_test_split(X,y,test_size=0.3)


print('Data Shape::',Xt.shape,yt.shape,Xs.shape,ys.shape)

#Fitting
clf = NaiveBayes()
clf.fit(Xt,yt)

#Prediction
ytp = clf.predict(Xt)
ysp = clf.predict(Xs)

print('Training Accuracy : ',np.mean(ytp==yt))
print('Testing  Accuracy : ',np.mean(ysp==ys))


#Probabilities
ytpr = clf.predict_prob(Xt)
yspr = clf.predict_prob(Xs)
print('\nProbability')
print(ytpr[0])

#parameters
print('\nParameters\nClass 0: ')
print(clf.parameters[0])


#Visualising 
clf.set_class_labels(data['target_names'])
#clf.set_feature_names(data['feature_names'])


fig = plt.figure(1,figsize=(10,8))
clf.VizPx(nfeatures=range(16))
plt.show()

fig = plt.figure(2,figsize=(10,8))
clf.VizPx(nfeatures=range(16,30))
plt.show()
