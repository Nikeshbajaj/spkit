'''
Signal Processing Toolkit ->ML
Machine Learning from scrach
Naive Bayes - Example 3: Digit Classification

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
data = datasets.load_digits()
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
print(clf.predict(Xs[0]), clf.predict_prob(Xs[0]))

plt.figure(1)
plt.imshow(Xs[0].reshape([8,8]),cmap='gray')
plt.axis('off')
plt.title('Prediction'+str(clf.predict(Xs[0])))
plt.show()
print('Prediction',clf.predict(Xs[0]))


#parameters
print('\nParameters\nClass 0: ')
print(clf.parameters[0])


#Visualising 
clf.set_class_labels(data['target_names'])
#clf.set_feature_names(data['feature_names'])


fig = plt.figure(figsize=(12,10))
clf.VizPx(nfeatures=range(5,19))
plt.show()
