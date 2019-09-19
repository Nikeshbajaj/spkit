'''
Machine Learning from scrach
Example 2: Logistic Regression

@Author _ Nikesh Bajaj
PhD Student at Queen Mary University of London &
University of Genova
Conact _ http://nikeshbajaj.in 
n[dot]bajaj@qmul.ac.uk
bajaj[dot]nikkey@gmail.com
'''
print('running..')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from spkit.data import dataGen as ds
from spkit.ml import LR

plt.close('all')

dtype = ['MOONS','GAUSSIANS','LINEAR','SINUSOIDAL','SPIRAL']

X, y,_ = ds.create_dataset(200, dtype[4],0.01,varargin = 'PRESET')


#Normalizing
means = np.mean(X,0)
stds  = np.std(X,0)
X = (X-means)/stds

#In cureent version LR takes X and y as shape (nf,n) and (n,1)
X = X.T
y = y[None,:]

print(X.shape, y.shape)


clf = LR(X,y,alpha=0.0003,polyfit=True,degree=5,lambd=1.5)

plt.ion()
delay=0.01
fig=plt.figure(figsize=(10,7))
gs=GridSpec(3,2)
ax1=fig.add_subplot(gs[0:2,0])
ax2=fig.add_subplot(gs[0:2,1])
ax3=fig.add_subplot(gs[2,:])

for i in range(100):
    clf.fit(X,y,itr=10)
    ax1.clear()
    clf.Bplot(ax1,hardbound=True)
    ax2.clear()
    clf.LCurvePlot(ax2)
    ax3.clear()
    clf.Wplot(ax3)
    fig.canvas.draw()
    plt.pause(0.0001)
    #time.sleep(0.001)