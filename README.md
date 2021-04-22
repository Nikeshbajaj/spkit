# Signal Processing toolkit

### Links: **[Homepage](https://spkit.github.io)** | **[Documentation](https://spkit.readthedocs.io/)** | **[Github](https://github.com/Nikeshbajaj/spkit)**  |  **[PyPi - project](https://pypi.org/project/spkit/)** |     _ **Installation:** [pip install spkit](https://pypi.org/project/spkit/)
-----

[![Documentation Status](https://readthedocs.org/projects/spkit/badge/?version=latest)](https://spkit.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version fury.io](https://badge.fury.io/py/spkit.svg)](https://pypi.org/project/spkit/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/spkit.svg)](https://pypi.python.org/pypi/spkit/)
[![GitHub release](https://img.shields.io/github/release/nikeshbajaj/spkit.svg)](https://GitHub.com/nikeshbajaj/spkit/releases/)
[![PyPI format](https://img.shields.io/pypi/format/spkit.svg)](https://pypi.python.org/pypi/spkit/)
[![PyPI implementation](https://img.shields.io/pypi/implementation/spkit.svg)](https://pypi.python.org/pypi/spkit/)
[![HitCount](http://hits.dwyl.io/nikeshbajaj/spkit.svg)](http://hits.dwyl.io/nikeshbajaj/spkit)
![GitHub commit activity](https://img.shields.io/github/commit-activity/y/nikeshbajaj/spkit?style=plastic)
[![Percentage of issues still open](http://isitmaintained.com/badge/open/nikeshbajaj/spkit.svg)](http://isitmaintained.com/project/nikeshbajaj/spkit "Percentage of issues still open")
[![PyPI download month](https://img.shields.io/pypi/dm/spkit.svg)](https://pypi.org/project/spkit/)
[![PyPI download week](https://img.shields.io/pypi/dw/spkit.svg)](https://pypi.org/project/spkit/)


[![Generic badge](https://img.shields.io/badge/pip%20install-spkit-blue.svg)](https://pypi.org/project/spkit/)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](mailto:n.bajaj@qmul.ac.uk)

![PyPI - Downloads](https://img.shields.io/pypi/dm/spkit?style=social)


-----
## Table of contents
- [**New Updates**](#new-updates)
- [**Installation**](#installation)
- [**Function list: Signal Processing & ML**](#functions-list)
- [**Examples with Notebooks**](#examples)
    - [**Scalogram CWT**](https://github.com/Nikeshbajaj/spkit#scalogram-cwt)
    - [**Information Theory**](#information-theory)
    - [**Machine Learning**](#machine-learning)
        -[Logistic Regression](#logistic-regression---view-in-notebook)
        -[Naive Bayes](#naive-bayes---view-in-notebook)
        -[Decision Trees](#decision-trees---view-in-notebook)
    - [**Independent Component Analysis**](#independent-component-analysis)
    - [**Linear Feedback Shift Register**](#linear-feedback-shift-register)
-----

## New Updates
**<mark>Version: 0.0.9.2</mark>**
* **Added Scalogram with CWT functions**
<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/cwt_ex0.jpg" width="800"/>
</p>

**<mark>Version: 0.0.9.1</mark>**
* **Fixed the Import Error with python 2.7**
* **Logistic Regression with multiclass**
* **Updated Examples with 0.0.9 version [View Notebooks](https://nbviewer.jupyter.org/github/Nikeshbajaj/Notebooks/tree/master/spkit/0.0.9/ML/Trees/) | Run all the examples with [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Nikeshbajaj/Notebooks/master?urlpath=lab/tree/spkit/0.0.9/ML/Trees)**


#### <mark>New Updates</mark>:: Decision Tree [View Notebooks](https://nbviewer.jupyter.org/github/Nikeshbajaj/Notebooks/tree/master/spkit_ML/DecisionTree/)

**<mark>Version: 0.0.7</mark>**
* **Analysing the performance measure of trained tree at different depth - with ONE-TIME Training ONLY**
* **Optimize the depth of tree**
* **Shrink the trained tree with optimal depth**
* **Plot the Learning Curve**
* **Classification: Compute the probability and counts of label at a leaf for given example sample**
* **Regression: Compute the standard deviation and number of training samples at a leaf for given example sample**

<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/DTree_withKDepth1.png" width="500"/>
<img src="https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/DTree_LCurve.png" width="300"/>
</p>


* **<mark>Version: 0.0.6</mark>**: Works with catogorical features without converting them into binary vector
* **<mark>Version: 0.0.5</mark>**: Toy examples to understand the effect of incresing max_depth of Decision Tree
<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/DTree_withCatogoricalFeatures.png" width="300"/>
<img src="https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/tree_sinusoidal.png" width="500"/>
</p>

## Installation

**Requirement**:  numpy, matplotlib, scipy.stats, scikit-learn

### with pip

```
pip install spkit
```

### update with pip

```
pip install spkit --upgrade
```


### Build from the source
Download the repository or clone it with git, after cd in directory build it from source with

```
python setup.py install
```

## Functions list
#### Signal Processing Techniques
**Information Theory functions**  for real valued signals
* Entropy : Shannon entropy, Rényi entropy of order α, Collision entropy
* Joint entropy
* Conditional entropy
* Mutual Information
* Cross entropy
* Kullback–Leibler divergence
* Computation of optimal bin size for histogram using FD-rule
* Plot histogram with optimal bin size

**Matrix Decomposition**
* SVD
* ICA using InfoMax, Extended-InfoMax, FastICA & **Picard**

**Linear Feedback Shift Register**
* pylfsr

**Continuase Wavelet Transform** and other functions comming soon..

#### Machine Learning models - with visualizations
* Logistic Regression
* Naive Bayes
* Decision Trees
* DeepNet (to be updated)


# Examples
## Scalogram CWT

```
import numpy as np
import matplotlib.pyplot as plt
import spkit as sp
from spkit.cwt import ScalogramCWT
from spkit.cwt import compare_cwt_example

x,fs = sp.load_data.eegSample_1ch()
t = np.arange(len(x))/fs
print(x.shape, t.shape)
compare_cwt_example(x,t,fs=fs)
```
<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/cwt_examples.jpg" width="800"/>
</p>

```
f0 = np.linspace(0.1,10,100)
Q  = np.linspace(0.1,5,100)
XW,S = ScalogramCWT(x,t,fs=fs,wType='Gauss',PlotPSD=True,f0=f0,Q=Q)
```
<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/cwt_ex6_gauss.jpg" width="800"/>
</p>

## Information Theory
### [View in notebook](https://nbviewer.jupyter.org/github/Nikeshbajaj/Notebooks/blob/master/spkit_InfoTheory/1_Entropy_Example.ipynb)

```
import numpy as np
import matplotlib.pyplot as plt
import spkit as sp

x = np.random.rand(10000)
y = np.random.randn(10000)

#Shannan entropy
H_x= sp.entropy(x,alpha=1)
H_y= sp.entropy(y,alpha=1)

#Rényi entropy
Hr_x= sp.entropy(x,alpha=2)
Hr_y= sp.entropy(y,alpha=2)

H_xy= sp.entropy_joint(x,y)

H_x1y= sp.entropy_cond(x,y)
H_y1x= sp.entropy_cond(y,x)

I_xy = sp.mutual_Info(x,y)

H_xy_cross= sp.entropy_cross(x,y)

D_xy= sp.entropy_kld(x,y)


print('Shannan entropy')
print('Entropy of x: H(x) = ',H_x)
print('Entropy of y: H(y) = ',H_y)
print('-')
print('Rényi entropy')
print('Entropy of x: H(x) = ',Hr_x)
print('Entropy of y: H(y) = ',Hr_y)
print('-')
print('Mutual Information I(x,y) = ',I_xy)
print('Joint Entropy H(x,y) = ',H_xy)
print('Conditional Entropy of : H(x|y) = ',H_x1y)
print('Conditional Entropy of : H(y|x) = ',H_y1x)
print('-')
print('Cross Entropy of : H(x,y) = :',H_xy_cross)
print('Kullback–Leibler divergence : Dkl(x,y) = :',D_xy)



plt.figure(figsize=(12,5))
plt.subplot(121)
sp.HistPlot(x,show=False)

plt.subplot(122)
sp.HistPlot(y,show=False)
plt.show()
```

## Independent Component Analysis
### [View in notebook](https://nbviewer.jupyter.org/github/Nikeshbajaj/Notebooks/blob/master/spkit_SP/1_EEG_ICA_Example_spkit.ipynb)
```
from spkit import ICA
from spkit.data import load_data
X,ch_names = load_data.eegSample()

x = X[128*10:128*12,:]
t = np.arange(x.shape[0])/128.0

ica = ICA(n_components=14,method='fastica')
ica.fit(x.T)
s1 = ica.transform(x.T)

ica = ICA(n_components=14,method='infomax')
ica.fit(x.T)
s2 = ica.transform(x.T)

ica = ICA(n_components=14,method='picard')
ica.fit(x.T)
s3 = ica.transform(x.T)

ica = ICA(n_components=14,method='extended-infomax')
ica.fit(x.T)
s4 = ica.transform(x.T)
```

## Machine Learning
### [Logistic Regression](https://nbviewer.jupyter.org/github/Nikeshbajaj/Notebooks/blob/master/spkit_ML/LogisticRegression/1_LogisticRegression_examples_spkit.ipynb) - *View in notebook*
<p align="center"><img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/LogisticRegression/img/example5.gif" width="600"/></p>

### [Naive Bayes](https://nbviewer.jupyter.org/github/Nikeshbajaj/Notebooks/blob/master/spkit_ML/NaiveBayes/1_NaiveBayes_example_spkit.ipynb) - *View in notebook*
<p align="center"><img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/Probabilistic/img/FeatureDist.png" width="600"/></p>

### [Decision Trees](https://nbviewer.jupyter.org/github/Nikeshbajaj/Notebooks/blob/master/spkit_ML/DecisionTree/1_Tree_ClassificationRegression_spkitV0.0.5.ipynb) - *View in notebook*

[**[jupyter-notebooks]**](https://nbviewer.jupyter.org/github/Nikeshbajaj/Notebooks/tree/master/spkit/0.0.9/ML/Trees/) | **[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Nikeshbajaj/Notebooks/master?urlpath=lab/tree/spkit/0.0.9/ML/Trees)**
<p align="center">
<img src="https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/tree_sinusoidal.png" width="800"/>
<img src="https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/trees.png" width="800"/>
</p>


#### Plottng tree while training

<p align="center"><img src="https://raw.githubusercontent.com/Nikeshbajaj/MachineLearningFromScratch/master/Trees/img/a123_nik.gif" width="600"/></p>

[**view in repository **](https://github.com/Nikeshbajaj/Notebooks/tree/master/spkit_ML/DecisionTree)

## Linear Feedback Shift Register

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshbajaj/Linear_Feedback_Shift_Register/master/images/LFSR.jpg" width="300"/>
</p>

```
import numpy as np
from spkit.pylfsr import LFSR
## Example 1  ## 5 bit LFSR with x^5 + x^2 + 1
L = LFSR()
L.info()
L.next()
L.runKCycle(10)
L.runFullCycle()
L.info()
tempseq = L.runKCycle(10000)    # generate 10000 bits from current state
```
______________________________________

# Contacts:

* **Nikesh Bajaj**
* http://nikeshbajaj.in
* n.bajaj@qmul.ac.uk
* bajaj.nikkey@gmail.com
### PhD Student: Queen Mary University of London
______________________________________
