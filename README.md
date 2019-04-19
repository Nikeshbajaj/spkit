# Signal Processing toolkit

### **[Github](https://github.com/Nikeshbajaj/spkit)**
### **[PyPi - project](https://pypi.org/project/spkit/)**

**Information Theory functions**
* Entropy : Shannon entropy, Rényi entropy of order α, Collision entropy
* Joint entropy
* Conditional entropy
* Mutual Information
* Cross entropy
* Kullback–Leibler divergence
* Computation of optimal bin size for histogram using FD-rule
* Plot histogram with optimal bin size


**Continuase Wavelet Transform** and other functions comming soon..


### Requirement :  *numpy, matplotlib*

## Installation

### with pip

```
pip install spkit
```

### Build from the source
Download the repository or clone it with git, after cd in directory build it from source with

```
python setup.py install
```

## Example -  Information Theory

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


______________________________________

# Contacts:

* **Nikesh Bajaj**
* http://nikeshbajaj.in
* n.bajaj@qmul.ac.uk
* bajaj.nikkey@gmail.com
### PhD Student: Queen Mary University of London & University of Genoa
______________________________________
