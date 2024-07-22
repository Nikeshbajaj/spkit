# Signal Processing toolkit

### Links: **[Homepage](https://spkit.github.io)** | **[Documentation](https://spkit.readthedocs.io/)** | **[Github](https://github.com/Nikeshbajaj/spkit)**  |  **[PyPi - project](https://pypi.org/project/spkit/)** |     _ **Installation:** [pip install spkit](https://pypi.org/project/spkit/)
-----
![CircleCI](https://img.shields.io/circleci/build/github/Nikeshbajaj/spkit)
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

[![DOI](https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/figures/zenodo.4710694.svg)](https://doi.org/10.5281/zenodo.4710694)

<!--[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4710694.svg)](https://doi.org/10.5281/zenodo.4710694)
<a href="https://doi.org/10.5281/zenodo.4710694"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4710694.svg" alt="DOI"></a>
-->

-----

## Installation

**Requirement**:  numpy, matplotlib, scipy.stats, scikit-learn, seaborn

### with pip

```
pip install spkit
```

### update with pip

```
pip install spkit --upgrade
```
# New in 0.0.9.7:
## MEA Processing Toolkit

# New in 0.0.9.5:
## MEA Processing Toolkit
  * sp.mea
## Geometrical Functions
  * sp.gemetry
## More on signal processing
  * sp.core
## Statistics
  * sp.stats



# For updated list of contents and documentation check [github](https://GitHub.com/nikeshbajaj/spkit) or [Documentation](https://spkit.readthedocs.io/)

# List of all functions
# Signal Processing Techniques
## **Information Theory functions**
 **for real valued signals**
 * Entropy
   * Shannon entropy
   * Rényi entropy of order α, Collision entropy,
   * Joint entropy
   * Conditional entropy
   * Mutual Information
   * Cross entropy
   * Kullback–Leibler divergence
   * Spectral Entropy
   * Approximate Entropy
   * Sample Entropy
   * Permutation Entropy
   * SVD Entropy

* Plot histogram with optimal bin size
* Computation of optimal bin size for histogram using FD-rule
* Compute bin_width with various statistical measures
* Plot Venn Diagram- joint distribuation and normalized entropy values

## **Dispersion Entropy** --**for time series (physiological signals)**
* **Dispersion Entropy** (Advanced) - for time series signal
  * Dispersion Entropy
  * Dispersion Entropy - multiscale
  * Dispersion Entropy - multiscale - refined


## **Matrix Decomposition**
* SVD
* ICA using InfoMax, Extended-InfoMax, FastICA & **Picard**

## **Continuase Wavelet Transform**
* Gauss wavelet
* Morlet wavelet
* Gabor wavelet
* Poisson wavelet
* Maxican wavelet
* Shannon wavelet

## **Discrete Wavelet Transform**
* Wavelet filtering
* Wavelet Packet Analysis and Filtering

## **Basic Filtering**
* Removing DC/ Smoothing for multi-channel signals
* Bandpass/Lowpass/Highpass/Bandreject filtering for multi-channel signals

## Biomedical Signal Processing

### MEA Processing Toolkit

**Artifact Removal Algorithm**
* **ATAR Algorithm** [Automatic and Tunable Artifact Removal Algorithm for EEG from artical](https://www.sciencedirect.com/science/article/pii/S1746809419302058)
* **ICA based Algorith**

## Analysis and Synthesis Models
* **DFT Analysis & Synthesis**
* **STFT Analysis & Synthesis**
* **Sinasodal Model - Analysis & Synthesis**
  - to decompose a signal into sinasodal wave tracks
* **f0 detection**

## Ramanajum Methods for period estimation
* **Period estimation for a short length sequence using Ramanujam Filters Banks (RFB)**
* **Minizing sparsity of periods**

## Fractional Fourier Transform
* **Fractional Fourier Transform**
* **Fast Fractional Fourier Transform**



## Machine Learning models - with visualizations
* Logistic Regression
* Naive Bayes
* Decision Trees
* DeepNet (to be updated)

## **Linear Feedback Shift Register**
* pylfsr






# Cite As
```
@software{nikesh_bajaj_2021_4710694,
  author       = {Nikesh Bajaj},
  title        = {Nikeshbajaj/spkit: 0.0.9.4},
  month        = apr,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {0.0.9.4},
  doi          = {10.5281/zenodo.4710694},
  url          = {https://doi.org/10.5281/zenodo.4710694}
}
```
# Contacts:

* **Nikesh Bajaj**
* http://nikeshbajaj.in
* n.bajaj[AT]qmul.ac.uk, n.bajaj[AT]imperial[dot]ac[dot]uk
### Imperial College London
______________________________________
