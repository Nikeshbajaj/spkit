import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

top_dir, _ = os.path.split(os.path.abspath(__file__))
if os.path.isfile(os.path.join(top_dir, 'Version')):
    with open(os.path.join(top_dir, 'Version')) as f:
        version = f.readline().strip()
else:
    import urllib
    Vpath = 'https://raw.githubusercontent.com/Nikeshbajaj/spkit/master/Version'
    version = urllib.request.urlopen(Vpath).read().strip().decode("utf-8")


setuptools.setup(
    name="spkit",
    version= version,
    author="Nikesh Bajaj",
    author_email="nikkeshbajaj@gmail.com",
    description="SpKit: Signal Processing toolkit | Nikesh Bajaj |",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://spkit.github.io",
    download_url = 'https://github.com/Nikeshbajaj/spkit/tarball/' + version,
    packages=setuptools.find_packages(),
    license = 'MIT',
    keywords = 'Signal processing machine-learning entropy Rényi Kullback–Leibler divergence mutual information decision-tree logistic-regression naive-bayes LFSR ICA',
    classifiers=[
        "Programming Language :: Python :: 2",
		"Programming Language :: Python :: 2.7",
		"Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Natural Language :: English',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Development Status :: 5 - Production/Stable',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Multimedia',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Multimedia :: Sound/Audio :: Speech',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Visualization',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',

        'Development Status :: 5 - Production/Stable',
    ],
    project_urls={
    'Documentation': 'https://spkit.readthedocs.io/',
    'Say Thanks!': 'https://github.com/Nikeshbajaj',
    'Source': 'https://github.com/Nikeshbajaj/spkit',
    'Tracker': 'https://github.com/Nikeshbajaj/spkit/issues',
    },
    include_package_data=True,
    install_requires=['numpy','matplotlib','scipy','scikit-learn','python-picard']
)
