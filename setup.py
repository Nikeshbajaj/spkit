import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

top_dir, _ = os.path.split(os.path.abspath(__file__))

with open(os.path.join(top_dir, 'Version')) as f:
    version = f.readline().strip()

setuptools.setup(
    name="spkit",
    version= version,
    author="Nikesh Bajaj",
    author_email="bajaj.nikey@gmail.com",
    description="SpKit: Signal Processing toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nikeshbajaj/spkit",
    download_url = 'https://github.com/Nikeshbajaj/spkit/tarball/' + version,
    packages=setuptools.find_packages(),
    license = 'MIT',
    keywords = 'Signal processing entropy Rényi entropy Kullback–Leibler divergence Mutual Information',
    classifiers=[
        "Programming Language :: Python :: 2",
		"Programming Language :: Python :: 2.7",
		"Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',

        "License :: OSI Approved :: MIT License",

        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        'Development Status :: 5 - Production/Stable',
    ],
    include_package_data=True,
    install_requires=['numpy','matplotlib','scipy','scikit-learn','python-picard']
)
