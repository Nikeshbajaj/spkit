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
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    install_requires=['numpy','matplotlib','scipy','scikit-learn','python-picard']
)
