import setuptools
import os

DISTNAME = 'spkit'
DESCRIPTION = "SpKit: Signal Processing ToolKit"
MAINTAINER =  "Nikesh Bajaj"
MAINTAINER_EMAIL =  "nikkeshbajaj@gmail.com"
AUTHER = "Nikesh Bajaj"
AUTHER_EMAIL = "nikkeshbajaj@gmail.com"
URL = 'https://spkit.github.io'
LICENSE = 'BSD-3-Clause'
GITHUB_URL= 'https://github.com/Nikeshbajaj/spkit'


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


def parse_requirements_file(fname):
    requirements = list()
    with open(fname, 'r') as fid:
        for line in fid:
            req = line.strip()
            if req.startswith('#'):
                continue
            # strip end-of-line comments
            req = req.split('#', maxsplit=1)[0].strip()
            requirements.append(req)
    return requirements

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    install_requires = parse_requirements_file('requirements.txt')

    setuptools.setup(
        name=DISTNAME,
        version= version,
        author=AUTHER,
        author_email = AUTHER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type="text/markdown",
        url=URL,
        download_url = 'https://github.com/Nikeshbajaj/spkit/tarball/' + version,
        packages=setuptools.find_packages(),
        license = 'MIT',
        keywords = 'Signal processing machine-learning entropy Rényi Kullback–Leibler divergence mutual information decision-tree logistic-regression naive-bayes LFSR ICA EEG-signal-processing ATAR',
        classifiers=[
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

        platforms='any',
        python_requires='>=3.5',
        install_requires = install_requires,
        setup_requires=["numpy>1.8","setuptools>=45", "setuptools_scm>=6.2"],
        include_package_data=True,
    )
