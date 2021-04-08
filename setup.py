# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 19:08:52 2021

@author: majdi
"""

import os
import sys
import subprocess
from setuptools import setup, find_packages
from distutils.version import LooseVersion
from neorl.version import version
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

#check python version
if sys.version_info.major != 3:
    raise ValueError('--ERROR: This package is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))
    
# The directory containing this file
HERE = os.getcwd()

# The text of the README file
with open(os.path.join(HERE , "README.md"), encoding='utf-8') as f:
    README = f.read()
    
    
long_description = """
# NEORL

NEORL (**N**euro**E**volution **O**ptimisation with **R**einforcement **L**earning) is a set of implementations of hybrid algorithms combining neural networks and evolutionary computation based on a wide range of machine learning and evolutionary intelligence architectures. NEORL aims to solve large-scale optimisation problems relevant to operation & optimisation research, engineering, business, and other disciplines. 

NEORL can be used for multidisciplinary applications for research, industrial, academic, and/or teaching purposes. NEORL can be used as a standalone platform or an additional benchmarking tool to supplement or validate other optimisation packages. Our objective when we built NEORL is to give the user a simple and easy-to-use framework with an access to a wide range of covering both standalone and hybrid algorithms in evolutionary, swarm, supervised learning, deep learning, and reinforcement learning. We hope our implementation will allow beginners to enjoy more advanced optimisation and algorithms, without being involved in too many theoretical/implementation details.

## Copyright

This repository and its content are copyright of [Exelon Corporation](https://www.exeloncorp.com/) © in collaboration with [MIT](https://web.mit.edu/nse/) Nuclear Science and Engineering 2021. All rights reserved.

You can read the first successful application of NEORL for nuclear fuel optimisation in this [News Article](https://news.mit.edu/2020/want-cheaper-nuclear-energy-turn-design-process-game-1217).

## Links

Repository:
https://github.com/mradaideh/neorl

Main News Article:
https://news.mit.edu/2020/want-cheaper-nuclear-energy-turn-design-process-game-1217

Documentation:
https://neorl.readthedocs.io/en/latest/index.html

## Quick Example

Here is a quick example of how to use NEORL to minimize a 5-D sphere function:
```python
#---------------------------------
# Import packages
#---------------------------------
import numpy as np
import matplotlib.pyplot as plt
from neorl import DE, XNES

#---------------------------------
# Fitness
#---------------------------------
#Define the fitness function
def FIT(individual):
    '''Sphere test objective function.
        F(x) = sum_{i=1}^d xi^2
        d=1,2,3,...
        Range: [-100,100]
        Minima: 0
    '''
    
    return -sum(x**2 for x in individual)      #-1 is used to convert minimization to maximization

#---------------------------------
# Parameter Space
#---------------------------------
#Setup the parameter space (d=5)
nx=5
BOUNDS={}
for i in range(1,nx+1):
    BOUNDS['x'+str(i)]=['float', -100, 100]

#---------------------------------
# DE
#---------------------------------
de=DE(bounds=BOUNDS, fit=FIT, npop=60, CR=0.5, F=0.7, ncores=1, seed=1)
x_best, y_best, de_hist=de.evolute(ngen=100, verbose=0)

#---------------------------------
# NES
#---------------------------------
x0=[-50]*len(BOUNDS)
amat = np.eye(len(x0))
xnes = XNES(FIT, x0, amat, npop=40, bounds=BOUNDS, use_adasam=True, eta_bmat=0.04, eta_sigma=0.1, patience=9999, verbose=0, ncores=1)
x_best, y_best, nes_hist=xnes.evolute(100)

#---------------------------------
# Plot
#---------------------------------
#Plot fitness for both methods
plt.figure()
plt.plot(-np.array(de_hist), label='DE')               #multiply by -1 to covert back to a min problem
plt.plot(-np.array(nes_hist['fitness']), label='NES')  #multiply by -1 to covert back to a min problem
plt.xlabel('Step')
plt.ylabel('Fitness')
plt.legend()
plt.show()
plt.show()
```

## Citing the Project

To cite this repository in publications:

```
@misc{neorl,
  author = {Radaideh, Majdi I. and Seurin, Paul and Wang, Haijia and Shirvan, Koroush},
  title = {NEORL},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {https://github.com/mradaideh/neorl},
}
```

"""

# Read version from file
__version__ = version()
    
# Check tensorflow installation to avoid
# breaking pre-installed tf gpu ##(credit to @hill-a and stable-baselines)
def find_tf_dependency():
    install_tf, tf_gpu = False, False
    try:
        import tensorflow as tf
        if tf.__version__ < LooseVersion('1.8.0'):
            install_tf = True
            # check if a gpu version is needed
            tf_gpu = tf.test.is_gpu_available()
    except ImportError:
        install_tf = True
        # Check if a nvidia gpu is present
        for command in ['nvidia-smi', '/usr/bin/nvidia-smi', 'nvidia-smi.exe']:
            try:
                if subprocess.call([command]) == 0:
                    tf_gpu = True
                    break
            except IOError:  # command does not exist / is not executable
                pass
        if os.environ.get('USE_GPU') == 'True':  # force GPU even if not auto-detected
            tf_gpu = True

    tf_dependency = []
    if install_tf:
        tf_dependency = ['tensorflow-gpu>=1.8.0,<2.0.0'] if tf_gpu else ['tensorflow>=1.8.0,<2.0.0']
        if tf_gpu:
            print("A GPU was detected, tensorflow-gpu will be installed")

    return tf_dependency
    
# This call to setup() does all the work
setup(
    name="neorl",
    packages=[package for package in find_packages() if package.startswith('neorl')],
    include_package_data=True,
    package_data={'neorl': ['requirements.txt', 'version.txt']},
    install_requires=['tensorflow==1.13.1',
                      'numpy== 1.16.2',
                      'gym >= 0.15.4, < 0.17.0',
                      'scikit-optimize==0.8.1',
                      'cloudpickle >= 1.2.2',
                      'scipy',
                      'joblib',
                      'pandas',
                      'xlrd==1.2.0',
                      'matplotlib',
                      'pytest',
                      'pytest-cov',
                      'sphinx',
                      'sphinx-rtd-theme',
                      'sphinx-autobuild'] + find_tf_dependency(),               
     extras_require={'tests': ['pytest', 'pytest-cov', 'pytest-env', 'pytest-xdist', 'pytype'],
                     'docs': ['sphinx', 'sphinx-autobuild', 'sphinx-rtd-theme']},   
    
    description="NeuroEvolution Optimisation with Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mradaideh/neorl",
    author="Majdi I. Radaideh",
    author_email="radaideh@mit.edu",
    entry_points={
        "console_scripts": [
            "neorl=neorl.scripts:main ",
        ]
    },
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7"],
    version= __version__,
)
