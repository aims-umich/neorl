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
    
    
long_description = r"""
# NEORL

NEORL (**N**euro**E**volution **O**ptimization with **R**einforcement **L**earning) is a set of implementations of hybrid algorithms combining neural networks and evolutionary computation based on a wide range of machine learning and evolutionary intelligence architectures. NEORL aims to solve large-scale optimization problems relevant to operation & optimization research, engineering, business, and other disciplines. 

NEORL can be used for multidisciplinary applications for research, industrial, academic, and/or teaching purposes. NEORL can be used as a standalone platform or an additional benchmarking tool to supplement or validate other optimization packages. Our objective when we built NEORL is to give the user a simple and easy-to-use framework with an access to a wide range of algorithms, covering both standalone and hybrid algorithms in evolutionary, swarm, supervised learning, deep learning, and reinforcement learning. We hope NEORL will allow beginners to enjoy more advanced optimization and algorithms, without being involved in too many theoretical/implementation details, and give experts an opportunity to solve large-scale optimization problems.
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

    return sum(x**2 for x in individual)

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
de=DE(mode='min', bounds=BOUNDS, fit=FIT, npop=50, CR=0.5, F=0.7, ncores=1, seed=1)
x_best, y_best, de_hist=de.evolute(ngen=120, verbose=0)
print('---DE Results---', )
print('x:', x_best)
print('y:', y_best)

#---------------------------------
# NES
#---------------------------------
x0=[-50]*len(BOUNDS)
amat = np.eye(nx)
xnes=XNES(mode='min', bounds=BOUNDS, fit=FIT, npop=50, eta_mu=0.9,
          eta_sigma=0.5, adapt_sampling=True, seed=1)
x_best, y_best, nes_hist=xnes.evolute(120, x0=x0, verbose=0)
print('---XNES Results---', )
print('x:', x_best)
print('y:', y_best)


#---------------------------------
# Plot
#---------------------------------
#Plot fitness for both methods
plt.figure()
plt.plot(np.array(de_hist), label='DE')
plt.plot(np.array(nes_hist['fitness']), label='NES')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.show()
```

## Citing the Project

To cite this repository in publications:

```
@misc{neorl,
  author = {Radaideh, Majdi I. and Du, Katelin and Seurin, Paul and Seyler, Devin and Gu, Xubo and Wang, Haijia and Shirvan, Koroush},
  title = {NEORL},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mradaideh/neorl}},
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
    install_requires=['tensorflow==1.14.0',
                      'numpy== 1.16.2',
                      'gym >= 0.15.4, < 0.17.0',
                      'scikit-optimize==0.8.1',
                      'cloudpickle >= 1.2.2',
                      'h5py < 2.10.0',    #for windows version (NHHO fails with model load)
                      'neat-python', 
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
