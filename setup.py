# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 19:08:52 2021

@author: majdi
"""

import os
import sys
import importlib.util
from setuptools import setup, find_packages
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

# Read version directly from neorl/version.py without importing the neorl
# package itself, since neorl/__init__.py imports tensorflow and other
# runtime dependencies that are not guaranteed to be installed yet at
# build-requirement-resolution time.
_version_spec = importlib.util.spec_from_file_location(
    "neorl_version", os.path.join(HERE, "neorl", "version.py"))
_version_module = importlib.util.module_from_spec(_version_spec)
_version_spec.loader.exec_module(_version_module)
__version__ = _version_module.version()

# This call to setup() does all the work
setup(
    name="neorl",
    packages=[package for package in find_packages() if package.startswith('neorl')],
    include_package_data=True,
    package_data={'neorl': ['requirements.txt', 'version.txt']},
    install_requires=['tensorflow==2.21.0',
                      'numpy>=2.2.6,<=2.5.2',
                      'gymnasium==1.3.0',
                      'scikit-optimize==0.10.2',
                      'cloudpickle==3.1.2',
                      'h5py==3.14.0',
					  'scikit-learn>=1.7.2,<=1.9.0',
                      'neat-python==2.0.0',
                      'xarray>=2025.6.1,<=2026.7.0',
                      'scipy>=1.15.3,<=1.18.1',
                      'joblib==1.6.0',
                      'pandas>=2.3.3,<=3.0.5',
                      'openpyxl==3.1.5',
                      'matplotlib>=3.10.9,<=3.11.2',
                      'autograd',
                      'pytest',
                      'pytest-cov',
                      'sphinx',
                      'sphinx-rtd-theme',
                      'sphinx-autobuild'],
     extras_require={'tests': ['pytest', 'pytest-cov', 'pytest-env', 'pytest-xdist', 'pytest-forked', 'pytype', 'autograd'],
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
    python_requires=">=3.10,<=3.13.15",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13"],
    version= __version__,
)
