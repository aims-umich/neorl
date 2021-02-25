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

# Read version from file
with open(os.path.join('neorl', 'version.txt'), 'r') as file_handler:
    __version__ = file_handler.read().strip()
    
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
    install_requires=['gym[atari,classic_control]>=0.11',
                      'scipy',
                      'joblib',
                      'cloudpickle>=0.5.5',
                      'opencv-python',
                      'numpy',
                      'pandas',
                      'matplotlib'] + find_tf_dependency(),               
     extras_require={'tests': ['pytest', 'pytest-cov', 'pytest-env', 'pytest-xdist', 'pytype'],
                     'docs': ['sphinx', 'sphinx-autobuild', 'sphinx-rtd-theme']},   
    
    description="NeuroEvolution Optimisation with Reinforcement Learning",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/mradaideh/neorl",
    author="Majdi I. Radaideh",
    author_email="radaideh@mit.edu",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7"],
    version= __version__,
)
