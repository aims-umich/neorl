<img src="docs/logo.png" align="right" width="40%"/>

<!---
[![Build Status](https://travis-ci.com/hill-a/stable-baselines.svg?branch=master)](https://travis-ci.com/hill-a/stable-baselines) [![Documentation Status](https://readthedocs.org/projects/stable-baselines/badge/?version=master)](https://stable-baselines.readthedocs.io/en/master/?badge=master) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/3bcb4cd6d76a4270acb16b5fe6dd9efa)](https://www.codacy.com/app/baselines_janitors/stable-baselines?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=hill-a/stable-baselines&amp;utm_campaign=Badge_Grade) [![Codacy Badge](https://api.codacy.com/project/badge/Coverage/3bcb4cd6d76a4270acb16b5fe6dd9efa)](https://www.codacy.com/app/baselines_janitors/stable-baselines?utm_source=github.com&utm_medium=referral&utm_content=hill-a/stable-baselines&utm_campaign=Badge_Coverage)
--->

# NEORL

NEORL (**N**euro**E**volution **O**ptimisation with **R**einforcement **L**earning) is a set of implementations of hybrid algorithms combining neural networks and evolutionary computation based on a wide range of machine learning and evolutionary intelligence architectures. NEORL aims to solve large-scale optimisation problems relevant to operation & optimisation research, engineering, business, and other disciplines. 

NEORL can be used for multidisciplinary applications for research, industrial, academic, and/or teaching purposes. NEORL can be used as a standalone platform or an additional benchmarking tool to supplement or validate other optimisation packages. Our objective when we built NEORL is to give the user a simple and easy-to-use framework with an access to a wide range of covering both standalone and hybrid algorithms in evolutionary, swarm, supervised learning, deep learning, and reinforcement learning. We hope our implementation will allow beginners to enjoy more advanced optimisation and algorithms, without being involved in too many theoretical/implementation details.

## Documentation

Documentation is available online: [https://neorl.readthedocs.io/en/latest/index.html](https://neorl.readthedocs.io/en/latest/index.html)

## Copyright

<img src="docs/copyright.png" align="right" width="40%"/>

This repository and its content are copyright of [Exelon Corporation](https://www.exeloncorp.com/) © in collaboration with [MIT](https://web.mit.edu/nse/) Nuclear Science and Engineering 2021. All rights reserved.

You can read the first successful application of NEORL for nuclear fuel optimisation in this [News Article](https://news.mit.edu/2020/want-cheaper-nuclear-energy-turn-design-process-game-1217).

## Basic Features

| **Features**                             | **NEORL**                         
| -----------------------------------------| ----------------------------------- 
| Reinforcement Learning (standalone)      | :heavy_check_mark:                |
| Evolutionary Computation (standalone)    | :heavy_check_mark:                |
| Hybrid Neuroevolution                    | :heavy_check_mark:                |
| Supervised Learning                      | :x:                               |
| Parallel processing                      | :heavy_check_mark:                |
| Combinatorial/Discrete Optimisation      | :heavy_check_mark:                |
| Continuous Optimisation                  | :heavy_check_mark:                |
| Mixed Discrete/Continuous Optimisation   | :heavy_check_mark:                |
| Ipython / Notebook friendly              | :heavy_check_mark:                |
| Detailed Documentation                   | :heavy_check_mark:                |
| Advanced logging                         | :heavy_check_mark:                |
| Optimisation Benchmarks                  | :heavy_check_mark:                |

### Knowledge Prerequisites

**Note: despite the simplicity of NEORL usage, most algorithms, especially the neuro-based, need some basic knowledge about the optimisation research and neural networks in supervised and reinforcement learning**. Using NEORL without sufficient knowledge may lead to undesirable results due to the poor selection of algorithm hyperparameters. You should not utilize this package without some knowledge. 

## Installation

**Note:** NEORL supports Tensorflow versions from 1.8.0 to 1.14.0 (Recommended). Please, make sure to have the proper TensorFlow installed on your machine before installing NEORL.

### Prerequisites
NEORL requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows

#### Ubuntu Prerequisites

```bash
sudo apt-get update && sudo apt-get install cmake python3-dev
```

#### Windows Prerequisites

To install NEORL on Windows, it is recommended to install Anaconda3 on the machine first to have some pre-installed packages, then open "Anaconda Prompt" as an administrator and use the instructions below for "Install using pip".

### Install using pip

For both Ubuntu and Windows, you can install NEORL via pip 
```
pip install neorl
```

## Example

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
    """Sphere test objective function.
            F(x) = sum_{i=1}^d xi^2
            d=1,2,3,...
            Range: [-100,100]
            Minima: 0
    """

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


<!---
## Enjoy NEORL with Colab Notebooks !

All the following examples can be executed online using Google colab notebooks:
- [Sphere](https://github.com/araffin/rl-tutorial-jnrr19)
- [Pressure Vessel](https://github.com/Stable-Baselines-Team/rl-colab-notebooks)
- [Welded Beam](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/master/stable_baselines_getting_started.ipynb)
--->


## Implemented Algorithms

NEORL offers a wide range of algorithms, where some algorithms could be used with a specific parameter space (e.g. discrete only).

| **Algorithm**       | **Discrete Space** | **Continous Space**| **Mixed Space**    | **Multiprocessing**|   
| ------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| PPO                 | :heavy_check_mark: | :heavy_check_mark: | :x:                | :heavy_check_mark: |
| A2C                 | :heavy_check_mark: | :heavy_check_mark: | :x:                | :heavy_check_mark: |
| DQN                 | :heavy_check_mark: | :x:                | :x:                | :x:                |
| ES                  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| DE                  | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: |
| PSO                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| XNES                | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: |
| GWO                 | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: |
| PESA                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| PESA2               | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: |

## Testing the installation
All unit tests in NEORL can be run using pytest runner. If pytest is not installed, please use
```
pip install pytest pytest-cov
```
Then run
```
neorl --test
```

## Projects/Papers Using NEORL

1- Radaideh, M. I., Wolverton, I., Joseph, J., Tusar, J. J., Otgonbaatar, U., Roy, N., Forget, B., Shirvan, K. (2021). Physics-informed reinforcement learning optimization of nuclear assembly design. *Nuclear Engineering and Design*, **372**, p. 110966.

2- Radaideh, M. I., Shirvan, K. (2021). Rule-based reinforcement learning methodology to inform evolutionary algorithms for constrained optimization of engineering applications. *Knowledge-Based Systems*, **217**, p. 106836.

3- Radaideh, M.I., Forget, B., Shirvan, K. (2021). Application of Reinforcement Learning Optimisation Methodology to BWR Assemblies ," In: *Proc. International Conference on Mathematics & Computational Methods Applied to Nuclear Science & Engineering (M&C 2021)*, Rayleigh, North Carolina, USA, October 3-7, 2021.

## Citing the Project

To cite this repository in publications:

```
@misc{neorl,
  author = {Radaideh, Majdi I. and Seurin, Paul and Wang, Haijia and Shirvan, Koroush},
  title = {NEORL},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mradaideh/neorl}},
}
```

## Maintainers

NEORL is currently maintained by [Majdi Radaideh](https://github.com/mradaideh) (aka @mradaideh), [Paul Seurin](https://github.com/pseur) (aka @pseur), [Koroush Shirvan](https://github.com/kshirvan) (aka @kshirvan)

**Important Note: We do not do technical support** and we do not answer personal questions per email.


## How To Contribute

To any interested in making NEORL better, there is an open undergraduate research position at MIT, see [Be Part of the Machine Learning-based Neuroevolution Optimization Framework](http://nurop.scripts.mit.edu/UROP/index.php)

## Acknowledgments

NEORL was established in MIT back to 2020 with feedback, validation, and usage of different colleagues: Issac Wolverton (MIT Quest for Intelligence), Joshua Joseph (MIT Quest for Intelligence), Benoit Forget (MIT Nuclear Science and Engineering), Ugi Otgonbaatar (Exelon Corporation).
