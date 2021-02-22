<img src="neorl_utils/logo.png" align="right" width="40%"/>

<!---
[![Build Status](https://travis-ci.com/hill-a/stable-baselines.svg?branch=master)](https://travis-ci.com/hill-a/stable-baselines) [![Documentation Status](https://readthedocs.org/projects/stable-baselines/badge/?version=master)](https://stable-baselines.readthedocs.io/en/master/?badge=master) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/3bcb4cd6d76a4270acb16b5fe6dd9efa)](https://www.codacy.com/app/baselines_janitors/stable-baselines?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=hill-a/stable-baselines&amp;utm_campaign=Badge_Grade) [![Codacy Badge](https://api.codacy.com/project/badge/Coverage/3bcb4cd6d76a4270acb16b5fe6dd9efa)](https://www.codacy.com/app/baselines_janitors/stable-baselines?utm_source=github.com&utm_medium=referral&utm_content=hill-a/stable-baselines&utm_campaign=Badge_Coverage)
--->

# NEORL

NEORL (**N**euro**E**volution **O**ptimsiation with **R**einforcement **L**earning) is a set of implementations of hybrid algorathims combining neural networks and evolutionary computation based on a wide range of machine learning and evolutionary intellgence archtectures. NEORL aims to solve large-scale optimasation problems relevant to operation & optimsiation research, engineering, bussiness, and other disciplins. 

You can read a detailed presentation of Stable Baselines in the [Medium article](https://medium.com/@araffin/stable-baselines-a-fork-of-openai-baselines-reinforcement-learning-made-easy-df87c4b2fc82).

These algorithms will make it easier for the research community and industry to replicate, refine, and identify new ideas, and will create good baselines to build projects on top of. We expect these tools will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. We also hope that the simplicity of these tools will allow beginners to experiment with a more advanced toolset, without being buried in implementation details.

## Copyright

<img src="neorl_utils/copyright.png" align="right" width="40%"/>

This repository and its content are copyright of [Exelon Corporation](https://www.exeloncorp.com/) © in collaboration with [MIT](https://web.mit.edu/nse/) Nuclear Science and Engineering 2021. All rights reserved.

You can read the first successful application of NEORL for nuclear fuel optimisation in this [News article](https://news.mit.edu/2020/want-cheaper-nuclear-energy-turn-design-process-game-1217).

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
| Detailed Documentation                   | :x:                               |
| Advanced logging                         | :heavy_check_mark:                |
| Advanced plotters                        | :heavy_check_mark:                |
| Optimisation Benchmarks                  | :heavy_check_mark:                |

## Documentation

Documentation is available online: [https://stable-baselines.readthedocs.io/](https://stable-baselines.readthedocs.io/)

### Knowledge Prerequisites

**Note: despite the simplicity of NEORL usage, most algorathims, especially the neuro-based, need some basic knowledge about the optimisation research and neural networks in supervised and reinforcement learning**. Using NEORL without sufficient knowledge may lead to undesirable results due to the poor selection of algorathims hyperparameters. You should not utilize this package without some knowledge. 

## Installation

**Note:** NEORL supports Tensorflow versions from 1.8.0 to 1.14.0 (Recommended). Please, make sure to have the proper TensorFlow installed on your machine before installing NEORL.

### Prerequisites
NEORL requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows

#### Ubuntu

```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

#### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```

#### Windows 10

To install stable-baselines on Windows, please look at the [documentation](https://stable-baselines.readthedocs.io/en/master/guide/install.html#prerequisites).

### Install using pip
Install the Stable Baselines package:
```
pip install stable-baselines[mpi]
```

This includes an optional dependency on MPI, enabling algorithms DDPG, GAIL, PPO1 and TRPO. If you do not need these algorithms, you can install without MPI:
```
pip install stable-baselines
```

Please read the [documentation](https://stable-baselines.readthedocs.io/) for more details and alternatives (from source, using docker).


## Example

Here is a quick example of how to use NEORL to minimize a 5-D sphere function:
```python
#---------------------------------
# Import packages
#---------------------------------
import numpy as np
import matplotlib.pyplot as plt
from neopy import DE, XNES

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
de=DE(bounds=BOUNDS, fit=FIT, npop=60, mutate=0.5, recombination=0.7, ncores=1, seed=1)
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

<!---
Please read the [documentation](https://stable-baselines.readthedocs.io/) for more examples.
--->

## Enjoy NEORL with Colab Notebooks !

All the following examples can be executed online using Google colab notebooks:
- [Sphere](https://github.com/araffin/rl-tutorial-jnrr19)
- [Pressure Vessel](https://github.com/Stable-Baselines-Team/rl-colab-notebooks)
- [Welded Beam](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/master/stable_baselines_getting_started.ipynb)

## Implemented Algorithms

| **Algorithm**       | **Discrete Space** | **Continous Space**| **Mixed Space**    | **Multiprocessing**|   
| ------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| GA                  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| SA                  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| PSO                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| PPO                 | :heavy_check_mark: | :heavy_check_mark: | :x:                | :heavy_check_mark: |
| A2C                 | :heavy_check_mark: | :heavy_check_mark: | :x:                | :heavy_check_mark: |
| PESA                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| ES                  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| DE                  | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: |
| NES                 | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: |

## Testing the installation
All unit tests in baselines can be run using pytest runner:
```
pip install pytest pytest-cov
make pytest
```

## Projects/Papers Using NEORL

1- Radaideh, M. I., Wolverton, I., Joseph, J., Tusar, J. J., Otgonbaatar, U., Roy, N., ... & Shirvan, K. (2021). Physics-informed reinforcement learning optimization of nuclear assembly design. *Nuclear Engineering and Design*, **372**, p. 110966.

2- Radaideh, M. I., Shirvan, K. (2021). Rule-based reinforcement learning methodology to inform evolutionary algorithms for constrained optimization of engineering applications. *Knowledge-Based Systems*, **217**, p. 106836.

3-

4-

5-

6-

## Citing the Project

To cite this repository in publications:

```
@misc{stable-baselines,
  author = {Radaideh, Majdi I. and Seurin, Paul and Wang, Haijia and Shirvan, Koroush},
  title = {NEORL},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/hill-a/stable-baselines}},
}
```

## Maintainers

NEORL is currently maintained by [Majdi Radaideh](https://github.com/hill-a) (aka @hill-a), [Antonin Raffin](https://araffin.github.io/) (aka [@araffin](https://github.com/araffin)), [Maximilian Ernestus](https://github.com/erniejunior) (aka @erniejunior), [Adam Gleave](https://github.com/adamgleave) (@AdamGleave) and [Anssi Kanervisto](https://github.com/Miffyli) (@Miffyli).

**Important Note: We do not do technical support** and we do not answer personal questions per email.


## How To Contribute

To any interested in making NEORL better, there is an open undergraduate research position at MIT, see [Be Part of the Machine Learning-based Neuroevolution Optimization Framework](http://nurop.scripts.mit.edu/UROP/index.php)

## Acknowledgments

NEORL was established in MIT back to 2020 with feedback, validation, and usage of different colleagues: Issac Wolverton (MIT Quest for Intelligence), Joshua Joseph (MIT Quest for Intelligence), Benoit Forget (MIT Nuclear Science and Engineering), Ugi Ugotonbar.
