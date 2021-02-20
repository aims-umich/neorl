<img src="neorl_utils/logo.png" align="right" width="40%"/>

[![Build Status](https://travis-ci.com/hill-a/stable-baselines.svg?branch=master)](https://travis-ci.com/hill-a/stable-baselines) [![Documentation Status](https://readthedocs.org/projects/stable-baselines/badge/?version=master)](https://stable-baselines.readthedocs.io/en/master/?badge=master) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/3bcb4cd6d76a4270acb16b5fe6dd9efa)](https://www.codacy.com/app/baselines_janitors/stable-baselines?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=hill-a/stable-baselines&amp;utm_campaign=Badge_Grade) [![Codacy Badge](https://api.codacy.com/project/badge/Coverage/3bcb4cd6d76a4270acb16b5fe6dd9efa)](https://www.codacy.com/app/baselines_janitors/stable-baselines?utm_source=github.com&utm_medium=referral&utm_content=hill-a/stable-baselines&utm_campaign=Badge_Coverage)

# NEORL

NEORL (**N**euro**E**volution **O**ptimsiation with **R**einforcement **L**earning) is a set of implementations of hybrid algorathims combining neural networks and evolutionary computation based on a wide range of machine learning and evolutionary intellgence archtectures. NEORL aims to solve large-scale optimasation problems relevant to operation & optimsiation research, engineering, bussiness, and other disciplins. 

You can read a detailed presentation of Stable Baselines in the [Medium article](https://medium.com/@araffin/stable-baselines-a-fork-of-openai-baselines-reinforcement-learning-made-easy-df87c4b2fc82).

These algorithms will make it easier for the research community and industry to replicate, refine, and identify new ideas, and will create good baselines to build projects on top of. We expect these tools will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. We also hope that the simplicity of these tools will allow beginners to experiment with a more advanced toolset, without being buried in implementation details.

## Copyright

<img src="neorl_utils/copyright.png" align="right" width="40%"/>

This repository and its content are copyright of [Exelon Corporation](https://www.exeloncorp.com/) © in collaboration with [MIT](https://web.mit.edu/nse/) Nuclear Science and Engineering 2021. All rights reserved.

You can read the first successful application of NEORL for nuclear fuel optimisation in this [News article](https://news.mit.edu/2020/want-cheaper-nuclear-energy-turn-design-process-game-1217).

## Main differences with OpenAI Baselines

This toolset is a fork of OpenAI Baselines, with a major structural refactoring, and code cleanups:
- Unified structure for all algorithms
- PEP8 compliant (unified code style)
- Documented functions and classes
- More tests & more code coverage
- Additional algorithms: SAC and TD3 (+ HER support for DQN, DDPG, SAC and TD3)


| **Features**                             | **NEORL**                         
| -----------------------------------------| ----------------------------------- 
| Reinforcement Learning (standalone)      | :heavy_check_mark:                |
| Evolutionary Computation (standalone)    | :heavy_check_mark:                |
| Hybrid Neuroevolution                    | :heavy_check_mark:                |
| Supervised Learning                      | :x:                               |
| Parallel processing                      | :heavy_check_mark:                |
| Custom fitness function                  | :heavy_check_mark:                |
| Ipython / Notebook friendly              | :heavy_check_mark:                |
| Documentation                            | :x:                               |
| Advanced logging                         | :heavy_check_mark:                |
| Advanced plotters                        | :heavy_check_mark:                |
| Optimisation Benchmarks                  | :heavy_check_mark:                |

## Documentation

Documentation is available online: [https://stable-baselines.readthedocs.io/](https://stable-baselines.readthedocs.io/)

### Knowledge Prerequisites

**Note: despite the simplicity of NEORL usage, most algorathims, especially the neuro-based, need some basic knowledge about the optimisation research and neural networks in supervised and reinforcement learning**. Using NEORL without sufficient knowledge may lead to undesirable results due to the poor selection of algorathims hyperparameters. You should not utilize this package without some knowledge. 

## Installation

**Note:** NEORL supports Tensorflow versions from 1.8.0 to 1.14.0. Please, make sure to have the proper TensorFlow installed on your machine.

### Prerequisites
Baselines requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows

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

Most of the library tries to follow a sklearn-like syntax for the Reinforcement Learning algorithms.

Here is a quick example of how to train and run PPO2 on a cartpole environment:
```python
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('CartPole-v1')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()
```

Or just train a model with a one liner if [the environment is registered in Gym](https://github.com/openai/gym/wiki/Environments) and if [the policy is registered](https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html):

```python
from stable_baselines import PPO2

model = PPO2('MlpPolicy', 'CartPole-v1').learn(10000)
```

Please read the [documentation](https://stable-baselines.readthedocs.io/) for more examples.


## Try it online with Colab Notebooks !

All the following examples can be executed online using Google colab notebooks:

- [Full Tutorial](https://github.com/araffin/rl-tutorial-jnrr19)
- [All Notebooks](https://github.com/Stable-Baselines-Team/rl-colab-notebooks)
- [Getting Started](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/master/stable_baselines_getting_started.ipynb)
- [Training, Saving, Loading](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/master/saving_loading_dqn.ipynb)
- [Multiprocessing](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/master/multiprocessing_rl.ipynb)
- [Monitor Training and Plotting](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/master/monitor_training.ipynb)
- [Atari Games](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/master/atari_games.ipynb)
- [RL Baselines Zoo](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/master/rl-baselines-zoo.ipynb)


## Implemented Algorithms

| **Name**            | **Refactored**<sup>(1)</sup> | **Recurrent**      | ```Box```          | ```Discrete```     | ```MultiDiscrete``` | ```MultiBinary```  | **Multi Processing**              |
| ------------------- | ---------------------------- | ------------------ | ------------------ | ------------------ | ------------------- | ------------------ | --------------------------------- |
| A2C                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:                |
| ACER                | :heavy_check_mark:           | :heavy_check_mark: | :x: <sup>(5)</sup> | :heavy_check_mark: | :x:                 | :x:                | :heavy_check_mark:                |
| ACKTR               | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                 | :x:                | :heavy_check_mark:                |
| DDPG                | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: <sup>(4)</sup>|
| DQN                 | :heavy_check_mark:           | :x:                | :x:                | :heavy_check_mark: | :x:                 | :x:                | :x:                               |
| GAIL <sup>(2)</sup> | :heavy_check_mark:           | :x:                | :heavy_check_mark: |:heavy_check_mark:| :x:                 | :x:                | :heavy_check_mark: <sup>(4)</sup> |
| HER <sup>(3)</sup>  | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                 | :heavy_check_mark:| :x:                               |
| PPO1                | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: <sup>(4)</sup> |
| PPO2                | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:                |
| SAC                 | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :x:                | :x:                 | :x:                | :x:                               |
| TD3                 | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :x:                | :x:                 | :x:                | :x:                               |
| TRPO                | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: <sup>(4)</sup> |

<sup><sup>(1): Whether or not the algorithm has be refactored to fit the ```BaseRLModel``` class.</sup></sup><br>
<sup><sup>(2): Only implemented for TRPO.</sup></sup><br>
<sup><sup>(3): Re-implemented from scratch, now supports DQN, DDPG, SAC and TD3</sup></sup><br>
<sup><sup>(4): Multi Processing with [MPI](https://mpi4py.readthedocs.io/en/stable/).</sup></sup><br>
<sup><sup>(5): TODO, in project scope.</sup></sup>

NOTE: Soft Actor-Critic (SAC) and Twin Delayed DDPG (TD3) were not part of the original baselines and HER was reimplemented from scratch.

Actions ```gym.spaces```:
 * ```Box```: A N-dimensional box that containes every point in the action space.
 * ```Discrete```: A list of possible actions, where each timestep only one of the actions can be used.
 * ```MultiDiscrete```: A list of possible actions, where each timestep only one action of each discrete set can be used.
 * ```MultiBinary```: A list of possible actions, where each timestep any of the actions can be used in any combination.


## Testing the installation
All unit tests in baselines can be run using pytest runner:
```
pip install pytest pytest-cov
make pytest
```

## Projects/Papers Using NEORL

1-

2- Radaideh, M. I., Shirvan, K. (2021). Rule-based reinforcement learning methodology to inform evolutionary algorithms for constrained optimization of engineering applications. *Knowledge-Based Systems*, **217**, p. 106836.

3-

4-

## Citing the Project

To cite this repository in publications:

```
@misc{stable-baselines,
  author = {Hill, Ashley and Raffin, Antonin and Ernestus, Maximilian and Gleave, Adam and Kanervisto, Anssi and Traore, Rene and Dhariwal, Prafulla and Hesse, Christopher and Klimov, Oleg and Nichol, Alex and Plappert, Matthias and Radford, Alec and Schulman, John and Sidor, Szymon and Wu, Yuhuai},
  title = {Stable Baselines},
  year = {2018},
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
