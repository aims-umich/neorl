# neorl
Nuclear Engineering Optimisation with Reinforcement Learning

Majdi Radaideh NEORL is a python-based software inspired by the need for a framework that houses classical (evolutionary), modern (reinforcement learning), and their hybrid form to solve combinatorial optimization problems that we face in all engineering disciplines as well as computer science. NEORL provides an easy-to-use interface with access to variety of algorithms. The user needs to build the environment according to a certain template (OpenAI Gym environments are currently supported). The environment is case-dependent and it has the cost/reward function that will be optimized. Afterward, the user completes an easy input file that provides hyperparameters for the optimization methods being used. NEORL provides on-the-fly monitoring of the training and optimization performance to provide users constant feedback on the performance. The current structure of NEORL is as follows:  
neorl.py: the main file
src: contains all source files, categorized as follows: 
1.	rl: reinforcement learning algorithms. Currently, Deep Q Learning (DQN), Proximal Policy Optimization (PPO), and Advantage Actor Critic (A2C) are supported based on stable-baselines backend.  
2.	evolu: evolutionary algorithms. Currently, genetic algorithm (GA) is supported. Particle swarm (PSO) and simulated annealing (SA) will be added soon.  
3.	parsers: contains the parser for the master input file (PARSER.py) and the list of all NEORL variables that can be defined by the user (ParamList.py).
4.	utils: contains different utilities such as agent testing, logging, file initializers, multiprocessor engine, etc.
envs: contains pre-built environments, all are Gym-based environments from nuclear engineering:
1.	6x6 boiling water reactor assembly (BWR) based upon CASMO4 neutronic solver. No burnup is considered (Nuclear Engineering). 
2.	10x10 boiling water reactor assembly (BWR) based upon CASMO4 neutronic solver. Burnup is included (Nuclear Engineering).
3.	1/8 pressurized water reactor core based upon SIMULATE3 core simulator. Multi-cycle is included. 
examples: contains input files used as unit tests:
•	Descriptions to be added later ….
other files: 
1.	install.txt: instructions for installation, will be updated continuously as the package is improved.  
2.	LICENSE.md: open-source MIT license.
3.	neorl.spec: pyinstaller make file, currently not working. 
4.	requirements.txt: external packages needed to be installed via pip. 
5.	README.md: this file! 

