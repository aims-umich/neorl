.. _rneat:

Recurrent Neuroevolution of Augmenting Topologies (RNEAT)
===========================================================

Neuroevolution of Augmenting Topologies (NEAT) uses evolutionary genetic algorithms to evolve neural architectures, where the best optimized neural network is selected according to certain criteria. For NEORL, NEAT tries to build a neural network that minimizes or maximizes an objective function by following {action, state, reward} terminology of reinforcement learning. In RNEAT, genetic algorithms evolve Recurrent neural networks for optimization purposes in a reinforcement learning context.

Original paper: Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. Evolutionary computation, 10(2), 99-127.

What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ❌
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ❌

Parameters
------------

.. autoclass:: neorl.hybrid.rneat.RNEAT
  :members:
  :inherited-members:

Example
-------

Train a RNEAT agent to optimize the 5-D sphere function

.. code-block:: python

	from neorl import RNEAT
	import numpy as np 
	
	def Sphere(individual):
	    """Sphere test objective function.
	            F(x) = sum_{i=1}^d xi^2
	            d=1,2,3,...
	            Range: [-100,100]
	            Minima: 0
	    """
	    return sum(x**2 for x in individual)
	
	nx=5
	lb=-100
	ub=100
	bounds={}
	for i in range(1,nx+1):
	        bounds['x'+str(i)]=['float', -100, 100]
	
	# modify your own NEAT config
	config = {
	    'pop_size': 50,
	    'num_hidden': 1,
	    'activation_mutate_rate': 0.1,
	    'survival_threshold': 0.3,
	    }
	
	# model config
	rneat=RNEAT(fit=Sphere, bounds=bounds, mode='min', config= config, ncores=1, seed=1)
	#A random initial guess (provide one individual)
	x0 = np.random.uniform(lb,ub,nx)
	x_best, y_best, rneat_hist=rneat.evolute(ngen=200, x0=x0, 
	                                         verbose=True, checkpoint_itv=None, 
	                                         startpoint=None)

Notes
--------

- The following major hyperparameters can be changed when you define the ``config`` dictionary:
    +--------------------------------+---------------------------------------------------------------------------------------------------+
    |Hyperparameter                  | Description                                                                                       |
    +================================+===================================================================================================+
    |- pop_size                      |- The number of individuals in each generation (30)                                                |
    |- num_hidden                    |- The number of hidden nodes to add to each genome in the initial population (1)                   |
    |- elitism                       |- The number of individuals to survive from one generation to the next (1)                         |
    |- survival_threshold            |- The fraction for each species allowed to reproduce each generation(0.3)                          |
    |- min_species_size              |- The minimum number of genomes per species after reproduction (2)                                 |
    |- activation_mutate_rate        |- The probability that mutation will replace the node’s activation function (0.05)                 |
    |- aggregation_mutate_rate       |- The probability that mutation will replace the node’s aggregation function  (0.05)               |
    |- weight_mutate_rate            |- The probability that mutation will change the connection weight by adding a random value (0.5)   |
    |- bias_mutate_rate              |- The probability that mutation will change the bias of a node by adding a random value (0.7)      |
    +--------------------------------+---------------------------------------------------------------------------------------------------+
                                      
Acknowledgment
-----------------

Thanks to our fellows in NEAT-Python, as we have used their NEAT implementation to leverage our optimization classes. 

https://github.com/CodeReclaimers/neat-python