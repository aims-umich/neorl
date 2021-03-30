.. _evolu:

Evolutionary Search
=====================

A module of evolutionary search for hyperparameter tuning of NEORL algorithms based upon evolution strategies.

Original paper: E. Bochinski, T. Senst and T. Sikora, "Hyper-parameter optimization for convolutional neural network committees based on evolutionary algorithms," 2017 IEEE International Conference on Image Processing (ICIP), Beijing, China, 2017, pp. 3924-3928, doi: 10.1109/ICIP.2017.8297018.

We have used a compact evolution strategy (ES) module for the purpose of tuning hyperparameter of NEORL algorithms. See the :ref:`ES algorithm <es>` section for more details about the (:math:`\mu,\lambda`) algorithm. To reduce the burden on the users, we specified and adapt all ES tuner hyperparameters, so the user needs to specify the hyperparameter space similar to grid, random, and other search methods. ES tuner leverages a population of individuals, where each individual represents a sample from the hyperparameter space. ES uses recombination, crossover, and mutation operations to improve the individuals from generation to the other. The best of the best individuals in all generations are reported as the top hyperparameter sets for the algorithm (See the Figure below). 

.. image:: ../images/genetic.png
   :scale: 30%
   :alt: alternate text
   :align: center

For example, to tune few hyperparameters of DQN with evolutionary search, the parameter space can be defined as:

| ``learning_rate`` = :math:`Real(low=0.0001, high=0.001) \rightarrow` Continuous hyperparameter
| ``batch_size`` = :math:`Categorical (categories=(16, 32, 64)) \rightarrow` Categorical hyperparameter
| ``target_network_update_freq`` = :math:`Integer(low=1, high=4) \rightarrow` Discrete hyperparameter
| ``exploration_fraction`` = :math:`Real(low=0.05, high=0.35) \rightarrow` Continuous hyperparameter

  
The cost of evolutionary search is determined by the total number of evaluated individuals in the population over all generations (``ngen *  npop``).


What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete/Continuous/Mixed spaces: ✔️
-  Reinforcement Learning Algorithms: ✔️
-  Evolutionary Algorithms: ✔️
-  Hybrid Neuroevolution Algorithms: ✔️

Parameters
----------

.. autoclass:: PSO
  :members:
  :inherited-members:
  
Example
-------

.. code-block:: python

	from neorl import GA

	#Define the fitness function
	def FIT(individual):
		"""Sphere test objective function.
			F(x) = sum_{i=1}^d xi^2
			d=1,2,3,...
			Range: [-100,100]
			Minima: 0
		"""
		y=sum(x**2 for x in individual)
		return -y  #-1 to convert min to max problem

	#Setup the parameter space (d=5)
	nx=5
	BOUNDS={}
	for i in range(1,nx+1):
		BOUNDS['x'+str(i)]=['float', -100, 100]

	ga=GA(bounds=BOUNDS, fit=FIT, npop=60, mutate=0.25, 
	     cx='2point', cxpb=0.7, chi=0.1, 
		 ncores=1, seed=1)
	x_best, y_best, ga_hist=ga.evolute(ngen=100, verbose=0)

Notes
-----
- A suggestion to increase ``npop`` from 10 to 60 under the same value of ``ngen`` to minimize the cost of evolutionary search. 
- In this ES tuner, 50\% of the population survive to the next generation, i.e. :math:`\mu=0.5\lambda`.  
- The strategy and individual vectors in the ES tuner are updated similarly to the ES algorithm module described :ref:`here <es>`.    
- For difficult problems, the analyst can start with a random search first to narrow the choices of the important hyperparameters. Then, an evolutionary search can be executed on those important parameters to refine their values. 



 
