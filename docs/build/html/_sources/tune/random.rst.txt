.. _random:

Random Search
===============

A module for random search of hyperparameters of NEORL algorithms. 

Original paper: Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of machine learning research, 13(2).

Random search is a technique where random combinations of the hyperparameters are used to find the best solution for the algorithm used. Random search tries random combinations of the hyperparameter set, where the cost function is evaluated at these random sets in the parameter space. As indicated by the reference above, the chances of finding the optimal hyperparameters are comparatively higher in random search than grid search, because of the random search pattern as the algorithm might end up being used on the optimised hyperparameters without any aliasing or wasting of resources.

.. image:: ../images/random.png
   :scale: 40 %
   :alt: alternate text
   :align: center

For example, to tune few hyperparameters of DQN, the following grids can be defined:

| ``learning_rate`` = :math:`\mathcal{N}(0.0005, 0.0001)` (Continuous normal distribution)
| ``batch_size`` = :math:`\mathcal{U}  \{16, 64\}` (Discrete uniform distribution)
| ``target_network_update_freq`` = :math:`\mathcal{U}  \{100, 2000\}`(Discrete uniform distribution)
| ``exploration_fraction`` = :math:`\mathcal{N}(0.25, 0.05)` (Continuous normal distribution)

  
The cost of random search is determined by the total number of random evaluations provided by the user (``n_calls``).


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

- Random search struggles with dimensionality if there are large number of hyperparameters to tune. Therefore, it is always recommended to do a preliminary sensitivity study to exclude or fix the hyperparameters with small impact.      
- To determine an optimal ``n_calls``, try to setup your problem for grid search on paper, calculate the grid search cost, and go for 50\% of this cost. Achieving similar performance with 50\% cost is a promise for random search.  
- For difficult problems, the analyst can start with a random search first to narrow the choices of the important hyperparameters. Then, a grid search can be executed on those important parameters with more refined and narrower grids. 


 
