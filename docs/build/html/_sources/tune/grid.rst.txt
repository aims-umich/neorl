.. _grid:

Grid Search
=============

A module for grid search of hyperparameters of NEORL algorithms. 

Original paper: Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of machine learning research, 13(2).

Grid Search is an exhaustive search for selecting an optimal set of algorithm hyperparameters. In Grid Search, the analyst sets up a grid of hyperparameter values. A multi-dimensional full grid of all hyperparameters is constructed, which contains all possible combinations of hyperparameters. Afterwards, every combination of hyperparameter values is tested in serial/parallel, where the optimisation score (e.g. fitness) is estimated. Grid search can be very expensive for fine grids as well as large number of hyperparameters to tune. 

.. image:: ../images/grid.png
   :scale: 40 %
   :alt: alternate text
   :align: center

For example, to tune few hyperparameters of DQN, the following grids can be defined:

| ``learning_rate`` =[0.0001, 0.00025, 0.0005, 0.00075, 0.001]
| ``batch_size`` =[16, 32, 64]
| ``target_network_update_freq`` =[100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
| ``exploration_fraction`` =[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]

  
The full grid has a size of = 5*3*9*7= 945 (A total of 945 hyperparameter combinations will be evaluated). Therefore, the cost of grid search is:

.. math::

	Cost = k_1 \times k_2 \times ... \times k_d, 

where :math:`k_i` is the number of nodes in the hyperparameter :math:`i` and :math:`d` is the number of hyperparameters to tune 

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

- If there are large number of hyperparameters to tune (large :math:`d`), try nested grid search. First, run a grid search on few parameters first, then fix them to their best, and start another grid search for the next group of hyperparameters, and so on.    
- Always start with coarse grid for all hyperparameters (small :math:`k_i`) to obtain an impression about thier sensitivity. Then, refine the grids for those hyperparameters showing more impact, and run a more detailed grid search.  
- Grid search is ideal to use when the analyst has prior experience on the feasible range of each hyperparameter and the most important hyperparameters to tune. 
