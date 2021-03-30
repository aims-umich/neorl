.. _bayes:

Bayesian Search
=========================

A module of Bayesian optimisation search for hyperparameter tuning of NEORL algorithms based upon ``scikit-optimise``. 

Original paper: https://arxiv.org/abs/1012.2599

Bayesian search, in contrast to grid and random searches, keeps track of past evaluation results. Bayesian uses past evaluations to form a probabilistic model mapping hyperparameters to a probability of a score on the objective function (e.g. max fitness, max reward). Bayesian optimization excels when the objective functions are expensive to evaluate, when we do not have access to derivatives, or when the problem at hand is non-convex. 

The heart of Bayesian optimization is Bayes theorem, which updates our prior beliefs (e.g. hyperparameter values) after new evidence/data is observed (e.g. new fitness values found by the algorithm of interest). The updated beliefs are represented by the posterior distribution, which is used to guide the next round of hyperparameter sampling. Also, Bayesian optimization combines the concepts of "surrogate" models (e.g. Gaussian processes) to accelerate the search, and the "acquisition" function to guide sampling from the posterior  distribution, which both can effectively make a robust search toward the global optima of the cost function (see the Figure below). The sequential-nature of Bayesian optimisation makes its parallelization complex and not natural as grid/random/evolutionary search, which is the obvious downside of Bayesian optimisation.   


.. image:: ../images/bayes.png
   :scale: 50%
   :alt: alternate text
   :align: center

For example, to tune few hyperparameters of DQN by Bayesian search, the parameter space can be defined as:

| ``learning_rate`` = :math:`Real(low=0.0001, high=0.001) \rightarrow` Continuous hyperparameter
| ``batch_size`` = :math:`Categorical (categories=(16, 32, 64)) \rightarrow` Categorical hyperparameter
| ``target_network_update_freq`` = :math:`Integer(low=1, high=4) \rightarrow` Discrete hyperparameter
| ``exploration_fraction`` = :math:`Real(low=0.05, high=0.35) \rightarrow` Continuous hyperparameter

  
The cost of Bayesian search is determined by the total number of fitness evaluations provided by the user (``n_calls``).


What can you use?
--------------------

-  Multi processing: ❌
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

- To allow a weak parallelization of Bayesian search, the user can start different search sessions with different initial guesses ``x0``. The convergence of most sessions to the same hyperparameter set would imply a high-performance by the Bayesian algorithm. Conversely, if the parallel sessions converge to different values, the user can compare them and pick the best hyperparameter set. **The automation of this parallel capability is on track of NEORL development**.     
- Keep ``n_calls > 11`` to avoid internal error raise. It is good to start with ``n_calls=50``, check the optimizer convergence, and increase as needed.
- Relying on ``Categorical`` variables can accelerate the search by a wide margin. Therefore, if the user is aware of certain values of the discrete (``Integer``) or the continuous(``Real``) hyperparameters, it is good to convert them to ``Categorical``.
- Try to pick a reasonable initial guess ``x0`` for the Bayesian search. Starting from the center of the range of each hyperparameter can be a good starting choice. 
