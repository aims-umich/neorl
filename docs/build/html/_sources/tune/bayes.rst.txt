.. _bayes:

.. automodule:: neorl.tune.bayestune

Bayesian Search
=========================

A module of Bayesian optimisation search for hyperparameter tuning of NEORL algorithms based upon ``scikit-optimise``. 

Original paper: https://arxiv.org/abs/1012.2599

Bayesian search, in contrast to grid and random searches, keeps track of past evaluation results. Bayesian uses past evaluations to form a probabilistic model mapping hyperparameters to a probability of a score on the objective function (e.g. max fitness, max reward). Bayesian optimization excels when the objective functions are expensive to evaluate, when we do not have access to derivatives, or when the problem at hand is non-convex. 

What can you use?
--------------------

-  Multi processing: ❌ (Multithreading in a single processor is available via ``nthreads``)
-  Discrete/Continuous/Mixed spaces: ✔️
-  Reinforcement Learning Algorithms: ✔️
-  Evolutionary Algorithms: ✔️
-  Hybrid Neuroevolution Algorithms: ✔️

Parameters
----------

.. autoclass:: BAYESTUNE
  :members:
  :inherited-members:
  
Example
-------

.. code-block:: python

	from neorl.tune import BAYESTUNE
	from neorl import ES
	
	#**********************************************************
	# Part I: Original Problem Settings
	#**********************************************************
	
	#Define the fitness function (for original optimisation)
	def sphere(individual):
	    y=sum(x**2 for x in individual)
	    return y
	
	#*************************************************************
	# Part II: Define fitness function for hyperparameter tuning
	#*************************************************************
	def tune_fit(cxpb, mu, alpha, cxmode):
	
	    #--setup the parameter space
	    nx=5
	    BOUNDS={}
	    for i in range(1,nx+1):
	            BOUNDS['x'+str(i)]=['float', -100, 100]
	
	    #--setup the ES algorithm
	    es=ES(mode='min', bounds=BOUNDS, fit=sphere, lambda_=80, mu=mu, mutpb=0.1, alpha=alpha,
	             cxmode=cxmode, cxpb=cxpb, ncores=1, seed=1)
	
	    #--Evolute the ES object and obtains y_best
	    #--turn off verbose for less algorithm print-out when tuning
	    x_best, y_best, es_hist=es.evolute(ngen=100, verbose=0)
	
	    return y_best #returns the best score
	
	#*************************************************************
	# Part III: Tuning
	#*************************************************************
	#Setup the parameter space
	#VERY IMPORTANT: The order of these parameters MUST be similar to their order in tune_fit
	#see tune_fit
	param_grid={
	#def tune_fit(cxpb, mu, alpha, cxmode):
	'cxpb': [[0.1, 0.9],'float'],             #cxpb is first (low=0.1, high=0.8, type=float/continuous)
	'mu':   [[30, 60],'int'],                 #mu is second (low=30, high=60, type=int/discrete)
	'alpha':[[0.1, 0.2, 0.3, 0.4],'grid'],    #alpha is third (grid with fixed values, type=grid/categorical)
	'cxmode':[['blend', 'cx2point'],'grid']}  #cxmode is fourth (grid with fixed values, type=grid/categorical)
	
	#setup a bayesian tune object
	btune=BAYESTUNE(param_grid=param_grid, fit=tune_fit, ncases=15)
	#tune the parameters with method .tune
	bayesres=btune.tune(nthreads=1, csvname='bayestune.csv', verbose=True)
	print(bayesres)

Notes
-----

- We allow a weak parallelization of Bayesian search via multithreading. The user can start independent Bayesian search with different seeds by increasing ``nthreads``. However, all threads will be executed on a single processor, which will slow down every Bayesian sequence. Therefore, this option is recommended when each hyperparameter case is fast-to-evaluate and does not require intensive CPU power. 
- If the user sets ``nthreads=4`` and sets ``ncases=15``, a total of 60 hyperparameter cases are evaluated, where each thread uses 25\% of the CPU power. **The extension to multiprocessing/multi-core capability is on track in future**.     
- Keep ``ncases >= 11``. If ncases < 11 is given, the optimiser resets ``ncases=11``. It is good to start with ``ncases=30``, check the optimizer convergence, and increase as needed.
- Relying on ``grid/categorical`` variables can accelerate the search by a wide margin. Therefore, if the user is aware of certain values of the (``int/discrete``) or the (``float/continuous``) hyperparameters, it is good to convert them to ``grid/categorical``.

Acknowledgment
-----------------

Thanks to our fellows in `scikit-optimize <https://github.com/scikit-optimize/scikit-optimize>`_, as we used their ``gp_minimize`` implementation to leverage our Bayesian search module in this page.

Head, Tim, Gilles Louppe MechCoder, and Iaroslav Shcherbatyi. "scikit-optimize/scikit-optimize: v0.7.1"(2020).


