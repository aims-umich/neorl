.. _ash:

.. automodule:: neorl.tune.ashtune

Asynchronous Successive Halving
=================================

A module for asynchronous successive halving for search of hyperparameters for expensive NEORL applications. 

Original paper: Li, Liam, et al. "A system for massively parallel hyperparameter tuning." arXiv preprint arXiv:1810.05934 (2018).

Coming Soon!!!

.. 
	Random search is a technique where random combinations of the hyperparameters are used to find the best solution for the algorithm used. Random search tries random combinations of the hyperparameter set, where the cost function is evaluated at these random sets in the parameter space. As indicated by the reference above, the chances of finding the optimal hyperparameters are comparatively higher in random search than grid search, because of the random search pattern as the algorithm might end up being used on the optimised hyperparameters without any aliasing or wasting of resources.
	
	What can you use?
	--------------------
	
	-  Multi processing: ✔️
	-  Discrete/Continuous/Mixed spaces: ✔️
	-  Reinforcement Learning Algorithms: ✔️
	-  Evolutionary Algorithms: ✔️
	-  Hybrid Neuroevolution Algorithms: ✔️
	
	Parameters
	----------
	
	.. autoclass:: RANDTUNE
	  :members:
	  :inherited-members:
	  
	Example
	-------
	
	Example of using random search to tune three ES hyperparameters for solving the 5-d Sphere function 
	
	.. code-block:: python
	
		from neorl.tune import RANDTUNE
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
		
		#setup a random tune object
		rtune=RANDTUNE(param_grid=param_grid, fit=tune_fit, ncases=25, seed=1)
		#view the generated cases before running them
		print(rtune.hyperparameter_cases)
		#tune the parameters with method .tune
		randres=rtune.tune(ncores=1, csvname='tune.csv')
		print(randres)
	
	Notes
	-----
	
	- For ``ncores > 1``, the parallel tuning engine starts. **Make sure to run your python script from Terminal NOT from an IDE (e.g. Spyder, Jupyter Notebook)**. IDEs usually crash when running parallel problems with packages like ``joblib`` or ``multiprocessing``. For ``ncores = 1``, IDEs seem to work fine.    
	- Random search struggles with dimensionality if there are large number of hyperparameters to tune. Therefore, it is always recommended to do a preliminary sensitivity study to exclude or fix the hyperparameters with small impact.      
	- To determine an optimal ``ncases``, try to setup your problem for grid search on paper, calculate the grid search cost, and go for 50\% of this cost. Achieving similar performance with 50\% cost is a promise for random search.  
	- For difficult problems, the analyst can start with a random search first to narrow the choices of the important hyperparameters. Then, a grid search can be executed on those important parameters with more refined and narrower grids. 


 
