.. _grid:

.. automodule:: neorl.tune.gridtune

Grid Search
=============

A module for grid search of hyperparameters of NEORL algorithms. 

Original paper: Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of machine learning research, 13(2).

Grid Search is an exhaustive search for selecting an optimal set of algorithm hyperparameters. In Grid Search, the analyst sets up a grid of hyperparameter values. A multi-dimensional full grid of all hyperparameters is constructed, which contains all possible combinations of hyperparameters. Afterwards, every combination of hyperparameter values is tested in serial/parallel, where the optimisation score (e.g. fitness) is estimated. Grid search can be very expensive for fine grids as well as large number of hyperparameters to tune. 

What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete/Continuous/Mixed spaces: ✔️
-  Reinforcement Learning Algorithms: ✔️
-  Evolutionary Algorithms: ✔️
-  Hybrid Neuroevolution Algorithms: ✔️

Parameters
----------

.. autoclass:: GRIDTUNE
  :members:
  :inherited-members:
  
Example
-------

Example of using grid search to tune three ES hyperparameters for solving the 5-d Sphere function 

.. code-block:: python

	from neorl.tune import GRIDTUNE
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
	def tune_fit(cxpb, mutpb, alpha):
	
	    #--setup the parameter space
	    nx=5
	    BOUNDS={}
	    for i in range(1,nx+1):
	        BOUNDS['x'+str(i)]=['float', -100, 100]
	
	    #--setup the ES algorithm
	    es=ES(mode='min', bounds=BOUNDS, fit=sphere, lambda_=80, mu=40, mutpb=mutpb, alpha=alpha,
	         cxmode='blend', cxpb=cxpb, ncores=1, seed=1)
	
	    #--Evolute the ES object and obtains y_best
	    #--turn off verbose for less algorithm print-out when tuning
	    x_best, y_best, es_hist=es.evolute(ngen=100, verbose=0)
	
	    return y_best #returns the best score
	
	#*************************************************************
	# Part III: Tuning
	#*************************************************************
	#Setup the parameter space
	#VERY IMPORTANT: The order of these grids MUST be similar to their order in tune_fit
	#see tune_fit
	param_grid={
	#def tune_fit(cxpb, mutpb, alpha):
	'cxpb': [0.2, 0.4],  #cxpb is first
	'mutpb': [0.05, 0.1],  #mutpb is second
	'alpha': [0.1, 0.2, 0.3, 0.4]}  #alpha is third
	
	#setup a grid tune object
	gtune=GRIDTUNE(param_grid=param_grid, fit=tune_fit)
	#view the generated cases before running them
	print(gtune.hyperparameter_cases)
	#tune the parameters with method .tune
	gridres=gtune.tune(ncores=1, csvname='tune.csv')
	print(gridres)

Notes
-----

- For ``ncores > 1``, the parallel tuning engine starts. **Make sure to run your python script from Terminal NOT from an IDE (e.g. Spyder, Jupyter Notebook)**. IDEs usually crash when running parallel problems with packages like ``joblib`` or ``multiprocessing``. For ``ncores = 1``, IDEs seem to work fine.    
- If there are large number of hyperparameters to tune (large :math:`d`), try nested grid search. First, run a grid search on few parameters first, then fix them to their best, and start another grid search for the next group of hyperparameters, and so on.    
- Always start with coarse grid for all hyperparameters (small :math:`k_i`) to obtain an impression about thier sensitivity. Then, refine the grids for those hyperparameters showing more impact, and run a more detailed grid search.  
- Grid search is ideal to use when the analyst has prior experience on the feasible range of each hyperparameter and the most important hyperparameters to tune. 
