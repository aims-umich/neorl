.. _evolsearch:

.. automodule:: neorl.tune.estune

Evolutionary Search
=====================

A module of evolutionary search for hyperparameter tuning of NEORL algorithms based upon evolution strategies.

Original paper: E. Bochinski, T. Senst and T. Sikora, "Hyper-parameter optimization for convolutional neural network committees based on evolutionary algorithms," 2017 IEEE International Conference on Image Processing (ICIP), Beijing, China, 2017, pp. 3924-3928, doi: 10.1109/ICIP.2017.8297018.

We have used a compact evolution strategy (ES) module for the purpose of tuning hyperparameter of NEORL algorithms. See the :ref:`ES algorithm <es>` section for more details about the (:math:`\mu,\lambda`) algorithm. To reduce the burden on the users, we specified and adapt all ES tuner hyperparameters, so the user needs to specify the hyperparameter space similar to grid, random, and other search methods. ES tuner leverages a population of individuals, where each individual represents a sample from the hyperparameter space. ES uses recombination, crossover, and mutation operations to improve the individuals from generation to the other. The best of the best individuals in all generations are reported as the top hyperparameter sets for the algorithm. 


What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete/Continuous/Mixed spaces: ✔️
-  Reinforcement Learning Algorithms: ✔️
-  Evolutionary Algorithms: ✔️
-  Hybrid Neuroevolution Algorithms: ✔️

Parameters
----------

.. autoclass:: ESTUNE
  :members:
  :inherited-members:
  
Example
-------

.. code-block:: python

	from neorl.tune import ESTUNE
	from neorl import PSO
	
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
	def tune_fit(x):
	    
	    npar=x[0]
	    c1=x[1] 
	    c2=x[2]
	    if x[3] == 1:
	        speed_mech='constric'
	    elif x[3] == 2:
	        speed_mech='timew'
	    elif x[3] == 3:
	        speed_mech='globw'
	
	    #--setup the parameter space
	    nx=5
	    BOUNDS={}
	    for i in range(1,nx+1):
	            BOUNDS['x'+str(i)]=['float', -100, 100]
	
	    #--setup the PSO algorithm
	    pso=PSO(mode='min', bounds=BOUNDS, fit=sphere, npar=npar, c1=c1, c2=c2,
	             speed_mech=speed_mech, ncores=1, seed=1)
	
	    #--Evolute the PSO object and obtains y_best
	    #--turn off verbose for less algorithm print-out when tuning
	    x_best, y_best, pso_hist=pso.evolute(ngen=30, verbose=0)
	
	    return y_best #returns the best score
	
	#*************************************************************
	# Part III: Tuning
	#*************************************************************
	#Setup the parameter space
	#VERY IMPORTANT: The order of these parameters MUST be similar to their order in tune_fit
	#see tune_fit
	param_grid={
	#def tune_fit(npar, c1, c2, speed_mech):
	'npar': ['int', 40, 60],        #npar is first (low=30, high=60, type=int/discrete)
	'c1': ['float', 2.05, 2.15],    #c1 is second (low=2.05, high=2.15, type=float/continuous)
	'c2': ['float', 2.05, 2.15],    #c2 is third (low=2.05, high=2.15, type=float/continuous)
	'speed_mech': ['int', 1, 3]}    #speed_mech is fourth (categorial variable encoded as integer, see tune_fit)
	
	#setup a evolutionary tune object
	etune=ESTUNE(mode='min', param_grid=param_grid, fit=tune_fit, ngen=10) #total cases is ngen * 10
	#tune the parameters with method .tune
	evolures=etune.tune(ncores=1, csvname='evolutune.csv', verbose=True)
	evolures = evolures.sort_values(['score'], axis='index', ascending=True) #rank the scores from min to max
	print(evolures)
	etune.plot_results(pngname='evolu_conv')

Notes
-----
- Evolutionary search uses fixed values for ``lambda_=10`` and ``mu=10``. 
- Therefore, total cost of evolutionary search or total number of hyperparameter tests is ``ngen * 10``.
- For categorical variables, use integers to encode them to convert to integer variables. See how ``speed_mech`` is handled in the example above.  
- The strategy and individual vectors in the ES tuner are updated similarly to the ES algorithm module described :ref:`here <es>`.    
- For difficult problems, the analyst can start with a random search first to narrow the choices of the important hyperparameters. Then, an evolutionary search can be executed on those important parameters to refine their values. 



 
