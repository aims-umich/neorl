.. _sa:

.. automodule:: neorl.evolu.sa

Simulated Annealing (SA)
==================================

A module for parallel Simulated Annealing. A Synchronous Approach with Occasional Enforcement of Best Solution. 

Original paper: Onbaşoğlu, E., Özdamar, L. (2001). Parallel simulated annealing algorithms in global optimization. Journal of global optimization, 19(1), 27-50..

What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ✔️
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ✔️

Parameters
----------

.. autoclass:: SA
  :members:
  :inherited-members:
  
Example
-------

.. code-block:: python

	from neorl import SA
	import matplotlib.pyplot as plt
	import random
	
	#Define the fitness function
	def FIT(individual):
	    """Sphere test objective function.
	    """
	    y=sum(x**2 for x in individual)
	    return y
	
	#Setup the parameter space (d=5)
	nx=5
	BOUNDS={}
	for i in range(1,nx+1):
	    BOUNDS['x'+str(i)]=['float', -100, 100]
	
	#define a custom moving function
	def my_move(x, **kwargs):
	    #-----
	    #this function selects two random indices in x and perturb their values
	    #-----
	    x_new=x.copy()
	    indices=random.sample(range(0,len(x)), 2)
	    for i in indices:
	        x_new[i] = random.uniform(BOUNDS['x1'][1],BOUNDS['x1'][2])
	    
	    return x_new
	
	#setup and evolute SA
	sa=SA(mode='min', bounds=BOUNDS, fit=FIT, chain_size=30,chi=0.2, 
	      move_func=my_move, reinforce_best=True, ncores=1, seed=1)
	x_best, y_best, sa_hist=sa.evolute(ngen=100, verbose=1)
	
	#plot different statistics
	plt.figure()
	plt.plot(sa_hist['accept'], '-o', label='Acceptance')
	plt.plot(sa_hist['reject'], '-s', label='Rejection')
	plt.plot(sa_hist['improve'], '-^', label='Improvement')
	plt.xlabel('Generation')
	plt.ylabel('Rate (%)')
	plt.legend()
	plt.show()

Notes
-----

- Temperature is annealed between ``Tmax`` and ``Tmin`` using three different ``cooling`` schedules: ``fast``, ``cauchy``, and ``boltzmann``. 
- Custom ``move_func`` is allowed by following the input/output format in the example above. If ``None`` the default moving function is used, which is controlled by the ``chi`` parameter. Therefore, ``chi`` is used ONLY if ``move_func=None``.
- ``chi`` controls the probability of perturbing an attribute of the individual. For example, for ``d=4``, :math:`\vec{x}=[x_1,x_2,x_3,x_4]`, for every :math:`x_i`, a uniform random number :math:`U[0,1]` is compared to ``chi``, if ``U[0,1] < chi``, the attribute is perturbed. Otherwise, it remains fixed.   
- For every generation, a total of ``chain_size`` individuals are executed. Therefore, look for an optimal balance between ``chain_size`` and ``ngen``.
- Total number of cost evaluations for SA is ``chain_size`` * ``ngen``.
- If ``ncores > 1``, parallel SA chains are initialized to accelerate the calculations of all ``chain_size`` * ``ngen``. 
- If ``ncores > 1`` and ``move_func=None``, parallel SA chains can have different ``chi`` values, provided as a list/vector.
- Option ``reinforce_best=True`` allows enforcing the best solution from previous generation to use as a chain startup in the next generation. For example, if ``chain_size=10``, the best of the 10 individuals in the first generation is used to initialize the chain in the second generation, and so on. Different ``ncores`` chains have different best individuals (initial guess) to avoid biasing the parallel search toward a specific chain.  