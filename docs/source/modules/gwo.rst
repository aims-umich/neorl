.. _gwo:

.. automodule:: neorl.evolu.gwo


Grey Wolf Optimizer (GWO)
===============================================

A module for the Grey Wolf Optimizer with parallel computing support. 

Original paper: Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer. Advances in engineering software, 69, 46-61.

.. image:: ../images/gwo.jpg
   :scale: 40%
   :alt: alternate text
   :align: center


What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ✔️
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ✔️

Parameters
----------

.. autoclass:: GWO
  :members:
  :inherited-members:
  
Example
-------

.. code-block:: python

	from neorl import GWO
	import matplotlib.pyplot as plt
	    
	#Define the fitness function
	def FIT(individual):
	    """Sphere test objective function.
	            F(x) = sum_{i=1}^d xi^2
	            d=1,2,3,...
	            Range: [-100,100]
	            Minima: 0
	    """
	    y=sum(x**2 for x in individual)
	    return y
	    
	#Setup the parameter space (d=5)
	nx=5
	BOUNDS={}
	for i in range(1,nx+1):
	    BOUNDS['x'+str(i)]=['float', -100, 100]
	
	nwolves=5
	gwo=GWO(mode='min', fit=FIT, bounds=BOUNDS, nwolves=nwolves, ncores=1, seed=1)
	x_best, y_best, gwo_hist=gwo.evolute(ngen=100, verbose=1)
	
	#-----
	#or with fixed initial guess for all wolves (uncomment below)
	#-----
	#x0=[[-90, -85, -80, 70, 90] for i in range(nwolves)]
	#x_best, y_best, gwo_hist=gwo.evolute(ngen=100, x0=x0)
	
	plt.figure()
	plt.plot(gwo_hist['alpha_wolf'], label='alpha_wolf')
	plt.plot(gwo_hist['beta_wolf'], label='beta_wolf')
	plt.plot(gwo_hist['delta_wolf'], label='delta_wolf')
	plt.plot(gwo_hist['fitness'], label='best')
	plt.xlabel('Generation')
	plt.ylabel('Fitness')
	plt.legend()
	plt.show()

Notes
-----

- GWO assigns the best fitness to the first wolf (called **Alpha**), second best fitness to **Beta** wolf, third best fitness to **Delta** wolf, while the remaining wolves in the group are called **Omega**, which follow the leadership and position of Alpha, Beta, and Delta.  
- ``ncores`` argument evaluates the fitness of all wolves in the group in parallel. Therefore, set ``ncores <= nwolves`` for most optimal resource allocation.
- Look for an optimal balance between ``nwolves`` and ``ngen``, it is recommended to minimize the number of ``nwolves`` to allow for more updates and more generations.
- Total number of cost evaluations for GWO is ``nwolves`` * ``ngen``.