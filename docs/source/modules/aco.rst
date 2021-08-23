.. _aco:

.. automodule:: neorl.evolu.aco


Ant Colony Optimization (ACO)
===============================================

A module for the Ant Colony Optimization with parallel computing support and continuous optimization ability. 

Original paper: Socha, K., & Dorigo, M. (2008). Ant colony optimization for continuous domains. European journal of operational research, 185(3), 1155-1173.

.. image:: ../images/aco.jpg
   :scale: 30%
   :alt: alternate text
   :align: center

What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ❌
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ❌

Parameters
----------

.. autoclass:: ACO
  :members:
  :inherited-members:
  
Example
-------

.. code-block:: python

	from neorl import ACO 
	import random
	
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
	
	nants=40
	x0=[[random.uniform(-100,100)]*nx for item in range(nants)]
	acor = ACO(mode='min', fit=FIT, bounds=BOUNDS, nants=nants, narchive=10, 
	           Q=0.5, Z=1, ncores=1, seed=1)
	x_best, y_best, acor_hist=acor.evolute(ngen=100, x0=x0, verbose=1)

Notes
-----

- ACO is inspired from the cooperative behavior and food search of ants. Several ants cooperatively search for food in different directions in an attempt to find the global optima (richest food source).
- For ACO, the archive of best ants is ``narchive``, where it must be less than the population size ``nants``.
- The factor ``Q`` controls the rate of exploration/exploitation of ACO. When ``Q`` is small, the best-ranked solutions are strongly preferred next (more exploitation), and when ``Q`` is large, the probability of all solutions become more uniform (more exploration).
- The factor ``Z`` is the pheromone evaporation rate, which controls search behavior. As ``Z`` increases, the search becomes less biased towards the points of the search space that have been already explored, which are kept in the archive. In general, the higher the value of ``Z``, the lower the convergence speed of ACO.
- ``ncores`` argument evaluates the fitness of all ants in parallel after the position update. Therefore, set ``ncores <= nants`` for most optimal resource allocation.
- Look for an optimal balance between ``nants`` and ``ngen``, it is recommended to minimize the number of ``nants`` to allow for more updates and more generations.
- Total number of cost evaluations for ACO is ``nants`` * ``ngen``.