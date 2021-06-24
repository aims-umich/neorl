.. _mfo:

.. automodule:: neorl.evolu.mfo


Moth-flame Optimization (MFO)
===============================================

A module for the Moth-flame Optimization with parallel computing support. 

Original paper: Mirjalili, S. (2015). Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm. Knowledge-based systems, 89, 228-249.

.. image:: ../images/mfo.jpg
   :scale: 11%
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

.. autoclass:: MFO
  :members:
  :inherited-members:
  
Example
-------

.. code-block:: python

	from neorl import MFO
	
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
	
	#setup and evolute MFO
	mfo=MFO(mode='min', bounds=BOUNDS, fit=FIT, nmoths=50, ncores=1, seed=1)
	x_best, y_best, mfo_hist=mfo.evolute(ngen=200, verbose=1)

Notes
-----

- MFO mimics the navigation behavior of moths in nature. Moths fly in night by maintaining a fixed angle with respect to the moon to travel in a straight line for long distances. However, the moths may get trapped in a deadly spiral path around artificial lights (i.e. called flames). This algorithm models this behavior to perform optimization by escaping the local/deadly regions during search.
- MFO creates two equal arrays of moth and flame positions. The moths are actual search agents that move around the search space, whereas flames are the best position of moths that obtains so far. Therefore, the flame can be seen as checkpoint of the best solutions found by the moths during the search. 
- A logarithmic spiral is used as the main update mechanism of moths, which is controlled by the parameter ``b``.
- MFO emphasizes exploitation through annealing an internal parameter ``r`` between -1 and -2. The value of ``r`` plays a factor in convergence as the moths prioritize their best solutions as we approach the value of ``ngen``.
- ``ncores`` argument evaluates the fitness of all moths in parallel. Therefore, set ``ncores <= nmoths`` for most optimal resource allocation.
- Look for an optimal balance between ``nmoths`` and ``ngen``, it is recommended to minimize the number of ``nmoths`` to allow for more updates and more generations.
- Total number of cost evaluations for MFO is ``nmoths`` * ``ngen``.