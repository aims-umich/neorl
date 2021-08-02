.. _de:

.. automodule:: neorl.evolu.de


Differential Evolution (DE)
===========================

A module for differential evolution (DE) that optimizes a problem by iteratively trying to improve a candidate solution. DE maintains the population of candidate solutions and creating new candidate solutions by combining existing ones according to simple combination methods. The candidate solution with the best score/fitness is reported by DE.

Original paper: Storn, Rainer, and Kenneth Price. "Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces." Journal of global optimization 11.4 (1997): 341-359.


What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ✔️
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ✔️

Parameters
----------

.. autoclass:: DE
  :members:
  :inherited-members:

Example
-------

.. code-block:: python

	from neorl import DE

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
	
	de=DE(mode='min', bounds=BOUNDS, fit=FIT, npop=60, F=0.5, CR=0.7, ncores=1, seed=1)
	x_best, y_best, de_hist=de.evolute(ngen=100, verbose=1)

Notes
-----

- Start with a crossover probability ``CR`` considerably lower than one (e.g. 0.2-0.3). If no convergence is achieved, increase to higher levels (e.g. 0.8-0.9) 
- ``F`` is usually chosen between [0.5, 1].
- The higher the population size ``npop``, the lower one should choose the weighting factor ``F``
- You may start with ``npop`` =10*d, where d is the number of input parameters to optimise (degrees of freedom).