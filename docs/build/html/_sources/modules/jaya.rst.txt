.. _jaya:

.. automodule:: neorl.evolu.jaya


JAYA Algorithm
===============================================

A module for the JAYA Algorithm with parallel computing support. 

Original paper: Rao, R. (2016). Jaya: A simple and new optimization algorithm for solving constrained and unconstrained optimization problems. International Journal of Industrial Engineering Computations, 7(1), 19-34.


What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ❌
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ❌

Parameters
----------

.. autoclass:: JAYA
  :members:
  :inherited-members:
  
Example
-------

.. code-block:: python

	from neorl import JAYA
	
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
	
	#setup and evolute JAYA
	jaya=JAYA(mode='min', bounds=BOUNDS, fit=FIT, npop=60, ncores=1, seed=1)
	x_best, y_best, jaya_hist=jaya.evolute(ngen=200, verbose=1)

Notes
-----

- JAYA concept is very simple that any optimization algorithm should look for solutions that move towards the best solution and should avoid the worst solution. Therefore, JAYA keeps tracking of both the best and worst solutions and varies the population accordingly.
- JAYA is free of special hyperparameters, therefore, the user only needs to specify the size of the population ``npop``.
- ``ncores`` argument evaluates the fitness of all individuals in the population in parallel. Therefore, set ``ncores <= npop`` for most optimal resource allocation.
- Look for an optimal balance between ``npop`` and ``ngen``, it is recommended to minimize the number of ``npop`` to allow for more updates and more generations.
- Total number of cost evaluations for JAYA is ``npop`` * ``ngen``.