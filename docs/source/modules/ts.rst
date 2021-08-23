.. _ts:

.. automodule:: neorl.evolu.ts


Tabu Search (TS)
===============================================

A module for the tabu search with long-term memory for discrete/combinatorial optimization.

Original papers: 

- Glover, F. (1989). Tabu search—part I. ORSA Journal on computing, 1(3), 190-206.

- Glover, F. (1990). Tabu search—part II. ORSA Journal on computing, 2(1), 4-32.

What can you use?
--------------------

-  Multi processing: ❌
-  Discrete spaces: ✔️
-  Continuous spaces: ❌
-  Mixed Discrete/Continuous spaces: ❌

Parameters
----------

.. autoclass:: TS
  :members:
  :inherited-members:
  
Example
-------

.. code-block:: python

	from neorl import TS
	
	#Define the fitness function
	def Sphere(individual):
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
	for i in range(nx):
	    BOUNDS['x'+str(i)]=['int', -100, 100]
	
	#setup and evolute TS
	x0=[-25,50,100,-75,-100]  #initial guess
	ts=TS(mode = "min", bounds = BOUNDS, fit = Sphere, tabu_tenure=60, 
	      penalization_weight = 0.8, swap_mode = "perturb", ncores=1, seed=1)
	x_best, y_best, ts_hist=ts.evolute(ngen = 700, x0=x0, verbose=1)

Notes
-----

- Tabu search (TS) is a metaheuristic algorithm that can be used for solving combinatorial optimization problems (problems where an optimal ordering and selection of options is desired). Also, we adapted TS to solve bounded discrete problems for which the candidate solution needs to be perturbed and bounded.
- For ``swap_mode``, choose ``perturb`` for problems that have lower/upper bounds at which the individual is perturbed between them to find optimal solution (e.g. Sphere function). Choose ``swap`` for combinatorial problems where the elements of the individual are swapped (not perturbed) to find the optimal solution (e.g. Travel Salesman, Job Scheduling).
- ``tabu_tenure`` refers to a short-term set of the solutions that have been visited in the recent past time steps. For example ``tabu_tenure=6`` stores the list of the previous 6 found solutions, which can be reused by the algorithm.
- ``penalization_weight`` represents the importance/frequency of a certain action performed in the search. Large values of ``penalization_weight`` reduces the frequency of using the same action again in the search. 