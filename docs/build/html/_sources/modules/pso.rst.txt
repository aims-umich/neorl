.. _pso:

.. automodule:: neorl.evolu.pso


Particle Swarm Optimisation (PSO)
==================================

A module for classical genetic algorithms with constant mutation strengths on population and individual levels. 

Original paper: Bäck, T., Fogel, D. B., Michalewicz, Z. (Eds.). (2018). Evolutionary computation 1: Basic algorithms and operators. CRC press.


What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ✔️
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ✔️

Parameters
----------

.. autoclass:: PSO
  :members:
  :inherited-members:
  
Example
-------

.. code-block:: python

	from neorl import GA

	#Define the fitness function
	def FIT(individual):
		"""Sphere test objective function.
			F(x) = sum_{i=1}^d xi^2
			d=1,2,3,...
			Range: [-100,100]
			Minima: 0
		"""
		y=sum(x**2 for x in individual)
		return -y  #-1 to convert min to max problem

	#Setup the parameter space (d=5)
	nx=5
	BOUNDS={}
	for i in range(1,nx+1):
		BOUNDS['x'+str(i)]=['float', -100, 100]

	ga=GA(bounds=BOUNDS, fit=FIT, npop=60, mutate=0.25, 
	     cx='2point', cxpb=0.7, chi=0.1, 
		 ncores=1, seed=1)
	x_best, y_best, ga_hist=ga.evolute(ngen=100, verbose=0)

Notes
-----

- Too large mutation rate "``mutate``" could destroy the population, the recommended range for this variable is between 0.01-0.4. 
- Too large mutation rate "``chi``" could destroy the individual and hence the population, the recommended range for this variable is 0.01-0.3. 
