.. _ga:

.. automodule:: neorl.evolu.ga


Genetic Algorithms (GA)
===========================

A module for classical genetic algorithms with constant mutation strengths on population and individual levels. The user can control the number of individuals from the population to participate in the next generation offspring (``mu`` <= ``npop``).

Original paper: Bäck, T., Fogel, D. B., Michalewicz, Z. (Eds.). (2018). Evolutionary computation 1: Basic algorithms and operators. CRC press.


What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ✔️
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ✔️

Parameters
----------

.. autoclass:: GA
  :members:
  :inherited-members:

Example
-------

.. code-block:: python

	from neorl import GA

	#Define the sphere (fitness) function
	def FIT(individual):
		y=sum(x**2 for x in individual)
		return -y  #-1 to convert min to max problem

	#Setup the parameter space (d=5)
	nx=5
	BOUNDS={}
	for i in range(1,nx+1):
		BOUNDS['x'+str(i)]=['float', -100, 100]

	ga=GA(bounds=BOUNDS, fit=FIT, npop=60, mutate=0.2, 
	     cx='2point', cxpb=0.7, chi=0.1, 
		 ncores=1, seed=1)
	x_best, y_best, ga_hist=ga.evolute(ngen=100, verbose=0)

Notes
-----

- Too large population mutation rate ``mutate`` could destroy the population, the recommended range for this variable is between 0.01-0.4. 
- Too large individual mutation rate ``chi`` could destroy the individual and hence the population, the recommended range for this variable is 0.01-0.3. 
- Too small ``cxpb`` and ``mutate`` reduce GA exploration, and increase the likelihood of falling in a local optima.
- Usually, population size ``npop`` between 40-80 shows good performance. 
- Look for an optimal balance between ``npop`` and ``ngen``, it is recommended to minimize population size to allow for more generations.
- Total number of cost evaluations for GA is ``npop`` * ``ngen``.
