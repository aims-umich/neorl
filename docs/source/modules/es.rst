.. _es:

.. automodule:: neorl.evolu.es


Evolution Strategies (ES)
============================

A module for the evolution strategies (:math:`\mu,\lambda`) with adaptive strategy vectors. 

Original paper: Bäck, T., Fogel, D. B., Michalewicz, Z. (Eds.). (2018). Evolutionary computation 1: Basic algorithms and operators. CRC press.


What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ✔️
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ✔️

Parameters
----------

.. autoclass:: ES
  :members:
  :inherited-members:
  
Example
-------

.. code-block:: python

	from neorl import ES
	
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
	
	ga=ES(bounds=BOUNDS, fit=FIT, lambda_=80, mu=40, mutpb=0.25,
	     cxmode='blend', cxpb=0.7, ncores=1, seed=1)
	x_best, y_best, es_hist=ga.evolute(ngen=100, verbose=0)

Notes
-----

- Too large population mutation rate ``mutpb`` could destroy the population, the recommended range for this variable is between 0.01-0.4. 
- Too large ``smax`` will allow the individual to be perturbed in a large rate.  
- Too small ``cxpb`` and ``mutpb`` reduce ES exploration, and increase the likelihood of falling in a local optima.
- Usually, population size ``lambda_`` between 60-100 shows good performance along with ``mu=0.5*lambda_``. 
- Look for an optimal balance between ``lambda_`` and ``ngen``, it is recommended to minimize population size to allow for more generations.
- Total number of cost evaluations for ES is ``npop`` * ``ngen``.
- ``cxmode='blend'`` with ``alpha=0.5`` may perform better than ``cxmode='cx2point'``.