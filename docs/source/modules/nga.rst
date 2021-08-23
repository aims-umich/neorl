.. _nga:

.. automodule:: neorl.hybrid.nga


Neural Genetic Algorithms (NGA)
====================================

A module for the surrogate-based genetic algorithms trained by offline data-driven tri-training approach. The surrogate model used is radial basis function networks (RBFN).

Original paper: Huang, P., Wang, H., & Jin, Y. (2021). Offline data-driven evolutionary optimization based on tri-training. Swarm and Evolutionary Computation, 60, 100800.


What can you use?
--------------------

-  Multi processing: ❌
-  Discrete spaces: ❌
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ❌

Parameters
----------

.. autoclass:: NGA
  :members:
  :inherited-members:
  
Example
-------

.. code-block:: python

	from neorl import NGA
	
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
	
	nga = NGA(mode='min', bounds=BOUNDS, fit=FIT, npop=40, num_warmups=200, 
	          hidden_shape=10, seed=1)
	individuals, surrogate_fit = nga.evolute(ngen=100, verbose=False)
	
	#make evaluation of the best individuals using the real fitness function
	real_fit=[FIT(item) for item in individuals]
	
	#print the best individuals/fitness found
	min_index=real_fit.index(min(real_fit))
	print('------------------------ Final Summary --------------------------')
	print('Best real individual:', individuals[min_index])
	print('Best real fitness:', real_fit[min_index])
	print('-----------------------------------------------------------------')

Notes
-----

- Tri-training concept uses semi-supervised learning to leverage surrogate models that approximate the real fitness function to accelerate the optimization process for expensive fitness functions. Three RBFN models are trained, which are used to determine the best individual from one generation to the next, which is added to retrain the three surrogate models. The real fitness function ``fit`` is ONLY used to evaluate ``num_warmups``. Afterwards, the three RBFN models are used to guide the genetic algorithm optimizer.
- For ``num_warmups``, choose a reasonable value to accommodate the number of design variables ``x`` in your problem. If ``None``, the default value of warmup samples is 20 times the size of ``x``. 
- For ``hidden_shape``, large number of hidden layers can slow down surrogate training, small number can lead to underfitting. If ``None``, the default value of ``hidden_shape`` is :math:`int(\sqrt{int(num_{warmups}/3)})`.
- The ``kernel`` can play a significant role in surrogate training. Four options are available: Gaussian function (``gaussian``), Reflected function (``reflect``), Multiquadric function (``mul``), and Inverse multiquadric function (``inmul``).
- Total number of cost evaluations via the real fitness function ``fit`` for NGA is ``num_warmups``.
- Total number of cost evaluations via the surrogate model for NGA is ``npop`` * ``ngen``.