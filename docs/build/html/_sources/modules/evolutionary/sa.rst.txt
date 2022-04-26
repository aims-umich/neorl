.. _sa:

.. automodule:: neorl.evolu.sa

Simulated Annealing (SA)
==================================

A module for parallel Simulated Annealing. A Synchronous Approach with Occasional Enforcement of Best Solution. 

Original paper: Onbaşoğlu, E., Özdamar, L. (2001). Parallel simulated annealing algorithms in global optimization. Journal of global optimization, 19(1), 27-50..

.. image:: ../../images/sa.jpeg
   :scale: 30%
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

.. autoclass:: SA
  :members:
  :inherited-members:
  
Example
-------

.. literalinclude :: ../../scripts/ex_sa.py
   :language: python

Notes
-----

- Temperature is annealed between ``Tmax`` and ``Tmin`` if any of the following ``cooling`` schedules is used: ``fast``, ``cauchy``, and ``boltzmann``. 
- A special cooling schedule ``equilibrium`` is supported, which is associated with the following arguments:

	1. The initial temperature is determined by :math:`T_0=Max (\alpha * STD(\vec{E}_0), Tmin)`, where :math:`STD(\vec{E}_0)` is the standard deviation of the finesses of the initial chains. A tolerance is applied to the temperature if the standard deviation converged to zero.   
	2. ``alpha`` is the parameter above that controls the initial temperature of SA.  
	3. ``lmbda`` expresses the cooling rate or the speed of the temperature decay. Larger values lead to faster cooling. 
	4.  The ``threshold`` (in \%) expresses the acceptance rate threshold under which the SA stops running. For example, if ``threshold=10``, when the mean of acceptance rate of all chains falls below 10\%, SA terminates. For zero threshold, SA will terminate when all chains no longer accept any new solution.
	5. The ``equilibrium`` cooling option is activated for parallel SA chains only, i.e. when ``ncores > 1``.
	
- Custom ``move_func`` is allowed by following the input/output format in the example above. If ``None`` the default moving function is used, which is controlled by the ``chi`` parameter. Therefore, ``chi`` is used ONLY if ``move_func=None``.
- ``chi`` controls the probability of perturbing an attribute of the individual. For example, for ``d=4``, :math:`\vec{x}=[x_1,x_2,x_3,x_4]`, for every :math:`x_i`, a uniform random number :math:`U[0,1]` is compared to ``chi``, if ``U[0,1] < chi``, the attribute is perturbed. Otherwise, it remains fixed.   
- For every generation, a total of ``chain_size`` individuals are executed. Therefore, look for an optimal balance between ``chain_size`` and ``ngen``.

- Option ``reinforce_best`` allows enforcing a solution from the previous generation to use as a chain startup in the next generation. Three options are available for this argument:

	1. ``None``: No solution is enforced. The last chain state is preserved to the next generation.
	2. ``hard``: the best individual in the chain is used as the initial starting point.   
	3. ``soft``: an energy-based sampling approach is utilized to draw an individual from the chain to start the next generation. 

- Total number of cost evaluations for SA is ``chain_size`` * ``(ngen + 1)``.
- If ``ncores > 1``, parallel SA chains are initialized to accelerate the calculations. 
- If ``ncores > 1`` and ``move_func=None``, parallel SA chains can have different ``chi`` values, provided as a list/vector.