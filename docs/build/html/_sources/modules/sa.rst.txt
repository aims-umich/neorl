.. _sa:

.. automodule:: neorl.evolu.sa

Simulated Annealing (SA)
==================================

A module for parallel Simulated Annealing. A Synchronous Approach with Occasional Enforcement of Best Solution. 

Original paper: Onbaşoğlu, E., Özdamar, L. (2001). Parallel simulated annealing algorithms in global optimization. Journal of global optimization, 19(1), 27-50..

.. image:: ../images/sa.jpeg
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

.. literalinclude :: ../scripts/ex_sa.py
   :language: python

Notes
-----

- Temperature is annealed between ``Tmax`` and ``Tmin`` using three different ``cooling`` schedules: ``fast``, ``cauchy``, and ``boltzmann``. 
- Custom ``move_func`` is allowed by following the input/output format in the example above. If ``None`` the default moving function is used, which is controlled by the ``chi`` parameter. Therefore, ``chi`` is used ONLY if ``move_func=None``.
- ``chi`` controls the probability of perturbing an attribute of the individual. For example, for ``d=4``, :math:`\vec{x}=[x_1,x_2,x_3,x_4]`, for every :math:`x_i`, a uniform random number :math:`U[0,1]` is compared to ``chi``, if ``U[0,1] < chi``, the attribute is perturbed. Otherwise, it remains fixed.   
- For every generation, a total of ``chain_size`` individuals are executed. Therefore, look for an optimal balance between ``chain_size`` and ``ngen``.
- Total number of cost evaluations for SA is ``chain_size`` * ``(ngen + 1)``.
- If ``ncores > 1``, parallel SA chains are initialized to accelerate the calculations of all ``chain_size`` * ``ngen``. 
- If ``ncores > 1`` and ``move_func=None``, parallel SA chains can have different ``chi`` values, provided as a list/vector.
- Option ``reinforce_best=True`` allows enforcing the best solution from previous generation to use as a chain startup in the next generation. For example, if ``chain_size=10``, the best of the 10 individuals in the first generation is used to initialize the chain in the second generation, and so on. Different ``ncores`` chains have different best individuals (initial guess) to avoid biasing the parallel search toward a specific chain.  