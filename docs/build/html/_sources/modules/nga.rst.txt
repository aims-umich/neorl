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

.. literalinclude :: ../scripts/ex_nga.py
   :language: python

Notes
-----

- Tri-training concept uses semi-supervised learning to leverage surrogate models that approximate the real fitness function to accelerate the optimization process for expensive fitness functions. Three RBFN models are trained, which are used to determine the best individual from one generation to the next, which is added to retrain the three surrogate models. The real fitness function ``fit`` is ONLY used to evaluate ``num_warmups``. Afterwards, the three RBFN models are used to guide the genetic algorithm optimizer.
- For ``num_warmups``, choose a reasonable value to accommodate the number of design variables ``x`` in your problem. If ``None``, the default value of warmup samples is 20 times the size of ``x``. 
- For ``hidden_shape``, large number of hidden layers can slow down surrogate training, small number can lead to underfitting. If ``None``, the default value of ``hidden_shape`` is :math:`int(\sqrt{int(num_{warmups}/3)})`.
- The ``kernel`` can play a significant role in surrogate training. Four options are available: Gaussian function (``gaussian``), Reflected function (``reflect``), Multiquadric function (``mul``), and Inverse multiquadric function (``inmul``).
- Total number of cost evaluations via the real fitness function ``fit`` for NGA is ``num_warmups``.
- Total number of cost evaluations via the surrogate model for NGA is ``npop`` * ``ngen``.