.. _jaya:

.. automodule:: neorl.evolu.jaya


JAYA Algorithm
===============================================

A module for the JAYA Algorithm with parallel computing support. 

Original paper: Rao, R. (2016). Jaya: A simple and new optimization algorithm for solving constrained and unconstrained optimization problems. International Journal of Industrial Engineering Computations, 7(1), 19-34.


What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ✔️
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ✔️

Parameters
----------

.. autoclass:: JAYA
  :members:
  :inherited-members:
  
Example
-------

.. literalinclude :: ../../scripts/ex_jaya.py
   :language: python

Notes
-----

- JAYA concept is very simple that any optimization algorithm should look for solutions that move towards the best solution and should avoid the worst solution. Therefore, JAYA keeps tracking of both the best and worst solutions and varies the population accordingly.
- JAYA is free of special hyperparameters, therefore, the user only needs to specify the size of the population ``npop``.
- ``ncores`` argument evaluates the fitness of all individuals in the population in parallel. Therefore, set ``ncores <= npop`` for most optimal resource allocation.
- Look for an optimal balance between ``npop`` and ``ngen``, it is recommended to minimize the number of ``npop`` to allow for more updates and more generations.
- Total number of cost evaluations for JAYA is ``npop`` * ``(ngen + 1)``.