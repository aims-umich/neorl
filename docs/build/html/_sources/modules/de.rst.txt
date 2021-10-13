.. _de:

.. automodule:: neorl.evolu.de


Differential Evolution (DE)
===========================

A module for differential evolution (DE) that optimizes a problem by iteratively trying to improve a candidate solution. DE maintains the population of candidate solutions and creating new candidate solutions by combining existing ones according to simple combination methods. The candidate solution with the best score/fitness is reported by DE.

Original paper: Storn, Rainer, and Kenneth Price. "Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces." Journal of global optimization 11.4 (1997): 341-359.


What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ✔️
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ✔️

Parameters
----------

.. autoclass:: DE
  :members:
  :inherited-members:

Example
-------

.. literalinclude :: ../scripts/ex_de.py
   :language: python

Notes
-----

- Start with a crossover probability ``CR`` considerably lower than one (e.g. 0.2-0.3). If no convergence is achieved, increase to higher levels (e.g. 0.8-0.9) 
- ``F`` is usually chosen between [0.5, 1].
- The higher the population size ``npop``, the lower one should choose the weighting factor ``F``
- You may start with ``npop`` =10*d, where d is the number of input parameters to optimise (degrees of freedom).
- Total number of cost evaluations for DE is ``2 * npop * ngen``.