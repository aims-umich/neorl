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
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ❌

Parameters
----------

.. autoclass:: TS
  :members:
  :inherited-members:
  
Example
-------

.. literalinclude :: ../scripts/ex_ts.py
   :language: python

Notes
-----

- Tabu search (TS) is a metaheuristic algorithm that can be used for solving combinatorial optimization problems (problems where an optimal ordering and selection of options is desired). Also, we adapted TS to solve bounded discrete problems for which the candidate solution needs to be perturbed and bounded.
- For ``swap_mode``, choose ``perturb`` for problems that have lower/upper bounds at which the individual is perturbed between them to find optimal solution (e.g. Sphere function). Choose ``swap`` for combinatorial problems where the elements of the individual are swapped (not perturbed) to find the optimal solution (e.g. Travel Salesman, Job Scheduling).
- ``tabu_tenure`` refers to a short-term set of the solutions that have been visited in the recent past time steps. For example ``tabu_tenure=6`` stores the list of the previous 6 found solutions, which can be reused by the algorithm.
- ``penalization_weight`` represents the importance/frequency of a certain action performed in the search. Large values of ``penalization_weight`` reduces the frequency of using the same action again in the search. 