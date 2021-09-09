.. _aco:

.. automodule:: neorl.evolu.aco


Ant Colony Optimization (ACO)
===============================================

A module for the Ant Colony Optimization with parallel computing support and continuous optimization ability. 

Original paper: Socha, K., & Dorigo, M. (2008). Ant colony optimization for continuous domains. European journal of operational research, 185(3), 1155-1173.

.. image:: ../images/aco.jpg
   :scale: 30%
   :alt: alternate text
   :align: center

What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ❌
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ❌

Parameters
----------

.. autoclass:: ACO
  :members:
  :inherited-members:
  
Example
-------

.. literalinclude :: ../scripts/ex_aco.py
   :language: python

Notes
-----

- ACO is inspired from the cooperative behavior and food search of ants. Several ants cooperatively search for food in different directions in an attempt to find the global optima (richest food source).
- For ACO, the archive of best ants is ``narchive``, where it must be less than the population size ``nants``.
- The factor ``Q`` controls the rate of exploration/exploitation of ACO. When ``Q`` is small, the best-ranked solutions are strongly preferred next (more exploitation), and when ``Q`` is large, the probability of all solutions become more uniform (more exploration).
- The factor ``Z`` is the pheromone evaporation rate, which controls search behavior. As ``Z`` increases, the search becomes less biased towards the points of the search space that have been already explored, which are kept in the archive. In general, the higher the value of ``Z``, the lower the convergence speed of ACO.
- ``ncores`` argument evaluates the fitness of all ants in parallel after the position update. Therefore, set ``ncores <= nants`` for most optimal resource allocation.
- Look for an optimal balance between ``nants`` and ``ngen``, it is recommended to minimize the number of ``nants`` to allow for more updates and more generations.
- Total number of cost evaluations for ACO is ``nants`` * ``ngen``.