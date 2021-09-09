.. _pso:

.. automodule:: neorl.evolu.pso


Particle Swarm Optimisation (PSO)
==================================

A module for particle swarm optimisation with three different speed mechanisms. 

Original papers: 

- Kennedy, J., Eberhart, R. (1995). Particle swarm optimization. In: Proceedings of ICNN'95-international conference on neural networks (Vol. 4, pp. 1942-1948), IEEE.
- Kennedy, J., & Eberhart, R. C. (1997). A discrete binary version of the particle swarm algorithm. In: 1997 IEEE International conference on systems, man, and cybernetics. Computational cybernetics and simulation (Vol. 5, pp. 4104-4108), IEEE.
- Clerc, M., Kennedy, J. (2002). The particle swarm-explosion, stability, and convergence in a multidimensional complex space. IEEE transactions on Evolutionary Computation, 6(1), 58-73.

.. image:: ../images/pso.jpg
   :scale: 45%
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

.. autoclass:: PSO
  :members:
  :inherited-members:
  
Example
-------

.. literalinclude :: ../scripts/ex_pso.py
   :language: python

Notes
-----

- Always try the three speed mechanisms via ``speed_mech`` when you solve any problem. 
- Keep c1, c2 > 2.0 when using ``speed_mech='constric'``. 
- ``speed_mech=timew`` uses a time-dependent inertia factor, where inertia ``w`` is annealed over PSO generations.
- ``speed_mech=globw`` uses a ratio of swarm global position to local position to define inertia factor, and this factor is updated every generation.
- Look for an optimal balance between ``npar`` and ``ngen``, it is recommended to minimize particle size to allow for more generations.
- Total number of cost evaluations for PSO is ``npar`` * ``ngen``.
