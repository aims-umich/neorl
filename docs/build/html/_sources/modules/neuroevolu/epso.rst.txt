.. _epso:

.. automodule:: neorl.hybrid.epso


Ensemble Particle Swarm Optimization (EPSO)
=============================================

A powerful hybrid ensemble of five particle swarm optimization variants: classical inertia weight particle swarm optimization (PSO), self-organizing hierarchical particle swarm optimizer withtime-varying acceleration coefficients (HPSO-TVAC), Fitness-Distance-Ratio based PSO (FDR-PSO), Distance-based locally informed PSO (LIPS), and Comprehensive Learning PSO (CLPSO).

Original paper: Lynn, N., Suganthan, P. N. (2017). Ensemble particle swarm optimizer. Applied Soft Computing, 55, 533-548.


What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ✔️
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ✔️

Parameters
----------

.. autoclass:: EPSO
  :members:
  :inherited-members:
  
Example
-------

.. literalinclude :: ../../scripts/ex_epso.py
   :language: python

Notes
-----

- The number of particles in the exploration subgroup (``g1``) and exploitation subgroup (``g2``) are needed for EPSO. In the original algorithm, ``g1`` tends to be smaller than ``g2``. 
- For EPSO, in the first 90\% of the generations, both exploration and exploitation subgroups are involved, where ``g1`` is controlled by CLPSO and ``g2`` is controlled by all five variants. In the last 10\% of the generations, the search focuses on exploitation only, where both ``g1 + g2`` are controlled by the five variants.
- The value of LP represents the learning period at which the success and fail memories are updated to calculate the success rate for each PSO variant. The success rate represents the probability for each PSO variant to update the position and velocity of the next particle in the group. ``LP=3`` means the update will occur every 3 generations. 
- Look for an optimal balance between ``g1``, ``g2``, and ``ngen``, it is recommended to minimize particle size to allow for more generations.
- Total number of cost evaluations for EPSO is ``(g1 + g2)`` * ``(ngen + 1)``.
