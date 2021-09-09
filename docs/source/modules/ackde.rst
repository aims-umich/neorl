.. _ackde:

.. automodule:: neorl.hybrid.ackde

RL-informed Differential Evolution (ACKTR-DE)
================================================

The Actor Critic using Kronecker-Factored Trust Region (ACKTR) algorithm starts the search to collect some individuals given a fitness function through a RL environment. In the second step, the best ACKTR individuals are used to guide differential evolution (DE), where RL individuals are randomly introduced into the DE population to enrich their diversity by replacing the worst DE individuals. The user first runs ACKTR search followed by DE, the best results of both stages are reported to the user. 
 
Original papers: 

- Radaideh, M. I., & Shirvan, K. (2021). Rule-based reinforcement learning methodology to inform evolutionary algorithms for constrained optimization of engineering applications. Knowledge-Based Systems, 217, 106836.

What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ✔️
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ✔️

Parameters
----------

.. autoclass:: ACKDE
  :members:
  :inherited-members:

.. autoclass:: neorl.rl.make_env.CreateEnvironment

Example
-------

Train a ACKTR-DE agent to optimize the 5-D sphere function

.. literalinclude :: ../scripts/ex_ackde.py
   :language: python