.. _ppoes:

.. automodule:: neorl.hybrid.ppoes

RL-informed Evolution Strategies (PPO-ES)
================================================

The Proximal Policy Optimization algorithm starts the search to collect some individuals given a fitness function through a RL environment. In the second step, the best PPO individuals are used to guide evolution strategies (ES), where RL individuals are randomly introduced into the ES population to enrich their diversity. The user first runs PPO search followed by ES, the best results of both stages are reported to the user. 
 
Original papers: 

- Radaideh, M. I., & Shirvan, K. (2021). Rule-based reinforcement learning methodology to inform evolutionary algorithms for constrained optimization of engineering applications. Knowledge-Based Systems, 217, 106836.

- Radaideh, M. I., Forget, B., & Shirvan, K. (2021). Large-scale design optimisation of boiling water reactor bundles with neuroevolution. Annals of Nuclear Energy, 160, 108355.

What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ✔️
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ✔️

Parameters
----------

.. autoclass:: PPOES
  :members:
  :inherited-members:

.. autoclass:: neorl.rl.make_env.CreateEnvironment
   :noindex:

Example
-------

Train a PPO-ES agent to optimize the 5-D sphere function

.. literalinclude :: ../../scripts/ex_ppoes.py
   :language: python