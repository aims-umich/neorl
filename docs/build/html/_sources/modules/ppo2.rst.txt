.. _ppo2:

.. automodule:: neorl.rl.baselines.ppo2

Proximal Policy Optimisation (PPO)
===================================

The `Proximal Policy Optimization <https://arxiv.org/abs/1707.06347>`_ algorithm combines ideas from A2C (having multiple workers) and TRPO (it uses a trust region to improve the actor). The idea is that after an update, the new policy should be not be too far from the old policy.
For that, PPO uses clipping to avoid too large update.

Original paper: https://arxiv.org/abs/1707.06347

What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ✔️
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ✔️

Parameters
----------

.. autoclass:: PPO2
  :members:
  :inherited-members:

.. autoclass:: neorl.rl.make_env.CreateEnvironment

.. autoclass:: neorl.utils.neorlcalls.RLLogger

Example
-------

Train a PPO agent to optimize the 5-D sphere function

.. literalinclude :: ../scripts/ex_ppo.py
   :language: python

Acknowledgment
-----------------

Thanks to our fellows in `stable-baselines <https://github.com/hill-a/stable-baselines>`_, as we used their standalone RL implementation, which is utilized as a baseline to leverage advanced neuroevolution algorithms. 

Hill, Ashley, et al. "Stable baselines." (2018).