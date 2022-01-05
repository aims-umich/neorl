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
   :noindex:

.. autoclass:: neorl.utils.neorlcalls.RLLogger
   :noindex:

Example
-------

Train a PPO agent to optimize the 5-D sphere function

.. literalinclude :: ../../scripts/ex_ppo.py
   :language: python

Notes
-------	

- PPO is the most popular RL algorithm due to its robustness. PPO is parallel and supports all types of spaces.
- PPO shows sensitivity to ``n_steps``, ``vf_coef``, ``ent_coef``, and ``lam``. It is always good to consider tuning these hyperparameters before using for optimization. In particular, ``n_steps`` is considered the most important parameter to tune for PPO. Always start with small ``n_steps`` and increase as needed. 
- For PPO, always ensure that ``ncores`` * ``n_steps`` is divisible by ``nminibatches``. For example, if ``nminibatches=4``, then ``ncores=12``/``n_steps=5`` setting works, while ``ncores=5``/``n_steps=5`` will fail. For tuning purposes, it is recommended to choose ``ncores`` divisible by ``nminibatches`` so that you can change ``n_steps`` more freely.  
- The cost of PPO equals to the ``total_timesteps`` in the ``learn`` function, where the original fitness function will be accessed ``total_timesteps`` times.
- See how PPO is used to solve two common combinatorial problems in :ref:`TSP <ex1>` and :ref:`KP <ex10>`.

Acknowledgment
-----------------

Thanks to our fellows in `stable-baselines <https://github.com/hill-a/stable-baselines>`_, as we used their standalone RL implementation, which is utilized as a baseline to leverage advanced neuroevolution algorithms. 

Hill, Ashley, et al. "Stable baselines." (2018).