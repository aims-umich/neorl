.. _a2c:

.. automodule:: neorl.rl.baselines.a2c


Advantage Actor Critic (A2C)
==============================

A synchronous, deterministic variant of `Asynchronous Advantage Actor Critic (A3C) <https://arxiv.org/abs/1602.01783>`_.
It uses multiple workers to avoid the use of a replay buffer.

Original paper:  https://arxiv.org/abs/1602.01783

What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ✔️
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ✔️

Parameters
----------

.. autoclass:: A2C
  :members:
  :inherited-members:
  
.. autoclass:: neorl.rl.make_env.CreateEnvironment

.. autoclass:: neorl.utils.neorlcalls.RLLogger

Example
-------

Train an A2C agent to optimize the 5-D sphere function

.. literalinclude :: ../scripts/ex_a2c.py
   :language: python

Notes
-------	

- A2C belongs to the actor-critic family, and usually considered as the state-of-the-art in the reinforcement learning domain. A2C is parallel and supports all types of spaces.
- A2C shows sensitivity to ``n_steps``, ``vf_coef``, ``ent_coef``, and ``learning_rate``. It is always good to consider tuning these hyperparameters before using for optimization. In particular, ``n_steps`` is considered the most important parameter to tune for A2C. Always start with small ``n_steps`` and increase as needed. 
- The cost of A2C equals to the ``total_timesteps`` in the ``learn`` function, where the original fitness function will be accessed ``total_timesteps`` times.
- See how A2C is used to solve two common combinatorial problems in :ref:`TSP <ex1>` and :ref:`KP <ex10>`.
  
Acknowledgment
-----------------

Thanks to our fellows in `stable-baselines <https://github.com/hill-a/stable-baselines>`_, as we used their standalone RL implementation, which is utilized as a baseline to leverage advanced neuroevolution algorithms. 

Hill, Ashley, et al. "Stable baselines." (2018).


