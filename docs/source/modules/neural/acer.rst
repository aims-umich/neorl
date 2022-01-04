.. _acer:

.. automodule:: neorl.rl.baselines.acer

Actor-Critic with Experience Replay (ACER)
===========================================

Sample Efficient Actor-Critic with Experience Replay (ACER) combines concepts of parallel agents from A2C and provides a replay memory as in DQN. ACER also includes truncated importance sampling with bias correction, stochastic dueling network architectures, and a new trust region policy optimization method.

Original paper: https://arxiv.org/abs/1611.01224

What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ✔️
-  Continuous spaces: ❌
-  Mixed Discrete/Continuous spaces: ❌

Parameters
----------

.. autoclass:: ACER
  :members:
  :inherited-members:

.. autoclass:: neorl.rl.make_env.CreateEnvironment
   :noindex:

.. autoclass:: neorl.utils.neorlcalls.RLLogger
   :noindex:

Example
-------

Train an ACER agent to optimize the 5-D discrete sphere function

.. literalinclude :: ../../scripts/ex_acer.py
   :language: python
  
Notes
-------	

- ACER can be observed as the parallel version of DQN with additional enhancements. ACER is also restricted to discrete spaces.
- ACER shows sensitivity to ``n_steps``, ``q_coef``, and ``ent_coef``. It is always good to consider tuning these hyperparameters before using for optimization. In particular, ``n_steps`` is considered the most important parameter to tune. 
- The cost of ACER equals to the ``total_timesteps`` in the ``learn`` function, where the original fitness function will be accessed ``total_timesteps`` times.
- See how ACER is used to solve two common combinatorial problems in :ref:`TSP <ex1>` and :ref:`KP <ex10>`.

Acknowledgment
-----------------

Thanks to our fellows in `stable-baselines <https://github.com/hill-a/stable-baselines>`_, as we used their standalone RL implementation, which is utilized as a baseline to leverage advanced neuroevolution algorithms. 

Hill, Ashley, et al. "Stable baselines." (2018).