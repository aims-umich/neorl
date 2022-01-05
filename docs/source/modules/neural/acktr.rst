.. _acktr:

.. automodule:: neorl.rl.baselines.acktr

Actor Critic using Kronecker-Factored Trust Region (ACKTR)
===========================================================

Actor Critic using Kronecker-Factored Trust Region (ACKTR) uses Kronecker-factored approximate curvature (K-FAC) for trust region optimization. ACKTR uses K-FAC to allow more efficient inversion of the covariance matrix of the gradient. ACKTR also extends the natural policy gradient algorithm to optimize value functions via Gauss-Newton approximation.

Original paper: https://arxiv.org/abs/1708.05144

What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ✔️
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ✔️

Parameters
----------

.. autoclass:: ACKTR
  :members:
  :inherited-members:

.. autoclass:: neorl.rl.make_env.CreateEnvironment
   :noindex:

.. autoclass:: neorl.utils.neorlcalls.RLLogger
   :noindex:

Example
-------

Train an ACKTR agent to optimize the 5-D sphere function

.. literalinclude :: ../../scripts/ex_acktr.py
   :language: python

Notes
-------	

- ACKTR belongs to the actor-critic family of reinforcement learning. ACKTR uses some methods to increase the efficiency of reinforcement learning gradient-based search. ACKTR is parallel and supports all types of spaces.
- ACKTR shows sensitivity to ``n_steps``, ``vf_fisher_coef``, ``vf_coef``, and ``learning_rate``. It is always good to consider tuning these hyperparameters before using for optimization. In particular, ``n_steps`` is considered the most important parameter to tune for ACKTR. Always start with small ``n_steps`` and increase as needed. 
- The cost of ACKTR equals to the ``total_timesteps`` in the ``learn`` function, where the original fitness function will be accessed ``total_timesteps`` times.
- See how ACKTR is used to solve two common combinatorial problems in :ref:`TSP <ex1>` and :ref:`KP <ex10>`.

Acknowledgment
-----------------

Thanks to our fellows in `stable-baselines <https://github.com/hill-a/stable-baselines>`_, as we used their standalone RL implementation, which is utilized as a baseline to leverage advanced neuroevolution algorithms. 

Hill, Ashley, et al. "Stable baselines." (2018).