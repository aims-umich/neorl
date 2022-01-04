.. _dqn:

.. automodule:: neorl.rl.baselines.deepq


Deep Q Learning (DQN)
=======================

`Deep Q Network (DQN) <https://arxiv.org/abs/1312.5602>`_
and its extensions (Double-DQN, Dueling-DQN, Prioritized Experience Replay).

Original papers:

- DQN paper: https://arxiv.org/abs/1312.5602
- Dueling DQN: https://arxiv.org/abs/1511.06581
- Double-Q Learning: https://arxiv.org/abs/1509.06461
- Prioritized Experience Replay: https://arxiv.org/abs/1511.05952

What can you use?
--------------------

-  Multi processing: ❌
-  Discrete spaces: ✔️
-  Continuous spaces: ❌
-  Mixed Discrete/Continuous spaces: ❌

Parameters
----------

.. autoclass:: DQN
  :members:
  :inherited-members:

.. autoclass:: neorl.rl.make_env.CreateEnvironment
   :noindex:

.. autoclass:: neorl.utils.neorlcalls.RLLogger
   :noindex:

Example
-------

Train a DQN agent to optimize the 5-D discrete sphere function

.. literalinclude :: ../../scripts/ex_dqn.py
   :language: python

Notes
-------	

- DQN is the most limited RL algorithm in the package with no multiprocessing and only restricted to discrete spaces. Nevertheless, DQN is considered the first and the heart of many deep RL algorithms. 
- For parallel RL algorithm with Q-value support like DQN, use ACER. 
- DQN shows sensitivity to ``exploration_fraction``, ``train_freq``, and ``target_network_update_freq``. It is always good to consider tuning these hyperparameters before using for optimization. 
- Activating ``prioritized_replay`` seems to improve DQN performance.
- The cost for DQN equals to the ``total_timesteps`` in the ``learn`` function, where the original fitness function will be accessed ``total_timesteps`` times.
- See how DQN is used to solve two common combinatorial problems in :ref:`TSP <ex1>` and :ref:`KP <ex10>`.

Acknowledgment
-----------------

Thanks to our fellows in `stable-baselines <https://github.com/hill-a/stable-baselines>`_, as we used their standalone RL implementation, which is utilized as a baseline to leverage advanced neuroevolution algorithms. 

Hill, Ashley, et al. "Stable baselines." (2018).