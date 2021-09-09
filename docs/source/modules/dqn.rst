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

.. autoclass:: neorl.utils.neorlcalls.RLLogger

Example
-------

Train a DQN agent to optimize the 5-D discrete sphere function

.. literalinclude :: ../scripts/ex_dqn.py
   :language: python
	

Acknowledgment
-----------------

Thanks to our fellows in `stable-baselines <https://github.com/hill-a/stable-baselines>`_, as we used their standalone RL implementation, which is utilized as a baseline to leverage advanced neuroevolution algorithms. 

Hill, Ashley, et al. "Stable baselines." (2018).