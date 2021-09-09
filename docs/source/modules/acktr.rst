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

.. autoclass:: neorl.utils.neorlcalls.RLLogger

Example
-------

Train an ACKTR agent to optimize the 5-D sphere function

.. literalinclude :: ../scripts/ex_acktr.py
   :language: python
	    
Acknowledgment
-----------------

Thanks to our fellows in `stable-baselines <https://github.com/hill-a/stable-baselines>`_, as we used their standalone RL implementation, which is utilized as a baseline to leverage advanced neuroevolution algorithms. 

Hill, Ashley, et al. "Stable baselines." (2018).