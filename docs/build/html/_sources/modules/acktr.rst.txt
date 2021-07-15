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

Example
-------

Train an ACKTR agent to optimize the 5-D sphere function

.. code-block:: python

	from neorl import ACKTR
	from neorl import MlpPolicy
	from neorl import RLLogger
	from neorl import CreateEnvironment
	
	def Sphere(individual):
	        """Sphere test objective function.
	                F(x) = sum_{i=1}^d xi^2
	                d=1,2,3,...
	                Range: [-100,100]
	                Minima: 0
	        """
	        return sum(x**2 for x in individual)
	
	nx=5
	bounds={}
	for i in range(1,nx+1):
	        bounds['x'+str(i)]=['float', -10, 10]
	
	#create an enviroment class
	env=CreateEnvironment(method='acktr', fit=Sphere, 
	                      bounds=bounds, mode='min', episode_length=50)
	#create a callback function to log data
	cb=RLLogger(check_freq=1, mode='min')
	#create an acktr object based on the env object
	acktr = ACKTR(MlpPolicy, env=env, n_steps=12, seed=1)
	#optimise the enviroment class
	acktr.learn(total_timesteps=2000, callback=cb)
	#print the best results
	print('--------------- ACKTR results ---------------')
	print('The best value of x found:', cb.xbest)
	print('The best value of y found:', cb.rbest)

Acknowledgment
-----------------

Thanks to our fellows in `stable-baselines <https://github.com/hill-a/stable-baselines>`_, as we used their standalone RL implementation, which is utilized as a baseline to leverage advanced neuroevolution algorithms. 

Hill, Ashley, et al. "Stable baselines." (2018).