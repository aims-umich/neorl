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

Example
-------

Train a PPO agent to optimize the 5-D sphere function

.. code-block:: python

	from neorl import PPO2
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
	env=CreateEnvironment(method='ppo', fit=Sphere, 
	                      bounds=bounds, mode='min', episode_length=50)
	
	#create a callback function to log data
	cb=RLLogger(check_freq=1, mode='min')
	#create a RL object based on the env object
	ppo = PPO2(MlpPolicy, env=env, n_steps=12, seed=1)
	#optimise the enviroment class
	ppo.learn(total_timesteps=2000, callback=cb)
	#print the best results
	print('--------------- PPO results ---------------')
	print('The best value of x found:', cb.xbest)
	print('The best value of y found:', cb.rbest)


Acknowledgment
-----------------

Thanks to our fellows in `stable-baselines <https://github.com/hill-a/stable-baselines>`_, as we used their standalone RL implementation, which is utilized as a baseline to leverage advanced neuroevolution algorithms. 

Hill, Ashley, et al. "Stable baselines." (2018).