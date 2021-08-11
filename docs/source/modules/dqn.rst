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

.. code-block:: python

	from neorl import DQN
	from neorl import DQNPolicy
	from neorl import RLLogger
	from neorl import CreateEnvironment
	
	def Sphere(individual):
	        """Sphere test objective function.
	                F(x) = sum_{i=1}^d xi^2
	                d=1,2,3,...
	                Range: [-100,100]
	                Minima: 0
	        """
	        #print(individual)
	        return sum(x**2 for x in individual)
	
	nx=5
	bounds={}
	for i in range(1,nx+1):
	        bounds['x'+str(i)]=['int', -100, 100]
	
	#create an enviroment class
	env=CreateEnvironment(method='dqn', 
	                      fit=Sphere, 
	                      bounds=bounds, 
	                      mode='min', 
	                      episode_length=50)
	#create a callback function to log data
	cb=RLLogger(check_freq=1)
	#create a RL object based on the env object
	dqn = DQN(DQNPolicy, env=env, seed=1)
	#optimise the enviroment class
	dqn.learn(total_timesteps=2000, callback=cb)
	#print the best results
	print('--------------- DQN results ---------------')
	print('The best value of x found:', cb.xbest)
	print('The best value of y found:', cb.rbest)
	

Acknowledgment
-----------------

Thanks to our fellows in `stable-baselines <https://github.com/hill-a/stable-baselines>`_, as we used their standalone RL implementation, which is utilized as a baseline to leverage advanced neuroevolution algorithms. 

Hill, Ashley, et al. "Stable baselines." (2018).