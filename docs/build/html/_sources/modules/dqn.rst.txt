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

Example
-------

.. code-block:: python

	import gym
	import numpy as np
	from gym.spaces import Discrete, Box
	from neorl import DQN
	from neorl import DQNPolicy
	from neorl.tools import RLLogger
	
	#Define a Gym-class containing your function to optimise
	#follow the template below
	class Sphere(gym.Env):
	
	    def __init__(self):
	        lb=-100
	        ub=100
	        self.nx=5
	        self.action_space = Discrete(201)
	        self.real_actions=list(range(lb,ub+1))
	        self.observation_space = Box(low=min(self.real_actions), high=max(self.real_actions), shape=(self.nx,), dtype=int)
	        self.episode_length=50
	        self.reset()
	        self.done=False
	        self.counter = 0
	
	    def step(self, action):
	        individual=[self.real_actions[action]]*self.nx
	        print(individual)
	        reward=self.fit(individual=individual)
	        self.counter += 1
	        if self.counter == self.episode_length:
	            self.done=True
	            self.counter = 0
	        
	        return individual, reward, self.done, {'x':action}
	 
	    def fit(self, individual):
	            """Sphere test objective function.
	                    F(x) = sum_{i=1}^d xi^2
	                    d=1,2,3,...
	                    Range: [-100,100]
	                    Minima: 0
	            """
	            #-1 is used to convert minimization to maximization
	            return -sum(x**2 for x in individual)    
	
	    def reset(self):
	        self.done=False
	        ac=self.action_space.sample()
	        individual=[self.real_actions[ac]]*self.nx
	        return individual
	
	    def render(self, mode='human'):
	        pass
	
	#create an object from the class
	env=Sphere()  
	#create a callback function to log data
	cb=RLLogger(check_freq=1, log_dir='dqn_log')
	#create an a2c object based on the env object
	dqn = DQN(DQNPolicy, env=env, verbose=1)
	#optimise the enviroment class
	dqn.learn(total_timesteps=2000, callback=cb)
	
Parameters
----------

.. autoclass:: DQN
  :members:
  :inherited-members:

.. _deepq_policies:

Notes
-----