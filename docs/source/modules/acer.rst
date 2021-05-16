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

Example
-------

Train an ACER agent to optimize the 5-D discrete sphere function

.. code-block:: python

	import gym
	from gym.spaces import Discrete, Box
	from neorl import ACER
	from neorl import MlpPolicy
	from neorl import RLLogger
	
	#--------------------------------------------------------
	# Fitness class based on OpenAI Gym
	#--------------------------------------------------------
	#Define a Gym-class containing your function to optimise
	#see the template below for the Sphere function
	#We will build automatic templates for RL in the near future to simplify fitness definition
	class IntegerSphere(gym.Env):
	    #An integer/discrete form of the sphere function
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
	        reward=self.fit(individual=individual)
	        self.counter += 1
	        if self.counter == self.episode_length:
	            self.done=True
	            self.counter = 0
	        
	        return individual, reward, self.done, {'x':individual}
	
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
	
	#--------------------------------------------------------
	# RL Optimisation
	#--------------------------------------------------------
	#create an object from the class
	env=IntegerSphere()
	#create a callback function to log data
	cb=RLLogger(check_freq=1)
	#create an acer object based on the env object
	acer = ACER(MlpPolicy, env=env, n_steps=25, q_coef=0.55, ent_coef=0.02)
	#optimise the enviroment class
	acer.learn(total_timesteps=2000, callback=cb)
	#print the best results
	print('--------------- ACER results ---------------')
	print('The best value of x found:', cb.xbest)
	print('The best value of y found:', cb.rbest)


Acknowledgment
-----------------

Thanks to our fellows in `stable-baselines <https://github.com/hill-a/stable-baselines>`_, as we used their standalone RL implementation, which is utilized as a baseline to leverage advanced neuroevolution algorithms. 

Hill, Ashley, et al. "Stable baselines." (2018).