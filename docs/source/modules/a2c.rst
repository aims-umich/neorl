.. _a2c:

.. automodule:: neorl.rl.baselines.a2c


Advantage Actor Critic (A2C)
==============================

A synchronous, deterministic variant of `Asynchronous Advantage Actor Critic (A3C) <https://arxiv.org/abs/1602.01783>`_.
It uses multiple workers to avoid the use of a replay buffer.

Original paper:  https://arxiv.org/abs/1602.01783

What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ✔️
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ✔️

Parameters
----------

.. autoclass:: A2C
  :members:
  :inherited-members:
  
.. autoclass:: neorl.rl.make_env.CreateEnvironment

.. autoclass:: neorl.utils.neorlcalls.RLLogger

Example
-------

Train an A2C agent to optimize the 5-D sphere function

.. code-block:: python

	from neorl import A2C
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
	        bounds['x'+str(i)]=['float', -100, 100]
	
	#create an enviroment class
	env=CreateEnvironment(method='a2c', fit=Sphere, 
	                      bounds=bounds, mode='min', episode_length=50)
	#create a callback function to log data
	cb=RLLogger(check_freq=1)
	#create an optimizer object based on the env object
	a2c = A2C(MlpPolicy, env=env, n_steps=8, seed=1)
	#optimise the enviroment class
	a2c.learn(total_timesteps=2000, callback=cb)
	#print the best results
	print('--------------- A2C results ---------------')
	print('The best value of x found:', cb.xbest)
	print('The best value of y found:', cb.rbest)

Here is an example for a parallel A2C
	
.. code-block:: python

	from neorl import A2C
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
	
	if __name__=='__main__':
	    nx=5
	    bounds={}
	    for i in range(1,nx+1):
	            bounds['x'+str(i)]=['float', -100, 100]
	    
	    #create an environment class
	    env=CreateEnvironment(method='a2c', fit=Sphere, ncores=8, 
	                          bounds=bounds, mode='min', episode_length=50)
	    
	    #create a callback function to log data
	    cb=RLLogger(check_freq=1)
	    #create a RL object based on the env object
	    a2c = A2C(MlpPolicy, env=env, n_steps=8, seed=1)
	    #optimise the environment class
	    a2c.learn(total_timesteps=2000, callback=cb)
	    #print the best results
	    print('--------------- A2C results ---------------')
	    print('The best value of x found:', cb.xbest)
	    print('The best value of y found:', cb.rbest)
  
Acknowledgment
-----------------

Thanks to our fellows in `stable-baselines <https://github.com/hill-a/stable-baselines>`_, as we used their standalone RL implementation, which is utilized as a baseline to leverage advanced neuroevolution algorithms. 

Hill, Ashley, et al. "Stable baselines." (2018).


