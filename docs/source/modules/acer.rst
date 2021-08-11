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

.. autoclass:: neorl.rl.make_env.CreateEnvironment

.. autoclass:: neorl.utils.neorlcalls.RLLogger

Example
-------

Train an ACER agent to optimize the 5-D discrete sphere function

.. code-block:: python

	from neorl import ACER
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
	        bounds['x'+str(i)]=['int', -100, 100]
	
	#create an enviroment class
	env=CreateEnvironment(method='acer', fit=Sphere, 
	                      bounds=bounds, mode='min', episode_length=50)
	#create a callback function to log data
	cb=RLLogger(check_freq=1)
	#create an acer object based on the env object
	acer = ACER(MlpPolicy, env=env, n_steps=25, q_coef=0.55, ent_coef=0.02, seed=1)
	#optimise the enviroment class
	acer.learn(total_timesteps=2000, callback=cb)
	#print the best results
	print('--------------- ACER results ---------------')
	print('The best value of x found:', cb.xbest)
	print('The best value of y found:', cb.rbest)
	
Here is an example of parallel ACER

.. code-block:: python

	from neorl import ACER
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
	
	if __name__ == '__main__':
	    nx=5
	    bounds={}
	    for i in range(1,nx+1):
	            bounds['x'+str(i)]=['int', -100, 100]
	    
	    #create an enviroment class
	    env=CreateEnvironment(method='acer', fit=Sphere, ncores=10,
	                          bounds=bounds, mode='min', episode_length=50)
	    #create a callback function to log data
	    cb=RLLogger(check_freq=1)
	    #create an acer object based on the env object
	    acer = ACER(MlpPolicy, env=env, n_steps=25, q_coef=0.55, ent_coef=0.02, seed=1)
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