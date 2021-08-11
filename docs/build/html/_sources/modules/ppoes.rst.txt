.. _ppoes:

.. automodule:: neorl.hybrid.ppoes

RL-informed Evolution Strategies (PPO-ES)
================================================

The Proximal Policy Optimization algorithm starts the search to collect some individuals given a fitness function through a RL environment. In the second step, the best PPO individuals are used to guide evolution strategies (ES), where RL individuals are randomly introduced into the ES population to enrich their diversity. The user first runs PPO search followed by ES, the best results of both stages are reported to the user. 
 
Original papers: 

- Radaideh, M. I., & Shirvan, K. (2021). Rule-based reinforcement learning methodology to inform evolutionary algorithms for constrained optimization of engineering applications. Knowledge-Based Systems, 217, 106836.

- Radaideh, M. I., Forget, B., & Shirvan, K. (2021). Large-scale design optimisation of boiling water reactor bundles with neuroevolution. Annals of Nuclear Energy, 160, 108355.

What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ✔️
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ✔️

Parameters
----------

.. autoclass:: PPOES
  :members:
  :inherited-members:

.. autoclass:: neorl.rl.make_env.CreateEnvironment

Example
-------

Train a PPO-ES agent to optimize the 5-D sphere function

.. code-block:: python

	from neorl import PPOES
	from neorl import CreateEnvironment
	
	def Sphere(individual):
	    """Sphere test objective function.
	            F(x) = sum_{i=1}^d xi^2
	            d=1,2,3,...
	            Range: [-100,100]
	            Minima: 0
	    """
	    y=sum(x**2 for x in individual)
	    return y
	
	
	#Setup the parameter space (d=5)
	nx=5
	BOUNDS={}
	for i in range(1,nx+1):
	    BOUNDS['x'+str(i)]=['float', -100, 100]
	
	if __name__=='__main__':  #use this block for parallel PPO!
	    #create an enviroment class for RL/PPO
	    env=CreateEnvironment(method='ppo', fit=Sphere, ncores=1,  
	                          bounds=BOUNDS, mode='min', episode_length=50)
	    
	    #change hyperparameters of PPO/ES if you like (defaults should be good to start with)
	    h={'cxpb': 0.8,
	       'mutpb': 0.2,
	       'n_steps': 24,
	       'lam': 1.0}
	    
	    #Important: `mode` in CreateEnvironment and `mode` in PPOES must be consistent
	    #fit is needed to be passed again for ES, must be same as the one used in env
	    ppoes=PPOES(mode='min', fit=Sphere, 
	                env=env, npop_rl=4, init_pop_rl=True, 
	                bounds=BOUNDS, hyperparam=h, seed=1)
	    #first run RL for some timesteps
	    rl=ppoes.learn(total_timesteps=2000, verbose=True)
	    #second run ES, which will use RL data for guidance
	    ppoes_x, ppoes_y, ppoes_hist=ppoes.evolute(ngen=20, ncores=1, verbose=True) #ncores for ES