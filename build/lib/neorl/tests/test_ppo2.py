import numpy as np
import gym
from gym.spaces import Box
from neorl import PPO2
from neorl import MlpPolicy
from neorl import RLLogger

#--------------------------------------------------------
# Fitness class based on OpenAI Gym
#--------------------------------------------------------
#Define a Gym-class containing your function to optimise
#see the template below for the Sphere function
#We will build automatic templates for RL in the near future to simplify fitness definition
class Sphere(gym.Env):

    def __init__(self):
        lb=np.array([-5.12,-5.12,-5.12,-5.12,-5.12])
        ub=np.array([5.12,5.12,5.12,5.12,5.12])
        self.nx=len(lb)
        self.action_space = Box(low=lb, high=ub, dtype=np.float32)
        self.observation_space = Box(low=lb, high=ub, dtype=np.float32)
        self.episode_length=5
        self.reset()
        self.done=False
        self.counter = 0

    def step(self, action):
        reward=self.fit(individual=action)
        self.counter += 1
        if self.counter == self.episode_length:
            self.done=True
            self.counter = 0

        return action, reward, self.done, {'x':action}

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
        return self.action_space.sample()

    def render(self, mode='human'):
        pass

#--------------------------------------------------------
# RL Optimisation
#--------------------------------------------------------
def test_ppo():
    #create an object from the class
    env=Sphere()
    #create a callback function to log data
    cb=RLLogger(check_freq=1)
    #create an a2c object based on the env object
    ppo = PPO2(MlpPolicy, env=env, n_steps=12)
    #optimise the enviroment class
    ppo.learn(total_timesteps=2000, callback=cb)
    #print the best results
    print('--------------- PPO results ---------------')
    print('The best value of x found:', cb.xbest)
    print('The best value of y found:', cb.rbest)
    
    return 
	
test_ppo()