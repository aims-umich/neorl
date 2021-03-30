# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:37:17 2021

@author: majdi
"""


import gym
import numpy as np
from gym.spaces import Discrete, Box

from neorl import DQN
from neorl import DQNPolicy
#from neorl.tools import RLLogger

import os 
from neorl.rl.baselines.shared.callbacks import BaseCallback
import matplotlib.pyplot as plt

class RLLogger(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(RLLogger, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.rbest = -np.inf
        self.r_hist=[]
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(self.n_calls)
            #print('---------my call--------------------')
            try:
                rwd=self.locals['rewards'][0]
                x=self.locals['infos'][0]['x']
                self.r_hist.append(rwd)
                if rwd > self.rbest:
                    self.xbest=x.copy()
                    self.rbest=rwd
                #print(self.locals['rewards'][0])
                #print()
            except:
                self.r_hist.append(self.locals['rewards'])
                #print(self.locals['rewards'])
                #print(self.locals['infos']['x'])
            
            
        return True


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