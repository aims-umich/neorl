# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 19:12:55 2021

@author: majdi
"""

import sys, uuid
from neorl.evolu.discrete import encode_grid_to_discrete, decode_discrete_to_grid
import numpy as np
import gym
from gym.spaces import Box, MultiDiscrete, Discrete
import random


def globalize(func):
   """
   multiprocessing trick to paralllelize nested functions in python 
   (un-picklable objects!)
   """
   def result(*args, **kwargs):
       return -func(*args, **kwargs)
   result.__name__ = result.__qualname__ = uuid.uuid4().hex
   setattr(sys.modules[result.__module__], result.__name__, result)
   return result


def action_map(norm_action, ub, lb, ub_norm, lb_norm):
    """
    Map a nromalized action `norm_action` from a small range [lb_norm, ub_norm] 
    to a new range [lb, ub].
    Ex: Convert norm_action from [-1, 1] to [-100, 100]
    """
    d=len(norm_action)
    NormRange = np.array([ub_norm-lb_norm]*d)
    OrigRange = ub - lb
    NormMin=np.array([lb_norm]*d)  
    OrigMin = lb
    new_action = ((norm_action - NormMin) * OrigRange / NormRange) + OrigMin
    
    return new_action
            
def ensure_discrete(action, var_type):
    """
    Check the variables in a vector `vec` and convert the discrete ones to integer
    based on their type saved in `var_type`
    """
    vec=[]
    for dim in range(len(action)):
        if var_type[dim] == 'int':
            vec.append(int(action[dim]))
        else:
            vec.append(action[dim])
    return vec


def convert_multidiscrete_discrete(bounds):
    """
    For DQN/ACER, convert the multidiscrete vector to a single discrete one
    to be compatible with DQN/ACER.
    Provide the bounds dict for all variables.
    """
    discrete_list=[]
    for item in bounds:
        discrete_list = discrete_list + list(range(bounds[item][1],bounds[item][2]+1))
    space=list(set(discrete_list))
    
    bounds_map={}
    for i in range(len(space)):
        bounds_map[i]=space[i]
        
    return bounds_map

class CreateEnvironment(gym.Env):

    def __init__(self, method, fit, bounds, mode='max', episode_length=50):

        if method not in ['ppo', 'a2c', 'acer', 'acktr', 'dqn', 'neat']:
            raise ValueError ('--error: unknown RL method is provided, choose from: ppo, a2c, acer, acktr, dqn, neat')
        self.episode_length=episode_length
        self.var_type = np.array([bounds[item][0] for item in bounds])
        self.nx=len(bounds)
        self.method=method
        
        #mir-grid
        if "grid" in self.var_type:
            self.grid_flag=True
            self.orig_bounds=bounds  #keep original bounds for decoding
            print('--debug: grid parameter type is found in the space')
            self.bounds, self.bounds_map=encode_grid_to_discrete(bounds) #encoding grid to int
            #define var_types again by converting grid to int
            self.var_type = np.array([self.bounds[item][0] for item in self.bounds])
        else:
            self.grid_flag=False
            self.bounds = bounds

        self.lb=np.array([self.bounds[item][1] for item in self.bounds])
        self.ub=np.array([self.bounds[item][2] for item in self.bounds])
        
        #for PPO/A2C/ACKTR gaussian policy, keep action range normalized, then map it back
        #Recommended---> -1/1 or -2/2        
        self.lb_norm=-1
        self.ub_norm=1
        if all([item == 'int' for item in self.var_type]):
            if method in ['ppo', 'a2c', 'acktr', 'neat']:
                self.action_space = MultiDiscrete([100,100,100,100,100])
                self.observation_space = Box(low=self.lb, high=self.ub, dtype=int)
                self.map_flag=False
            elif method in ['acer', 'dqn']:
                self.discrete_map=convert_multidiscrete_discrete(self.bounds)
                self.action_space = Discrete(len(self.discrete_map))
                self.observation_space = Box(low=self.lb, high=self.ub, dtype=int)
                self.map_flag=False
        else:
            if method in ['ppo', 'a2c', 'acktr', 'neat']:
                self.action_space = Box(low=self.lb_norm, high=self.ub_norm, shape=(self.nx,))
                self.observation_space = Box(low=self.lb, high=self.ub)
                self.map_flag=True
            elif method in ['acer', 'dqn']:
                self.map_flag=False
                raise Exception ('--error: the method {} does not support continious spaces, please use ppo, a2c, or acktr'.format(method))
                
        #--mir
        self.mode=mode
        if mode == 'max':
            self.fit=fit
        elif mode == 'min':
            self.fit = globalize(lambda x: fit(x))  #use the function globalize to serialize the nested fit
        else:
            raise ValueError('--error: The mode entered by user is invalid, use either `min` or `max`')
        
        self.reset()
        self.done=False
        self.counter = 0
        self.index=0

    def step(self, action):
        action=self.action_mapper(action)
        reward=self.fit(action)
        self.counter += 1
        if self.counter == self.episode_length:
            self.done=True
            self.counter = 0
        
        #print(self.state, action, reward)
        return self.state, reward, self.done, {'x':action}
    
    def reset(self):
        self.done=False
        if self.method in ['ppo', 'a2c', 'acktr', 'neat']:
            init_state=self.action_space.sample()
        elif self.method in ['acer', 'dqn']:
            init_state = self.observation_space.sample()
            self.full_action=init_state.copy()
        else:
            pass
        return init_state

    def render(self, mode='human'):
        pass
    
    def action_mapper(self, action):
        
        if self.method in ['ppo', 'a2c', 'acktr', 'neat']:
            #--------------------------
            # cont./discrete methods
            #---------------------------
            if self.map_flag:
                action=action_map(norm_action=action, 
                                  lb=self.lb, ub=self.ub, 
                                  lb_norm=self.lb_norm, ub_norm=self.ub_norm)
            if 'int' in self.var_type:
                action=ensure_discrete(action=action, var_type=self.var_type)
            if self.grid_flag:
                #decode the individual back to the int/float/grid mixed space
                action=decode_discrete_to_grid(action,self.orig_bounds,self.bounds_map)
            
            self.state=action.copy()
            
        elif self.method in ['acer', 'dqn']:
            #--------------------------
            # discrete methods
            #---------------------------
            if self.index < self.nx:
                decoded_action=self.discrete_map[action]
                
                if decoded_action >= self.lb[self.index] and decoded_action <= self.ub[self.index]:
                    self.full_action[self.index]=decoded_action
                else:
                    #print(self.index, decoded_action, 'random guess')
                    self.full_action[self.index]=random.randint(self.lb[self.index],self.ub[self.index] )
                self.index += 1
            else:
                self.index = 0   #start a new loop over the individual
                self.full_action=self.observation_space.sample()
    
            if self.grid_flag:
                #decode the individual back to the int/float/grid mixed space
                self.decoded_action=decode_discrete_to_grid(self.full_action,self.orig_bounds,self.bounds_map) #convert integer to categorical
                action=self.decoded_action.copy()   #returned the decoded action for reward evaluation
            else:
                action=self.full_action.copy()   #returned the original action for reward evaluation
            
            
            self.state=self.full_action.copy()   #take the unconverted original anyway as next state
            
        return action