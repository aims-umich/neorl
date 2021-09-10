#    This file is part of NEORL.

#    Copyright (c) 2021 Exelon Corporation and MIT Nuclear Science and Engineering
#    NEORL is free software: you can redistribute it and/or modify
#    it under the terms of the MIT LICENSE

#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

# -*- coding: utf-8 -*-
#"""
#Created on Mon Jul 12 19:12:55 2021
#
#@author: majdi
#"""

import sys, uuid
from neorl.evolu.discrete import encode_grid_to_discrete, decode_discrete_to_grid
from neorl.rl.baselines.shared import set_global_seeds
from neorl.rl.baselines.shared.vec_env import SubprocVecEnv
import numpy as np
import gym
from gym.spaces import Box, MultiDiscrete, Discrete
import random


def globalize(func):
   #"""
   #multiprocessing trick to paralllelize nested functions in python 
   #(un-picklable objects!)
   #"""
   def result(*args, **kwargs):
       return -func(*args, **kwargs)
   result.__name__ = result.__qualname__ = uuid.uuid4().hex
   setattr(sys.modules[result.__module__], result.__name__, result)
   return result


def action_map(norm_action, ub, lb, ub_norm, lb_norm):
    #"""
    #Map a nromalized action `norm_action` from a small range [lb_norm, ub_norm] 
    #to a new range [lb, ub].
    #Ex: Convert norm_action from [-1, 1] to [-100, 100]
    #"""
    d=len(norm_action)
    NormRange = np.array([ub_norm-lb_norm]*d)
    OrigRange = ub - lb
    NormMin=np.array([lb_norm]*d)  
    OrigMin = lb
    new_action = ((norm_action - NormMin) * OrigRange / NormRange) + OrigMin
    
    return new_action
            
def ensure_discrete(action, var_type):
    #"""
    #Check the variables in a vector `vec` and convert the discrete ones to integer
    #based on their type saved in `var_type`
    #"""
    vec=[]
    for dim in range(len(action)):
        if var_type[dim] == 'int':
            vec.append(int(action[dim]))
        else:
            vec.append(action[dim])
    return vec


def convert_multidiscrete_discrete(bounds):
    #"""
    #For DQN/ACER, convert the multidiscrete vector to a single discrete one
    #to be compatible with DQN/ACER.
    #Input: Provide the bounds dict for all variables.
    #"""
    discrete_list=[]
    for item in bounds:
        discrete_list = discrete_list + list(range(bounds[item][1],bounds[item][2]+1))
    space=list(set(discrete_list))
    
    bounds_map={}
    for i in range(len(space)):
        bounds_map[i]=space[i]
        
    return bounds_map

def convert_actions_multidiscrete(bounds):
    #"""
    #For PPO/ACKTR/A2C, convert the action provided by the user to a multidiscrete vector
    #to be compatible with OpenAI gym multidiscrete space.
    #Input: Provide the bounds dict for all variables.
    #Returns: action_bounds (list) for encoding and bounds_map (dict) for decoding
    #"""
    action_bounds=[]
    
    for item in bounds:
        action_bounds.append(len(list(range(bounds[item][1],bounds[item][2]+1))))
    
    bounds_map={}
    for var, act in zip(bounds,action_bounds):
        bounds_map[var]={}
        act_list=list(range(bounds[var][1],bounds[var][2]+1))
        for i in range(act):
            bounds_map[var][i] = act_list[i]
    
    return action_bounds, bounds_map 

def convert_multidiscrete_actions(action, int_bounds_map):
    #"""
    #For PPO/ACKTR/A2C, convert the action in multidiscrete form 
    #to the real action space defined by the user
    #Input: Provide the action in multidiscrete, and the integer bounds map
    #Returns: decoded action (list)
    #"""
    decoded_action=[]
    
    for act, key in zip(action, int_bounds_map):
        decoded_action.append(int_bounds_map[key][act])
        
    return decoded_action 

class BaseEnvironment(gym.Env):
    #"""
    #A module to construct a fitness environment for certain algorithms 
    #that follow reinforcement learning approach of optimization
    #
    #:param method: (str) the supported algorithms, choose either: ``dqn``, ``ppo``, ``acktr``, ``acer``, ``a2c``, ``rneat``, ``fneat``
    #:param fit: (function) the fitness function
    #:param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    #:param mode: (str) problem type, either ``min`` for minimization problem or ``max`` for maximization (RL is default to ``max``)
    #:param episode_length: (int): number of individuals to evaluate before resetting the environment to random initial guess. 
    #"""
    def __init__(self, method, fit, bounds, mode='max', episode_length=50):

        if method not in ['ppo', 'a2c', 'acer', 'acktr', 'dqn', 'neat', 'rneat', 'fneat']:
            raise ValueError ('--error: unknown RL method is provided, choose from: ppo, a2c, acer, acktr, dqn, neat or rneat, fneat')
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
        if all([item == 'int' for item in self.var_type]):   #discrete optimization
            if method in ['ppo', 'a2c', 'acktr', 'rneat', 'fneat']:
                self.action_bounds, self.int_bounds_map=convert_actions_multidiscrete(self.bounds)
                self.action_space = MultiDiscrete(self.action_bounds)
                self.observation_space = Box(low=self.lb, high=self.ub, dtype=int)
                self.cont_map_flag=False
                self.int_map_flag=True
            elif method in ['acer', 'dqn']:
                self.discrete_map=convert_multidiscrete_discrete(self.bounds)
                self.action_space = Discrete(len(self.discrete_map))
                self.observation_space = Box(low=self.lb, high=self.ub, dtype=int)
                self.cont_map_flag=False
                self.int_map_flag=False
        else:
            if method in ['ppo', 'a2c', 'acktr', 'rneat', 'fneat']:
                self.action_space = Box(low=self.lb_norm, high=self.ub_norm, shape=(self.nx,))
                self.observation_space = Box(low=self.lb, high=self.ub)
                self.cont_map_flag=True
                self.int_map_flag=False
            elif method in ['acer', 'dqn']:
                self.cont_map_flag=False
                self.int_map_flag=False
                raise Exception ('--error: the method {} does not support continuous spaces, please use ppo, a2c, or acktr'.format(method))
                
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

    def seed(self, seed_id):
        np.random.seed(seed_id)
        random.seed(seed_id)
        
    def step(self, action):
        state, action, reward=self.action_mapper(action)
        self.counter += 1
        if self.counter == self.episode_length:
            self.done=True
            self.counter = 0
        
        #print(state, action, reward)
        return state, reward, self.done, {'x':action}
    
    def reset(self):
        self.done=False
        if self.method in ['ppo', 'a2c', 'acktr', 'rneat', 'fneat']:
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
        
        if self.method in ['ppo', 'a2c', 'acktr', 'rneat', 'fneat']:
            #--------------------------
            # cont./discrete methods
            #---------------------------
            if self.cont_map_flag:
                action=action_map(norm_action=action, 
                                  lb=self.lb, ub=self.ub, 
                                  lb_norm=self.lb_norm, ub_norm=self.ub_norm)
                
            if self.int_map_flag:  #this flag converts multidiscrete action to the real space
                action=convert_multidiscrete_actions(action=action, int_bounds_map=self.int_bounds_map)
                
            if 'int' in self.var_type:
                action=ensure_discrete(action=action, var_type=self.var_type)
            
            if self.grid_flag:
                #decode the individual back to the int/float/grid mixed space
                decoded_action=decode_discrete_to_grid(action,self.orig_bounds,self.bounds_map)
                reward=self.fit(decoded_action)  #calc reward based on decoded action
                state=action.copy()   #save the state as the original undecoded action (for further procecessing)
                action=decoded_action  #now for logging, return the action as the decoded action
            else:
                #calculate reward and use state as action
                reward=self.fit(action)  
                state=action.copy()
                
            
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
                reward=self.fit(self.decoded_action)  #calc reward based on decoded action
                state=self.full_action.copy()        #save the state as the original undecoded action (for further procecessing)   
                action=self.decoded_action.copy()   #now for logging, return the action as the decoded action
            else:
                action=self.full_action.copy()   #returned the full action for logging
                reward=self.fit(action)          #calc reward based on the action (float + int)
                state=action.copy()              #save the state as the action
            
        return state, action, reward
    
    

def CreateEnvironment(method, fit, bounds, ncores=1, mode='max', episode_length=50):
    """
    A module to construct a fitness environment for certain algorithms 
    that follow reinforcement learning approach of optimization
    
    :param method: (str) the supported algorithms, choose either: ``dqn``, ``ppo``, ``acktr``, ``acer``, ``a2c``.
    :param fit: (function) the fitness function
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param ncores: (int) number of parallel processors
    :param mode: (str) problem type, either ``min`` for minimization problem or ``max`` for maximization (RL is default to ``max``)
    :param episode_length: (int): number of individuals to evaluate before resetting the environment to random initial guess. 
    """
    
    def make_env(rank, seed=0):
        #"""
        #Utility function for multiprocessed env.
        # 
        #:param num_env: (int) the number of environment you wish to have in subprocesses
        #:param seed: (int) the inital seed for RNG
        #:param rank: (int) index of the subprocess
        #"""
        def _init():
            env=BaseEnvironment(method=method, fit=fit, 
                          bounds=bounds, mode=mode, episode_length=episode_length)
            env.seed(seed + rank)
            return env
        set_global_seeds(seed)
        return _init
    
    if ncores > 1:
        env = SubprocVecEnv([make_env(i) for i in range(ncores)])
    else:
        env=BaseEnvironment(method=method, fit=fit, 
                      bounds=bounds, mode=mode, episode_length=episode_length)
    return env