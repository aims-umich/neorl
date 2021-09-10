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
#Created on Tue Jul 13 21:06:40 2021
#
#@author: Paul Seurin
#"""

########################
#
# Import Packages
#
########################

import gym
from gym.spaces import Discrete, Box, MultiDiscrete
import numpy as np
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

##############################
#
# Define Knapsack Function
#        (Discrete)
#
##############################

class KP(gym.Env):

  def __init__ (self,obj_list,weight_capacity,optimum_knapsack = None, episode_length = None, method = None):                
    self.obj_library = {}
    self.weight_capacity = weight_capacity
    self.obj_weight_to_value = {}
    
    for i in range(len(obj_list)):
        self.obj_library["%d"%(i + 1)] = [obj_list[i][0],obj_list[i][1]]
        w_to_v = obj_list[i][1] / obj_list[i][0]
        self.obj_weight_to_value["%d"%(i + 1)] = w_to_v
    
    self.number_of_objects = len(obj_list)
    self.state = - 10**6.0 * np.ones((self.number_of_objects,3), dtype = float) # values/weight/weight to value ratio
    
    self.episode_length = episode_length # Number of kp solved per episodes
    self.obj_id = copy.deepcopy(list(self.obj_library.keys())) # list with the ids of the objects
    self.obj_list = [] # List representing a bag of objects
    self.full_obj_list = copy.deepcopy(np.transpose(obj_list)) # store all the objects for plotting purposes
    self.counter = 0 # counter to record a knap
    self.subcounter = 0 # counter to record each knap per episode
    self.method = method #RL method used for optimization
    if self.method not in ['ppo', 'a2c', 'acktr', 'neat', 'dqn', 'acer']:
        raise ValueError("--error: The only methods available are 'ppo', 'a2c', 'acktr', 'neat', 'dqn', or 'acer' not {}".format(self.method))
    if self.method in ['ppo', 'a2c', 'acktr', 'neat']:
        self.action_bounds = [i for i in range(self.number_of_objects,0,-1)]
        self.action_space = MultiDiscrete(self.action_bounds) # action space 
    elif self.method in ['acer', 'dqn']:
        self.action_bounds = self.number_of_objects
        self.action_space = Discrete(self.action_bounds) # action space 
    self.observation_space = Box(low=0, high=self.number_of_objects, shape=(self.number_of_objects * 3,), dtype=int) # state space  
    self.best_knap = -10**6 # variable to store the value of the best knapsack
    self._iter_episode = 0 # record current episode (for logger)
    self.reset()
    self.done = False
    if optimum_knapsack is not None: # record the (known) best knapsack map and plot it
        optimum_knap_map = - 10**6.0 * np.ones((self.number_of_objects,2), dtype = float) 
        for increment in range(0,len(optimum_knapsack)):
            optimum_knap_map[:,0][increment] = self.obj_library['%d'%(optimum_knapsack[increment])][0]
            optimum_knap_map[:,1][increment] = self.obj_library['%d'%(optimum_knapsack[increment])][1]
        optimum_knap_value = self.Compute_knap_value(knap = optimum_knap_map)
        title_map = "Sum of the values : {} \n".format(optimum_knap_value)
        title_map += "Sum of the weights : {} \n".format(self._check_weight_cap(optimum_knap_map))
        title_map += "Weight's limit : {} \n".format(self.weight_capacity)
        _plot_knap_map(optimum_knap_map,flag = False, name = "Optimum_knap_map_%d.png"%(len(obj_list)), title_map = title_map, all_objects=np.transpose(obj_list))
  def step(self, x): 
    if self.method in ['ppo', 'a2c', 'acktr', 'neat']:
        self.counter = 0 # initialize the per knapsack counter
        for action in x: 
            if self.counter == 0:
                self.reset()
            self.counter += 1
            if action <= len(self.obj_id) - 1:  
                new_obj = self.obj_id[action] # recover the objects corresponding to action 'action'
            else: # ensure a valid action considering that the action space is fixed
                action = random.randint(0,len(self.obj_id) - 1)
                new_obj = self.obj_id[action]

            # place the objects in hte knapsack and fill the state space
            self.state[:,0][self.counter - 1] = self.obj_library[new_obj][0]# --- put as embedding the value
            self.state[:,1][self.counter - 1] = self.obj_library[new_obj][1]# --- put as embedding the weights
            self.state[:,2][self.counter - 1] = self.state[:,1][self.counter - 1] / self.state[:,0][self.counter - 1]
    
            index1 = self.obj_id.index(self.obj_id[action]) # remove the objects chosen by action : 'action' from the action space
            self.obj_id.pop(index1)
            self.obj_list.append(new_obj) # list of objects in the knapsack (for neorl callback)

            if  self._check_weight_cap(self.state) > self.weight_capacity:
                self.state[:,0][self.counter - 1] = - 10**6.0 # --- remove last object
                self.state[:,1][self.counter - 1] = - 10**6.0
                self.state[:,2][self.counter - 1] = - 10**6.0
                break # stop filling the knapsack and evaluate the reward
            elif self.obj_list == []:
                raise ValueError("--error: the weight capacity is higher than the sum of the weights of all the objects. The problem is trivial")

        reward = self._get_stats()
        self.subcounter += 1 
        if self.subcounter != self.episode_length + 1: # generate the new knapsack
            self.obj_id = copy.deepcopy(list(self.obj_library.keys()))
            self.obj_list.pop(-1)
            individual = copy.deepcopy(self.obj_list)
            self.obj_list = []
            self.state[:,0] = - 10**6.0 * np.ones(self.number_of_objects)
            self.state[:,1] = - 10**6.0 * np.ones(self.number_of_objects)
            self.state[:,2] = - 10**6.0 * np.ones(self.number_of_objects)     
    
    elif self.method in ['acer', 'dqn']:
        action = x
        if action <= len(self.obj_id) - 1:  
            new_obj = self.obj_id[action] # recover the object corresponding to action 'action'
        else: # ensure a valid action considering that the action space is fixed
            action = random.randint(0,len(self.obj_id) - 1)
            new_obj = self.obj_id[action]
        self.counter += 1 # --- increment the global counter
        # place the objects in hte knapsack and fill the state space
        self.state[:,0][self.counter - 1] = self.obj_library[new_obj][0]# --- put as embedding the value
        self.state[:,1][self.counter - 1] = self.obj_library[new_obj][1]# --- put as embedding the weights
        self.state[:,2][self.counter - 1] = self.state[:,1][self.counter - 1] / self.state[:,0][self.counter - 1]
    
        index1 = self.obj_id.index(self.obj_id[action]) # remove the objects chosen by action : 'action' from the action space
        self.obj_id.pop(index1)
        self.obj_list.append(new_obj) # list of objects in the knapsack (for neorl callback)
        reward = 0
        if  self._check_weight_cap(self.state) > self.weight_capacity:
            self.state[:,0][self.counter - 1] = - 10**6.0 # --- remove last object
            self.state[:,1][self.counter - 1] = - 10**6.0
            self.state[:,2][self.counter - 1] = - 10**6.0
            reward = self._get_stats()
            self.subcounter += 1
            self.counter = 0
            if self.subcounter != self.episode_length + 1: # generate the new knapsack 
                self.obj_id = copy.deepcopy(list(self.obj_library.keys()))
                self.obj_list.pop(-1)
                individual = copy.deepcopy(self.obj_list)
                self.obj_list = []
                self.state[:,0] = - 10**6.0 * np.ones(self.number_of_objects)
                self.state[:,1] = - 10**6.0 * np.ones(self.number_of_objects)
                self.state[:,2] = - 10**6.0 * np.ones(self.number_of_objects)
        else:
            individual = copy.deepcopy(self.obj_list) # a knapsack is not complete.

    if self.subcounter == self.episode_length:# episode terminates
        self.done = True
        self._iter_episode += 1
        self.subcounter = 0
    return ([self.state.flatten(),reward, self.done, {'x':individual}])

  def reset(self):
    self.done = False
    self.obj_id = copy.deepcopy(list(self.obj_library.keys()))
    self.state[:,0] = - 10**6.0 * np.ones(self.number_of_objects)
    self.state[:,1] = - 10**6.0 * np.ones(self.number_of_objects)
    self.state[:,2] = - 10**6.0 * np.ones(self.number_of_objects)
    return (self.state.flatten())

  def Compute_knap_value(self, knap = None):
    if knap is None:
        knap = copy.deepcopy(self.state)        
    value = 0
    limit_knap = np.where(knap[:,0] == -10**6.0)[0] # the list of objects is either -10**6.0 or a reasonable weights
    if len(limit_knap) > 0:
        limit_knap = limit_knap[0]
    else:
        limit_knap = self.numlocs 
    for increment in range(0,limit_knap): # sum the values of the object chosen
        value += knap[:,0][increment]    
    return value

  def _check_weight_cap(self, knap = None):
    if knap is None:
        knap = copy.deepcopy(self.state) 
    value = 0
    limit_knap = np.where(knap[:,0] == -10**6.0)[0] # the list of objects is either -10**6.0 or a reasonable weights
    if len(limit_knap) > 0:
        limit_knap = limit_knap[0]
    else:
        limit_knap = self.numlocs 
    for increment in range(0,limit_knap): # sum the weights of the object chosen and verify that it does not exceed the weight limit
        value += knap[:,1][increment]    
    return value 

  def _get_stats(self,flag = True):
      score = self.Compute_knap_value() 
      if flag: # compute the best knapsack and plot it only for complete knapsack
        if score - self.best_knap > 0:
            self.best_knap = score
            current_knap_value = self.Compute_knap_value(knap = self.state)
            title_map = " Episode: {}, Value of the Knapsack : {} \n".format(self._iter_episode + 1, np.ceil(current_knap_value))
            title_map += "Sum of the weights : {} \n".format(self._check_weight_cap(self.state))
            _plot_knap_map(self.state,flag = False, name = "Best_Knap_map_%d_%s.png"%(len(np.transpose(self.full_obj_list)),self.method), title_map = title_map, all_objects = self.full_obj_list)

      return score
            
  def render(self, mode = 'human'):
    pass  

def _plot_knap_map(knap_embed,flag = False, name ="Obj_dist.png", title_map = None,all_objects = None):
    """
      Plot Object distribution stored in : knap_embed
    """
    Full_map = copy.deepcopy(knap_embed)
    X_or = Full_map[:,0]
    Y_or = Full_map[:,1]
    limit_knap = np.where(X_or == -10**6.0)[0][0]
    plt.figure()
    if title_map is not None:
        plt.title(title_map)
    else:
        plt.title('Items distribution')
    plt.ylabel('weights [-]')
    plt.xlabel('values [-]')
    plt.scatter(all_objects[0],all_objects[1],color = 'blue', label = "Remaining items")
    plt.scatter(X_or[:limit_knap],Y_or[:limit_knap], color = 'orange' ,label = "Chosen items")
    plt.legend()
    plt.savefig("%s"%(name),format='png' ,dpi=300, bbox_inches="tight")
    plt.close()