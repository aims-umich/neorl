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
import bisect
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

####################################
#
# Define Traveling Salesman Function
#           (Discrete)
#
####################################

class TSP(gym.Env):

  def __init__ (self,city_loc_list,optimum_tour_city = None, episode_length = None, method = None):                
    self.city_library = {}
    self.city_dist_to_origin = {}
    self.city_clost_to_each_other = {}
    self.city_clost_to_each_other_for_helperfunc = {}
    self.number_of_cities = len(city_loc_list)
    self.state = - 10**6.0 * np.ones((self.number_of_cities,4), dtype = float) # initialize the state space with a single number 
    for i in range(len(city_loc_list)): # initialize the dictionnaries
        self.city_library["%d"%(i + 1)] = [city_loc_list[i][0],city_loc_list[i][1]]
        loc1 = np.array([city_loc_list[i][0],city_loc_list[i][1]])
        dist = np.sqrt(np.sum(np.power(loc1,2)))
        self.city_dist_to_origin["%d"%(i + 1)] = dist # get the distance of each city to the origin and store it in a dictionnary
        close_dist = []
        i_close_dist = []
        for index in range(0,self.number_of_cities):
            if index != i:
                loc2 = np.array([city_loc_list[index][0],city_loc_list[index][1]])
                dist2 = np.sqrt(np.sum(np.power(loc1 - loc2,2)))
                bisect.insort_left(close_dist,dist2) 
                new_index = np.where(close_dist == dist2)[0][0]
                i_close_dist = np.insert(i_close_dist,new_index,"%d"%(index+1))
        self.city_clost_to_each_other["%d"%(i + 1)] = close_dist 
        self.city_clost_to_each_other_for_helperfunc["%d"%(i + 1)] = i_close_dist #  evaluate the closest city to each city and store it in a dictionnary
    
    self.episode_length = episode_length # Number of tsp solved per episodes
    self.city_id = copy.deepcopy(list(self.city_library.keys())) # list with the ids of the cities
    self.city_list = [] # List representing a tour
    
    self.counter = 0 # counter to record a tour
    self.subcounter = 0 # counter to record each tour per episode
    self.method = method #RL method used for optimization
    if self.method not in ['ppo', 'a2c', 'acktr', 'neat', 'dqn', 'acer']:
        raise ValueError("--error: The only methods available are 'ppo', 'a2c', 'acktr', 'neat', 'dqn', or 'acer' not {}".format(self.method))
    if self.method in ['ppo', 'a2c', 'acktr', 'neat']:
        self.action_bounds = [i for i in range(self.number_of_cities,0,-1)]
        self.action_space = MultiDiscrete(self.action_bounds) # action space 
    elif self.method in ['acer', 'dqn']:
        self.action_bounds = self.number_of_cities
        self.action_space = Discrete(self.action_bounds) # action space 
    self.observation_space = Box(low=0, high=self.number_of_cities, shape=(self.number_of_cities * 4,), dtype=int) # state space  
    self.best_tour = -10**6 # variable to store the value of the best tour
    self._iter_episode = 0 # record current episode (for logger)
    self.reset()
    self.done = False
    if optimum_tour_city is not None: # record the (known) best tour map and plot it
        optimum_tour_map = - 10**6.0 * np.ones((self.number_of_cities,4), dtype = float) 
        for increment in range(0,len(list(self.city_library.keys()))):
            optimum_tour_map[:,0][increment] = self.city_library['%d'%(optimum_tour_city[increment])][0]
            optimum_tour_map[:,1][increment] = self.city_library['%d'%(optimum_tour_city[increment])][1] 
        optimum_tour_value = np.ceil(-1 * self.Compute_tour_cost(tour = optimum_tour_map))
        title_map = " Length of Tour : {} \n".format(optimum_tour_value)
        _plot_tour_map(optimum_tour_map,flag = False, name = "Optimum_Tour_map_%d.png"%(len(self.city_id)), title_map = title_map)

  def step(self, x): 
    if self.method in ['ppo', 'a2c', 'acktr', 'neat']:
        self.counter = 0 # initialize the per tour counter
        for action in x: 
            if self.counter == 0:
                self.reset()
            self.counter += 1

            if action <= len(self.city_id) - 1:  
                new_city = self.city_id[action] # recover the city corresponding to action 'action'
            else: # ensure a valid action considering that the action space is fixed
                action = random.randint(0,len(self.city_id) - 1)
                new_city = self.city_id[action]

            if len(np.where(self.state[:,0] != -10**6.0)[0]) != 0:
                non_zero = np.where(self.state[:,0] != -10**6.0)[0]
                max_cost = -10**6.0
                for k in range(len(non_zero)): # place the city chosen in the location that reduces the partial tour
                    temp_state_0 = copy.deepcopy(self.state[:,0])
                    temp_state_1 = copy.deepcopy(self.state[:,1])
                    self.state[:,0] = np.delete(np.insert(self.state[:,0],k,self.city_library[new_city][0]),-1)
                    self.state[:,1] = np.delete(np.insert(self.state[:,1],k,self.city_library[new_city][1]),-1)
                    temp_score = self.Compute_tour_cost()# evalaute the cost of a partial tour 
                    if temp_score > max_cost:
                        best_k = k
                        max_cost = temp_score
                    self.state[:,0] = temp_state_0
                    self.state[:,1] = temp_state_1

                # fill the state space
                self.state[:,0] = np.delete(np.insert(self.state[:,0],best_k,self.city_library[new_city][0]),-1) 
                self.state[:,1] = np.delete(np.insert(self.state[:,1],best_k,self.city_library[new_city][1]),-1)
                self.state[:,2][self.counter - 1] = self.city_clost_to_each_other[new_city][0]
                self.state[:,3][self.counter - 1] = self.city_dist_to_origin[new_city]

            else: # place the first city and fill the state space
                self.state[:,0][self.counter - 1] = self.city_library[new_city][0]
                self.state[:,1][self.counter - 1] = self.city_library[new_city][1] 
                self.state[:,2][self.counter - 1] = self.city_clost_to_each_other[new_city][0]
                self.state[:,3][self.counter - 1] = self.city_dist_to_origin[new_city]

            index1 = self.city_id.index(self.city_id[action]) # remove the city chosen by action : 'action' from the action space
            self.city_id.pop(index1)
            self.city_list.append(new_city) # sequence of cities in the tour (for neorl callback)

        reward = self._get_stats()
        self.subcounter += 1 
        if self.subcounter != self.episode_length + 1: # generate the new tour
            self.city_id = copy.deepcopy(list(self.city_library.keys()))
            individual = self.city_list
            self.city_list = []
            self.state[:,0] = - 10**6.0 * np.ones(self.number_of_cities)
            self.state[:,1] = - 10**6.0 * np.ones(self.number_of_cities)
            self.state[:,2] = - 10**6.0 * np.ones(self.number_of_cities)
            self.state[:,3] = - 10**6.0 * np.ones(self.number_of_cities)      
    
    elif self.method in ['acer', 'dqn']:
        action = x
        if action <= len(self.city_id) - 1:  
            new_city = self.city_id[action] # recover the city corresponding to action 'action'
        else: # ensure a valid action considering that the action space is fixed
            action = random.randint(0,len(self.city_id) - 1)
            new_city = self.city_id[action]
        self.counter += 1 # --- increment the global counter
        if len(np.where(self.state[:,0] != -10**6.0)[0]) != 0:
            non_zero = np.where(self.state[:,0] != -10**6.0)[0]
            max_cost = -10**6.0
            for k in range(len(non_zero)): # place the city chosen in the location that reduces the partial tour
                temp_state_0 = copy.deepcopy(self.state[:,0])
                temp_state_1 = copy.deepcopy(self.state[:,1])
                self.state[:,0] = np.delete(np.insert(self.state[:,0],k,self.city_library[new_city][0]),-1)
                self.state[:,1] = np.delete(np.insert(self.state[:,1],k,self.city_library[new_city][1]),-1)
                temp_score = self.Compute_tour_cost()# evalaute the cost of a partial tour 
                if temp_score > max_cost:
                    best_k = k
                    max_cost = temp_score
                self.state[:,0] = temp_state_0
                self.state[:,1] = temp_state_1
            # fill the state space
            self.state[:,0] = np.delete(np.insert(self.state[:,0],best_k,self.city_library[new_city][0]),-1) 
            self.state[:,1] = np.delete(np.insert(self.state[:,1],best_k,self.city_library[new_city][1]),-1)
            self.state[:,2][self.counter - 1] = self.city_clost_to_each_other[new_city][0]
            self.state[:,3][self.counter - 1] = self.city_dist_to_origin[new_city]
        else:
            self.state[:,0][self.counter - 1] = self.city_library[new_city][0]
            self.state[:,1][self.counter - 1] = self.city_library[new_city][1] 
            self.state[:,2][self.counter - 1] = self.city_clost_to_each_other[new_city][0]
            self.state[:,3][self.counter - 1] = self.city_dist_to_origin[new_city]


        index1 = self.city_id.index(self.city_id[action]) # remove the city chosen by action : 'action' from the action space
        self.city_id.pop(index1)
        self.city_list.append(new_city) # sequence of cities in the tour (for neorl callback)
        reward = self._get_stats(flag = False#self.state = - 10**6.0 * np.ones((self.number_of_cities,4), dtype = float)
    ) * -10**3 # evaluate a partial tour. Increase the reward for it to be always bigger than a full tour

        if len(np.where(self.state[:,0] == -10**6.0)[0]) == 0:# a tour is complete.
            reward = self._get_stats()
            self.subcounter += 1
            self.counter = 0
            if self.subcounter != self.episode_length + 1: # generate the new tour 
                self.city_id = copy.deepcopy(list(self.city_library.keys()))
                individual = self.city_list
                self.city_list = []
                #self.state = - 10**6.0* * np.ones((self.number_of_cities,4), dtype = float)# reinitialize the state space
                self.state[:,0] = - 10**6.0 * np.ones(self.number_of_cities)
                self.state[:,1] = - 10**6.0 * np.ones(self.number_of_cities)
                self.state[:,2] = - 10**6.0 * np.ones(self.number_of_cities)
                self.state[:,3] = - 10**6.0 * np.ones(self.number_of_cities)
        else:
            individual = self.city_list # a tour is not complete. No individuals are return (for neorl callback function)

    if self.subcounter == self.episode_length:# episode terminates
        self.done = True
        self._iter_episode += 1
        self.subcounter = 0
    return ([self.state.flatten(),reward, self.done, {'x':individual}])

  def reset(self):
    self.done = False
    self.city_id = copy.deepcopy(list(self.city_library.keys()))
    self.state[:,0] = - 10**6.0 * np.ones(self.number_of_cities)
    self.state[:,1] = - 10**6.0 * np.ones(self.number_of_cities)
    self.state[:,2] = - 10**6.0 * np.ones(self.number_of_cities)
    self.state[:,3] = - 10**6.0 * np.ones(self.number_of_cities)
    return (self.state.flatten())

  def Compute_tour_cost(self, tour = None):
    if tour is None:
        tour = copy.deepcopy(self.state)
    cost = 0
    limit_tour = np.where(tour[:,0] == -10**6.0)[0] # a tour ends with the traveling salesman circling back to the initial city
    if len(limit_tour) > 0:
        limit_tour = limit_tour[0] - 1
    else:
        limit_tour = self.number_of_cities - 1  
    for increment in range(0,limit_tour): # compute euclidean distance from cities to cities
        loc1 = np.array([tour[:,0][increment],tour[:,1][increment]])
        loc2 = np.array([tour[:,0][increment + 1],tour[:,1][increment + 1]])
        dist = np.sqrt(np.sum(np.power(loc1 - loc2,2)))
        cost += int(round(dist))
    loc1 = np.array([tour[:,0][limit_tour],tour[:,1][limit_tour]])
    loc2 = np.array([tour[:,0][0],tour[:,1][0]])
    dist = np.sqrt(np.sum(np.power(loc1 - loc2,2)))
    cost += int(round(dist)) 
    score  =  - cost 
    return score    

  def _get_stats(self,flag = True):
      score = self.Compute_tour_cost() 
      if flag: # compute the best tour and plot it only for complete tour
        if score - self.best_tour > 0:
            self.best_tour = score
            current_tour_cost = -1 * self.Compute_tour_cost(tour = self.state)
            title_map = " Episode: {}, Length of Tour : {} \n".format(self._iter_episode + 1, np.ceil(current_tour_cost))
            _plot_tour_map(self.state,flag = False, name = "Best_Tour_map_%d_%s.png"%(self.number_of_cities,self.method), title_map = title_map)

      return score
            
  def render(self, mode = 'human'):
    pass

def _plot_tour_map(tour_embed,dirname_2 = ".", flag = False, name ="Tour_map.png", title_map = None):
    """
      Plot a Tour stored in : tour_embed
    """
    Full_map = copy.deepcopy(tour_embed)
    X = np.append(Full_map[:,0], Full_map[:,0][0])
    Y = np.append(Full_map[:,1], Full_map[:,1][0])
    X_or = Full_map[:,0]
    Y_or = Full_map[:,1]
    plt.figure()
    if title_map is not None:
        plt.title(title_map)
    else:
        plt.title('Tour map')
    plt.ylabel('y [m]')
    plt.xlabel('x [m]')
    plt.scatter(X_or[0],Y_or[0], color = 'black',label = "Departure")
    plt.scatter(X_or[1:],Y_or[1:], color = 'orange' ,label = "Cities")
    plt.plot(X,Y, "-", label = "Tour")
    plt.legend()
    plt.savefig("%s/%s"%(dirname_2,name),format='png' ,dpi=300, bbox_inches="tight")
    plt.close()
