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
#Created on Thu Dec  3 14:42:29 2020
#
#@author: Majdi
#"""

import random
import numpy as np
import math
import time
import joblib
from neorl.evolu.discrete import mutate_discrete, encode_grid_to_discrete, decode_discrete_to_grid

class SSA(object):
    """
    Salp Swarm Algorithm
    
    :param mode: (str) problem type, either ``min`` for minimization problem or ``max`` for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param fit: (function) the fitness function 
    :param nsalps: (int): number of salps in the swarm
    :param c1: (float/list): a scalar value or a list of values with size ``ngen`` for the coefficient that controls exploration/exploitation. 
                            If ``None``, default annealing formula for ``c1`` is used (see **Notes** below for more info).
    :param int_transform: (str): method of handling int/discrete variables, choose from: ``nearest_int``, ``sigmoid``, ``minmax``.
    :param ncores: (int) number of parallel processors (must be ``<= nsalps``)
    :param seed: (int) random seed for sampling
    """
    def __init__(self, mode, bounds, fit, nsalps=5, c1=None, int_transform='nearest_int', ncores=1, seed=None):
        
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        
        assert ncores <= nsalps, '--error: ncores ({}) must be less than or equal than nsalps ({})'.format(ncores, nsalps)
        
        #--mir
        self.mode=mode
        if mode == 'min':
            self.fit=fit
        elif mode == 'max':
            def fitness_wrapper(*args, **kwargs):
                return -fit(*args, **kwargs) 
            self.fit=fitness_wrapper
        else:
            raise ValueError('--error: The mode entered by user is invalid, use either `min` or `max`')
          
        self.int_transform=int_transform
        if int_transform not in ["nearest_int", "sigmoid", "minmax"]:
            raise ValueError('--error: int_transform entered by user is invalid, must be `nearest_int`, `sigmoid`, or `minmax`')
            
        self.bounds=bounds
        self.ncores = ncores
        self.nsalps=nsalps
        self.c1=c1
        
        #infer variable types 
        self.var_type = np.array([bounds[item][0] for item in bounds])
        
        #mir-grid
        if "grid" in self.var_type:
            self.grid_flag=True
            self.orig_bounds=bounds  #keep original bounds for decoding
            print('--debug: grid parameter type is found in the space')
            self.bounds, self.bounds_map=encode_grid_to_discrete(self.bounds) #encoding grid to int
            #define var_types again by converting grid to int
            self.var_type = np.array([self.bounds[item][0] for item in self.bounds])
        else:
            self.grid_flag=False
            self.bounds = bounds
        
        self.dim = len(bounds)
        self.lb=np.array([self.bounds[item][1] for item in self.bounds])
        self.ub=np.array([self.bounds[item][2] for item in self.bounds])

    def init_sample(self, bounds):
    
        indv=[]
        for key in bounds:
            if bounds[key][0] == 'int':
                indv.append(random.randint(bounds[key][1], bounds[key][2]))
            elif bounds[key][0] == 'float':
                indv.append(random.uniform(bounds[key][1], bounds[key][2]))
            #elif bounds[key][0] == 'grid':
            #    indv.append(random.sample(bounds[key][1],1)[0])
            else:
                raise Exception ('unknown data type is given, either int, float, or grid are allowed for parameter bounds')   
        return np.array(indv)

    def eval_salps(self):
    
        #---------------------
        # Fitness calcs
        #---------------------
        core_lst=[]
        for case in range (0, self.Positions.shape[0]):
            core_lst.append(self.Positions[case, :])
    
        if self.ncores > 1:

            with joblib.Parallel(n_jobs=self.ncores) as parallel:
                fitness_lst=parallel(joblib.delayed(self.fit_worker)(item) for item in core_lst)
                
        else:
            fitness_lst=[]
            for item in core_lst:
                fitness_lst.append(self.fit_worker(item))
        
        return fitness_lst

    def select(self, pos, fit):
        
        best_fit=np.min(fit)
        min_idx=np.argmin(fit)
        best_pos=pos[min_idx,:].copy()
        
        return best_pos, best_fit 
        
    def ensure_bounds(self, vec):
    
        vec_new = []
        # cycle through each variable in vector 
        for i, (key, val) in enumerate(self.bounds.items()):
    
            # variable exceedes the minimum boundary
            if vec[i] < self.bounds[key][1]:
                vec_new.append(self.bounds[key][1])
    
            # variable exceedes the maximum boundary
            if vec[i] > self.bounds[key][2]:
                vec_new.append(self.bounds[key][2])
    
            # the variable is fine
            if self.bounds[key][1] <= vec[i] <= self.bounds[key][2]:
                vec_new.append(vec[i])
            
        return vec_new
    
    def fit_worker(self, x):
        #This worker is for parallel calculations
        
        # Clip the salp with position outside the lower/upper bounds and return same position
        x=self.ensure_bounds(x)
        
        if self.grid_flag:
            #decode the individual back to the int/float/grid mixed space
            x=decode_discrete_to_grid(x,self.orig_bounds,self.bounds_map)
        
        # Calculate objective function for each search agent
        fitness = self.fit(x)
        
        return fitness
    
    def ensure_discrete(self, vec):
        #"""
        #to mutate a vector if discrete variables exist 
        #handy function to be used within SSA phases

        #Params:
        #vec - salp position in vector/list form

        #Return:
        #vec - updated salp position vector with discrete values
        #"""
        
        for dim in range(self.dim):
            if self.var_type[dim] == 'int':
                vec[dim] = mutate_discrete(x_ij=vec[dim], 
                                               x_min=min(vec),
                                               x_max=max(vec),
                                               lb=self.lb[dim], 
                                               ub=self.ub[dim],
                                               alpha=self.a,
                                               method=self.int_transform,
                                               )
        
        return vec

    def UpdateSalps(self):

        for i in range(0, self.nsalps):

            self.Positions = np.transpose(self.Positions)

            if i < self.nsalps / 2:
                for j in range(0, self.dim):
                    c2 = random.random()
                    c3 = random.random()
                    if c3 < 0.5:
                        self.Positions[j, i] = self.best_position[j] + self.c1r * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])
                    else:
                        self.Positions[j, i] = self.best_position[j] - self.c1r * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])

            elif i >= self.nsalps / 2 and i < self.nsalps + 1:
                point1 = self.Positions[:, i - 1]
                point2 = self.Positions[:, i]

                self.Positions[:, i] = (point2 + point1) / 2
                
            self.Positions[:,i]=self.ensure_bounds(self.Positions[:,i])
            self.Positions[:, i] = self.ensure_discrete(self.Positions[: , i])
                
            self.Positions = np.transpose(self.Positions)
            

    def evolute(self, ngen, x0=None, verbose=True):
        """
        This function evolutes the SSA algorithm for number of generations.
        
        :param ngen: (int) number of generations to evolute
        :param x0: (list of lists) initial position of the salps (must be of same size as ``nsalps``)
        :param verbose: (bool) print statistics to screen
        
        :return: (dict) dictionary containing major SSA search results
        """
        self.history = {'local_fitness':[], 'global_fitness':[], 'c1': []}
        self.best_fitness=float("inf") 
        self.verbose=verbose
        self.Positions = np.zeros((self.nsalps, self.dim))
        if x0:
            assert len(x0) == self.nsalps, '--error: the length of x0 ({}) MUST equal the number of salps in the group ({})'.format(len(x0), self.nsalps)
            for i in range(self.nsalps):
                self.Positions[i,:] = x0[i]
        else:
            #self.Positions=self.init_sample(self.bounds)  #TODO, update later for mixed-integer optimisation
            # Initialize the positions of salps
            
            for i in range(self.nsalps):
                self.Positions[i,:]=self.init_sample(self.bounds)
        
        fitness0=self.eval_salps()
        
        self.best_position, self.best_fitness = self.select(self.Positions, fitness0)
                       
        for l in range(1, ngen+1):
            self.a= 1 - l * ((1) / ngen)  #mir: a decreases linearly between 1 to 0, for discrete mutation
            
            if self.c1 is None:
                self.c1r = 2 * math.exp(-((4 * l / ngen) ** 2))
            elif isinstance(self.c1, (float, int)):
                self.c1r=self.c1
            elif isinstance(self.c1, (list)):
                assert len(self.c1) == ngen, '--error: if c1 is a list of values, it must have equal size ({}) as ngen ({})'.format(len(self.c1), ngen)
                self.c1r=self.c1[l-1]
            else:
                raise ValueError ('--error: c1 should be either None, a scalar, or a vector of size ngen')
                
            #-----------------------------
            # Update Salp Positions
            #-----------------------------
            self.UpdateSalps()
                    
            #----------------------
            #  Evaluate New Salps
            #----------------------
            fitness=self.eval_salps()
            for i, fits in enumerate(fitness):
                #save the best of the best!!!
                if fits < self.best_fitness:
                    self.best_fitness=fits
                    self.best_position=self.Positions[i, :].copy()
                
            #--mir
            if self.mode=='max':
                self.fitness_best_correct=-self.best_fitness
                self.local_fitness=-np.min(fitness)
            else:
                self.fitness_best_correct=self.best_fitness
                self.local_fitness=np.min(fitness)
            
            self.history['local_fitness'].append(self.local_fitness)
            self.history['global_fitness'].append(self.fitness_best_correct)
            self.history['c1'].append(self.c1r)
            
            # Print statistics
            if self.verbose and i % self.nsalps:
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                print('SSA step {}/{}, nsalps={}, Ncores={}'.format((l)*self.nsalps, ngen*self.nsalps, self.nsalps, self.ncores))
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                print('Best Salp Fitness:', np.round(self.fitness_best_correct,6))
                if self.grid_flag:
                    self.salp_decoded = decode_discrete_to_grid(self.best_position, self.orig_bounds, self.bounds_map)
                    print('Best Salp Position:', self.salp_decoded)
                else:
                    print('Best Salp Position:', self.best_position)
                print('c1:', self.c1r)
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        #mir-grid
        if self.grid_flag:
            self.salp_correct = decode_discrete_to_grid(self.best_position, self.orig_bounds, self.bounds_map)
        else:
            self.salp_correct = self.best_position                

        if self.verbose:
            print('------------------------ SSA Summary --------------------------')
            print('Best fitness (y) found:', self.fitness_best_correct)
            print('Best individual (x) found:', self.salp_correct)
            print('--------------------------------------------------------------')  
            
        return self.salp_correct, self.fitness_best_correct, self.history

