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
#Created on Sun Aug 15 2021
#
#@author: Paul
#"""


import random
import numpy as np
import math
import time
import joblib
from neorl.evolu.discrete import mutate_discrete, encode_grid_to_discrete, decode_discrete_to_grid

class CS(object):
    """
    Cuckoo Search Algorithm
    
    :param mode: (str) problem type, either ``min`` for minimization problem or ``max`` for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param fit: (function) the fitness function 
    :param ncuckoos: (int) number of cuckoos or nests in the population: one cuckoos per nest. Default value is 15.
    :param pa: (float) a scalar value for the coefficient that controls exploration/exploitation, 
                            i.e. fraction of the cuckoos/nests that will be replaced by the new cuckoos/nests.
    :param int_transform: (str) method of handling int/discrete variables, choose from: ``nearest_int``, ``sigmoid``, ``minmax``.
    :param ncores: (int) number of parallel processors (must be ``<= ncuckoos``)
    :param seed: (int) random seed for sampling
    """
    def __init__(self, mode, bounds, fit, ncuckoos=15, pa=0.25, int_transform='nearest_int', ncores=1, seed=None):
        
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        assert ncores <= ncuckoos, '--error: ncores ({}) must be less than or equal than ncuckoos ({})'.format(ncores, ncuckoos)
        
        self.mode=mode #  mode for optimization: CS only solves a minimization problem.
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
        self.ncuckoos = ncuckoos
        self.pa = pa # Discovery rate of parasitic eggs/solutions
        
        self.var_type = np.array([bounds[item][0] for item in bounds])# infer variable types 
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
        #"""
        #Initialize the initial population of cuckoos/nests
        #"""
        indv=[]
        for key in bounds:
            if bounds[key][0] == 'int':
                indv.append(random.randint(bounds[key][1], bounds[key][2]))
            elif bounds[key][0] == 'float':
                indv.append(random.uniform(bounds[key][1], bounds[key][2]))
            else:
                raise Exception ('unknown data type is given, either int, float, or grid are allowed for parameter bounds')   
        return np.array(indv)

    def eval_cuckoos(self,newnest = None):
        #---------------------
        # Fitness calcs
        #---------------------
        core_lst=[]
        if newnest is None:
            for case in range (0, self.Positions.shape[0]):
                core_lst.append(self.Positions[case, :])
            if self.ncores > 1:
                with joblib.Parallel(n_jobs=self.ncores) as parallel:
                    fitness_lst=parallel(joblib.delayed(self.fit_worker)(item) for item in core_lst)
            else:
                fitness_lst=[]
                for item in core_lst:
                    fitness_lst.append(self.fit_worker(item))
        else:# fitness of new cuckoo versus old cuckoo must also be compared. newnest are the new cuckoos
            for case in range (0, newnest.shape[0]):
                core_lst.append(newnest[case, :])
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
        # Clip the cuckoo with position outside the lower/upper bounds and return same position
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
        #handy function to be used within CS phases

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

    def UpdateCuckoos(self):
        # perform Levy flights to generate 
        # self.ncuckoos new Cuckoos
        tempnest = np.zeros((self.ncuckoos, self.dim))
        tempnest = np.array(self.Positions).copy()
        beta = 3 / 2
        sigma = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        s = np.zeros(self.dim)
        for j in range(0, self.ncuckoos):
            s = self.Positions[j, :]
            u = np.random.randn(len(s)) * sigma
            v = np.random.randn(len(s))
            step = u / abs(v) ** (1 / beta)
            stepsize = 0.01 * (step * (s - self.best_position))
            s = s + stepsize * np.random.randn(len(s))
            tempnest[j,:]=self.ensure_bounds(s)
            tempnest[j,:] = self.ensure_discrete(s)
 
        return tempnest
        
    def evolute(self,ngen,x0=None, verbose=True):
        """
        This function evolutes the CS algorithm for number of generations
        
        :param ngen: (int) number of generations to evolute
        :param x0: (list of lists) initial position of the cuckoos (must be of same size as ``ncuckoos``)
        :param verbose: (bool) print statistics to screen
        
        :return: (tuple) (best cuckoo, best fitness, and dictionary of fitness history)
        """
        self.history = {'local_fitness':[], 'global_fitness':[]}
        self.best_fitness=float("inf") 
        self.verbose=verbose
        self.Positions = np.zeros((self.ncuckoos, self.dim))
        if x0:
            assert len(x0) == self.ncuckoos, '--error: the length of x0 ({}) MUST equal the number of cuckoos in the group ({})'.format(len(x0), self.ncuckoos)
            for i in range(self.ncuckoos):
                self.Positions[i,:] = x0[i]
        else:
            #self.Positions=self.init_sample(self.bounds)  #TODO, update later for mixed-integer optimisation
            # Initialize the positions of cuckoos
            for i in range(self.ncuckoos):
                self.Positions[i,:]=self.init_sample(self.bounds)

        fitness=self.eval_cuckoos() # evaluate the first cuckoos
        self.best_position, self.best_fitness = self.select(pos = self.Positions,fit = fitness) # find the initial best position and fitness
        for l in range(1, ngen+1):# Main loop
            #-----------------------------
            # Obtain new Cuckoo Positions by Lévy flights
            #-----------------------------
            newnest = self.UpdateCuckoos() # new cuckoos after Lévy Flights
            #----------------------
            #  Evaluate New Cuckoos
            #----------------------
            # Evaluating all new solutions
            tempnest = np.copy(self.Positions)
            fnew = self.eval_cuckoos(newnest)
            # and update current Cuckoos
            for j in range(0, self.ncuckoos):
                if fnew[j] <= fitness[j]:
                    fitness[j] = fnew[j]
                    tempnest[j, :] = newnest[j, :]
            self.Positions = tempnest.copy() # Take only the fittest individual
            #----------------------
            #  Discover a fraction ~pa of Cuckoos from the Cuckoos after Lévy flights
            #----------------------
            new_nest = np.zeros((self.ncuckoos, self.dim))
            K = np.random.uniform(0, 1, (self.ncuckoos, self.dim)) > self.pa
            stepsize = random.random() * (
                newnest[np.random.permutation(self.ncuckoos), :] - newnest[np.random.permutation(self.ncuckoos), :]
            )
            new_nest = newnest + stepsize * K # Update a fraction ~pa of Cuckoo after the Lévy flights
            #----------------------
            #  Re-evaluate the Cuckoos obtained and update to get the fittest individuals
            #----------------------
            tempnest = np.copy(self.Positions) # Will save the fittest individual
            fnew = self.eval_cuckoos(new_nest)
            for j in range(0, self.ncuckoos): # Compare Cuckoo fitness of newly generated Cuckoos (newnest) and current Cuckoos (self.positions)
                if fnew[j] <= fitness[j]:
                    fitness[j] = fnew[j]
                    tempnest[j, :] = new_nest[j, :]
            self.Positions = tempnest.copy() # Update the population
            
            #----------------------
            #  Logger related portion
            #----------------------
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
            # Print statistics
            if self.verbose and i % self.ncuckoos:
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                print('CS step {}/{}, ncuckoos={}, Ncores={}'.format((l)*self.ncuckoos, ngen*self.ncuckoos, self.ncuckoos, self.ncores))
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                print('Best Cuckoo/Nest Fitness:', np.round(self.fitness_best_correct,6))
                if self.grid_flag:
                    self.cuckoo_decoded = decode_discrete_to_grid(self.best_position, self.orig_bounds, self.bounds_map)
                    print('Best Cuckoo Position:', self.cuckoo_decoded)
                else:
                    print('Best Cuckoo Position:', self.best_position)
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        #mir-grid
        if self.grid_flag:
            self.cuckoo_correct = decode_discrete_to_grid(self.best_position, self.orig_bounds, self.bounds_map)
        else:
            self.cuckoo_correct = self.best_position                
        if self.verbose:
            print('------------------------ CS Summary --------------------------')
            print('Best fitness (y) found:', self.fitness_best_correct)
            print('Best individual (x) found:', self.cuckoo_correct)
            print('--------------------------------------------------------------')  
        return self.cuckoo_correct, self.fitness_best_correct, self.history