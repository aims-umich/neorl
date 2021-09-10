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
#Created on Fri Jun 18 19:45:06 2021
#
#@author: Devin Seyler and Majdi Radaideh
#"""

import random
import numpy as np
import math
import joblib
from neorl.evolu.discrete import mutate_discrete, encode_grid_to_discrete, decode_discrete_to_grid

#Main reference of the BAT algorithm:
#Xie, J., Zhou, Y., & Chen, H. (2013). A novel bat algorithm based on 
#differential operator and Levy flights trajectory. 
#Computational intelligence and neuroscience, 2013.

class BAT(object):
    """
    BAT Algorithm
    
    :param mode: (str) problem type, either ``min`` for minimization problem or ``max`` for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['float', 0.1, 0.8], 'x2': ['float', 2.2, 6.2]}``
    :param fit: (function) the fitness function 
    :param nbats: (int): number of bats in the population
    :param fmin: (float): minimum value of the pulse frequency
    :param fmax: (float): maximum value of the pulse frequency
    :param A: (float) initial value of the loudness rate
    :param r0: (float) asymptotic value of the pulse rate 
    :param alpha: (float) decay factor of loudness ``A``, i.e. A approaches 0 by the end of evolution if ``alpha < 1``
    :param gamma: (float) exponential factor of the pulse rate ``r``, i.e. ``r`` increases abruptly at the beginning and then converges to ``r0`` by the end of evolution
    :param levy: (bool): a flag to activate Levy flight steps of the bat to increase bat diversity
    :param int_transform: (str): method of handling int/discrete variables, choose from: ``nearest_int``, ``sigmoid``, ``minmax``.
    :param ncores: (int) number of parallel processors (must be ``<= nbats``)
    :param seed: (int) random seed for sampling
    """
    def __init__(self, mode, bounds, fit, nbats=50, fmin=0, 
                 fmax=1, A=0.5, r0=0.5, alpha=1.0, gamma=0.9, 
                 levy='False', int_transform='nearest_int', ncores=1, seed=None):
        
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        
        assert ncores <= nbats, '--error: ncores ({}) must be less than or equal to nbats ({})'.format(ncores, nbats)
        assert nbats >= 5, '--error: number of bats must be more than 5 for this algorithm'
        
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
        self.nbats=nbats
        
        self.fmax=fmax
        self.fmin=fmin
        self.A=A
        self.alpha=alpha
        self.gamma=gamma
        self.r0=r0
        self.levy_flight=levy
        
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
        #sample initializer
        indv=[]
        for key in bounds:
            if bounds[key][0] == 'int':
                indv.append(random.randint(bounds[key][1], bounds[key][2]))
            elif bounds[key][0] == 'float':
                indv.append(random.uniform(bounds[key][1], bounds[key][2]))
           # elif bounds[key][0] == 'grid':
           #     indv.append(random.sample(bounds[key][1],1)[0])
            else:
                raise Exception ('unknown data type is given, either int, float, or grid are allowed for parameter bounds')   
        return np.array(indv)
   
    def eval_bats(self, position_array):
        #---------------------
        # Fitness calcs
        #---------------------
        core_lst=[]
        for case in range (0, position_array.shape[0]):
            core_lst.append(position_array[case, :])
    
        if self.ncores > 1:

            with joblib.Parallel(n_jobs=self.ncores) as parallel:
                fitness_lst=parallel(joblib.delayed(self.fit_worker)(item) for item in core_lst)
                
        else:
            fitness_lst=[]
            for item in core_lst:
                fitness_lst.append(self.fit_worker(item))
        
        return fitness_lst

    def select(self, pos, fit):
        #this function selects the best fitness and position in a population
        best_fit=np.min(fit)
        min_idx=np.argmin(fit)
        best_pos=pos[min_idx,:]
        
        return best_pos, best_fit 
    
    def ensure_bounds(self, vec, bounds):
    
        vec_new = []
        # cycle through each variable in vector 
        for i, (key, val) in enumerate(bounds.items()):
    
            # variable exceedes the minimum boundary
            if vec[i] < bounds[key][1]:
                vec_new.append(bounds[key][1])
    
            # variable exceedes the maximum boundary
            if vec[i] > bounds[key][2]:
                vec_new.append(bounds[key][2])
    
            # the variable is fine
            if bounds[key][1] <= vec[i] <= bounds[key][2]:
                vec_new.append(vec[i])
                
        return vec_new
    
    def fit_worker(self, x):
        #This worker is for parallel calculations
        
        # Clip the bat with position outside the lower/upper bounds and return same position
        x=self.ensure_bounds(x,self.bounds)
        
        if self.grid_flag:
            #decode the individual back to the int/float/grid mixed space
            x=decode_discrete_to_grid(x,self.orig_bounds,self.bounds_map)
        
        # Calculate objective function for each search agent
        fitness = self.fit(x)
        
        return fitness
    
    def ensure_discrete(self, vec):
        #"""
        #to mutate a vector if discrete variables exist 
        #handy function to be used three times within BAT phases

        #Params:
        #vec - bat position in vector/list form

        #Return:
        #vec - updated bat position vector with discrete values
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
    
    def Levy(self, dim):
        #function to return levy step
        beta = 1.5
        sigma = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = 0.01 * np.random.randn(dim) * sigma
        v = np.random.randn(dim)
        zz = np.power(np.absolute(v), (1 / beta))
        step = np.divide(u, zz)
        return step

    def evolute(self, ngen, x0=None, verbose=True):
        """
        This function evolutes the BAT algorithm for number of generations.
        
        :param ngen: (int) number of generations to evolute
        :param x0: (list of lists) initial position of the bats (must be of same size as ``nbats``)
        :param verbose: (bool) print statistics to screen
        
        :return: (dict) dictionary containing major BAT search results
        """
        self.history = {'local_fitness':[], 'global_fitness':[], 'A': [], 'r': []}
        self.fbest=float("inf")
        self.verbose=verbose
        self.Positions = np.zeros((self.nbats, self.dim))
        self.r=self.r0
        if x0:
            assert len(x0) == self.nbats, '--error: the length of x0 ({}) MUST equal the number of bats in the group ({})'.format(len(x0), self.nbats)
            for i in range(self.nbats):
                self.Positions[i,:] = x0[i]
        else:
            # Initialize the positions of bats
            for i in range(self.nbats):
                self.Positions[i,:]=self.init_sample(self.bounds)
        
        #Initialize and evaluate the first bat population
        fitness0=self.eval_bats(position_array=self.Positions)
        x0, f0=self.select(pos=self.Positions,fit=fitness0)
        self.xbest=np.copy(x0)
        
        # Main BAT loop
        for l in range(0, ngen):
            self.a= 1 - l * ((1) / ngen)  #mir: a decreases linearly between 1 to 0, for discrete mutation
            #------------------------------------------------------
            # Stage 1A: Loop over all bats to update the positions
            #------------------------------------------------------
            for i in range(0, self.nbats):
                
                if self.levy_flight:
                    #Eq.(11) make a levy flight jump to increase population diversity
                    self.Positions[i,:]=self.Positions[i,:]+np.multiply(np.random.randn(self.dim), self.Levy(self.dim))
                
                #Eq.(8)-(10)
                f1=((self.fmin-self.fmax)*l/ngen+self.fmax)*random.random()
                f2=((self.fmax-self.fmin)*l/ngen+self.fmin)*random.random()
                betas=random.sample(list(range(0,self.nbats)),4)
                self.Positions[i, :]=self.xbest+f1*(self.Positions[betas[0],:]-self.Positions[betas[1],:])
                +f2*(self.Positions[betas[2],:]-self.Positions[betas[3],:])
                self.Positions[i, :] = self.ensure_bounds(self.Positions[i , :], self.bounds)
                self.Positions[i, :] = self.ensure_discrete(self.Positions[i , :])
            #-----------------------
            #Stage 1B: evaluation
            #-----------------------
            fitness1=self.eval_bats(position_array=self.Positions)
            x1, f1=self.select(pos=self.Positions,fit=fitness1)
            if f1 <= self.fbest:
                self.fbest=f1
                self.xbest=x1.copy()
            #---------------------------------
            #Stage 2A: Generate new positions
            #---------------------------------
            for i in range(0, self.nbats):
                # Pulse rate
                if random.random() > self.r:
                    self.Positions[i, :] = self.xbest + self.A * np.random.uniform(-1,1,self.dim)
                self.Positions[i, :] = self.ensure_bounds(self.Positions[i , :], self.bounds)
                self.Positions[i, :] = self.ensure_discrete(self.Positions[i , :])
            #-----------------------
            #Stage 2B: evaluation
            #-----------------------
            fitness2=self.eval_bats(position_array=self.Positions)
            x2, f2=self.select(pos=self.Positions,fit=fitness2)
            if f2 <= self.fbest:
                self.fbest=f2
                self.xbest=x2.copy()
            #---------------------------------
            #Stage 3A: Generate new positions
            #---------------------------------
            for i in range(0, self.nbats):
                # loudness effect
                if random.random() < self.A:
                    self.Positions[i, :] = self.xbest + self.r * np.random.uniform(-1,1,self.dim)
                self.Positions[i, :] = self.ensure_bounds(self.Positions[i , :], self.bounds)
                self.Positions[i, :] = self.ensure_discrete(self.Positions[i , :])
            #-----------------------
            #Stage 3B: evaluation
            #-----------------------
            fitness3=self.eval_bats(position_array=self.Positions)
            x3, f3=self.select(pos=self.Positions,fit=fitness3)
            if f3 <= self.fbest:
                self.fbest=f3
                self.xbest=x3.copy()
            #---------------------------------
            #Stage 4: Check and update A/r
            #---------------------------------                
            if min(f1, f2, f3) <= self.fbest:
                #update A
                self.A = self.alpha*self.A
                #update r
                self.r = self.r0*(1-math.exp(-self.gamma*l))  
            #---------------------------------
            #Stage 5: post-processing
            #---------------------------------             
            #--mir
            if self.mode=='max':
                self.fitness_best_correct=-self.fbest
                self.local_fitness = -min(f1 , f2 , f3)
            else:
                self.fitness_best_correct=self.fbest
                self.local_fitness = min(f1 , f2 , f3)
            
            self.best_position=self.xbest.copy()
            self.history['local_fitness'].append(self.local_fitness)
            self.history['global_fitness'].append(self.fitness_best_correct)
            self.history['A'].append(self.A)
            self.history['r'].append(self.r)   
            
            # Print statistics
            if self.verbose and i % self.nbats:
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                print('BAT step {}/{}, nbats={}, Ncores={}'.format((l+1)*self.nbats, ngen*self.nbats, self.nbats, self.ncores))
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                print('Best Bat Fitness:', np.round(self.fitness_best_correct,6))
                if self.grid_flag:
                    self.bat_decoded = decode_discrete_to_grid(self.best_position, self.orig_bounds, self.bounds_map)
                    print('Best Bat Position:', self.bat_decoded)
                else:
                    print('Best Bat Position:', self.best_position)
                print('Loudness A:', self.A)
                print('Pulse rate r:', self.r)
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        
        #mir-grid
        if self.grid_flag:
            self.bat_correct = decode_discrete_to_grid(self.best_position, self.orig_bounds, self.bounds_map)
        else:
            self.bat_correct = self.best_position
                
        if self.verbose:
            print('------------------------ BAT Summary --------------------------')
            print('Best fitness (y) found:', self.fitness_best_correct)
            print('Best individual (x) found:', self.bat_correct)
            print('--------------------------------------------------------------')  
            
        return self.bat_correct, self.fitness_best_correct, self.history