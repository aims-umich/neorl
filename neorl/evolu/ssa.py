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

class SSA(object):
    """
    Salp Swarm Algorithm
    
    :param mode: (str) problem type, either ``min`` for minimization problem or ``max`` for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param fit: (function) the fitness function 
    :param nsalps: (int): number of salps in the swarm
    :param c1: (float/list): a scalar value or a list of values with size ``ngen`` for the coefficient that controls exploration/exploitation. 
                            If ``None``, default annealing formula for ``c1`` is used (see **Notes** below for more info).
    :param ncores: (int) number of parallel processors (must be ``<= nsalps``)
    :param seed: (int) random seed for sampling
    """
    def __init__(self, mode, bounds, fit, nsalps=5, c1=None, ncores=1, seed=None):
        
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
            
        self.bounds=bounds
        self.ncores = ncores
        self.nsalps=nsalps
        self.c1=c1
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
            elif bounds[key][0] == 'grid':
                indv.append(random.sample(bounds[key][1],1)[0])
            else:
                raise Exception ('unknown data type is given, either int, float, or grid are allowed for parameter bounds')   
        return indv

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
        best_pos=pos[min_idx,:]
        
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
        
        # Calculate objective function for each search agent
        fitness = self.fit(x)
        
        return fitness

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
            
            for i in range(self.dim):
                self.Positions[:, i] = (np.random.uniform(0, 1, self.nsalps) * (self.ub[i] - self.lb[i]) + self.lb[i])
        
        fitness0=self.eval_salps()
        
        self.best_position, self.best_fitness = self.select(self.Positions, fitness0)
                       
        for l in range(1, ngen+1):
            
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
                print('Best Salp Position:', np.round(self.best_position,6))
                print('c1:', self.c1r)
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        if self.verbose:
            print('------------------------ SSA Summary --------------------------')
            print('Best fitness (y) found:', self.fitness_best_correct)
            print('Best individual (x) found:', self.best_position)
            print('--------------------------------------------------------------')  
            
        return self.best_position, self.fitness_best_correct, self.history