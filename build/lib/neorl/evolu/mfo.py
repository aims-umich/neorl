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
#@author: Xubo
#@email: guxubo@alumni.sjtu.edu
#"""
import random
import numpy as np
import math
import joblib
from neorl.evolu.discrete import mutate_discrete, encode_grid_to_discrete, decode_discrete_to_grid

class MFO:
    """
    Moth-flame Optimization (MFO)
    
    :param mode: (str) problem type, either "min" for minimization problem or "max" for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param fit: (function) the fitness function 
    :param nmoths: (int) number of moths in the population
    :param b: (float) constant for defining the shape of the logarithmic spiral
    :param int_transform: (str): method of handling int/discrete variables, choose from: ``nearest_int``, ``sigmoid``, ``minmax``.
    :param ncores: (int) number of parallel processors
    :param seed: (int) random seed for sampling
    """
    
    def __init__(self, mode, bounds, fit, nmoths=50, b=1, int_transform='nearest_int', ncores=1, seed=None):

        self.seed=seed
        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)

        assert ncores <= nmoths, '--error: ncores ({}) must be less than or equal to nmoths ({})'.format(ncores, nmoths)
        assert nmoths > 3, '--eror: size of nmoths must be more than 3'

        self.mode=mode
        if mode == 'min':
            self.fit=fit
        elif mode == 'max':
            def fitness_wrapper(*args, **kwargs):
                return -fit(*args, **kwargs)
            self.fit = fitness_wrapper
        else:
            raise ValueError('--error: The mode entered by user is invalid, use either `min` or `max`')

        self.int_transform=int_transform
        if int_transform not in ["nearest_int", "sigmoid", "minmax"]:
            raise ValueError('--error: int_transform entered by user is invalid, must be `nearest_int`, `sigmoid`, or `minmax`')
          
        self.npop= nmoths
        self.bounds=bounds
        self.ncores=ncores
        self.b=b
        
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
        
    def gen_indv(self, bounds): # individual 

        indv = []
        for key in bounds:
            if bounds[key][0] == 'int':
                indv.append(random.randint(bounds[key][1], bounds[key][2]))
            elif bounds[key][0] == 'float':
                indv.append(random.uniform(bounds[key][1], bounds[key][2]))
            #elif bounds[key][0] == 'grid':
            #    indv.append(random.sample(bounds[key][1],1)[0])
            else:
                raise Exception ('unknown data type is given, either int, float, or grid are allowed for parameter bounds')   
        return indv

    def init_population(self, x0=None): # population

        pop = []
        if x0: # have primary solution
            print('The first individual provided by the user:', x0[0])
            print('The last individual provided by the user:', x0[-1])
            for i in range(len(x0)):
                pop.append(x0[i])
        else: # random init
            for i in range(self.npop):
                indv=self.gen_indv(self.bounds)
                pop.append(indv)
        
        # array
        pop = np.array(pop)

        return pop

    def ensure_bounds(self, vec): # bounds check

        vec_new = []

        for i, (key, val) in enumerate(self.bounds.items()):
            # less than minimum 
            if vec[i] < self.bounds[key][1]:
                vec_new.append(self.bounds[key][1])
            # more than maximum
            if vec[i] > self.bounds[key][2]:
                vec_new.append(self.bounds[key][2])
            # fine
            if self.bounds[key][1] <= vec[i] <= self.bounds[key][2]:
                vec_new.append(vec[i])
        
        return vec_new

    def fit_worker(self, x):
        
        if self.grid_flag:
            #decode the individual back to the int/float/grid mixed space
            x=decode_discrete_to_grid(x,self.orig_bounds,self.bounds_map)
            
        fitness = self.fit(x)

        return fitness
    
    def ensure_discrete(self, vec):
        #"""
        #to mutate a vector if discrete variables exist 
        #handy function to be used within MFO phases

        #Params:
        #vec - moth position in vector/list form

        #Return:
        #vec - updated moth position vector with discrete values
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

    def evolute(self, ngen, x0=None, verbose=0):
        """
        This function evolutes the MFO algorithm for number of generations.
        
        :param ngen: (int) number of generations to evolute
        :param x0: (list of lists) the initial individuals of the population
        :param verbose: (bool) print statistics to screen
        
        :return: (dict) dictionary containing major MFO search results
        """
        
        self.history = {'local_fitness':[], 'global_fitness':[], 'r': []}
        self.best_fitness=float("inf")
        N = self.npop # population size
        dim = len(self.bounds) # individual length

        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)

        ## INITIALIZE
        #  moths
        if x0:
            assert len(x0) == N, '--error: the length of x0 ({}) (initial population) must equal to number of individuals npop ({})'.format(len(x0), self.npop)
            Moth_pos = self.init_population(x0=x0)
        else:
            Moth_pos = self.init_population()
            
        Moth_pos = Moth_pos * 1.0 #this is to account for mixed intger-cont. problems, data needs to be float
        Moth_fitness = np.full(N, float('inf'))  # set as worst result
        
        

        # sort moths
        sorted_population = np.copy(Moth_pos)
        fitness_sorted = np.zeros(N)
        # flames
        best_flames = np.copy(Moth_pos)
        best_flame_fitness = np.zeros(N)
        # moths+flames
        double_population = np.zeros((2 * N, dim))
        double_fitness = np.zeros(2 * N)     
        double_sorted_population = np.zeros((2*N, dim))
        double_fitness_sorted = np.zeros(2*N)
        # previous generation
        previous_population = np.zeros((N, dim))
        previous_fitness = np.zeros(N)
        
        ## main loop
        for gen in range(1, ngen+1):
            self.a= 1 - gen * ((1) / ngen)  #mir: a decreases linearly between 1 to 0, for discrete mutation
            Flame_no = round(N - gen*((N-1) / (ngen+1)))

            core_lst=[]
            for case in range (0, Moth_pos.shape[0]):
                core_lst.append(Moth_pos[case, :])
                    
            if self.ncores > 1: 
                with joblib.Parallel(n_jobs=self.ncores) as parallel:
                    Moth_fitness=parallel(joblib.delayed(self.fit_worker)(indv) for indv in core_lst) # 2d list
                Moth_pos = np.array(Moth_pos)
                Moth_fitness = np.array(Moth_fitness)
            else:
                Moth_fitness=[]
                for item in core_lst:
                    Moth_fitness.append(self.fit_worker(item))

            for i, fits in enumerate(Moth_fitness):
                #save the best of the best!!!
                if fits < self.best_fitness:
                    self.best_fitness=fits
                    self.best_position=Moth_pos[i, :].copy()
                                                            
            if gen == 1: # OF # equal to OM #
                # sort the moths
                fitness_sorted = np.sort(Moth_fitness) # default: (small -> large)
                #fitness_sorted = -(np.sort(-np.array(Moth_fitness)))  # descend (large -> small)
                I = np.argsort(np.array(Moth_fitness)) # index of sorted list 
                sorted_population = Moth_pos[I, :]

                # update flames
                best_flames = sorted_population
                best_flame_fitness = fitness_sorted

            else: # #OF may > #OM
                
                double_population = np.concatenate((previous_population, best_flames), axis=0)
                double_fitness = np.concatenate((previous_fitness, best_flame_fitness), axis=0)
                
                double_fitness_sorted = np.sort(double_fitness)
                I2 = np.argsort(double_fitness)
                double_sorted_population = double_population[I2, :]
                
                fitness_sorted = double_fitness_sorted[0:N]
                sorted_population = double_sorted_population[0:N, :]

                best_flames = sorted_population
                best_flame_fitness = fitness_sorted
            
            # record the best flame so far   
            Best_flame_score = fitness_sorted[0]
            Best_flame_pos = sorted_population[0, :]

            # previous
            previous_population = np.copy(Moth_pos)  # if not using np.copy(),changes of Moth_pos after this code will also change previous_population!  
            previous_fitness = np.copy(Moth_fitness) # because of the joblib..

            # r linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
            r = -1 + gen * ((-1) / ngen)

            # update moth position
            for i in range(0, N):
                for j in range(0,dim):
                    if i <= Flame_no:
                        distance_to_flame = abs(sorted_population[i,j]-Moth_pos[i,j])
                        t = (r-1)*random.random()+1
                        # eq. (3.12)
                        Moth_pos[i,j] = (
                            distance_to_flame*math.exp(self.b*t)*math.cos(t*2*math.pi)
                        + sorted_population[i,j] 
                        )

                    if i > Flame_no: 
                        distance_to_flame = abs(sorted_population[Flame_no,j]-Moth_pos[i,j])
                        t = (r-1)*random.random()+1     
                        # rebundant moths all fly to the last Flame_no
                        Moth_pos[i,j] = (
                            distance_to_flame*math.exp(self.b*t)*math.cos(t*2*math.pi)
                        + sorted_population[Flame_no,j] 
                        )
            
                
                Moth_pos[i,:]=self.ensure_bounds(Moth_pos[i,:])
                Moth_pos[i, :] = self.ensure_discrete(Moth_pos[i, :])
                
            #-----------------------------
            #Fitness saving 
            #-----------------------------
            gen_avg = sum(best_flame_fitness) / len(best_flame_fitness)  # current generation avg. fitness
                
            #--mir
            if self.mode=='max':
                self.fitness_best_correct=-self.best_fitness
                self.local_fitness=-Best_flame_score
            else:
                self.fitness_best_correct=self.best_fitness
                self.local_fitness=Best_flame_score

            self.history['local_fitness'].append(self.local_fitness)
            self.history['global_fitness'].append(self.fitness_best_correct)
            self.history['r'].append(r)

            if verbose:
                print('************************************************************')
                print('MFO step {}/{}, Ncores={}'.format(gen*self.npop, ngen*self.npop, self.ncores))
                print('************************************************************')
                print('Best fitness:', np.round(self.fitness_best_correct,6))
                if self.grid_flag:
                    self.moth_decoded = decode_discrete_to_grid(self.best_position, self.orig_bounds, self.bounds_map)
                    print('Best individual:', self.moth_decoded)
                else:
                    print('Best individual:', self.best_position)
                print('Average fitness:', np.round(gen_avg,6))
                print('************************************************************')

        #mir-grid
        if self.grid_flag:
            self.moth_correct = decode_discrete_to_grid(self.best_position, self.orig_bounds, self.bounds_map)
        else:
            self.moth_correct = self.best_position.copy()

        if verbose:
            print('------------------------ MFO Summary --------------------------')
            print('Best fitness (y) found:', self.fitness_best_correct)
            print('Best individual (x) found:', self.moth_correct)
            print('--------------------------------------------------------------')
                
        return self.moth_correct, self.fitness_best_correct, self.history