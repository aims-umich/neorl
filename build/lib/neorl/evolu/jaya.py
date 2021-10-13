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
import joblib
from neorl.evolu.discrete import mutate_discrete, encode_grid_to_discrete, decode_discrete_to_grid

class JAYA:
    """
    JAYA algorithm
    
    :param mode: (str) problem type, either "min" for minimization problem or "max" for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param fit: (function) the fitness function 
    :param npop: (int) number of individuals in the population
    :param int_transform: (str): method of handling int/discrete variables, choose from: ``nearest_int``, ``sigmoid``, ``minmax``.
    :param ncores: (int) number of parallel processors
    :param seed: (int) random seed for sampling
    """
    
    def __init__(self, mode, bounds, fit, npop=50, int_transform ='nearest_int', ncores=1, seed=None):

        self.seed=seed
        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)

        assert npop > 3, '--eror: size of npop must be more than 3'
        self.npop= npop
        self.bounds=bounds
        self.ncores=ncores

        self.int_transform=int_transform
        if int_transform not in ["nearest_int", "sigmoid", "minmax"]:
            raise ValueError('--error: int_transform entered by user is invalid, must be `nearest_int`, `sigmoid`, or `minmax`')


        self.mode=mode
        if mode == 'max':
            self.fit=fit
        elif mode == 'min':
            def fitness_wrapper(*args, **kwargs):
                return -fit(*args, **kwargs)
            self.fit = fitness_wrapper
        else:
            raise ValueError('--error: The mode entered by user is invalid, use either `min` or `max`')

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
        self.lb=[self.bounds[item][1] for item in self.bounds]
        self.ub=[self.bounds[item][2] for item in self.bounds]
        
    def gen_indv(self, bounds): # individual 

        indv = []
        for key in bounds:
            if bounds[key][0] == 'int':
                indv.append(random.randint(bounds[key][1], bounds[key][2]))
            elif bounds[key][0] == 'float':
                indv.append(random.uniform(bounds[key][1], bounds[key][2]))
            elif bounds[key][0] == 'grid':
                indv.append(random.sample(bounds[key][1],1)[0])
        return indv

    def init_population(self, x0=None): # population

        pop = []
        if x0: # have premary solution
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
    
    def eval_pop(self, pos_array):
        #"""
        #Evaluate fitness of the population with parallel processing.

        #Return:
        #list - pop fitnesses
        #"""
        if self.ncores > 1:
            with joblib.Parallel(n_jobs=self.ncores) as parallel:
                fitness_lst = parallel(joblib.delayed(self.fit_worker)(pos_array[i, :]) for i in range(self.npop))
        else:
            fitness_lst = []
            for i in range(self.npop):
                fitness_lst.append(self.fit_worker(pos_array[i, :]))
        return fitness_lst

    def fit_worker(self, x):
        
        # Clip the wolf with position outside the lower/upper bounds and return same position
        # x=self.ensure_bounds(x)
        
        if self.grid_flag:
            #decode the individual back to the int/float/grid mixed space
            x=decode_discrete_to_grid(x,self.orig_bounds,self.bounds_map)

        fitness = self.fit(x)

        return fitness
    
    def ensure_discrete(self, vec):
        #"""
        #to mutate a vector if discrete variables exist 

        #Params:
        #vec - position in vector/list form

        #Return:
        #vec - updated position vector with discrete values
        #"""
        
        for dim in range(self.dim):
            if self.var_type[dim] == 'int':
                vec[dim] = mutate_discrete(x_ij=vec[dim], 
                                               x_min=min(vec),
                                               x_max=max(vec),
                                               lb=self.lb[dim], 
                                               ub=self.ub[dim],
                                               alpha=self.b,
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
        N = self.npop # population size
        dim = len(self.bounds) # individual length

        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        fitness_mat = np.zeros(N)

        Best_pos = np.zeros(dim)
        Best_score = float('-inf') # find a maximum, so the larger the better
        Worst_pos = np.zeros(dim)
        Worst_score = float('inf')
        ## INITIALIZE
        #  population
        if x0:
            assert len(x0) == N, '--error: the length of x0 ({}) (initial population) must equal to number of individuals npop ({})'.format(len(x0), self.npop)
            pos = self.init_population(x0=x0)
        else:
            pos = self.init_population()

        pos = pos*1.0   #this is to account for mixed intger-cont. problems, data needs to be float
        
        # calulate fitness 
        fitness_mat=self.eval_pop(pos)
        for i in range(N):
            if fitness_mat[i] > Best_score:
                Best_score = fitness_mat[i]
                Best_pos = pos[i, :]
            if fitness_mat[i] < Worst_score:
                Worst_score = fitness_mat[i]
                Worst_pos = pos[i, :]

        ## main loop
        best_scores = []
        for gen in range(1, ngen+1):
            self.b= 1 - gen * ((1) / ngen)  #mir: b decreases linearly between 1 to 0, for discrete mutation
            new_pos = np.zeros((N,dim))

            # update pos
            for i in range(N):
                r1=np.random.random(dim)
                r2=np.random.random(dim)
                # Update pos
                new_pos[i,:] = (
                    pos[i,:] 
                    + r1*(Best_pos - abs(pos[i,:]))
                    - r2*(Worst_pos - abs(pos[i,:])) # !! minus
                )
                # check bounds            
                new_pos[i,:] = self.ensure_bounds(new_pos[i,:])
                new_pos[i,:] = self.ensure_discrete(new_pos[i,:])
            
            
            fitness_new=self.eval_pop(new_pos)
                        
            for i in range(N):
                if fitness_new[i] > fitness_mat[i]:
                    pos[i,:] = new_pos[i,:]
                    fitness_mat[i] = fitness_new[i]

            # update best_score and worst_score
            for i in range(N):
                if fitness_mat[i] > Best_score:
                    Best_score = fitness_mat[i]
                    Best_pos = pos[i, :]
                if fitness_mat[i] < Worst_score:
                    Worst_score = fitness_mat[i]
                    Worst_pos = pos[i, :]            

            #-----------------------------
            #Fitness saving 
            #-----------------------------
            gen_avg = sum(fitness_mat) / N                   # current generation avg. fitness
            y_best = Best_score                                # fitness of best individual
            x_best = Best_pos.copy()
            best_scores.append(y_best)
            
            #--mir  show the value wrt min/max
            if self.mode=='min':
                y_best_correct=-y_best
                gen_avg=-gen_avg
            else:
                y_best_correct=y_best

            if verbose:
                print('************************************************************')
                print('JAYA step {}/{}, Ncores={}'.format(gen*self.npop, ngen*self.npop, self.ncores))
                print('************************************************************')
                print('Best fitness:', np.round(y_best_correct,6))

                if self.grid_flag:
                    x_decoded = decode_discrete_to_grid(x_best, self.orig_bounds, self.bounds_map)
                    print('Best individual:', x_decoded)
                else:
                    print('Best individual:', x_best)
                
                print('Average fitness:', np.round(gen_avg,6))
                print('************************************************************')

        #mir-grid
        if self.grid_flag:
            x_best_correct = decode_discrete_to_grid(x_best, self.orig_bounds, self.bounds_map)
        else:
            x_best_correct = x_best
            
        if verbose:
            print('------------------------ JAYA Summary --------------------------')
            print('Best fitness (y) found:', y_best_correct)
            print('Best individual (x) found:', x_best_correct)
            print('--------------------------------------------------------------')
    
        if self.mode=='min':
            best_scores=[-item for item in best_scores]
                
        return x_best_correct, y_best_correct, best_scores