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
#Created on Mon Aug 17 2021
#
#@author: Paul
#"""


import random
import numpy as np
import joblib
from neorl.evolu.discrete import mutate_discrete, encode_grid_to_discrete, decode_discrete_to_grid
from itertools import combinations
import copy 

class TS(object):
    """
    Tabu Search Algorithm
    
    :param mode: (str) problem type, either ``min`` for minimization problem or ``max`` for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['int', -10, 10], 'x3': ['int', -100, 100]}``
    :param fit: (function) the fitness function 
    :param tabu_tenure: (int): Timestep under which a certain list of solution cannot be accessed (for diversification). Default value is 6.
    :param penalization_weight: (float): a scalar value for the coefficient that controls exploration/exploitation, 
                            i.e. importance of the frequency of a certain action performed in the search. The higher the value, the least likely is an action to be performed again after
                            multiple attempts.
    :param swap_mode: (str): either "swap" for swapping two elements of the input or "perturb" to perturb each input within certain bounds (see **Notes** below)
    :param ncores: (int) number of parallel processors (only ``ncores=1`` is supported now)
    :param seed: (int) random seed for sampling
    """
    def __init__(self, mode, bounds, fit, tabu_tenure=6, penalization_weight = 0.8, swap_mode = "perturb", ncores=1, seed=None):
        
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        assert ncores == 1,'-error: parallel implementaiton is not yet available. ncores ({}) should be equal to 1.'.format(ncores)
        #assert ncores <= len(bounds), '--error: ncores ({}) must be less than or equal than the length of an individual solution ({})'.format(ncores, len(bounds))
        
        int_transform='nearest_int'
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
        
        assert swap_mode in ["swap","perturb"],'--error: swap_mode must be either "swap" or "perturb" not ({})'.format(swap_mode)
        self.swap_mode = swap_mode # swapping method for characterizing a "move" in the tabu search
        self.tabu_tenure = tabu_tenure
        self.penalization_weight = penalization_weight
        self.ntabus = len(list(bounds.keys())) # the number of move performed depends on the size of the instance
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
        #Initialize the initial population of Tabu
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

        def eval_tabus(self,tabu_list = None):
            #---------------------
            # Fitness calcs
            #---------------------
            core_lst=[]
            if tabu_list is None:
                for case in range (0, self.Positions.shape[0]):
                    core_lst.append(self.Positions[case, :])
                if self.ncores > 1:
                    with joblib.Parallel(n_jobs=self.ncores) as parallel:
                        fitness_lst=parallel(joblib.delayed(self.fit_worker)(item) for item in core_lst)
                else:
                    fitness_lst=[]
                    for item in core_lst:
                        fitness_lst.append(self.fit_worker(item))
            else:
                for case in range (0, tabu_list.shape[0]):
                    core_lst.append(tabu_list[case, :])
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
        # Clip the tabu with position outside the lower/upper bounds and return same position
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
        #vec - tabu position in vector/list form

        #Return:
        #vec - updated tabu position vector with discrete values
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

    def UpdateTabu(self,Position, i, j):
        #""" 
        #Perturb the 'Position' vector with parameter i and j
        #if swap option: swap elements
        #if perturb option: perturb elements
        #"""
        tempposition = copy.deepcopy(Position)
        if self.swap_mode == "swap":
            # job index in the Position:
            i_index = tempposition.index(i)
            j_index = tempposition.index(j)
            tempposition[i_index], tempposition[j_index] = tempposition[j_index], tempposition[i_index]# Swap
        elif self.swap_mode == "perturb":
            if self.bounds['x'+str(i)][0] == 'int':
                tempposition[i] = random.randint(j[0],j[1])
            elif self.bounds['x'+str(i)][0] == 'float':
                tempposition[i] = random.uniform(j[0],j[1])
        return tempposition
        
    def evolute(self,ngen,x0=None, verbose=True):
        """
        This function evolutes the TS algorithm for number of generations
        
        :param ngen: (int) number of generations to evolute
        :param x0: (list) initial position of the tabu (vector size must be of same size as ``len(bounds)``)
        :param verbose: (bool) print statistics to screen
        
        :return: (tuple) (best tabu position, best fitness, and dictionary of fitness history)
        """
        self.history = {'local_fitness':[], 'global_fitness':[]}
        self.best_fitness=float("inf") 
        self.verbose=verbose
        self.Positions = np.zeros(self.dim)#np.zeros((self.ntabus, self.dim))
        if x0:
            assert len(x0) == self.ntabus, '--error: the length of x0 ({}) MUST equal the size of the problem ({})'.format(len(x0), self.ntabus)
            self.Positions = x0
        else:
            #self.Positions=self.init_sample(self.bounds)  #TODO, update later for mixed-integer optimisation
            # Initialize the positions of tabu
            if self.swap_mode == "swap":# no repetition of element if swap method
                self.Positions = list(range(1, self.ntabus + 1))
                random.shuffle(self.Positions)
            elif self.swap_mode == "perturb":
                self.Positions = self.init_sample(self.bounds)

        fitness=self.fit(self.Positions) # evaluate the initial tabu
        self.best_position, self.best_fitness = self.Positions.copy(), fitness#self.select(pos = self.Positions,fit = fitness) # find the initial best position and fitness
        current_solution = self.Positions.copy()
    
        tabu_structure = {} # record possible the tabu memory for all possible moves
        if self.swap_mode == "swap":
            for swap in combinations(range(1,self.ntabus + 1), 2):
                tabu_structure[swap] = {'tabu_time': 0, 'MoveValue': 0, 'freq': 0, 'Penalized_MV': 0}
        elif self.swap_mode == "perturb":
            for l in range(self.ntabus):
                tabu_structure[l] = {'tabu_time': 0, 'MoveValue': 0, 'freq': 0, 'Penalized_MV': 0}
                
        iter = 1    
        for l in range(1, ngen+1):# Main loop
            #-----------------------------
            # Performs multiple moves and evaluate the resulting tabu
            #-----------------------------
            increment = 0 # utilize to perturb the Position vector at the 'increment'th position
            temp_candidate = [] # store new candidate for perturbation to avoid the problem of change in perturbation when best move is called
            for move in tabu_structure:# Searching the whole neighborhood of the current solution:
                if self.swap_mode == "swap":
                    candidate_solution = self.UpdateTabu(self.Positions, move[0], move[1])
                elif self.swap_mode == "perturb":
                    candidate_solution = self.UpdateTabu(self.Positions, increment,[self.lb[increment], self.ub[increment]])
                    temp_candidate.append(candidate_solution)
                    increment +=1
                fitness = self.fit(candidate_solution)
                tabu_structure[move]['MoveValue'] = fitness
                tabu_structure[move]['Penalized_MV'] = fitness + (tabu_structure[move]['freq'] *
                                                                             self.penalization_weight)# Penalized fitness by simply adding freq to it (minimization):
            #----------------------
            #  Manipulate the tabu list
            #----------------------
            while True:# Admissible move
                best_move = min(tabu_structure, key =lambda x: tabu_structure[x]['Penalized_MV']) # select the move with the lowest Penalized fitness in the neighborhood (minimization)
                MoveValue = tabu_structure[best_move]["MoveValue"]
                tabu_time = tabu_structure[best_move]["tabu_time"]
                if tabu_structure[best_move]['Penalized_MV'] > 1e11:# no improvement
                    break
                if tabu_time < iter:# Not Tabu: the current move can be potentially added to the tabu list
                    # make the move
                    if self.swap_mode == "swap":
                        self.Positions = self.UpdateTabu(self.Positions, best_move[0], best_move[1])
                    elif self.swap_mode == "perturb":
                        best_loc = np.argmin([tabu_structure[x]['Penalized_MV'] for x in tabu_structure.keys()])
                        self.Positions = temp_candidate[best_loc].copy()
                    fitness = MoveValue#self.fit(self.Positions)
                    if MoveValue < self.best_fitness:# Best Improving move
                        self.best_position = self.Positions.copy()
                        self.best_fitness = fitness
                    # update tabu_time for the move and freq count
                    tabu_structure[best_move]['tabu_time'] = iter + self.tabu_tenure
                    tabu_structure[best_move]['freq'] += 1
                    iter += 1
                    break
                else:# If tabu
                    # Aspiration
                    if MoveValue < self.best_fitness:
                        # make the move
                        if self.swap_mode == "swap":
                            self.Positions = self.UpdateTabu(self.Positions, best_move[0], best_move[1])
                        elif self.swap_mode == "perturb":
                            best_loc = np.argmin([tabu_structure[x]['Penalized_MV'] for x in tabu_structure.keys()])
                            self.Positions = temp_candidate[best_loc].copy()
                        fitness = self.fit(self.Positions)
                        self.best_position = self.Positions.copy()
                        self.best_fitness = fitness
                        tabu_structure[best_move]['freq'] += 1
                        iter += 1
                        break
                    else:
                        tabu_structure[best_move]['Penalized_MV'] = float('inf')
                        continue
            
            #----------------------
            #  Logger related portion
            #----------------------
            #for i, fits in enumerate(fitness):  
            for i, fits in enumerate([fitness]): 
                #save the best of the best!!!
                if fits < self.best_fitness:
                    self.best_fitness=fits
                    #self.best_position=self.Positions[i, :].copy()
                    self.best_position=self.Positions.copy()
            
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
            if self.verbose and l % self.ntabus:
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                print('TS step {}/{}, ntabus={}, Ncores={}'.format((l)*self.ntabus, ngen*self.ntabus, self.ntabus, self.ncores))
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                print('Best Tabu Fitness:', np.round(self.fitness_best_correct,6))
                if self.grid_flag:
                    self.tabu_decoded = decode_discrete_to_grid(self.best_position, self.orig_bounds, self.bounds_map)
                    print('Best Tabu Position:', self.tabu_decoded)
                else:
                    print('Best Tabu Position:', self.best_position)
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        #mir-grid
        if self.grid_flag:
            self.tabu_correct = decode_discrete_to_grid(self.best_position, self.orig_bounds, self.bounds_map)
        else:
            self.tabu_correct = self.best_position                
        if self.verbose:
            print('------------------------ TS Summary --------------------------')
            print('Best fitness (y) found:', self.fitness_best_correct)
            print('Best individual (x) found:', self.tabu_correct)
            print('--------------------------------------------------------------')  
        return self.tabu_correct, self.fitness_best_correct, self.history