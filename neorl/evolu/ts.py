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
import math
from neorl.utils.seeding import set_neorl_seed
import sys
class TS2(object):
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
    :param kappa: (float) inverse temperature to control the softmax probability for 'soft' option in 'reinforce_best'.
    :param m: (float) control the probability for 'rank' option in 'reinforce_best'.
    :param reinforce_best: (str) an option to control the starting individual of the chain at every generation (See **Notes** below). choose 'None', 'hard', 'soft', 'rank', 'linear'
    :param chain_size: (int) number of individuals to evaluate in the chain every generation (e.g. like ``npop`` for other algorithms)
    :param ncores: (int) number of parallel processors (only ``ncores=1`` is supported now)
    :param seed: (int) random seed for sampling
    """
    def __init__(self, mode, bounds, fit, tabu_tenure=6, penalization_weight = 0.8,swap_mode = "perturb", kappa = 1, m = 2.0 ,reinforce_best = None, chain_size = 10,ncores=1, seed=None):
        
        set_neorl_seed(seed)
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
        self.reinforce_best=reinforce_best
        self.npop=chain_size
        self.kappa = kappa # for softmax-like sampling
        self.m = m # for ranking-based sampling
        self.seed=seed
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
        if self.swap_mode == "swap":# no repetition of element if swap method
            indv = list(range(1, self.ntabus + 1))
            random.shuffle(indv)
        elif self.swap_mode == "perturb":  
            indv=[]
            for key in bounds:
                if bounds[key][0] == 'int':
                    indv.append(random.randint(bounds[key][1], bounds[key][2]))
                elif bounds[key][0] == 'float':
                    indv.append(random.uniform(bounds[key][1], bounds[key][2]))
                else:
                    raise Exception ('unknown data type is given, either int, float, or grid are allowed for parameter bounds')   
        return np.array(indv)

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
            if self.bounds['x'+str(i + 1)][0] == 'int':
                tempposition[i] = random.randint(j[0],j[1])
            elif self.bounds['x'+str(i + 1)][0] == 'float':
                tempposition[i] = random.uniform(j[0],j[1])
        return tempposition

    def chain_object(self,inp):
        #"""
        #This function is a multiprocessing object, used to be passed to Pool, that respresents 
        #an individual SA chain. 
        #Input:
        #    inp: a list contains the following inputs in order
        #    inp[0] --> x0: initial guess to chain 
        #    inp[1] --> xbest: best of the chain 
        #    inp[2] --> E0: initial energy of x0
        #    inp[3] --> tabu0: initial tabu of all chain
        #    inp[4] --> iter0: initial iter of all chain
        #    inp[5] --> global Ebest: global best of this chain 
        #    inp[6] --> min_step: min step to start this chain 
        #    inp[7] --> max_step: max step to terminate this chain 
        #    inp[8] --> core_seed: seed for this chain
        #returns: 
        #    x_prev, E_prev: last obtained from this chain
        #    x_best, E_best: best obtained from this chain
        #    tabu_strucutre: tabu structure for this chain
        #    iter: iterate for this chain
        #"""
        x_prev=copy.deepcopy(inp[0])
        x_best=copy.deepcopy(inp[1])
        E_prev=inp[2]
        E_best = inp[3]
        tabu_structure = inp[4]
        iter = inp[5]
        min_step=inp[6]
        max_step=inp[7]
        core_seed=inp[8]
        
        if not (self.seed is None):
            random.seed(self.seed + core_seed)
            np.random.seed(self.seed + core_seed)
        k=min_step
        while k <= max_step:
            #-----------------------------
            # Performs multiple moves and evaluate the resulting tabu
            #-----------------------------
            increment = 0 # utilize to perturb the Position vector at the 'increment'th position
            temp_candidate = list() # store new candidate for perturbation to avoid the problem of change in perturbation when best move is called
            for move in tabu_structure:# Searching the whole neighborhood of the current solution:
                if self.swap_mode == "swap":
                    candidate_solution = self.UpdateTabu(x_prev, move[0], move[1])
                elif self.swap_mode == "perturb":
                    candidate_solution = self.UpdateTabu(x_prev, increment,[self.lb[increment], self.ub[increment]])
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
                    best_loc = np.argmin([tabu_structure[x]['Penalized_MV'] for x in tabu_structure.keys()])
                    x_prev = temp_candidate[best_loc].copy()
                    E_prev = MoveValue.copy()
                    if MoveValue < E_best:# Best Improving move
                        x_best = x_prev.copy()
                        E_best = copy.deepcopy(E_prev)
                    # update tabu_time for the move and freq count
                    tabu_structure[best_move]['tabu_time'] = iter + self.tabu_tenure
                    tabu_structure[best_move]['freq'] += 1
                    iter += 1
                    break
                else:# If tabu
                    # Aspiration
                    if MoveValue < E_best:
                        # make the move
                        best_loc = np.argmin([tabu_structure[x]['Penalized_MV'] for x in tabu_structure.keys()])
                        x_prev = temp_candidate[best_loc].copy()
                        E_prev = MoveValue.copy()
                        x_best = x_prev.copy()
                        E_best = copy.deepcopy(E_prev)
                        tabu_structure[best_move]['freq'] += 1
                        iter += 1
                        break
                    else:
                        tabu_structure[best_move]['Penalized_MV'] = float('inf')
                        continue
            k+=1
        return x_prev, E_prev, x_best, E_best, tabu_structure, iter
    
    def chain(self, x0, E0, tabu0, iter0, globalx0, globalE0, step0):
        #"""
        #This function creates ``ncores`` independent TS chains with same initial guess x0, E0, tabu0 and 
        #runs them via multiprocessing Pool.
        #Input:
        #    x0: initial input guess (comes from previous annealing chains or from replay memory)
        #    E0: energy/fitness value of x0
        #    tabu0: tabu list of each chain
        #    iter0: iterate for tabu check of each chain
        #    step0: is the first time step 
        #    npop: total number of individuals to be evaluated in this TS stage
        #returns: 
        #    x_best, E_best, Tabu obtained from this TS stage from all chains, iter for all chain
        #"""
        
        #Append and prepare the input list to be passed as input to Pool.map
        core_list=[]
        core_step_min=step0
        for j in range(1,self.ncores+1):
            core_step_max=step0+j*self.npop-1
            core_list.append([x0[j-1], globalx0[j-1], E0[j-1], globalE0[j-1], tabu0[j-1],iter0[j-1], core_step_min, core_step_max, j])
            core_step_min=core_step_max+1
        
        
        if self.ncores > 1:

            with joblib.Parallel(n_jobs=self.ncores) as parallel:
                results=parallel(joblib.delayed(self.chain_object)(item) for item in core_list)
        else:
            results=[]
            results.append(list(self.chain_object(core_list[0])))
                
        self.x_last=[item[0] for item in results]
        self.E_last=[item[1] for item in results]
        
        self.x_best, self.E_best=[item[2] for item in results], [item[3] for item in results]
        self.Tabu, self.iter=[item[4] for item in results], [item[5] for item in results]
        return self.x_last, self.E_last, self.x_best, self.E_best, self.Tabu, self.iter
    
    def InitChains(self, x0=None):
        
        #initialize the chain and run them in parallel (these samples will be used to initialize the tabu search process)
        #Establish the chain
        if x0:
            print('The first TS x0 individual provided by the user:', x0[0])
            print('The last TS x0 individual provided by the user:', x0[-1])
        else:
            x0=[]
            for i in range (self.ncores):
                x0.append(self.init_sample(self.bounds))
        
        
        if self.ncores > 1:  #evaluate chain in parallel
            core_list=[]
            for ind in x0:
                core_list.append(ind)
           
            with joblib.Parallel(n_jobs=self.ncores) as parallel:
                E0=parallel(joblib.delayed(self.fit)(item) for item in core_list)

            tabu_structure = {}
            for ind in range(self.ncores):
                tabu = {}
                if self.swap_mode == "swap":
                    for swap in combinations(range(1,self.ntabus + 1), 2):
                        tabu[swap] = {'tabu_time': 0, 'MoveValue': 0, 'freq': 0, 'Penalized_MV': 0}
                elif self.swap_mode == "perturb":
                    for l in range(self.ntabus):
                        tabu[l] = {'tabu_time': 0, 'MoveValue': 0, 'freq': 0, 'Penalized_MV': 0}
                tabu_structure[ind] = tabu
            iter0 = np.ones(self.ncores)  
        else: #evaluate chain in series
            E0=[]
            for ind in x0:
                fitness=self.fit(ind)
                E0.append(fitness)
        
            tabu_structure = {} # record possible the tabu memory for all possible moves
            tabu_structure[0] = {}
            if self.swap_mode == "swap":
                for swap in combinations(range(1,self.ntabus + 1), 2):
                    tabu_structure[0][swap] = {'tabu_time': 0, 'MoveValue': 0, 'freq': 0, 'Penalized_MV': 0}
            elif self.swap_mode == "perturb":
                for l in range(self.ntabus):
                    tabu_structure[0][l] = {'tabu_time': 0, 'MoveValue': 0, 'freq': 0, 'Penalized_MV': 0}
            iter0 = [1]
        return x0, E0, tabu_structure, iter0 #return initial guess and initial fitness      
   
    def evolute(self,ngen,x0=None, verbose=False):
        """
        This function evolutes the TS algorithm for number of generations
        
        :param ngen: (int) number of generations to evolute
        :param x0: (list) initial position of the tabu (vector size must be of same size as ``len(bounds)``)
        :param verbose: (bool) print statistics to screen
        
        :return: (tuple) (best individual, best fitness, and dictionary containing major search results)
        """
        self.history = {'x':[],'local_fitness':[], 'global_fitness':[]}
        E_opt=np.inf 
        self.verbose=verbose
        step0=1
        self.steps=ngen * self.npop
        
        if x0:
            if isinstance(x0[0], list):
                if not all(len(item) == len(x0[0]) for item in x0):
                    raise Exception ('--error: the variable x0 must be a list of lists, and all internal lists must have same length.')
            else:
                x0=[x0]
                
            assert len(x0) == self.ncores, '--error: Length of initial guesses x0 ({}) for chains do not equal to ncores or # of chains ({})'.format(len(x0), self.ncores)
            assert len(x0[0]) == len(self.bounds), '--error: Length of every list in x0 ({}) do not equal to the size of parameter space in bounds ({})'.format(len(x0[0]), len(self.bounds))
            xinit,Einit,Tabuinit,iterinit=self.InitChains(x0=x0)
        else:
            xinit,Einit,Tabuinit,iterinit=self.InitChains()

        x_next=copy.deepcopy(xinit)
        E_next=copy.deepcopy(Einit)
        globalx_best=copy.deepcopy(xinit)
        globalE_best=copy.deepcopy(Einit)
        Tabu_next=copy.deepcopy(Tabuinit)
        iter_next=copy.deepcopy(iterinit)
        ngen=int(ngen/self.ncores)
        for l in range(1, ngen+1):# Main loop
            x_next,E_next, x_best, E_best, Tabu_next, iter_next =self.chain(x0=x_next, E0=E_next,tabu0 = Tabu_next, iter0 = iter_next, globalx0 = globalx_best, globalE0 = globalE_best, step0=step0)
            globalE_best = E_best.copy()
            globalx_best = x_best.copy()
            step0=step0+self.npop*self.ncores
            arg_min=np.argmin(E_best)
            self.history['x'].append(x_best[arg_min])
            if self.mode=='max':
                self.history['global_fitness'].append(-min(E_best))
                self.history['local_fitness'].append(-min(E_next))
            else:
                self.history['global_fitness'].append(min(E_best))
                self.history['local_fitness'].append(min(E_next))
            
            if min(E_best) < E_opt:
                E_opt=min(E_best)
                x_opt=copy.deepcopy(x_best[arg_min])
            
            if self.reinforce_best == 'hard': # enfore best chain + heuristics not only the best x
                x_next = [x_next[arg_min]]*self.ncores
                globalx_best=[x_best[arg_min]]*self.ncores#[x_opt]*self.ncores
                E_next = [E_next[arg_min]]*self.ncores
                globalE_best=[E_best[arg_min]]*self.ncores#[E_opt]*self.ncores
                Tabu_next = [Tabu_next[arg_min]]*self.ncores
                iter_next = [iter_next[arg_min]]*self.ncores

            elif self.reinforce_best == 'soft':
                temp_E = copy.deepcopy(np.array(E_next))
                normalization = 1 / np.sum(np.exp(- temp_E * self.kappa)) # cte to generate a probability
                sampling = np.zeros(self.ncores)
                sampling[0] = np.exp(- temp_E[0] * self.kappa) # utilize last accepted solution to generate the sampling
                for i in range(1,self.ncores):
                    sampling[i] = sampling[i - 1] + np.exp(- temp_E[i] * self.kappa)
                sampling = sampling * normalization
                if self.ncores != 1:
                    rho = np.random.uniform(0,1,size = self.ncores) # sample random number between 0 and 1
                else:
                    rho = [1] # probability of choosing itself is 1    
                for count,prob in enumerate(rho):
                    if prob == 1:
                        index = self.ncores - 1
                    else:
                        if prob <= sampling[0]:
                            index = 0
                        elif math.isnan(sampling[count]):# exponantial can lead to numerical erros
                            index = np.argmax(E_next)
                        else:
                            index = np.where(sampling > prob)[0][0]
                    x_next[count] = x_next[index].copy() # re-initialize with the next x by the index'th Markov Chain
                    E_next[count] = copy.deepcopy(E_next[index])
                    globalx_best[count] = x_best[index].copy() # re-initialize with the best ever found by the index'th Markov Chain
                    globalE_best[count] = copy.deepcopy(E_best[index])
                    Tabu_next[count] = copy.deepcopy(Tabu_next[index])
                    iter_next[count] = copy.deepcopy(iter_next[index])
            elif self.reinforce_best == 'roulette':
                temp_E =  1 / copy.deepcopy(np.abs(E_next)) 
                normalization = 1 / np.sum(temp_E) # cte to generate a probability
                sampling = np.zeros(self.ncores)
                sampling[0] =  temp_E[0] # utilize last accepted solution to generate the sampling
                for i in range(1,self.ncores):
                    sampling[i] = sampling[i - 1] + temp_E[i]
                sampling = sampling * normalization
                rho = np.random.uniform(0,1,size = self.ncores) # sample random number between 0 and 1
                for count,prob in enumerate(rho):
                    if prob == 1:
                        index = self.ncores - 1
                    else:
                        if prob <= sampling[0]:
                            index = 0
                        else:
                            index = np.where(sampling > prob)[0][0]
                    x_next[count] = x_next[index].copy() # re-initialize with the next x by the index'th Markov Chain
                    E_next[count] = copy.deepcopy(E_next[index])
                    globalx_best[count] = x_best[index].copy() # re-initialize with the best ever found by the index'th Markov Chain
                    globalE_best[count] = copy.deepcopy(E_best[index])
                    Tabu_next[count] = copy.deepcopy(Tabu_next[index])
                    iter_next[count] = copy.deepcopy(iter_next[index])
            elif self.reinforce_best == 'rank':# linear ranking-based strategy
                if self.ncores != 1:
                    Energy_dict = dict(zip(list(map(int,np.linspace(0,self.ncores - 1,self.ncores))), E_next))
                    new_indices = sorted(Energy_dict,key = lambda k: Energy_dict[k],reverse=True)                    
                    
                    normalization = 1 / len(E_next) # cte to generate a probability
                    sampling = np.zeros(len(E_next))
                    sampling[0] =  2 - self.m + 2 * (self.m - 1) * (1 - 1) / (len(E_next) - 1) # utilize last accepted solution to generate the sampling
                    for i in range(1,self.ncores):
                        sampling[i] = sampling[i - 1] + 2 - self.m + 2 * (self.m - 1) * (i) / (len(E_next) - 1)
                    sampling = sampling * normalization
                    rho = np.random.uniform(0,1,size = self.ncores) # sample random number between 0 and 1
                    for count,prob in enumerate(rho):
                        if prob == 1:
                            index = self.ncores - 1
                        else:
                            if prob <= sampling[0]:
                                index = 0
                            else:
                                index = np.where(sampling > prob)[0][0]
                        #count, index = new_indices[count],new_indices[index]
                        index = new_indices[index]
                        x_next[count] = x_next[index].copy() # re-initialize with the next x by the index'th Markov Chain
                        E_next[count] = copy.deepcopy(E_next[index])
                        globalx_best[count] = x_best[index].copy() # re-initialize with the best ever found by the index'th Markov Chain
                        globalE_best[count] = copy.deepcopy(E_best[index])
                        Tabu_next[count] = copy.deepcopy(Tabu_next[index])
                        iter_next[count] = copy.deepcopy(iter_next[index])
            if verbose:
                print('************************************************************')
                print('TS step {}/{}, '.format(step0-1,self.steps))
                print('************************************************************')
                print('Statistics for the {} parallel chains'.format(self.ncores))
                if self.mode=='max': 
                    print('local fitness:', -np.round(E_next,6))
                    print('Best global fitness:',- np.round(E_best,6))
                else: 
                    print('local fitness:', np.round(E_next,6))
                    print('Best global fitness:', np.round(E_best,6))
                print('Best individual:', x_best)
                print('************************************************************')
            
        #--mir
        if self.mode=='max': # solve a minimization by default
            self.E_opt_correct=-E_opt
        else:
            self.E_opt_correct=E_opt
                
        if verbose:
            if self.grid_flag:
                x_opt = decode_discrete_to_grid(x_opt, self.orig_bounds, self.bounds_map)
            print('------------------------ TS Summary --------------------------')
            print('Best fitness (y) found:', self.E_opt_correct)
            print('Best individual (x) found:', x_opt)
            print('--------------------------------------------------------------')
            
        return x_opt, self.E_opt_correct, self.history