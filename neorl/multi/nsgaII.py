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
#Created on Tues May 24 19:37:04 2022
#
#@author: Paul
# The following implementation of NSGA-II is adapted to NEORL
# from DEAP implementation: https://github.com/DEAP/deap/blob/master/deap/tools/emo.py#L15 
#"""

import random
import numpy as np
from collections import defaultdict
import copy
import joblib
from neorl.evolu.crossover import cxES2point, cxESBlend
from neorl.evolu.discrete import encode_grid_to_discrete, decode_discrete_to_grid
from neorl.utils.seeding import set_neorl_seed

from neorl.evolu.es import ES
from itertools import chain
from neorl.multi.tools import sortNondominated, sortLogNondominated, assignCrowdingDist
from neorl.utils.tools import get_population_nsga

class NSGAII(ES):
    """
    Parallel Fast Non-dominated Sorting Gentic Algorithm - II
    
    Only the seleciton operator differ from classical GA implementation. Hence, we choose create a subclass
    of ES implementation

    :param mode: (str) problem type, either ``min`` for minimization problem or ``max`` for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param lambda\_: (int) total number of individuals in the population
    :param cxmode: (str): the crossover mode, either 'cx2point' or 'blend'
    :param alpha: (float) Extent of the blending between [0,1], the blend crossover randomly selects a child in the range [x1-alpha(x2-x1), x2+alpha(x2-x1)] (Only used for cxmode='blend')
    :param cxpb: (float) population crossover probability between [0,1]
    :param mutpb: (float) population mutation probability between [0,1] 
    :param smin: (float): minimum bound for the strategy vector
    :param smax: (float): maximum bound for the strategy vector
    :param ncores: (int) number of parallel processors
    :param seed: (int) random seed for sampling
    
    NSGA-II specific parameters:

    :param sorting: (str) sorting type, ``standard`` or ``log``. The latter should be faster and is used as default.#Paul
    """
    def __init__ (self, mode, bounds, fit, lambda_=60, cxmode='cx2point', 
                  alpha=0.5, cxpb=0.6, mutpb=0.3, smin=0.01, smax=0.5, clip=True, ncores=1, seed=None,sorting = 'log', **kwargs):  
        
        set_neorl_seed(seed)
        super().__init__(mode = mode, bounds = bounds, fit = fit, lambda_=lambda_, mu=lambda_, cxmode=cxmode, 
                  alpha=alpha, cxpb=cxpb, mutpb=mutpb, smin=smin, smax=smax, clip=clip, ncores=ncores, seed=seed)

        # new hyper-parameters #Paul
        self.sorting = sorting
        def fitness_wrapper(*args, **kwargs):
            fitness = fit(*args, **kwargs) 
            if isinstance(fitness,np.ndarray):
                if mode == 'max':
                    return fitness
                elif mode == 'min':
                    return -fitness
            elif isinstance(fitness,list):
                if mode == 'max':
                    return np.array(fitness)
                elif mode == 'min':
                    return -np.array(fitness)
            else:
                if mode == 'max':
                    return np.array([fitness])
                elif mode == 'min':
                    return -np.array([fitness])
        self.fit=fitness_wrapper   
    
    def fit_worker(self, x):
        #"""
        #Evaluates fitness of an individual.
        #"""
        
        
        #mir-grid
        if self.grid_flag:
            #decode the individual back to the int/float/grid mixed space
            x=decode_discrete_to_grid(x,self.orig_bounds,self.bounds_map) 
                    
        fitness = self.fit(x)
        return fitness
        
    def select(self,pop, k = 1, nd='standard'):
        """
        Apply NSGA-II selection operator on the *pop*. 

        :param pop: (dict) A list of pop to select from.
        :param k: (int) The number of pop to select.
        :param nd: (str) Specify the non-dominated algorithm to use: 'standard' or 'log'.
        :Returns best_dict: (dict) next population in dictionary structure
        """
        if nd == 'standard':
            pareto_fronts = sortNondominated(pop, k)
        elif nd == 'log':
            pareto_fronts = sortLogNondominated(pop, k)
        else:
            raise Exception('NSGA2: The choice of non-dominated sorting '
                            'method "{0}" is invalid.'.format(nd))

        #for front in pareto_fronts:
        #    assignCrowdingDist(front)

        chosen = list(chain(*pareto_fronts[:-1]))
        k = k - len(chosen)
        if k > 0:
            CrowdDist = assignCrowdingDist(pareto_fronts[-1])# Moved here. did not see need of computing it outside
            sorted_front = sorted(CrowdDist,key = lambda k: CrowdDist[k],reverse=True)    
            sorted_front = [(x , pop[x]) for x in sorted_front]
            #sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
            chosen.extend(sorted_front[:k])
        
        # re-cast into a dictionary to comply with NEORL 
        best_dict=defaultdict(list)
        index=0
        for key in chosen:
            best_dict[index] = key[1]
            index+=1
        return best_dict
    def GenOffspring(self, pop):
        #"""
        # 
        #This function generates the offspring by applying crossover, mutation **or** reproduction. 
        #The sum of both probabilities self.cxpb and self.mutpb must be in [0,1]
        #The reproduction probability is 1 - cxpb - mutpb
        #The new offspring goes for fitness evaluation
        
        #Inputs:
        #    pop (dict): population in dictionary structure
        #Returns:
        #    offspring (dict): new modified population in dictionary structure    
        #"""
        
        
        pop_indices=list(range(0,len(pop)))
        offspring = defaultdict(list)
        for i in range(self.lambda_):
            rn = random.random()
            #------------------------------
            # Crossover
            #------------------------------
            if rn < self.cxpb:            
                index1, index2 = random.sample(pop_indices,2)
                if self.cxmode.strip() =='cx2point':
                    ind1, ind2, strat1, strat2 = cxES2point(ind1=list(pop[index1][0]),ind2=list(pop[index2][0]), 
                                                            strat1=list(pop[index1][1]),strat2=list(pop[index2][1]))
                elif self.cxmode.strip() == 'blend':
                    ind1, ind2, strat1, strat2=cxESBlend(ind1=list(pop[index1][0]), ind2=list(pop[index2][0]), 
                                                                         strat1=list(pop[index1][1]),strat2=list(pop[index2][1]),
                                                                         alpha=self.alpha)
                else:
                    raise ValueError('--error: the cxmode selected (`{}`) is not available in ES, either choose `cx2point` or `blend`'.format(self.cxmode))
                
                ind1=self.ensure_bounds(ind1)
                ind2=self.ensure_bounds(ind2)
                
                ind1=self.ensure_discrete(ind1)  #check discrete variables after crossover
                ind2=self.ensure_discrete(ind2)  #check discrete variables after crossover
                
                offspring[i + len(pop)].append(ind1)
                offspring[i + len(pop)].append(strat1)
                #print('crossover is done for sample {} between {} and {}'.format(i,index1,index2))
            #------------------------------
            # Mutation
            #------------------------------
            elif rn < self.cxpb + self.mutpb:  # Apply mutation
                index = random.choice(pop_indices)
                ind, strat=self.mutES(ind=list(pop[index][0]), strat=list(pop[index][1]))
                offspring[i + len(pop)].append(ind)
                offspring[i + len(pop)].append(strat)
                #print('mutation is done for sample {} based on {}'.format(i,index))
            #------------------------------
            # Reproduction from population
            #------------------------------
            else:                         
                index=random.choice(pop_indices)
                pop[index][0]=self.ensure_discrete(pop[index][0])
                offspring[i + len(pop)].append(pop[index][0])
                offspring[i + len(pop)].append(pop[index][1])
                #print('reproduction is done for sample {} based on {}'.format(i,index))
                
        if self.clip:
            for item in offspring:
                offspring[item][1]=list(np.clip(offspring[item][1], self.smin, self.smax))
        
        return offspring
          
    def evolute(self, ngen, x0=None, verbose=False):
        """
        This function evolutes the ES algorithm for number of generations.
        
        :param ngen: (int) number of generations to evolute
        :param x0: (list of lists) the initial position of the swarm particles
        :param verbose: (bool) print statistics to screen
        
        :return: (tuple) (best individual, best fitness, and a list of fitness history)
        """
        self.es_hist={}
        self.es_hist['mean_strategy']=[]
        self.best_scores=[]
        self.best_indvs=[]
        if x0:    
            assert len(x0) == self.lambda_, '--error: the length of x0 ({}) (initial population) must equal to the size of lambda ({})'.format(len(x0), self.lambda_)
            self.population=self.init_pop(x0=x0, verbose=verbose)
        else:
            self.population=self.init_pop(verbose=verbose)
        self.y_opt=[-np.inf for i in range(len(self.population[1][2]))]#Paul
        self.x_opt=[[] for i in range(len(self.population[1][2]))]
        if len(self.population[1][2]) == 1:
            print("--warning: length of output is 1, the sorting method is changed to ``standard``.")
            self.sorting = "standard"
        # Begin the evolution process
        for gen in range(1, ngen + 1):
            # Vary the population and generate new offspring
            offspring = self.GenOffspring(pop=self.population)
            # Evaluate the individuals with an invalid fitness with multiprocessign Pool
            # create and run the Pool
            if self.ncores > 1:
                core_list=[]
                for key in offspring:
                    core_list.append(offspring[key][0])

                with joblib.Parallel(n_jobs=self.ncores) as parallel:
                    fitness=parallel(joblib.delayed(self.fit_worker)(item) for item in core_list)
                for ind in range(len(offspring)):
                    offspring[ind + len(self.population)].append(fitness[ind]) 
                
            else: #serial calcs
                
                for ind in offspring.keys():
                    fitness=self.fit_worker(offspring[ind][0])
                    offspring[ind].append(fitness)
        
            
            # Select the next generation population
            offspring.update(self.population) # concatenate offspring and parents dictionnaries
            self.population = copy.deepcopy(self.select(pop=offspring, k=self.mu, nd = self.sorting))
            if self.RLmode:  #perform RL informed ES
                self.population=self.mix_population(self.population)
                
            #Paul many changes: simply provide the pareto front (add a layer of calculation)
            if self.sorting == "standard":#Paul
                pareto_front = sortNondominated(self.population, len(self.population))[0]
            elif self.sorting == 'log':
                pareto_front = sortLogNondominated(self.population, len(self.population))[0]  
            inds_par, rwd_par=[i[1][0] for i in pareto_front], [i[1][2] for i in pareto_front]
            self.best_scores.append(rwd_par)
            if self.grid_flag:
                temp_indvs = []
                for count,elem in enumerate(inds_par):
                    temp_indvs.append(decode_discrete_to_grid(elem,self.orig_bounds,self.bounds_map))
                self.best_indvs.append(temp_indvs)
            else:
                self.best_indvs.append(inds_par)
            
            for fitn in range(len(np.min(rwd_par,axis=0))):
                if self.mode == 'min':
                    if  - np.min(rwd_par,axis=0)[fitn] > self.y_opt[fitn]:
                        self.y_opt[fitn] = np.min(rwd_par,axis=0)[fitn]
                        self.x_opt[fitn]=copy.deepcopy(inds_par[np.argmin(rwd_par,axis=0)[fitn]])
                elif self.mode == 'max':
                    if  np.min(rwd_par,axis=0)[fitn] > self.y_opt[fitn]:
                        self.y_opt[fitn] = np.min(rwd_par,axis=0)[fitn]
                        self.x_opt[fitn]=copy.deepcopy(inds_par[np.argmin(rwd_par,axis=0)[fitn]])
            #--mir
            if self.mode=='min':#Paul
                self.y_opt_correct=[- x for x in self.y_opt]
            else:
                self.y_opt_correct=self.y_opt

            #mir-grid
            
            if self.grid_flag:
                self.x_opt_correct = []
                for count,elem in enumerate([inds_par[x] for x in np.argmin(rwd_par,axis=0)]):
                    self.x_opt_correct.append(decode_discrete_to_grid(elem,self.orig_bounds,self.bounds_map))#self>x_opt
            else:
                self.x_opt_correct=[inds_par[x] for x in np.argmin(rwd_par,axis=0)]#inds_par#self.x_opt
            
            mean_strategy=[np.mean(self.population[i][1]) for i in self.population]
            self.es_hist['mean_strategy'].append(np.mean(mean_strategy))
            if verbose:
                print('##############################################################################')
                print('NSGA-II step {}/{}, CX={}, MUT={}, MU={}, LAMBDA={}, Ncores={}'.format(gen*self.lambda_,ngen*self.lambda_, np.round(self.cxpb,2), np.round(self.mutpb,2), self.mu, self.lambda_, self.ncores))
                print('##############################################################################')
                print('Statistics for generation {}'.format(gen))
                print('Best Fitness:', np.min(rwd_par,axis=0) if self.mode == 'max' else -np.min(rwd_par,axis=0))
                print('Best Individual(s):', self.x_opt_correct)
                print('Length of the pareto front / length of the population: {} / {}'.format(len(inds_par),len(self.population)))
                print('Max Strategy:', np.round(np.max(mean_strategy),3))
                print('Min Strategy:', np.round(np.min(mean_strategy),3))
                print('Average Strategy:', np.round(np.mean(mean_strategy),3))
                print('##############################################################################')
        if verbose:#Paul
            print('------------------------ NSGA-II Summary --------------------------')
            print('Best fitness (y) found:', self.y_opt_correct)
            print('Best individual (x) found:', self.x_opt_correct)
            #print('Length of the pareto front / length of the population: {} / {}'.format(len(self.y_opt_correct),len(self.population)))
            print('--------------------------------------------------------------') 

        #---update final logger
        self.es_hist['last_pop'] = get_population_nsga(self.population,mode = self.mode)
        if self.mode == 'min':
            self.es_hist['global_fitness'] = [- x for x in rwd_par]
            self.best_scores=[-np.array(item) for item in self.best_scores]
        else:
            self.es_hist['global_fitness'] = rwd_par # pareto of last population
        self.es_hist['global_pop'] = inds_par
        self.es_hist['local_fitness'] = self.best_scores # full history of pareto front
        self.es_hist['local_pop'] = self.best_indvs
        
        return self.x_opt_correct, self.y_opt_correct, self.es_hist
    
    
