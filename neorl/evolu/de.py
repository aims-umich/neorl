# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 13:45:54 2020

@author: Majdi
"""

#--- IMPORT DEPENDENCIES ------------------------------------------------------+

import random
from random import sample
from random import uniform
import pandas as pd
import numpy as np

#--- FUNCTIONS ----------------------------------------------------------------+

import multiprocessing
import multiprocessing.pool
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

class DE:
    def __init__ (self, bounds, fit, npop=50, mutate=0.5, recombination=0.7, ncores=1, seed=None):  
        """
        Parallel Differential Evolution:
        Inputs:
            bounds (dict): input paramter lower/upper bounds in dictionary form
            fit (function): fitness function 
            npop (int): number of individuals in the population group
            ncores (int): parallel cores
            mutate (float): mutation factor
            recombination (float): recombination factor 
            seed (int): random seeding for reproducibility
        """
        
        if seed:
            random.seed(seed)
            
        self.seed=seed
        self.npop=npop
        self.bounds=bounds
        self.ncores=ncores
        self.fit=fit
        self.mutate=mutate
        self.recombination=recombination
        
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

    def GenIndv(self, bounds):
        """
        Particle generator
        Input: 
            -bounds (dict): input paramter type and lower/upper bounds in dictionary form
        Returns: 
            -particle (list): particle position
            -speed (list): particle speed
        """
        
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

    def InitPopulation(self, x0=None):
        
        pop=[]
        #Establish the swarm
        if x0:
            print('The first individual provided by the user:', x0[0])
            print('The last individual provided by the user:', x0[-1])
            for i in range(len(x0)):
                pop.append(x0[i])
        else:
            for i in range (self.npop):
                indv=self.GenIndv(self.bounds)
                pop.append(indv)
        
        return pop

    def evolute(self, ngen, x0=None, verbose=0):
        
        #print('***************************************************************************')
        #print('***************************************************************************')
        #print('*******************Differential Evolution (DE)*****************************')
        #print('***************************************************************************')
        #print('***************************************************************************')
        if self.seed:
            random.seed(self.seed)
        #np.random.seed(self.seed)
        
        #--- INITIALIZE A POPULATION (step #1) ----------------+
        
        if x0:
            assert len(x0) == self.npop, '--error: the length of x0 ({}) (initial population) must equal to number of individuals npop ({})'.format(len(x0), self.npop)
            population = self.InitPopulation(x0=x0)
        else:
            population = self.InitPopulation()
                
        #--- SOLVE --------------------------------------------+
    
        # cycle through each generation (step #2)
        best_scores=[]
        for gen in range(1,ngen+1):
            
            #print(population)
            gen_scores = [] # score keeping
    
            # cycle through each individual in the population
            for j in range(0, self.npop):
    
                #--- MUTATION (step #3.A) ---------------------+
                
                # select three random vector index positions [0, popsize), not including current vector (j)
                candidates = list(range(0, self.npop))
                candidates.remove(j)
                random_index = sample(candidates, 3)
                            
                x_1 = population[random_index[0]]
                x_2 = population[random_index[1]]
                x_3 = population[random_index[2]]
                x_t = population[j]     # target individual
    
                # subtract x3 from x2, and create a new vector (x_diff)
                x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]
    
                # multiply x_diff by the mutation factor (F) and add to x_1
                v_donor = [x_1_i + self.mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
                v_donor = self.ensure_bounds(v_donor, bounds=self.bounds) #XXX check this line
    
                #--- RECOMBINATION (step #3.B) ----------------+
    
                v_trial = []
                for k in range(len(x_t)):
                    crossover = random.random()
                    if crossover <= self.recombination:
                        v_trial.append(v_donor[k])
    
                    else:
                        v_trial.append(x_t[k])
                        
                #--- GREEDY SELECTION (step #3.C) -------------+
                score_trial  = self.fit(v_trial)
                score_target = self.fit(x_t)
                    
                if score_trial > score_target:
                    population[j] = v_trial
                    gen_scores.append(score_trial)
    
                else:
                    gen_scores.append(score_target)
    
            #--- SCORE KEEPING --------------------------------+
            gen_avg = sum(gen_scores) / self.npop                         # current generation avg. fitness
            y_best = max(gen_scores)                                  # fitness of best individual
            x_best = population[gen_scores.index(max(gen_scores))]     # solution of best individual
            best_scores.append(y_best)
                    
#            if verbose:
#                print ("GENERATION:",gen)
#                print ('      > GENERATION AVERAGE:',gen_avg)
#                print ('      > GENERATION BEST:',gen_best)
#                print ('         > BEST SOLUTION:',gen_sol,'\n')

            if verbose:
                print('************************************************************')
                print('DE step {}/{}, Mutate={}, Recombination={}, Ncores={}'.format(gen*self.npop, ngen*self.npop, self.mutate, self.recombination, self.ncores))
                print('************************************************************')
                print('Best fitness:', np.round(y_best,6))
                print('Best individual:', x_best)
                print('Average fitness:', np.round(gen_avg,6))
                print('************************************************************')
            
        print('------------------------ DE Summary --------------------------')
        print('Best fitness (y) found:', y_best)
        print('Best individual (x) found:', x_best)
        print('--------------------------------------------------------------')
        return x_best, y_best, best_scores

#--- CONSTANTS ----------------------------------------------------------------+
