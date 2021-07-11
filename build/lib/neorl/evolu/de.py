# -*- coding: utf-8 -*-
#Created on Sun Jun 14 13:45:54 2020
#@author: Majdi Radaideh

import random
import numpy as np
import joblib 

class DE:
    """
    Parallel Differential Evolution
    
    :param mode: (str) problem type, either "min" for minimization problem or "max" for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param fit: (function) the fitness function 
    :param npop: (int) number of individuals in the population
    :param F: (float) differential/mutation weight between [0,2]
    :param CR: (float) crossover probability between [0,1]
    :param ncores: (int) number of parallel processors
    :param seed: (int) random seed for sampling
    """
    def __init__ (self, mode, bounds, fit, npop=50, F=0.5, CR=0.3, ncores=1, seed=None):  

        self.seed=seed
        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        assert npop > 4, '--error: size of npop must be more than 4'
        self.npop=npop
        self.bounds=bounds
        self.ncores=ncores
        #--mir
        self.mode=mode
        if mode == 'max':
            self.fit=fit
        elif mode == 'min':
            def fitness_wrapper(*args, **kwargs):
                return -fit(*args, **kwargs) 
            self.fit=fitness_wrapper
        else:
            raise ValueError('--error: The mode entered by user is invalid, use either `min` or `max`')
            
        self.F=F
        self.CR=CR
        
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
        #"""
        #Particle generator
        #Input: 
        #    -bounds (dict): input paramter type and lower/upper bounds in dictionary form
        #Returns: 
        #    -particle (list): particle position
        #    -speed (list): particle speed
        #"""
        
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

    def fit_worker(self, x):
        
        # Calculate objective function for each search agent
        fitness = self.fit(x)
        
        return fitness
    
    def evolute(self, ngen, x0=None, verbose=0):
        """
        This function evolutes the DE algorithm for number of generations.
        
        :param ngen: (int) number of generations to evolute
        :param x0: (list of lists) the initial individuals of the population
        :param verbose: (bool) print statistics to screen
        
        :return: (dict) dictionary containing major DE search results
        """
        
        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)

        #--- INITIALIZE the population
        
        if x0:
            assert len(x0) == self.npop, '--error: the length of x0 ({}) (initial population) must equal to number of individuals npop ({})'.format(len(x0), self.npop)
            population = self.InitPopulation(x0=x0)
        else:
            population = self.InitPopulation()
                
        with joblib.Parallel(n_jobs=self.ncores) as parallel:
            # loop through all generations
            best_scores=[]
            for gen in range(1,ngen+1):
                
                #print(population)
                gen_scores = [] # score keeping
                
                x_t_lst=[]
                v_trial_lst = []
                
                # cycle through each individual in the population
                for j in range(0, self.npop):
                    
                    #-----------------------------
                    #Mutation
                    #-----------------------------
                    # select three random vector index positions [0, popsize), not including current vector (j)
                    candidates = list(range(0, self.npop))
                    candidates.remove(j)
                    random_index = random.sample(candidates, 3)
                                
                    x_1 = population[random_index[0]]
                    x_2 = population[random_index[1]]
                    x_3 = population[random_index[2]]
                    x_t = population[j]     # target individual
        
                    # subtract x3 from x2, and create a new vector (x_diff)
                    x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]
        
                    # multiply x_diff by the mutation factor (F) and add to x_1
                    v_donor = [x_1_i + self.F * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
                    v_donor = self.ensure_bounds(v_donor, bounds=self.bounds) #XXX check this line
        
                    #-----------------------------
                    #Recombination
                    #-----------------------------
        
                    v_trial = []
                    for k in range(len(x_t)):
                        crossover = random.random()
                        if crossover <= self.CR:
                            v_trial.append(v_donor[k])
        
                        else:
                            v_trial.append(x_t[k])
                    
                    x_t_lst.append(x_t)
                    v_trial_lst.append(v_trial)
                
                #paralell evaluation
                if self.ncores > 1:
    
                    with joblib.Parallel(n_jobs=self.ncores) as parallel:
                        score_trial_lst=parallel(joblib.delayed(self.fit_worker)(item) for item in v_trial_lst)
                        score_target_lst=parallel(joblib.delayed(self.fit_worker)(item) for item in x_t_lst)
                        
                else:
                    score_trial_lst=[]
                    score_target_lst=[]
                    for item in v_trial_lst:
                        score_trial_lst.append(self.fit_worker(item))  
                    for item in x_t_lst:
                        score_target_lst.append(self.fit_worker(item))  
                #-----------------------------
                #Selection
                #-----------------------------
                #print('trial=', score_trial_lst)
                #print('target=', score_target_lst)
                index=0
                for (score_trial, score_target, v_trial) in zip(score_trial_lst, score_target_lst, v_trial_lst):
                    if score_trial > score_target:
                        population[index] = v_trial
                        gen_scores.append(score_trial)
                    else:
                        gen_scores.append(score_target)
                    
                    index+=1
                
                
                #-----------------------------
                #Fitness saving 
                #-----------------------------
                gen_avg = sum(gen_scores) / self.npop                   # current generation avg. fitness
                y_best = max(gen_scores)                                # fitness of best individual
                x_best = population[gen_scores.index(max(gen_scores))]  # solution of best individual
                best_scores.append(y_best)
                
                #--mir
                if self.mode=='min':
                    y_best_correct=-y_best
                    gen_avg=-gen_avg
                else:
                    y_best_correct=y_best
    
                if verbose:
                    print('************************************************************')
                    print('DE step {}/{}, F={}, CR={}, Ncores={}'.format(gen*self.npop, ngen*self.npop, self.F, self.CR, self.ncores))
                    print('************************************************************')
                    print('Best fitness:', np.round(y_best_correct,6))
                    print('Best individual:', x_best)
                    print('Average fitness:', np.round(gen_avg,6))
                    print('************************************************************')
        
        if verbose:
            print('------------------------ DE Summary --------------------------')
            print('Best fitness (y) found:', y_best_correct)
            print('Best individual (x) found:', x_best)
            print('--------------------------------------------------------------')

        #--mir
        if self.mode=='min':
            best_scores=[-item for item in best_scores]
            
        return x_best, y_best_correct, best_scores