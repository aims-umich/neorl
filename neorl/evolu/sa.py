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
#Created on Sun Jun 14 13:45:54 2020
#
#@author: Majdi and Paul
#"""

import random
import math
import numpy as np
import copy
import joblib
from neorl.evolu.discrete import mutate_discrete, encode_grid_to_discrete 
from neorl.evolu.discrete import decode_discrete_to_grid, encode_grid_indv_to_discrete
from neorl.utils.seeding import set_neorl_seed
from neorl.utils.tools import get_population, check_mixed_individual

class SA:
    """
    Parallel Simulated Annealing
    
    :param mode: (str) problem type, either ``min`` for minimization problem or ``max`` for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param fit: (function) the fitness function 
    :param cooling: (str) cooling schedule, choose ``fast``, ``boltzmann``, ``cauchy``,``equilibrium``. The ``equilibrium`` mode is only valid with ``ncores > 1`` (See **Notes** below)
    :param chain_size: (int) number of individuals to evaluate in the chain every generation (e.g. like ``npop`` for other algorithms)
    :param Tmax: (int) initial/maximum temperature
    :param Tmin: (int) final/minimum temperature
    :param chi: (float or list of floats) probability of perturbing every attribute of the input ``x``, ONLY used if ``move_func=None``. 
                For ``ncores > 1``, if a scalar is provided, constant value is used across all ``ncores``. If a list of size ``ncores``
                is provided, each core/chain uses different value of ``chi`` (See **Notes** below)
    :param move_func: (function) custom self-defined function that controls how to perturb the input space during annealing (See **Notes** below)
    :param reinforce_best: (str) an option to control the starting individual of the chain at every generation. Choose ``None``, ``hard``, ``soft`` (See **Notes** below).
    :param lmbda: (float) ONLY used if ``cooling = equilibrium``, control the cooling rate, and the speed at which the algorithm converges.
    :param alpha: (float) ONLY used if ``cooling = equilibrium``, control the initial temperature of the cooling schedule.
    :param threshold: (float) ONLY used if ``cooling = equilibrium``. The threshold (in %) for the acceptance rate of solution under which the algorithm stops running. 
    :param ncores: (int) number of parallel processors (``ncores > 1`` is required for ``cooling = equilibrium``)
    :param seed: (int) random seed for sampling
    """
    def __init__ (self, mode, bounds, fit, cooling='fast', chain_size=10,  
                  Tmax=10000, Tmin=1, chi=0.1, move_func=None, reinforce_best='soft', 
                  lmbda = 1.5, alpha = 1.5, threshold = 10 ,ncores=1, seed=None):  

        set_neorl_seed(seed)
        
        self.seed=seed
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
            
        self.bounds=bounds
        self.reinforce_best=reinforce_best
        self.ncores=ncores
        self.npop=chain_size
        self.threshold = threshold
        #assert npop % self.ncores == 0, 'The number of population (npop) to run must be divisible by ncores, {} mod {} != 0'.format(npop,self.ncores)
        self.Tmax=Tmax
        self.Tmin=Tmin
        if isinstance(chi, list):
            assert len(chi) == ncores, 'The list of chi values ({}) MUST equal ncores ({})'.format(len(chi),self.ncores) 
            self.chi=chi
        elif type(chi) == float or type(chi) == int:
            self.chi=[chi]*ncores
        else:
            raise Exception ('for chi, either list of floats or scalar float are allowed')
        
        
        assert cooling in ['fast', 'boltzmann', 'cauchy', 'equilibrium'], '--error: invalid cooling is provided'
        assert reinforce_best in ['soft', 'hard', None], '--error: invalid `reinforce_best` option is provided'
        
        self.cooling=cooling 
        self.equilib_deactivate=False
        if self.cooling == 'equilibrium' and self.ncores == 1:#Paul. The equilibrium cooling is only available with multiple chains working at the same time.
            print("-- warning: equilibrium cooling is implemented ONLY for ncores > 1. The cooling is changed to default cooling --> 'fast'")
            self.cooling = 'fast'
            self.equilib_deactivate=True
        if not self.cooling=='equilibrium':# Paul
            self.T=Tmax #initialize T
        else:
            self.lmbda = lmbda
            self.alpha = alpha
        self.move_func=move_func
        if self.move_func is None:
            self.move=self.def_move
        else:
            self.move=move_func
            
            
        #infer variable types 
        self.var_type = np.array([bounds[item][0] for item in bounds])
        
        #mir-grid
        if "grid" in self.var_type:
            self.grid_flag=True
            self.orig_bounds=bounds  #keep original bounds for decoding
            #print('--debug: grid parameter type is found in the space')
            self.bounds, self.bounds_map=encode_grid_to_discrete(self.bounds) #encoding grid to int
            #define var_types again by converting grid to int
            self.var_type = np.array([self.bounds[item][0] for item in self.bounds])
        else:
            self.grid_flag=False
            self.bounds = bounds
            self.orig_bounds=bounds
        
        self.dim = len(bounds)
        self.lb=[self.bounds[item][1] for item in self.bounds]
        self.ub=[self.bounds[item][2] for item in self.bounds]
            
    def GenInd(self, bounds):
        #"""
        #Individual generator
        #Input: 
        #    -bounds (dict): input paramter type and lower/upper bounds in dictionary form
        #Returns: 
        #    -individual (list): individual position
        #"""
        content=[]
        for key in bounds:
            if bounds[key][0] == 'int':
                content.append(random.randint(bounds[key][1], bounds[key][2]))
            elif bounds[key][0] == 'float':
                content.append(random.uniform(bounds[key][1], bounds[key][2]))
            elif bounds[key][0] == 'grid':
                content.append(random.sample(bounds[key][1],1)[0])
            else:
                raise Exception ('unknown data type is given, either int, float, or grid are allowed for parameter bounds')   
        ind=list(content)
        return ind

    def sampler(self,bound):
        #"""
        #This function takes input as [type, low, high] and returns a sample  
        #This is to sample the input parameters during optimisation
        #"""
        if bound[0] == 'int':
            sample=random.randint(bound[1], bound[2])
        elif bound[0] == 'float':
           sample=random.uniform(bound[1], bound[2])
        elif bound[0] == 'grid':
            sample=random.sample(bound[1],1)[0]
        else:
            raise Exception ('unknown data type is given, either int, float, or grid are allowed for parameter bounds')
                
        return sample

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
        #This worker is for parallel calculations of the GWO
        
        # Clip the wolf with position outside the lower/upper bounds and return same position
        x=self.ensure_bounds(x)
        
        if self.grid_flag:
            #decode the individual back to the int/float/grid mixed space
            x=decode_discrete_to_grid(x,self.orig_bounds,self.bounds_map)
        
        # Calculate objective function for each search agent
        fitness = self.fit(x)
        
        return fitness
    
    def def_move(self, x, chi):
        #"""
        #This function is to perturb x attributes by probability chi
        #Inputs:
        #    x: input vector 
        #    chi: perturbation probablity between 0 and 1
        #Returns: perturbed vector x
        #"""
        i=0
        x_new=x.copy()
        for item in self.bounds:
            if random.random() < chi:
                sample = self.sampler(self.bounds[item])
                while x_new[i] == sample and (self.bounds[item][1] != self.bounds[item][2]): 
                    sample = self.sampler(self.bounds[item])
                x_new[i] = sample
            i+=1
        return x_new
        
    def temp(self,step):
        #"""
        # Function to anneal temperature
        #"""

        if self.cooling=='fast':
            Tfac = -np.log(float(self.Tmax) / self.Tmin)
            T = self.Tmax * np.exp( Tfac * step / self.steps)
        elif self.cooling=='boltzmann':
            T = self.Tmax / np.log(step + 1)
        elif self.cooling=='cauchy':
            T = self.Tmax / (step + 1)
        # Paul
        elif self.cooling=='equilibrium':
            accept_rate = np.mean(self.accepts)
            sigma = np.std(self.accepted_energy)
            g_rho = 4 * accept_rate * (1 - accept_rate)**2 / ((2 - accept_rate)**2)
            T = np.max([1 / (1 / self.T + self.lmbda * (1 / sigma) * (self.T / sigma)**2 * g_rho),self.Tmin])
        else:
            raise Exception ('--error: unknown cooling mode is entered, fast, boltzmann, cauchy, or equilibrium are ONLY allowed')
        
        return T
    
    def chain_object(self,inp):
        #"""
        #This function is a multiprocessing object, used to be passed to Pool, that respresents 
        #an individual SA chain. 
        #Input:
        #    inp: a list contains the following inputs in order
        #    inp[0] --> x0: initial guess to chain 
        #    inp[1] --> E0: initial energy of x0
        #    inp[2] --> min_step: min step to start this chain 
        #    inp[3] --> max_step: max step to terminate this chain 
        #    inp[4] --> core_seed: seed for this chain
        #returns: 
        #    x_best, E_best: best obtained from this chain
        #    T: last temperature for this chain
        #    accepts, rejects, improves for this chain
        #"""
        x_prev=copy.deepcopy(inp[0])
        x_best=copy.deepcopy(x_prev)
        E_prev=inp[1]
        E_best=inp[1]
        min_step=inp[2]
        max_step=inp[3]
        core_seed=inp[4]
        
        if not (self.seed is None):
            random.seed(self.seed + core_seed)
            np.random.seed(self.seed + core_seed)
        
        rejects=0; accepts=0; improves=0
        k=min_step
        
        if self.cooling == 'equilibrium':#Paul.
            T = copy.deepcopy(self.T)#Paul
            accepted_energy = []#Paul
        while k <= max_step:
            if self.cooling != 'equilibrium': #Paul. if updated here may not return nan in first few steps
                T=self.temp(step=k)
                
            if self.move_func is None:
                x=copy.deepcopy(self.move(x=x_prev,chi=self.chi[core_seed-1]))
            else:
                x=copy.deepcopy(self.move(x=x_prev))
            
            x=self.ensure_bounds(x)
            #SA is programmed to maximize reward
            E=self.fit_worker(x)
            dE = E - E_prev        
            #-----------------------------------
            # Improve/Accept/Reject
            #-----------------------------------
            if dE > 0: #improvement
                improves += 1
                accepts += 1
                x_prev = copy.deepcopy(x)
                E_prev = E
                if E > E_best:
                    x_best = copy.deepcopy(x)
                    E_best = E     
                if self.cooling == 'equilibrium':# Paul
                    accepted_energy.append(E)
            elif np.exp(dE/T) >= random.random(): #accept the state
                accepts += 1
                x_prev = copy.deepcopy(x)
                E_prev = E
                if self.cooling == 'equilibrium':# Paul
                    accepted_energy.append(E)
            else:
                # Reject the new solution (maintain the current state!)
                rejects += 1
            k+=1
        if self.cooling == 'equilibrium':#Paul.
            return x_prev, E_prev, T, accepts, rejects, improves, x_best, E_best,accepted_energy
        else:
            return x_prev, E_prev, T, accepts, rejects, improves, x_best, E_best
        
    def chain(self, x0, E0, step0):
        #"""
        #This function creates ``ncores`` independent SA chains with same initial guess x0, E0 and 
        #runs them via multiprocessing Pool.
        #Input:
        #    x0: initial input guess (comes from previous annealing chains or from replay memory)
        #    E0: energy/fitness value of x0
        #    step0: is the first time step to use for temperature annealing
        #    npop: total number of individuals to be evaluated in this annealing stage
        #returns: 
        #    x_best, E_best, and T obtained from this annealing stage from all chains
        #"""
        
        #Append and prepare the input list to be passed as input to Pool.map
        core_list=[]
        core_step_min=step0
        for j in range(1,self.ncores+1):
            core_step_max=step0+j*self.npop-1
            core_list.append([x0[j-1], E0[j-1], core_step_min, core_step_max, j])
            core_step_min=core_step_max+1
            
        
        if self.ncores > 1:

            with joblib.Parallel(n_jobs=self.ncores) as parallel:
                results=parallel(joblib.delayed(self.chain_object)(item) for item in core_list)
        else:
            results=[]
            results.append(list(self.chain_object(core_list[0])))
                
        # Determine the index and the best solution from all chains
        #best_index=[y[0] for y in results].index(min([y[0] for y in results]))
        self.x_last=[item[0] for item in results]
        self.E_last=[item[1] for item in results]
        self.T=results[-1][2] # get the temperature of the last chain
        self.accepts=[np.round(item[3]/self.npop*100,1) for item in results] #convert to rate
        self.rejects=[np.round(item[4]/self.npop*100,1) for item in results] #convert to rate
        self.improves=[np.round(item[5]/self.npop*100,1) for item in results] #convert to rate
        
        if self.cooling == 'equilibrium':#Paul
            self.accepted_energy = [item[8] for item in results]#Paul
            self.accepted_energy = [elem for energy in self.accepted_energy for elem in energy]
            if np.std(self.accepted_energy) == 0: # prevent the std to be 0
                pass
            else:
                self.T=self.temp(step=None)
            
        self.x_best, self.E_best=[item[6] for item in results], [item[7] for item in results]
        
        return self.x_last, self.E_last, self.T, self.accepts, self.rejects, self.improves, self.x_best, self.E_best
    
    def InitChains(self, x0=None):
        
        #initialize the chain and run them in parallel (these samples will be used to initialize the annealing process)
        #Establish the chain
        if x0:
            if self.ncores==1:  
                if not any(isinstance(el, list) for el in x0):  #ensure a list of list is submitted for ncores=1
                    x0=[x0]
            assert len(x0) == self.ncores, '--error: the length of x0 ({}) MUST equal the number of ncores/parallel chains ({})'.format(len(x0), self.ncores)

            for i in range(len(x0)):
                check_mixed_individual(x=x0[i], bounds=self.orig_bounds) #assert the type provided is consistent
                if self.grid_flag:
                    x0[i] = encode_grid_indv_to_discrete(x0[i], bounds=self.orig_bounds, bounds_map=self.bounds_map)       
        
        else:
            x0=[]
            for i in range (self.ncores):
                x0.append(self.GenInd(self.bounds))
        
        #Evaluate the swarm
        if self.ncores > 1:  #evaluate chain in parallel
            core_list=[]
            for ind in x0:
                core_list.append(ind)
           
            with joblib.Parallel(n_jobs=self.ncores) as parallel:
                E0=parallel(joblib.delayed(self.fit_worker)(item) for item in core_list)
            if self.cooling == 'equilibrium': # Paul
                self.T = np.max([self.alpha * np.std(E0),self.Tmin,1e-10])

        else: #evaluate swarm in series
            E0=[]
            for ind in x0:
                fitness=self.fit_worker(ind)
                E0.append(fitness)
        

        return x0, E0 #return initial guess and initial fitness      
    
    def evolute(self, ngen, x0=None, verbose=False):
        """
        This function evolutes the SA algorithm for number of generations.
        
        :param ngen: (int) number of generations to evolute
        :param x0: (list of lists) initial samples to start the evolution (``len(x0)`` must be equal to ``ncores``)
        :param verbose: (int) print statistics to screen
        
        :return: (tuple) (best individual, best fitness, and dictionary containing major search results)
        """
        #chain statistics
        #self.accept=0
        #self.reject=0
        #self.improve=0
        
        stat={'x':[], 'fitness':[], 'T':[], 'accept':[], 'reject':[], 'improve':[]}
        E_opt=-np.inf
        
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
            xinit, Einit=self.InitChains(x0=x0)
        else:
            xinit, Einit=self.InitChains()
            
        x_next=copy.deepcopy(xinit)
        E_next=copy.deepcopy(Einit)
        
        ngen=int(ngen/self.ncores)
        for i in range (ngen):
            #if self.cooling == 'equilibrium':# Paul
            #    self.accepted_energy = [] # initialize list of accepted energy to empty at step i
            x_next,E_next,self.T, acc, rej, imp, x_best, E_best=self.chain(x0=x_next, E0=E_next, step0=step0)
            step0=step0+self.npop*self.ncores
            arg_max=np.argmax(E_best)
            stat['x'].append(x_best[arg_max])
            if self.mode=='max':
                stat['fitness'].append(max(E_best))
            else:
                stat['fitness'].append(-max(E_best))
            stat['T'].append(self.T)
            stat['accept'].append(acc[arg_max])
            stat['reject'].append(rej[arg_max])
            stat['improve'].append(imp[arg_max])
            
            if max(E_best) > E_opt:
                E_opt=max(E_best)
                x_opt=copy.deepcopy(x_best[arg_max])
            
            if self.cooling == 'equilibrium': # Paul
                if np.mean(self.accepts) <= self.threshold: # help prevent the std to be 0
                    print("--warning: The SA stopped because the average acceptance rate throughout the chain {} % falls below the threshold {} %".format(np.mean(self.accepts),self.threshold))
                    break
                elif np.mean(self.accepts) == 0:
                    print("--warning: The SA stopped because the average acceptance rate throughout the chain {} % reaches 0%".format(np.mean(self.accepts),self.threshold))
                    break
            # Paul
            if self.reinforce_best == 'hard':
                x_next=[x_opt]*self.ncores
                E_next=[E_opt]*self.ncores
            elif self.reinforce_best == 'soft':
                sampling = np.zeros(self.ncores)
                normalization = 1 / np.sum(np.exp(- np.abs(E_next) / self.T)) # cte to generate a probability
                sampling[0] = np.exp(- abs(E_next[0]) / self.T) # utilize last accepted solution to generate the sampling
                if self.ncores != 1:
                    rho = np.random.uniform(0,1,size = self.ncores) # sample random number between 0 and 1
                else:
                    rho = [1] # probability of choosing itself is 1    
                for i in range(1,self.ncores):
                    sampling[i] = sampling[i - 1] + np.exp(- abs(E_next[i]) / self.T)
                sampling = sampling * normalization
                for count,prob in enumerate(rho):
                    if prob == 1:
                        index = self.ncores - 1
                    elif math.isnan(sampling[count]):# exponantial can lead to numerical erros
                        index = np.argmax(E_next)
                    else:
                        if prob <= sampling[0]:
                            index = 0
                        else:
                            index = np.where(sampling > rho)[0][0]
                    x_next[count] = x_best[index] # re-initialize with the best ever found by the index'th Markov Chain
                    E_next[count] = E_best[index]

            #mir-grid
            if self.grid_flag:
                x_opt_correct = decode_discrete_to_grid(x_opt, self.orig_bounds, self.bounds_map)
            else:
                x_opt_correct = x_opt
                    
            if verbose:
                print('***********************************************************************')
                print('SA step {}/{}, T={}, Ncores={}, Cooling={}, Reinforce={}'.format(step0-1,self.steps,np.round(self.T), self.ncores, self.cooling, self.reinforce_best))
                print('***********************************************************************')
                print('Statistics for the {} parallel chains'.format(self.ncores))
                if self.mode=='max': 
                    print('Best fitness:', np.round(max(E_best),6))
                else: 
                    print('Best fitness:', -np.round(max(E_best),6))
                print('Best individual:', x_opt_correct)
                print('Acceptance Rate (%):', acc)
                print('Rejection Rate (%):', rej)
                print('Improvment Rate (%):', imp)
                print('***********************************************************************')

            
        #--mir
        if self.mode=='max':
            self.E_opt_correct=E_opt
        else:
            self.E_opt_correct=-E_opt
                
        if verbose:
            print('------------------------ SA Summary --------------------------')
            print('Best fitness (y) found:', self.E_opt_correct)
            print('Best individual (x) found:', x_opt_correct)
            print('--------------------------------------------------------------')

        if self.equilib_deactivate: #Paul.
            print("-- warning: equilibrium cooling is implemented ONLY for ncores > 1. The cooling is changed to default cooling --> 'fast'")
            
        return x_opt_correct, self.E_opt_correct, stat
