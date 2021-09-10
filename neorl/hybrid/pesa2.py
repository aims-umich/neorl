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
#Created on Sun Jun 28 18:21:05 2020
#
#@author: Majdi Radaideh
#"""

from neorl.hybrid.pesacore.er import ExperienceReplay
from neorl.hybrid.pesacore.gwo import GWOmod
from neorl.hybrid.pesacore.de import DEmod
from neorl.hybrid.pesacore.woa import WOAmod
from neorl.hybrid.pesacore.es import ESMod
from copy import deepcopy
from multiprocessing import Process, Queue
import random
import numpy as np
from collections import defaultdict
import time
import sys
import uuid
from neorl.evolu.discrete import mutate_discrete, encode_grid_to_discrete, decode_discrete_to_grid


#multiprocessing trick to paralllelize nested functions in python (un-picklable objects!)
def globalize(func):
  def result(*args, **kwargs):
    return -func(*args, **kwargs)
  result.__name__ = result.__qualname__ = uuid.uuid4().hex
  setattr(sys.modules[result.__module__], result.__name__, result)
  return result


class PESA2(ExperienceReplay):

    """
    Prioritized replay for Evolutionary Swarm Algorithms: PESA 2 (Modern Version) 
    A hybrid algorithm of GWO, DE, and WOA
    
    *PESA2 Major Parameters*
    
    :param mode: (str) problem type, either "min" for minimization problem or "max" for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param fit: (function) the fitness function 
    :param R_frac: (int) fraction of ``npop``, ``nwolves``, ``nwhales`` to survive to the next generation.
                     Also, ``R_frac`` equals to the number of individuals to replay from the memory
    :param memory_size: (int) max size of the replay memory (if None, ``memory_size`` is built to accommodate all samples during search) 
    :param alpha_init: (float) initial value of the prioritized replay coefficient (See **Notes** below)
    :param alpha_end: (float) final value of the prioritized replay coefficient (See **Notes** below)
    
    *PESA2 Auxiliary Parameters (for the internal algorithms)*

    :param npop: (int) for **DE**, total number of individuals in DE population
    :param CR: (float) for **DE**, crossover probability between [0,1]
    :param F: (float) for **DE**, differential/mutation weight between [0,2]
    :param nwolves: (float) for **GWO**, number of wolves for GWO
    :param nwhales: (float) for **WOA**, number of whales in the population of WOA
    
    *PESA2 Misc. Parameters*
    
    :param int_transform: (str): method of handling int/discrete variables, choose from: ``nearest_int``, ``sigmoid``, ``minmax``.    
    :param ncores: (int) number of parallel processors
    :param seed: (int) random seed for sampling
    """
    
    def __init__ (self, mode, bounds, fit, R_frac=0.5, #general parameters
                  memory_size=None, alpha_init=0.1, alpha_end=1, #replay parameters
                  nwolves=5, #GOW parameters
                  npop=50, CR=0.7, F=0.5,  #DE parameters
                  nwhales=10, #WOA parameters
                  int_transform ='nearest_int', ncores=1, seed=None): #misc parameters
        
        if seed:
            random.seed(seed)
            np.random.seed(seed)

        #--mir
        self.mode=mode
        if mode == 'max':
            self.FIT=fit
        elif mode == 'min':
            self.FIT = globalize(lambda x: fit(x))  #use the function globalize to serialize the nested fit
        else:
            raise ValueError('--error: The mode entered by user is invalid, use either `min` or `max`')

        self.int_transform=int_transform
        if int_transform not in ["nearest_int", "sigmoid", "minmax"]:
            raise ValueError('--error: int_transform entered by user is invalid, must be `nearest_int`, `sigmoid`, or `minmax`')
            
        if ncores > 1:
            if nwolves >= npop:
                nwolves=npop
            else:
                assert npop % nwolves==0, '--error: since ncores > 1, for max efficiency of PESA2, choose npop ({}) and nwolves ({}) that are either equal or divisible, e.g. npop=60, nwolves=5'.format(npop, nwolves)
           
            if nwhales >= npop:
                nwhales=npop
            else:
                assert npop % nwhales==0, '--error: since ncores > 1, for max efficiency of PESA2, choose npop ({}) and nwhales ({}) that are either equal or divisible, e.g. npop=60, nwhales=5'.format(npop, nwhales)

        
        self.GWO_gen=int(npop/nwolves)
        self.WOA_gen=int(npop/nwhales)
        if self.GWO_gen < 1:
            self.GWO_gen=1
        if self.WOA_gen < 1:
            self.WOA_gen=1
            
        self.bounds=bounds
        self.NPOP=npop
        self.ncores=ncores

        if ncores <= 3:
            self.NCORES=1
            # option for first-level parallelism
            self.PROC=False
        else:
            # option for first-level parallelism
            self.PROC=True
            self.NCORES=int(ncores/3)
            
        self.SEED=seed
        #--------------------
        #Experience Replay
        #--------------------
        self.MODE='prior'
        self.ALPHA0=alpha_init
        self.ALPHA1=alpha_end
        #--------------------
        # GWO hyperparameters
        #--------------------
        self.NWOLVES=nwolves
        
        #--------------------
        # WOA HyperParameters
        #--------------------
        self.NWHALES=nwhales

        #--------------------
        # DE HyperParameters
        #--------------------
        self.F=F   
        self.CR=CR  
        #-------------------------------
        #Memory Supply for each method
        #-------------------------------        
        assert 0 <= R_frac <= 1, '--error: The value of R_frac ({}) MUST be between 0 and 1'.format(R_frac)
        self.MU_DE=int(R_frac*self.NPOP)
        self.MU_WOA=int(R_frac*self.NWHALES)
        self.MU_GWO=int(R_frac*self.NWOLVES)
        #--------------------
        # Fixed/Derived parameters 
        #--------------------
        self.nx=len(self.bounds)  #all
        self.memory_size=memory_size
        
        #infer variable types 
        self.datatype = np.array([bounds[item][0] for item in bounds])
        
        #mir-grid
        if "grid" in self.datatype:
            self.grid_flag=True
            self.orig_bounds=bounds  #keep original bounds for decoding
            print('--debug: grid parameter type is found in the space')
            self.bounds, self.bounds_map=encode_grid_to_discrete(self.bounds) #encoding grid to int
            #define var_types again by converting grid to int
            self.datatype = np.array([self.bounds[item][0] for item in self.bounds])
        else:
            self.grid_flag=False
            self.bounds = bounds
        
        self.lb = np.array([self.bounds[item][1] for item in self.bounds])
        self.ub = np.array([self.bounds[item][2] for item in self.bounds])

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
        #"""
        #Evaluates fitness of an individual.
        #"""
        
        x=self.ensure_bounds(x)
        
        #mir-grid
        if self.grid_flag:
            #decode the individual back to the int/float/grid mixed space
            x=decode_discrete_to_grid(x,self.orig_bounds,self.bounds_map) 
                    
        fitness = self.FIT(x)
        return fitness
        
    def evolute(self, ngen, x0=None, replay_every=1, warmup=100, verbose=True):
        """
        This function evolutes the PESA2 algorithm for number of generations.
        
        :param ngen: (int) number of generations to evolute
        :param x0: (list of lists) initial samples to start the replay memory (``len(x0)`` must be equal or more than ``npop``)
        :param replay_every: (int) perform memory replay every number of generations, default: replay after every generation     
        :param warmup: (int) number of random warmup samples to initialize the replay memory and must be equal or more than ``npop`` (only used if ``x0=None``)
        :param verbose: (int) print statistics to screen, 0: no print, 1: PESA print, 2: detailed print
        
        :return: (dict) dictionary containing major PESA search results
        """
        
        
        assert ngen >= 2, '--error: use ngen > 2 for PESA2'
        
        self.verbose=verbose
        self.NGEN=int(ngen/replay_every)
        self.STEPS=self.NGEN*self.NPOP #all 
        if self.memory_size:
            self.MEMORY_SIZE=self.memory_size
        else:
            self.MEMORY_SIZE=self.STEPS*3+1 #PESA
            
        #-------------------------------------------------------
        # Check if initial pop is provided as initial guess 
        #-------------------------------------------------------
        if x0: 
            # use provided initial guess
            warm=ESMod(bounds=self.bounds, fit=self.fit_worker, mu=self.NPOP, lambda_=self.LAMBDA, ncores=self.ncores)
            x0size=len(x0)
            assert x0size >= self.NPOP, 'the number of lists in x0 ({}) must be more than or equal npop ({})'.format(x0size, self.NPOP)
            self.pop0=warm.init_pop(warmup=x0size, x_known=x0)  #initial population for all methods (use external ES modules for initialization)
        else:
            #create initial guess 
            assert warmup >= self.NPOP, 'the number of warmup samples ({}) must be more than or equal npop ({})'.format(warmup, self.NPOP)
            warm=ESMod(bounds=self.bounds, fit=self.fit_worker, mu=self.NPOP, lambda_=self.NPOP, ncores=self.ncores)
            self.pop0=warm.init_pop(warmup=warmup)  #initial population for all methods (use external ES modules for initialization)
        
        self.fit_hist=[]
        #------------------------------
        # Step 1: Initialize the memory
        #------------------------------
        self.mymemory=ExperienceReplay(size=self.MEMORY_SIZE) #memory object
        xvec0, obj0=[self.pop0[item][0] for item in self.pop0], [self.pop0[item][2] for item in self.pop0]  #parse the initial samples
        self.mymemory.add(xvec=xvec0, obj=obj0, method=['na']*len(xvec0)) # add initial samples to the replay memory
        
        #--------------------------------
        # Step 2: Initialize all methods
        #--------------------------------
        # Obtain initial population for all methods
        x0_gwo, fit0_gwo, x0_de, fit0_de, x0_woa, fit0_woa=self.init_guess(pop0=self.pop0)
        # Initialize GWO class
        gwo=GWOmod(mode='max', bounds=self.bounds, fit=self.fit_worker, int_transform=self.int_transform,
                   nwolves=self.NWOLVES, ncores=self.NCORES, seed=self.SEED)
        # Initialize DE class
        de=DEmod(bounds=self.bounds, fit=self.fit_worker, npop=self.NPOP, 
                 F=self.F, int_transform=self.int_transform,
                 CR=self.CR, ncores=self.NCORES, seed=self.SEED)
        # Initialize WOA class
        woa=WOAmod(mode='max', bounds=self.bounds, fit=self.fit_worker, int_transform=self.int_transform,
                   nwhales=self.NWHALES, ncores=self.NCORES, seed=self.SEED)
            
        #--------------------------------
        # Step 3: Initialize PESA engine
        #--------------------------------
        #Use initial samples as first guess
        self.gwo_next=deepcopy(x0_gwo)
        self.de_next=deepcopy(x0_de)
        self.woa_next=deepcopy(x0_woa) 
        self.STEP0=1  #step counter
        self.ALPHA=self.ALPHA0  #set alpha to alpha0
        
        #--------------------------------
        # Step 4: PESA evolution
        #--------------------------------
        for gen in range(1,self.NGEN+1):
            #-------------------------------------------------------------------------------------------------------------------
            # Step 5: evolute all methods for 1 generation 
            #-------------------------------------------------------------------------------------------------------------------
            #**********************************
            #--Step 5A: Complete PARALEL calcs 
            # via multiprocess.Process
            #*********************************
            
            if self.PROC:
                
                QGWO = Queue(); QDE=Queue(); QWOA=Queue()
                def gwo_worker():
                    xgwo_best, ygwo_best, gwo_new= gwo.evolute(ngen=self.GWO_gen*replay_every, x0=self.gwo_next, verbose=0)
                                                                                                 
                    QGWO.put((xgwo_best, ygwo_best, gwo_new))
                def de_worker():
                    random.seed(self.SEED)
                    xde_best, yde_best, de_new=de.evolute(ngen=1*replay_every,x0=self.de_next, verbose=0)
                    QDE.put((xde_best, yde_best, de_new))
                
                def woa_worker():
                    random.seed(self.SEED)
                    xwoa_best, ywoa_best, woa_new=woa.evolute(ngen=self.WOA_gen*replay_every,x0=self.woa_next, verbose=0)
                    QWOA.put((xwoa_best, ywoa_best, woa_new))
                    
                Process(target=gwo_worker).start()
                Process(target=de_worker).start()
                Process(target=woa_worker).start()
                
                #get the values from the Queue
                self.gwo_best, self.ygwo_best, self.gwo_next=QGWO.get()
                self.de_best, self.yde_best, self.de_next=QDE.get()
                self.woa_best, self.ywoa_best, self.woa_next=QWOA.get()
                
            #*********************************
            #--Step 5B: Complete Serial calcs
            #*********************************
            else:  
                self.gwo_best, self.ygwo_best, self.gwo_next= gwo.evolute(ngen=self.GWO_gen*replay_every, x0=self.gwo_next, verbose=0)
                self.de_best, self.yde_best, self.de_next=de.evolute(ngen=1*replay_every,x0=self.de_next, verbose=0)
                self.woa_best, self.ywoa_best, self.woa_next=woa.evolute(ngen=self.WOA_gen*replay_every, x0=self.woa_next, verbose=0)
            
            #*********************************************************
            # Step 5C: Obtain relevant statistics for this generation 
            #*********************************************************
            self.gwo_next=self.select(pop=self.gwo_next, k=self.MU_GWO)
            self.de_next=self.select(pop=self.de_next, k=self.MU_DE)
            self.woa_next=self.select(pop=self.woa_next, k=self.MU_WOA)
            
            self.STEP0=self.STEP0+self.NPOP  #update step counter
            if self.verbose==2:
                self.printout(mode=1, gen=gen)
            #-------------------------------------------------------------------------------------------------------------------
            #-------------------------------------------------------------------------------------------------------------------
            
            #-----------------------------
            # Step 6: Update the memory
            #-----------------------------
            self.memory_update()
            
            #-----------------------------------------------------------------
            # Step 7: Sample from the memory and prepare for next Generation 
            #-----------------------------------------------------------------
            self.resample()
            
            #--------------------------------------------------------
            # Step 8: Anneal Alpha if priortized replay is used
            #--------------------------------------------------------
            if self.MODE=='prior': #anneal alpha between alpha0 (lower) and alpha1 (upper) 
                self.ALPHA=self.linear_anneal(step=self.STEP0, total_steps=self.STEPS, a0=self.ALPHA0, a1=self.ALPHA1)
            
            #--------------------------------------------------------
            # Step 9: Calculate the memory best and print PESA summary 
            #--------------------------------------------------------
            self.pesa_best=self.mymemory.sample(batch_size=1,mode='greedy')[0]  #`greedy` will sample the best in memory
            self.fit_hist.append(self.pesa_best[1])
            self.memory_size=len(self.mymemory.storage) #memory size so far
            
            
            #--mir
            if self.mode=='min':
                self.fitness_best=-self.pesa_best[1]
            else:
                self.fitness_best=self.pesa_best[1]

            #mir-grid
            if self.grid_flag:
                self.xbest_correct=decode_discrete_to_grid(self.pesa_best[0],self.orig_bounds,self.bounds_map)
            else:
                self.xbest_correct=self.pesa_best[0]

            if self.verbose:  #print summary data to screen
                self.printout(mode=2, gen=gen)
            
        #--mir
        if self.mode=='min':
            self.fit_hist=[-item for item in self.fit_hist]

        if self.verbose:
            print('------------------------ PESA2 Summary --------------------------')
            print('Best fitness (y) found:', self.fitness_best)
            print('Best individual (x) found:', self.xbest_correct)
            print('--------------------------------------------------------------') 
        
        return self.xbest_correct, self.fitness_best, self.fit_hist

    def linear_anneal(self, step, total_steps, a0, a1):
        #"""
        #Anneal parameter between a0 and a1 
        #:param step: current time step
        #:param total_steps: total numbe of time steps
        #:param a0: lower bound of alpha/parameter
        #:param a0: upper bound of alpha/parameter
        #:return
        #  - annealed value of alpha/parameter
        #"""
        fraction = min(float(step) / total_steps, 1.0)
        return a0 + fraction * (a1 - a0)

    def select(self, pop, k=1):
        #"""
        #Select function sorts the population from max to min based on fitness and select k best
        #Inputs:
        #    pop (dict): population in dictionary structure
        #    k (int): top k individuals are selected
        #Returns:
        #    best_dict (dict): the new ordered dictionary with top k selected 
        #"""
        
        pop=list(pop.items())
        pop.sort(key=lambda e: e[1][1], reverse=True)
        sorted_dict=dict(pop[:k])
        
        #This block creates a new dict where keys are reset to 0 ... k in order to avoid unordered keys after sort
        best_dict=defaultdict(list)
        index=0
        for key in sorted_dict:
            best_dict[index].append(sorted_dict[key][0])
            best_dict[index].append(sorted_dict[key][1])
            index+=1
        
        sorted_dict.clear()
        return best_dict
    
    def memory_update(self):
        #"""
        #This function updates the replay memory with the samples of GWO, DE, and WOA (if used)
        #then remove the duplicates from the memory
        #"""
        gwo_x, gwo_y=[self.gwo_next[item][0] for item in self.gwo_next], [self.gwo_next[item][1] for item in self.gwo_next]
        de_x, de_y=[self.de_next[item][0] for item in self.de_next], [self.de_next[item][1] for item in self.de_next]
        woa_x, woa_y=[self.woa_next[item][0] for item in self.woa_next], [self.woa_next[item][1] for item in self.woa_next]
        
        self.mymemory.add(xvec=tuple(gwo_x), obj=gwo_y, method=['gwo']*len(gwo_x))
        self.mymemory.add(xvec=tuple(de_x), obj=de_y, method=['de']*len(de_x))
        self.mymemory.add(xvec=tuple(woa_x), obj=woa_y, method=['woa']*len(woa_x))

    def resample(self):
        #"""
        #This function samples data from the memory and prepares the chains for SA
        #the population for ES, and the swarm for PSO for the next generation
        #    -SA: initial guess for the parallel chains are sampled from the memroy
        #    -ES: a total of ES_MEMORY (or MU) individuals are sampled from the memory and appended to ES population 
        #    -PSO: a total of PSO_MEMORY (or MU) particles are sampled from the memory and appended to PSO swarm 
        #For SA: x_next and E_next particpate in next generation
        #For PSO: swm_next, local_pso_next, and local_fit_next particpate in next generation
        #For ES: pop_next particpates in next generation
        #"""
        #update the dictionary with new samples for GWO
        gwo_replay=self.mymemory.sample(batch_size=self.NWOLVES-self.MU_GWO,mode=self.MODE,alpha=self.ALPHA)
        index=self.MU_GWO
        for sample in range(self.NWOLVES-self.MU_GWO):
            self.gwo_next[index].append(gwo_replay[sample][0])
            self.gwo_next[index].append(gwo_replay[sample][1])
            index+=1
        
        #update the dictionary with new samples for DE
        de_replay=self.mymemory.sample(batch_size=self.NPOP-self.MU_DE,mode=self.MODE,alpha=self.ALPHA)
        index=self.MU_DE
        for sample in range(self.NPOP-self.MU_DE):
            self.de_next[index].append(de_replay[sample][0])
            self.de_next[index].append(de_replay[sample][1])
            index+=1
        
        #update the dictionary with new samples for WOA
        woa_replay=self.mymemory.sample(batch_size=self.NWHALES-self.MU_WOA,mode=self.MODE,alpha=self.ALPHA)
        index=self.MU_WOA
        for sample in range(self.NWHALES-self.MU_WOA):
            self.woa_next[index].append(woa_replay[sample][0])
            self.woa_next[index].append(woa_replay[sample][1])
            index+=1
                 
        #get *_next back to a list of lists for the next loop
        self.gwo_next=[self.gwo_next[item][0] for item in self.gwo_next]    
        self.de_next=[self.de_next[item][0] for item in self.de_next]
        self.woa_next=[self.woa_next[item][0] for item in self.woa_next]        

    def init_guess(self, pop0):
        #"""
        #This function takes initial guess pop0 and returns initial guesses for GWO, DE, and WOA 
        #to start PESA evolution
        #"""
        pop0=list(pop0.items())
        pop0.sort(key=lambda e: e[1][2], reverse=True)
        sorted_gwo=dict(pop0[:self.NWOLVES])
        x0_gwo, fit0_gwo=[sorted_gwo[key][0] for key in sorted_gwo], [sorted_gwo[key][2] for key in sorted_gwo] # initial guess for GWO
        
        sorted_de=dict(pop0[:self.NPOP])
        x0_de, fit0_de=[sorted_de[key][0] for key in sorted_de], [sorted_de[key][2] for key in sorted_de] # initial guess for DE
        
        sorted_woa=dict(pop0[:self.NWHALES])
        x0_woa, fit0_woa=[sorted_woa[key][0] for key in sorted_woa], [sorted_woa[key][2] for key in sorted_woa] # initial guess for WOA
        
        return x0_gwo, fit0_gwo, x0_de, fit0_de, x0_woa, fit0_woa

    def printout(self, mode, gen):
        #"""
        #Print statistics to screen
        #inputs:
        #    mode (int): 1 to print for individual algorathims and 2 to print for PESA 
        #    gen (int): current generation number 
        #"""
        if mode == 1:
            print('***********************************************************************************************')
            print('#############################################################################')
            print('GWO step {}/{}, NWolves={}, Ncores={}'.format(self.STEP0-1,self.STEPS, self.NWOLVES, self.NCORES))
            print('#############################################################################')
            print('Statistics for generation {}'.format(gen))
            print('Best Wolf Fitness:', np.round(np.max(self.ygwo_best),4) if self.mode == 'max' else -np.round(np.max(self.ygwo_best),4))
            print('Best Wolf Position:', np.round(self.gwo_best,3) if not self.grid_flag else decode_discrete_to_grid(self.gwo_best,self.orig_bounds,self.bounds_map))
            print('#############################################################################')
                  
            print('*****************************************************************************')
            print('DE step {}/{}, NPOP={}, F={}, CR={}, Ncores={}'.format(self.STEP0-1,self.STEPS,self.NPOP,np.round(self.F), self.CR, self.NCORES))
            print('****************************************************************************')
            print('Statistics for generation {}'.format(gen))
            print('Best Individual Fitness:', np.round(np.max(self.yde_best),4) if self.mode == 'max' else -np.round(np.max(self.yde_best),4))
            print('Best Individual Position:', np.round(self.de_best,3) if not self.grid_flag else decode_discrete_to_grid(self.de_best,self.orig_bounds,self.bounds_map))
            print('****************************************************************************')
            
            print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
            print('WOA step {}/{}, NWhales={}, Ncores={}'.format(self.STEP0-1,self.STEPS, self.NWHALES, self.NCORES))
            print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
            print('Statistics for generation {}'.format(gen))
            print('Best Whale Fitness:', np.round(np.max(self.ywoa_best),4) if self.mode == 'max' else -np.round(np.max(self.ywoa_best),4))
            print('Best Whale Position:', np.round(self.woa_best,3) if not self.grid_flag else decode_discrete_to_grid(self.woa_best,self.orig_bounds,self.bounds_map)) 
            print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        
        if mode == 2:
            print('------------------------------------------------------------')
            print('PESA2 step {}/{}, Ncores={}'.format(self.STEP0-1,self.STEPS, self.ncores))
            print('------------------------------------------------------------')
            print('PESA statistics for generation {}'.format(gen))
            print('Best Fitness:', self.fitness_best)
            print('Best Individual:', self.xbest_correct)
            print('ALPHA:', np.round(self.ALPHA,3))
            print('Memory Size:', self.memory_size)
            print('------------------------------------------------------------')
            
            print('***********************************************************************************************')