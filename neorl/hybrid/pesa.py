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
from neorl.hybrid.pesacore.sa import SAMod
from neorl.hybrid.pesacore.es import ESMod
from neorl.hybrid.pesacore.pso import PSOMod
from copy import deepcopy
from multiprocessing import Process, Queue
import random
import numpy as np
from collections import defaultdict
import time
import sys
import uuid
from neorl.evolu.discrete import mutate_discrete, encode_grid_to_discrete 
from neorl.evolu.discrete import decode_discrete_to_grid, encode_grid_indv_to_discrete
from neorl.utils.seeding import set_neorl_seed
from neorl.utils.tools import get_population, check_mixed_individual

#multiprocessing trick to paralllelize nested functions in python (un-picklable objects!)
def globalize(func):
  def result(*args, **kwargs):
    return -func(*args, **kwargs)
  result.__name__ = result.__qualname__ = uuid.uuid4().hex
  setattr(sys.modules[result.__module__], result.__name__, result)
  return result

class PESA(ExperienceReplay):

    """
    *PESA Major Parameters*
    
    :param mode: (str) problem type, either "min" for minimization problem or "max" for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param fit: (function) the fitness function 
    :param npop: (int) total number of individuals in each group. So for ES, PSO, and SA, full population is ``npop*3``.
    :param mu: (int) number of individuals to survive to the next generation. 
                     Also, ``mu`` equals to the number of individuals to sample from the memory. If None, ``mu=int(npop/2)``.
                     So 1/2 of PESA population comes from previous generation, and 1/2 comes from the replay memory (See **Notes** below for more info)
    :param memory_size: (int) max size of the replay memory (if None, ``memory_size`` is built to accommodate all samples during search) 
    :param alpha_init: (float) initial value of the prioritized replay coefficient (See **Notes** below)
    :param alpha_end: (float) final value of the prioritized replay coefficient (See **Notes** below)
    :param alpha_backdoor: (float) backdoor greedy replay rate/probability to sample from the memory for SA instead of random-walk (See **Notes** below)
    
    *PESA Auxiliary Parameters (for the internal algorithms)*
    
    :param cxpb: (float) for **ES**, population crossover probability between [0,1]
    :param mutpb: (float) for **ES**, population mutation probability between [0,1] 
    :param c1: (float) for **PSO**, cognitive speed constant 
    :param c2: (float) for **PSO**, social speed constant
    :param speed_mech: (str) for **PSO**, type of speed mechanism for to update particle velocity, choose between ``constric``, ``timew``, ``globw``.	
    :param Tmax: (float) for **SA**, initial/max temperature to start the annealing process 
    :param chi: (float) for **SA**, probability to perturb an attribute during SA annealing (occurs when ``rand(0,1) < chi``).	
    
    *PESA Misc. Parameters*
    
    :param ncores: (int) number of parallel processors
    :param seed: (int) random seed for sampling
    """
    
    def __init__ (self, mode, bounds, fit, npop, mu=None, #general parameters
                  memory_size=None, alpha_init=0.1, alpha_end=1, alpha_backdoor=0.1, #replay parameters
                  Tmax=10000, chi=0.1, #SA parameters
                  cxpb=0.7, mutpb=0.1,  #ES parameters
                  c1=2.05, c2=2.05, speed_mech='constric', #PSO parameters
                  ncores=1, seed=None): #misc parameters
        
        #--------------------
        #General Parameters
        #--------------------
        set_neorl_seed(seed)
        
        self.bounds=bounds
        #--mir
        self.mode=mode
        if mode == 'max':
            self.FIT=fit
        elif mode == 'min':
            self.FIT = globalize(lambda x: fit(x))  #use the function globalize to serialize the nested fit
        else:
            raise ValueError('--error: The mode entered by user is invalid, use either `min` or `max`')
        
        self.ncores=ncores
        self.NPOP=npop
        self.pso_flag=True
        self.ncores=ncores
        if ncores <= 3:
            self.NCORES=1
            self.PROC=False
        else:
            self.PROC=True
            if self.pso_flag:
                self.NCORES=int(ncores/3)
            else:
                self.NCORES=int(ncores/2)
        # option for first-level parallelism       
        #self.PROC=True
        self.SEED=seed
        
        #--------------------
        #Experience Replay
        #--------------------
        self.MODE='prior';  self.ALPHA0=alpha_init;   self.ALPHA1=alpha_end
        #--------------------
        # SA hyperparameters
        #--------------------
        self.TMAX=Tmax;  self.CHI=chi;  self.REPLAY_RATE=alpha_backdoor
        
        #--------------------
        # ES HyperParameters
        #--------------------
        if mu:    
            assert mu < npop, '--error: The value of mu ({}) MUST be less than npop ({})'.format(mu, npop)
            self.MU=mu
        else:
            self.MU=int(npop/2)
        
        self.CXPB=cxpb;  self.MUTPB=mutpb;   self.INDPB=1.0
        
        #--------------------
        # PSO HyperParameters
        #--------------------
        self.C1=c1;   self.C2=c2;   self.SPEED_MECH=speed_mech
        
        #-------------------------------
        #Memory Supply for each method
        #-------------------------------
        self.ES_MEMORY=self.MU
        self.SA_MEMORY=self.NCORES
        self.PSO_MEMORY=self.NPOP-self.MU
        #--------------------
        # Fixed/Derived parameters 
        #--------------------
        self.nx=len(bounds)  #all
        self.memory_size=memory_size
        
        self.COOLING='fast' #SA
        self.TMIN=1  #SA
        self.LAMBDA=self.NPOP #ES
        self.NPAR=self.NPOP #PSO
        self.SMIN = 1/self.nx #ES
        self.SMAX = 0.5  #ES
        self.v0=0.1 #constant to initialize PSO speed, not very important

        #infer variable types 
        self.datatype = np.array([bounds[item][0] for item in bounds])
        
        #mir-grid
        if "grid" in self.datatype:
            self.grid_flag=True
            self.orig_bounds=bounds  #keep original bounds for decoding
            #print('--debug: grid parameter type is found in the space')
            self.bounds, self.bounds_map=encode_grid_to_discrete(self.bounds) #encoding grid to int
            #define var_types again by converting grid to int
            self.datatype = np.array([self.bounds[item][0] for item in self.bounds])
        else:
            self.grid_flag=False
            self.bounds = bounds
            self.orig_bounds=bounds
        
        self.lb = np.array([self.bounds[item][1] for item in self.bounds])
        self.ub = np.array([self.bounds[item][2] for item in self.bounds])

    def fit_worker(self, x):
        #"""
        #Evaluates fitness of an individual.
        #"""
        
        
        #mir-grid
        if self.grid_flag:
            #decode the individual back to the int/float/grid mixed space
            x=decode_discrete_to_grid(x,self.orig_bounds,self.bounds_map) 
                    
        fitness = self.FIT(x)
        return fitness

    def evolute(self, ngen, x0=None, warmup=100, verbose=0):
        """
        This function evolutes the PESA algorithm for number of generations.
        
        :param ngen: (int) number of generations to evolute
        :param x0: (list of lists) initial samples to start the replay memory (``len(x0)`` must be equal or more than ``npop``)
        :param warmup: (int) number of random warmup samples to initialize the replay memory and must be equal or more than ``npop`` (only used if ``x0=None``)
        :param verbose: (int) print statistics to screen, 0: no print, 1: PESA print, 2: detailed print
        
        :return: (tuple) (best individual, best fitness, and a list of fitness history)
        """
        
        self.verbose=verbose
        self.NGEN=ngen
        self.STEPS=self.NGEN*self.NPOP #all 
        if self.memory_size:
            self.MEMORY_SIZE=self.memory_size
        else:
            self.MEMORY_SIZE=self.STEPS*3+1 #PESA
        
        #-------------------------------------------------------
        # Check if initial pop is provided as initial guess 
        #-------------------------------------------------------
        if x0: 
            self.x0=x0.copy()
            # use provided initial guess
            warm=ESMod(bounds=self.bounds, fit=self.fit_worker, mu=self.MU, lambda_=self.LAMBDA, ncores=self.ncores)
            x0size=len(self.x0)
            assert x0size >= self.NPOP, 'the number of lists in x0 ({}) must be more than or equal npop ({})'.format(x0size, self.NPOP)
            for i in range(x0size):
                check_mixed_individual(x=self.x0[i], bounds=self.orig_bounds) #assert the type provided is consistent
                if self.grid_flag:
                    self.x0[i] = encode_grid_indv_to_discrete(self.x0[i], bounds=self.orig_bounds, bounds_map=self.bounds_map)    
            
            self.pop0=warm.init_pop(warmup=x0size, x_known=self.x0)  #initial population for ES
        else:
            #create initial guess 
            assert warmup > self.NPOP, 'the number of warmup samples ({}) must be more than npop ({})'.format(warmup, self.NPOP)
            warm=ESMod(bounds=self.bounds, fit=self.fit_worker, mu=self.MU, lambda_=self.LAMBDA, ncores=self.ncores)
            self.pop0=warm.init_pop(warmup=warmup)  #initial population for ES
            
        self.partime={}
        self.partime['pesa']=[]
        self.partime['es']=[]
        self.partime['pso']=[]
        self.partime['sa']=[]
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
        espop0, swarm0, swm_pos0, swm_fit0, local_pos, local_fit, x0, E0=self.init_guess(pop0=self.pop0)
        # Initialize ES class
        es=ESMod(bounds=self.bounds, fit=self.fit_worker, mu=self.MU, lambda_=self.LAMBDA, ncores=self.NCORES, indpb=self.INDPB, 
                 cxpb=self.CXPB, mutpb=self.MUTPB, smin=self.SMIN, smax=self.SMAX)
        # Initialize SA class
        sa=SAMod(bounds=self.bounds, memory=self.mymemory, fit=self.fit_worker, steps=self.STEPS, ncores=self.NCORES, 
                 chi=self.CHI, replay_rate=self.REPLAY_RATE, cooling=self.COOLING, Tmax=self.TMAX, Tmin=self.TMIN)
        # Initialize PSO class (if USED)
        if self.pso_flag:
            pso=PSOMod(bounds=self.bounds, fit=self.fit_worker, npar=self.NPAR, swm0=[swm_pos0,swm_fit0], 
                       ncores=self.NCORES, c1=self.C1, c2=self.C2, speed_mech=self.SPEED_MECH)
            
        #--------------------------------
        # Step 3: Initialize PESA engine
        #--------------------------------
        #Use initial samples as first guess for SA, ES, and PSO
        self.pop_next=deepcopy(espop0) # x0 for ES
        self.x_next, self.E_next=deepcopy(x0), deepcopy(E0) # x0 for SA
        if self.pso_flag:
            self.swm_next, self.local_pos_next, self.local_fit_next=deepcopy(swarm0), deepcopy(local_pos), deepcopy(local_fit) # x0 for PSO (if used)
        self.STEP0=1  #step counter
        self.ALPHA=self.ALPHA0  #set alpha to alpha0
        
        #--------------------------------
        # Step 4: PESA evolution
        #--------------------------------
        for gen in range(1,self.NGEN+1):
            
            caseids=['es_gen{}_ind{}'.format(gen,ind+1) for ind in range(self.LAMBDA)] # save caseids for ES 
            if self.pso_flag:
                pso_caseids=['pso_gen{}_par{}'.format(gen+1,ind+1) for ind in range(self.NPAR)] # save caseids for PSO 
            #-------------------------------------------------------------------------------------------------------------------
            # Step 5: evolute all methods for 1 generation 
            #-------------------------------------------------------------------------------------------------------------------
            #**********************************
            #--Step 5A: Complete PARALEL calcs 
            # via multiprocess.Process
            #*********************************
            if self.PROC:
                t0=time.time()
                
                QSA = Queue(); QES=Queue(); QPSO=Queue()
                def sa_worker():
                    random.seed(self.SEED)
                    x_new, E_new, self.T, self.acc, self.rej, self.imp, x_best, E_best, sa_partime= sa.anneal(ngen=1,npop=self.NPOP, x0=self.x_next, 
                                                                                                  E0=self.E_next, step0=self.STEP0)
                    QSA.put((x_new, E_new, self.T, self.acc, self.rej, self.imp, x_best, E_best, sa_partime))
                def es_worker():
                    random.seed(self.SEED)
                    pop_new, es_partime=es.evolute(population=self.pop_next,ngen=1,caseids=caseids)
                    QES.put((pop_new, es_partime))
                def pso_worker():
                    random.seed(self.SEED)
                    if gen > 1:
                        swm_new, swm_pos_new, swm_fit_new, pso_partime=pso.evolute(ngen=1, swarm=self.swm_next, local_pos=self.local_pos_next, local_fit=self.local_fit_next, 
                                                                      swm_best=[self.swm_pos, self.swm_fit], mu=self.MU, exstep=self.STEP0, exsteps=self.STEPS, 
                                                                      caseids=pso_caseids, verbose=0)
                    else:
                        swm_new, swm_pos_new, swm_fit_new, pso_partime=pso.evolute(ngen=1, swarm=self.swm_next, local_pos=self.local_pos_next, 
                                                                      local_fit=self.local_fit_next, mu=self.MU, exstep=self.STEP0, exsteps=self.STEPS, 
                                                                      caseids=pso_caseids, verbose=0)
                    QPSO.put((swm_new, swm_pos_new, swm_fit_new, pso_partime))
                Process(target=sa_worker).start()
                Process(target=es_worker).start()
                
                if self.pso_flag:
                    Process(target=pso_worker).start()
                    self.swm_next, self.swm_pos, self.swm_fit, pso_partime=QPSO.get()
                    self.local_pos_next=[self.swm_next[key][3] for key in self.swm_next]
                    self.local_fit_next=[self.swm_next[key][4] for key in self.swm_next]
                     
                self.x_next, self.E_next, self.T, self.acc, self.rej, self.imp, self.x_best, self.E_best, sa_partime=QSA.get()
                self.pop_next, es_partime=QES.get()
                #self.partime.append(time.time()-t0)
                self.partime['pesa'].append(time.time()-t0)
                self.partime['pso'].append(pso_partime)
                self.partime['es'].append(es_partime)
                self.partime['sa'].append(sa_partime)
                                
            #*********************************
            #--Step 5B: Complete Serial calcs
            #*********************************
            else:  
                self.pop_next, _ =es.evolute(population=self.pop_next,ngen=1,caseids=caseids) #ES serial
                self.x_next, self.E_next, self.T, self.acc, self.rej, self.imp, self.x_best, self.E_best, _ = sa.anneal(ngen=1,npop=self.NPOP, x0=self.x_next, 
                                                                                                     E0=self.E_next, step0=self.STEP0) #SA serial
                if self.pso_flag:
                    self.swm_next, self.swm_pos, self.swm_fit, _ =pso.evolute(ngen=1, swarm=self.swm_next, local_pos=self.local_pos_next, 
                                                                          local_fit=self.local_fit_next, exstep=self.STEP0, exsteps=self.STEPS,
                                                                          caseids=pso_caseids, mu=self.MU, verbose=0)
                    self.local_pos_next=[self.swm_next[key][3] for key in self.swm_next]
                    self.local_fit_next=[self.swm_next[key][4] for key in self.swm_next]
            

            #*********************************************************
            # Step 5C: Obtain relevant statistics for this generation 
            #*********************************************************
            self.STEP0=self.STEP0+self.NPOP  #update step counter
            self.inds, self.rwd=[self.pop_next[i][0] for i in self.pop_next], [self.pop_next[i][2] for i in self.pop_next]  #ES statistics
            self.mean_strategy=[np.mean(self.pop_next[i][1]) for i in self.pop_next]  #ES statistics 
            if self.pso_flag:
                self.pars, self.fits=[self.swm_next[i][0] for i in self.swm_next], [self.swm_next[i][2] for i in self.swm_next]  #PSO statistics 
                self.mean_speed=[np.mean(self.swm_next[i][1]) for i in self.swm_next]
                
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

        if self.verbose:
            print('------------------------ PESA Summary --------------------------')
            print('Best fitness (y) found:', self.fitness_best)
            print('Best individual (x) found:', self.xbest_correct)
            print('--------------------------------------------------------------') 
                
        #--mir
        if self.mode=='min':
            self.fit_hist=[-item for item in self.fit_hist]
        
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
    
    def memory_update(self):
        #"""
        #This function updates the replay memory with the samples of SA, ES, and PSO (if used)
        #then remove the duplicates from the memory
        #"""
        self.mymemory.add(xvec=tuple(self.x_next), obj=self.E_next, method=['sanext']*len(self.x_next))
        self.mymemory.add(xvec=tuple(self.x_best), obj=self.E_best, method=['sabest']*len(self.x_best))
        self.mymemory.add(xvec=tuple(self.inds), obj=self.rwd, method=['es']*len(self.inds))
        if self.pso_flag:
            self.mymemory.add(xvec=tuple(self.pars), obj=self.fits, method=['pso']*len(self.pars))
        #self.mymemory.remove_duplicates()  #remove all duplicated samples in memory to avoid biased sampling
        
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
        es_replay=self.mymemory.sample(batch_size=self.ES_MEMORY,mode=self.MODE,alpha=self.ALPHA)
        index=self.MU
        for sample in range(self.ES_MEMORY):
            self.pop_next[index].append(es_replay[sample][0])
            self.pop_next[index].append([random.uniform(self.SMIN,self.SMAX) for _ in range(self.nx)])
            self.pop_next[index].append(es_replay[sample][1])
            index+=1
        
        if self.pso_flag:
            pso_replay=self.mymemory.sample(batch_size=self.PSO_MEMORY,mode=self.MODE,alpha=self.ALPHA)
            for key in self.swm_next:
                del self.swm_next[key][3:]
                
            index=self.MU
            for sample in range(self.PSO_MEMORY):
                self.swm_next[index].append(pso_replay[sample][0])
                #self.swm_next[index].append([random.uniform(self.SPMIN,self.SPMAX) for _ in range(self.nx)])
                self.swm_next[index].append(list(self.v0*np.array(pso_replay[sample][0])))
                self.swm_next[index].append(pso_replay[sample][1])
                self.local_pos_next.append(pso_replay[sample][0])
                self.local_fit_next.append(pso_replay[sample][1])
                index+=1
                
        sa_replay=self.mymemory.sample(batch_size=self.SA_MEMORY,mode=self.MODE,alpha=self.ALPHA)
        self.x_next, self.E_next=[item[0] for item in sa_replay], [item[1] for item in sa_replay]
        
    def init_guess(self, pop0):
        #"""
        #This function takes initial guess pop0 and returns initial guesses for SA, PSO, and ES 
        #to start PESA evolution
        #inputs:
        #    pop0 (dict): dictionary contains initial population to start with for all methods
        #returns:
        #    espop0 (dict): initial population for ES
        #    swarm0 (dict): initial swarm for PSO 
        #    swm_pos (list), swm_fit (float): initial guess for swarm best position and fitness for PSO
        #    local_pos (list of lists), local_fit (list): initial guesses for local best position of each particle and their fitness for PSO
        #    x0 (list of lists), E0 (list): initial input vectors and their initial fitness for SA
        #"""
        pop0=list(pop0.items())
        pop0.sort(key=lambda e: e[1][2], reverse=True)
        sorted_sa=dict(pop0[:self.NCORES])
        #sorted_dict=dict(sorted(pop0.items(), key=lambda e: e[1][2], reverse=True)[:self.NCORES]) # sort the initial samples for SA
        x0, E0=[sorted_sa[key][0] for key in sorted_sa], [sorted_sa[key][2] for key in sorted_sa] # initial guess for SA
        
        #sorted_pso=dict(sorted(pop0.items(), key=lambda e: e[1][2], reverse=True)[:self.NPAR]) # sort the initial samples for PSO
        #sorted_es=dict(sorted(pop0.items(), key=lambda e: e[1][2], reverse=True)[:self.LAMBDA]) # sort the initial samples for ES
        sorted_pso=dict(pop0[:self.NPAR])
        sorted_es=dict(pop0[:self.LAMBDA])
        
        swarm0=defaultdict(list)
        espop0=defaultdict(list)
        
        local_pos=[]
        local_fit=[]
        index=0
        for key in sorted_pso:
            swarm0[index].append(sorted_pso[key][0])
            swarm0[index].append(list(self.v0*np.array(sorted_pso[key][0])))
            swarm0[index].append(sorted_pso[key][2])
            local_pos.append(sorted_pso[key][0])
            local_fit.append(sorted_pso[key][2])
            index+=1

        swm_pos=swarm0[0][0]
        swm_fit=swarm0[0][2]
        
        index=0
        for key in sorted_es:
            espop0[index].append(sorted_es[key][0])
            espop0[index].append(sorted_es[key][1])
            espop0[index].append(sorted_es[key][2])
            index+=1
            
        return espop0, swarm0, swm_pos, swm_fit, local_pos, local_fit, x0, E0  

    def printout(self, mode, gen):
        #"""
        #Print statistics to screen
        #inputs:
        #    mode (int): 1 to print for individual algorathims and 2 to print for PESA 
        #    gen (int): current generation number 
        #"""
        if mode == 1:
            print('***********************************************************************************************')
            print('############################################################')
            print('ES step {}/{}, CX={}, MUT={}, MU={}, LAMBDA={}'.format(self.STEP0-1,self.STEPS, np.round(self.CXPB,2), np.round(self.MUTPB,2), self.MU, self.LAMBDA))
            print('############################################################')
            print('Statistics for generation {}'.format(gen))
            print('Best Fitness:', np.round(np.max(self.rwd),4) if self.mode == 'max' else -np.round(np.max(self.rwd),4))
            print('Max Strategy:', np.round(np.max(self.mean_strategy),3))
            print('Min Strategy:', np.round(np.min(self.mean_strategy),3))
            print('Average Strategy:', np.round(np.mean(self.mean_strategy),3))
            print('############################################################')
                  
            print('************************************************************')
            print('SA step {}/{}, T={}'.format(self.STEP0-1,self.STEPS,np.round(self.T)))
            print('************************************************************')
            print('Statistics for the {} parallel chains'.format(self.NCORES))
            print('Fitness:', np.round(self.E_next,4) if self.mode == 'max' else -np.round(self.E_next,4))
            print('Acceptance Rate (%):', self.acc)
            print('Rejection Rate (%):', self.rej)
            print('Improvment Rate (%):', self.imp)
            print('************************************************************')
            
            if self.pso_flag:
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                print('PSO step {}/{}, C1={}, C2={}, Particles={}'.format(self.STEP0-1,self.STEPS, np.round(self.C1,2), np.round(self.C2,2), self.NPAR))
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                print('Statistics for generation {}'.format(gen))
                print('Best Swarm Fitness:', np.round(self.swm_fit,4) if self.mode == 'max' else -np.round(self.swm_fit,4))
                print('Best Swarm Position:', self.swm_pos if not self.grid_flag else decode_discrete_to_grid(self.swm_pos,self.orig_bounds,self.bounds_map))
                print('Max Speed:', np.round(np.max(self.mean_speed),3))
                print('Min Speed:', np.round(np.min(self.mean_speed),3))
                print('Average Speed:', np.round(np.mean(self.mean_speed),3))
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        
        if mode == 2:
            print('------------------------------------------------------------')
            print('PESA step {}/{}, Ncores={}'.format(self.STEP0-1,self.STEPS, self.ncores))
            print('------------------------------------------------------------')
            print('PESA statistics for generation {}'.format(gen))
            print('Best Fitness:', self.fitness_best if self.mode == 'max' else -self.fitness_best)
            print('Best Individual:', self.xbest_correct)
            print('ALPHA:', np.round(self.ALPHA,3))
            print('Memory Size:', self.memory_size)
            print('------------------------------------------------------------')
            
            print('***********************************************************************************************')