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
"""
Created on Sun Jun 14 13:45:54 2020

@author: Majdi
"""

import random
import numpy as np
import copy
from neorl.hybrid.pesacore.er import ExperienceReplay
import time
import multiprocessing
import multiprocessing.pool
import joblib
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

class SAMod(ExperienceReplay):
    
    def __init__ (self, bounds, fit, steps, memory=None, ncores=1, chi=0.1, replay_rate=0, cooling='fast', Tmax=10000, Tmin=1):  
        """
        Parallel SA:
        A Synchronous Approach with Occasional Enforcement of Best Solution-Fixed Intervals
        Inputs:
            bounds (dict): input paramter lower/upper bounds in dictionary form
            memory (class): a class object created by ExperienceReplay
            fit (function): fitness function 
            ncores (int): parallel cores
            chi (float/list): a float or list of floats representing perturbation probablity for each parallel chain 
                 if float, one chi used for all chains 
                 else a list of chi's with size ``ncores`` are used for each chain
            cooling (str): cooling schedule, either fast, boltzmann, or cauchy are allowed
            Tmax (int): maximum temperature
            Tmin (int): minimum temperature
        """
        
        random.seed(1)
        
        if memory:
            self._memory=memory
        else:
            self._memory=False
            
        self.batch_size=10
        self.replay_rate=replay_rate
        self.bounds=bounds
        self.ncores=ncores
        self.Tmax=Tmax
        self.Tmin=Tmin
        if type(chi) == list:
            assert len(chi) == ncores, 'The list of chi values ({}) MUST equal ncores ({})'.format(len(chi),self.ncores) 
            self.chi=chi
        elif type(chi) == float or type(chi) == int:
            self.chi=[chi]*ncores
        else:
            raise Exception ('for chi, either list of floats or scalar float are allowed')
            
        self.cooling=cooling
        self.fit=fit
        self.steps=steps
        self.T=Tmax #initialize T
        
        #chain statistics
        self.accept=0
        self.reject=0
        self.improve=0

    def sampler(self,bound):
        """
        This function takes input as [type, low, high] and returns a sample  
        This is to sample the input parameters during optimisation
        """
        if bound[0] == 'int':
            sample=random.randint(bound[1], bound[2])
        elif bound[0] == 'float':
           sample=random.uniform(bound[1], bound[2])
        elif bound[0] == 'grid':
            sample=random.sample(bound[1],1)[0]
        else:
            raise Exception ('unknown data type is given, either int, float, or grid are allowed for parameter bounds')
                
        return sample
    
    def move(self, x, chi):
        """
        This function is to perturb x attributes by probability chi
        Inputs:
            x: input vector 
            chi: perturbation probablity between 0 and 1
        Returns: perturbed vector x
        """
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
        
        
        if self.cooling=='fast':
            Tfac = -np.log(float(self.Tmax) / self.Tmin)
            T = self.Tmax * np.exp( Tfac * step / self.steps)
        elif self.cooling=='boltzmann':
            T = self.Tmax / np.log(step + 1)
        elif self.cooling=='cauchy':
            T = self.Tmax / (step + 1)
        else:
            raise Exception ('--error: unknown cooling mode is entered, fast, boltzmann, or cauchy are ONLY allowed')
        
        return T
    
    def chain_object (self,inp):
        """
        This function is a multiprocessing object, used to be passed to Pool, that respresents 
        an individual SA chain. 
        Input:
            inp: a list contains the following inputs in order
            inp[0] --> x0: initial guess to chain 
            inp[1] --> E0: initial energy of x0
            inp[2] --> min_step: min step to start this chain 
            inp[3] --> max_step: max step to terminate this chain 
            inp[4] --> core_seed: seed for this chain
        returns: 
            x_best, E_best: best obtained from this chain
            T: last temperature for this chain
            accepts, rejects, improves for this chain
        """
        
        x_prev=copy.deepcopy(inp[0])
        x_best=copy.deepcopy(x_prev)
        E_prev=inp[1]
        E_best=inp[1]
        min_step=inp[2]
        max_step=inp[3]
        core_seed=inp[4]
        random.seed(core_seed)
        
        rejects=0; accepts=0; improves=0
        k=min_step
        while k <= max_step:
            T=self.temp(step=k)
            
            if random.random() < self.replay_rate and self._memory: #replay memory
                x, E, _=self._memory.sample(batch_size=1, mode='greedy', seed=core_seed)[0]
                E=self.fit(x)
                #if core_seed==1 or core_seed==10:
                #    print('memory sample {}, step {}, E {}'.format(core_seed, k, np.round(E,2)))
            else: #random-walk
                x=copy.deepcopy(self.move(x=x_prev,chi=self.chi[core_seed-1]))
                #SA is programmed to maximize reward
                E=self.fit(x)
            
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
                
            elif np.exp(dE/T) >= random.random(): #accept the state
                accepts += 1
                x_prev = copy.deepcopy(x)
                E_prev = E
            else:
                # Reject the new solution (maintain the current state!)
                rejects += 1
            k+=1
        
            #debugging lines
            #if core_seed==1 or core_seed==10:
            #    print('E_prev '+str(core_seed), np.round(E_prev,2))
            #    print('memory_SA', len(self._memory.storage))
            #assert self.fit(x_prev) == E_prev
        return x_prev, E_prev, T, accepts, rejects, improves, x_best, E_best
        
    def chain(self, x0, E0, step0, npop):
        """
        This function creates ``ncores`` independent SA chains with same initial guess x0, E0 and 
        runs them via multiprocessing Pool.
        Input:
            x0: initial input guess (comes from previous annealing chains or from replay memory)
            E0: energy/fitness value of x0
            step0: is the first time step to use for temperature annealing
            npop: total number of individuals to be evaluated in this annealing stage
        returns: 
            x_best, E_best, and T obtained from this annealing stage from all chains
        """
        assert npop % self.ncores == 0, 'The number of communications to run must be divisible by ncores, {} mod {} != 0'.format(npop,self.ncores)
        core_npop =int(npop/self.ncores)
        
        #Append and prepare the input list to be passed as input to Pool.map
        core_list=[]
        core_step_min=step0
        for j in range(1,self.ncores+1):
            core_step_max=step0+j*core_npop-1
            core_list.append([x0[j-1], E0[j-1], core_step_min, core_step_max, j])
            core_step_min=core_step_max+1
        
        if self.ncores > 1:
            # create and run the Pool
            t0=time.time()
            p=MyPool(self.ncores)
            results = p.map(self.chain_object, core_list)
            p.close()
            p.join()
            
            #with joblib.Parallel(n_jobs=self.ncores) as parallel:
            #    results=parallel(joblib.delayed(self.chain_object)(item) for item in core_list)
            self.partime=time.time()-t0
            #print('SA:', self.partime)
        else:
            results=[]
            results.append(list(self.chain_object(core_list[0])))
            self.partime=0
                
        # Determine the index and the best solution from all chains
        #best_index=[y[0] for y in results].index(min([y[0] for y in results]))
        self.x_last=[item[0] for item in results]
        self.E_last=[item[1] for item in results]
        self.T=results[-1][2] # get the temperature of the last chain
        self.accepts=[np.round(item[3]/core_npop*100,1) for item in results] #convert to rate
        self.rejects=[np.round(item[4]/core_npop*100,1) for item in results] #convert to rate
        self.improves=[np.round(item[5]/core_npop*100,1) for item in results] #convert to rate
        
        self.x_best, self.E_best=[item[6] for item in results], [item[7] for item in results]
        
        return self.x_last, self.E_last, self.T, self.accepts, self.rejects, self.improves, self.x_best, self.E_best
    
    def anneal(self, ngen, npop, x0, E0, step0, verbose=0):
        """
        Perform annealing over total ``steps`` by updating chains every ``npop``
        Returns the best ``x`` and ``energy`` over the whole stage 
        """
        assert len(x0) == self.ncores, 'Length of initial guesses x0 ({}) for chains do not equal to ncores or # of chains ({})'.format(len(x0), self.ncores)
        assert len(E0) == self.ncores, 'Length of initial fitness E0 ({}) for chains do not equal to ncores or # of chains ({})'.format(len(E0), self.ncores)
        x_next=copy.deepcopy(x0)
        E_next=copy.deepcopy(E0)
        steps=ngen*npop
        for i in range (ngen):
            x_next,E_next,self.T, acc, rej, imp, x_best, E_best=self.chain(x0=x_next, E0=E_next, step0=step0, npop=npop)
            step0=step0+npop
            arg_max=np.argmax(E_best)
            if verbose:
                print('************************************************************')
                print('SA step {}/{}, T={}'.format(step0-1,self.steps,np.round(self.T)))
                print('************************************************************')
                print('Statistics for the {} parallel chains'.format(self.ncores))
                print('Best fitness:', np.round(E_best,2))
                print('Best individual:', x_best[arg_max])
                print('Acceptance Rate (%):', acc)
                print('Rejection Rate (%):', rej)
                print('Improvment Rate (%):', imp)
                print('************************************************************')
            
        return x_next, E_next, self.T, acc, rej, imp, x_best, E_best, self.partime