# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 13:45:54 2020

@author: Majdi
"""

import random
import numpy as np
import copy
import time
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

class SA:
    def __init__ (self, bounds, fit, npop=50, ncores=1, chi=0.1, cooling='fast', Tmax=10000, Tmin=1, seed=None):  
        """
        Parallel SA:
        A Synchronous Approach with Occasional Enforcement of Best Solution-Fixed Intervals
        Inputs:
            bounds (dict): input paramter lower/upper bounds in dictionary form
            fit (function): fitness function 
            npop (int): number of individuals in the population group (must be divisble by ncores)
            ncores (int): parallel cores
            chi (float/list): a float or list of floats representing perturbation probablity for each parallel chain 
                 if float, one chi used for all chains 
                 else a list of chi's with size ``ncores`` are used for each chain
            cooling (str): cooling schedule, either fast, boltzmann, or cauchy are allowed
            Tmax (int): maximum temperature
            Tmin (int): minimum temperature
            seed (int): random seeding for reproducibility
        """
        
        if seed:
            random.seed(seed)
            
        self.seed=seed
        self.npop=npop
        self.batch_size=10
        self.bounds=bounds
        self.ncores=ncores
        assert npop % self.ncores == 0, 'The number of population (npop) to run must be divisible by ncores, {} mod {} != 0'.format(npop,self.ncores)
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
        self.T=Tmax #initialize T
        
        #chain statistics
        self.accept=0
        self.reject=0
        self.improve=0
        
    def GenInd(self, bounds):
        """
        Individual generator
        Input: 
            -bounds (dict): input paramter type and lower/upper bounds in dictionary form
        Returns: 
            -individual (list): individual position
        """
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
        for item in self.bounds:
            if random.random() < chi:
                sample = self.sampler(self.bounds[item])
                while x[i] == sample and (self.bounds[item][1] != self.bounds[item][2]): 
                    sample = self.sampler(self.bounds[item])
                x[i] = sample
            i+=1
        return x
        
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
        if self.seed:
            random.seed(self.seed + core_seed)
        
        rejects=0; accepts=0; improves=0
        k=min_step
        while k <= max_step:
            T=self.temp(step=k)
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
    
    def InitChains(self, x0=None):
        
        #initialize the chain and run them in parallel (these samples will be used to initialize the annealing process)
        
        #Establish the chain
        if x0:
            print('The first individual provided by the user:', x0[0])
            print('The last individual provided by the user:', x0[-1])
        else:
            x0=[]
            for i in range (self.ncores):
                x0.append(self.GenInd(self.bounds))
        
        #Evaluate the swarm
        if self.ncores > 1:  #evaluate swarm in parallel
            core_list=[]
            for ind in x0:
                core_list.append(ind)
           
            p=MyPool(self.ncores)
            E0 = p.map(self.fit, core_list)
            p.close(); p.join()
            
        else: #evaluate swarm in series
            E0=[]
            for ind in x0:
                fitness=self.fit(ind)
                E0.append(fitness)
        
        return x0, E0 #return initial guess and initial fitness      
    
    def anneal(self, ngen, x0=None, verbose=0):
        """
        Perform annealing over total ``ngen`` by updating chains every ``npop``
        Returns the best ``x`` and ``E`` over the whole stage along with relvant statistics 
        """
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
                
            assert len(x0) == self.ncores, 'Length of initial guesses x0 ({}) for chains do not equal to ncores or # of chains ({})'.format(len(x0), self.ncores)
            xinit, Einit=self.InitChains(x0=x0)
        else:
            xinit, Einit=self.InitChains()
            
        x_next=copy.deepcopy(xinit)
        E_next=copy.deepcopy(Einit)
        
        for i in range (ngen):
            x_next,E_next,self.T, acc, rej, imp, x_best, E_best=self.chain(x0=x_next, E0=E_next, step0=step0, npop=self.npop)
            step0=step0+self.npop
            arg_max=np.argmax(E_best)
            stat['x'].append(x_best[arg_max])
            stat['fitness'].append(E_best[arg_max])
            stat['T'].append(self.T)
            stat['accept'].append(acc[arg_max])
            stat['reject'].append(rej[arg_max])
            stat['improve'].append(imp[arg_max])
            
            if E_best[arg_max] > E_opt:
                E_opt=E_best[arg_max]
                x_opt=copy.deepcopy(x_best[arg_max])
            
            if verbose:
                print('************************************************************')
                print('SA step {}/{}, T={}'.format(step0-1,self.steps,np.round(self.T)))
                print('************************************************************')
                print('Statistics for the {} parallel chains'.format(self.ncores))
                print('Best fitness:', np.round(E_best,6))
                print('Best individual:', x_best[arg_max])
                print('Acceptance Rate (%):', acc)
                print('Rejection Rate (%):', rej)
                print('Improvment Rate (%):', imp)
                print('************************************************************')
        
        print('------------------------ SA Summary --------------------------')
        print('Best fitness (y) found:', E_opt)
        print('Best individual (x) found:', x_opt)
        print('--------------------------------------------------------------')
            
        return x_opt, E_opt, stat

