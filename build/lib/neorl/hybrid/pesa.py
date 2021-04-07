# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 18:21:05 2020

@author: Majdi Radaideh
"""

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

class PESA(ExperienceReplay):
    
    def __init__ (self, bounds, fit, ngen, npop, memory_size=None, pop0=None, ncores=1, mode='prior', alpha0=0.1, 
                  alpha1=1, warmup=200, chi=0.1, replay_rate=0.1, Tmax=10000, mu=40, cxpb=0.4, mutpb=0.1, indpb=0.1, 
                  c1=2, c2=2, speed_mech='constric', seed=1, pso_flag=True, verbose=True): 
        """
        parameters:
            -bounds (dict): type(int,float)/lower/upper bounds of input attributes in a dictionary form
            -fit (func): fitness function
            -ngen (int): number of generations to run
            -npop (int): number of population per generation (size of ES pop and size of PSO Swarm)
            -memory_size (int) : max size of the memory
            -pop0 (dict): initial population (if any) to start the memory
            -ncores (int): TOTAL number of cores to run for all methods
            -mode (string): `greedy`, `uniform`, `prior` for priortized replay with alpha
            -alpha0 (float): lower bound of priortized coefficient for mode=`prior`
            -alpha1 (float): upper bound of priortized coefficient for mode= `prior`
            -warmup (int): if pop0 is omitted, this forms initial population to warmup the algorathim
            -chi (float): probablity to perturb each input attribute for SA during random-walk
            -replay_rate (float): probablity to sample from the memory for SA instead of random-walk
            -Tmax (int): maximum/initial annealing temperature for SA
            -mu (int): number of inidividuals (ES)/particles (PSO)  to survive for next generation
                      Also, mu equals to the number of inidividuals (ES)/particles (PSO) to sample from the memory
                      in next generation. So 1/2 popultation comes from previous generation, and 1/2 comes from the memory
            -cxpb (float): population crossover probablity for ES
            -mutpb (float): population mutation probablity for ES 
            -c1 (float): cognative speed constant for PSO
            -c2 (float): social speed constant for PSO
            -speed_mech (string): mechanism to adapt inertia weight: constric, timew, globw
            -seed: random seed for the PESA algorathim
            -pso_flag (bool): whether to activate PSO or not
            -verbose (bool): to print summary to terminal or not
        """
        #--------------------
        #General Parameters
        #--------------------
        self.BOUNDS=bounds
        self.FIT=fit
        self.NGEN=ngen
        self.NPOP=npop
        self.pso_flag=pso_flag
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
        self.verbose=verbose
        self.SEED=seed
        
        #--------------------
        #Experience Replay
        #--------------------
        self.MODE=mode;  self.ALPHA0=alpha0;   self.ALPHA1=alpha1
        #--------------------
        # SA hyperparameters
        #--------------------
        self.TMAX=Tmax;  self.CHI=chi;  self.REPLAY_RATE=replay_rate
        
        #--------------------
        # ES HyperParameters
        #--------------------
        self.CXPB=cxpb;  self.MUTPB=mutpb;   self.MU=mu;   self.INDPB=indpb
        
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
        self.STEPS=self.NGEN*self.NPOP #all 
        self.nx=len(self.BOUNDS)  #all
        if memory_size:
            self.MEMORY_SIZE=memory_size
        else:
            self.MEMORY_SIZE=self.STEPS*3+1 #PESA
        self.COOLING='fast' #SA
        self.TMIN=1  #SA
        self.LAMBDA=self.NPOP #ES
        self.NPAR=self.NPOP #PSO
        self.SMIN = 1/self.nx #ES
        self.SMAX = 0.5  #ES
        self.v0=0.1 #constant to initialize PSO speed, not very important
        #-------------------------------------------------------
        # Check if initial pop is provided as initial guess 
        #-------------------------------------------------------
        if pop0: 
            # use provided ones
            self.pop0=pop0
        else:
            #create initial guess 
            warm=ESMod(bounds=self.BOUNDS, fit=self.FIT, mu=self.MU, lambda_=self.LAMBDA, ncores=self.NCORES)
            self.pop0=warm.init_pop(warmup=warmup)  #initial population for ES 

    def evolute(self):
        """
        Run PESA!
        No input to the function
        Returns: x_best (pesa_best[0]) and y_best(pesa_best[1]) found by PESA 
        """
        self.partime={}
        self.partime['pesa']=[]
        self.partime['es']=[]
        self.partime['pso']=[]
        self.partime['sa']=[]
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
        es=ESMod(bounds=self.BOUNDS, fit=self.FIT, mu=self.MU, lambda_=self.LAMBDA, ncores=self.NCORES, indpb=self.INDPB, 
                 cxpb=self.CXPB, mutpb=self.MUTPB, smin=self.SMIN, smax=self.SMAX)
        # Initialize SA class
        sa=SAMod(bounds=self.BOUNDS, memory=self.mymemory, fit=self.FIT, steps=self.STEPS, ncores=self.NCORES, 
                 chi=self.CHI, replay_rate=self.REPLAY_RATE, cooling=self.COOLING, Tmax=self.TMAX, Tmin=self.TMIN)
        # Initialize PSO class (if USED)
        if self.pso_flag:
            pso=PSOMod(bounds=self.BOUNDS, fit=self.FIT, npar=self.NPAR, swm0=[swm_pos0,swm_fit0], 
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
            
            #XXX remove
            #if gen == 67:
            #    print('-- early return')
            #    return self.pesa_best[0], self.pesa_best[1]
            
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
                
                #print(self.partime)
                
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
#            if self.verbose:
#                self.printout(mode=1, gen=gen)
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
            self.memory_size=len(self.mymemory.storage) #memory size so far
            if self.verbose:  #print summary data to screen
                self.printout(mode=2, gen=gen)
        
        return self.pesa_best[0], self.pesa_best[1]

    def linear_anneal(self, step, total_steps, a0, a1):
        """
        Anneal parameter between a0 and a1 
        :param step: current time step
        :param total_steps: total numbe of time steps
        :param a0: lower bound of alpha/parameter
        :param a0: upper bound of alpha/parameter
        :return
          - annealed value of alpha/parameter
        """
        fraction = min(float(step) / total_steps, 1.0)
        return a0 + fraction * (a1 - a0)
    
    def memory_update(self):
        """
        This function updates the replay memory with the samples of SA, ES, and PSO (if used)
        then remove the duplicates from the memory
        """
        self.mymemory.add(xvec=tuple(self.x_next), obj=self.E_next, method=['na']*len(self.x_next))
        self.mymemory.add(xvec=tuple(self.x_best), obj=self.E_best, method=['na']*len(self.x_best))
        self.mymemory.add(xvec=tuple(self.inds), obj=self.rwd, method=['na']*len(self.inds))
        if self.pso_flag:
            self.mymemory.add(xvec=tuple(self.pars), obj=self.fits, method=['na']*len(self.pars))
        #self.mymemory.remove_duplicates()  #remove all duplicated samples in memory to avoid biased sampling

    def resample(self):
        """
        This function samples data from the memory and prepares the chains for SA
        the population for ES, and the swarm for PSO for the next generation
            -SA: initial guess for the parallel chains are sampled from the memroy
            -ES: a total of ES_MEMORY (or MU) individuals are sampled from the memory and appended to ES population 
            -PSO: a total of PSO_MEMORY (or MU) particles are sampled from the memory and appended to PSO swarm 
        For SA: x_next and E_next particpate in next generation
        For PSO: swm_next, local_pso_next, and local_fit_next particpate in next generation
        For ES: pop_next particpates in next generation
        """
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
        """
        This function takes initial guess pop0 and returns initial guesses for SA, PSO, and ES 
        to start PESA evolution
        inputs:
            pop0 (dict): dictionary contains initial population to start with for all methods
        returns:
            espop0 (dict): initial population for ES
            swarm0 (dict): initial swarm for PSO 
            swm_pos (list), swm_fit (float): initial guess for swarm best position and fitness for PSO
            local_pos (list of lists), local_fit (list): initial guesses for local best position of each particle and their fitness for PSO
            x0 (list of lists), E0 (list): initial input vectors and their initial fitness for SA
        """
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
        """
        Print statistics to screen
        inputs:
            mode (int): 1 to print for individual algorathims and 2 to print for PESA 
            gen (int): current generation number 
        """
        if mode == 1:
            print('***********************************************************************************************')
            print('############################################################')
            print('ES step {}/{}, CX={}, MUT={}, MU={}, LAMBDA={}'.format(self.STEP0-1,self.STEPS, np.round(self.CXPB,2), np.round(self.MUTPB,2), self.MU, self.LAMBDA))
            print('############################################################')
            print('Statistics for generation {}'.format(gen))
            print('Best Fitness:', np.round(np.max(self.rwd),4))
            print('Max Strategy:', np.round(np.max(self.mean_strategy),3))
            print('Min Strategy:', np.round(np.min(self.mean_strategy),3))
            print('Average Strategy:', np.round(np.mean(self.mean_strategy),3))
            print('############################################################')
                  
            print('************************************************************')
            print('SA step {}/{}, T={}'.format(self.STEP0-1,self.STEPS,np.round(self.T)))
            print('************************************************************')
            print('Statistics for the {} parallel chains'.format(self.NCORES))
            print('Fitness:', np.round(self.E_next,4))
            print('Acceptance Rate (%):', self.acc)
            print('Rejection Rate (%):', self.rej)
            print('Improvment Rate (%):', self.imp)
            print('************************************************************')
            
            if self.pso_flag:
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                print('PSO step {}/{}, C1={}, C2={}, Particles={}'.format(self.STEP0-1,self.STEPS, np.round(self.C1,2), np.round(self.C2,2), self.NPAR))
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                print('Statistics for generation {}'.format(gen))
                print('Best Swarm Fitness:', np.round(self.swm_fit,4))
                print('Best Swarm Position:', np.round(self.swm_pos,2))
                print('Max Speed:', np.round(np.max(self.mean_speed),3))
                print('Min Speed:', np.round(np.min(self.mean_speed),3))
                print('Average Speed:', np.round(np.mean(self.mean_speed),3))
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        
        if mode == 2:
            print('------------------------------------------------------------')
            print('PESA step {}/{}'.format(self.STEP0-1,self.STEPS))
            print('------------------------------------------------------------')
            print('PESA statistics for generation {}'.format(gen))
            print('Best Fitness:', self.pesa_best[1])
            print('Best Individual:', np.round(self.pesa_best[0],2))
            print('ALPHA:', np.round(self.ALPHA,3))
            print('Memory Size:', self.memory_size)
            print('------------------------------------------------------------')
            
            print('***********************************************************************************************')