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
Created on Mon Jun 15 19:37:04 2020

@author: Majdi
"""

import random
import numpy as np
from collections import defaultdict
import copy
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

class PSOMod:
    def __init__ (self, bounds, fit, npar, swm0=None, ncores=1, c1=2.05, c2=2.05, speed_mech='constric'):  
        """
        Particle Swarm Optimisaion (PSO)
        Parallel mixed discrete/continuous PSO module
        Inputs:
            -bounds (dict): input paramter type and lower/upper bounds in dictionary form
                            Example:
                                {'x1': ['int', 1, 4],
                                 'x2': ['float', 0.1, 0.8],
                                 'x3': ['float', 2.2, 6.2]}
            -fit (function): fitness function 
            -npar (int): number of particles in the swarm
            -swm0 (list of 2): swm0[0] --> initial position of the swarm (list)
                               swm0[1] --> initial fitness of the swarm (float)
            -ncores (int): parallel cores
            -c1 (float): cognative speed constant 
            -c2 (float): social speed constant 
            -w (float): constant inertia weight (how much to weigh the previous velocity)
        """
        random.seed(1)
        self.bounds=bounds
        self.npar=npar
        self.fit=fit
        self.ncores=ncores
        self.speed_mech=speed_mech
        self.c1=c1
        self.c2=c2
        self.size=len(bounds)
        self.datatype=[bounds[key][0] for key in bounds]
        self.low=[bounds[key][1] for key in bounds]
        self.up=[bounds[key][2] for key in bounds]
        self.v0=0.1 # factor to intialize the speed
        if not swm0:
            self.swm_pos, _ =self.GenParticle(bounds=bounds)
            self.swm_fit=self.fit(self.swm_pos)
        else:
            self.swm_pos=swm0[0]
            self.swm_fit=swm0[1]
        
        if self.speed_mech=='constric':
            phi=self.c1+self.c2
            self.w=2/np.abs(2-phi-np.sqrt(phi**2-4*phi))
        elif self.speed_mech=='timew':
            self.wmin=0.4
            self.wmax=0.9
            self.w=self.wmax
        elif self.speed_mech=='globw':
            pass
        else:
            raise ('only timew, globw, or constric are allowed for speed_mech, the mechanism used is not defined')
        
        assert self.ncores >=1, "Number of cores must be more than or equal 1"
            
    def GenParticle(self, bounds):
        """
        Particle generator
        Input: 
            -bounds (dict): input paramter type and lower/upper bounds in dictionary form
        Returns: 
            -particle (list): particle position
            -speed (list): particle speed
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
        particle=list(content)
        speed = list(self.v0*np.array(content))
        return particle, speed
    
    def UpdateParticle(self, particle, local_pos, local_fit):
        """
        Function that updates the particle speed and position based on the 
        best local positon of the particle and best global position of the swarm
        The function works with both int or float variables
        Input: 
            particle (list of lists):
                    particle[0] (list) = current position
                    particle[1] (list) = speed 
                    particle[2] (list) = current fit 
            local_pos (list): best local position observed for this particle
        
        Return:
            new_particle (list of lists): modified particle with same structure as `particle` 
        """        
        new_particle=copy.deepcopy(particle)
        for i in range (self.size):
            r1=random.random()
            r2=random.random()
            
            #**********************
            #Update Speed
            #**********************
            speed_cognitive=self.c1*r1*(local_pos[i]-particle[0][i])
            speed_social=self.c2*r2*(self.swm_pos[i]-particle[0][i])
            
            if self.speed_mech=='constric':
                new_particle[1][i]=self.w*(particle[1][i]+speed_cognitive+speed_social)
                #print('constric', self.w)
            elif self.speed_mech=='timew':
                new_particle[1][i]=self.w*particle[1][i]+speed_cognitive+speed_social
            elif self.speed_mech=='globw':
                self.w=1.11-local_fit/self.swm_fit
                #print('globw', self.w)
                new_particle[1][i]=self.w*particle[1][i]+speed_cognitive+speed_social
            else:
                raise ('only constric, timew, globw, are allowed for speed_mech, the mechanism used is not defined')
            
            #***********************************   
            #Update Position based on data type
            #************************************
            if self.datatype[i].strip() == 'float':
                new_particle[0][i]=particle[0][i]+new_particle[1][i]
            
                # adjust maximum position if necessary
                if new_particle[0][i] > self.up[i]:
                    new_particle[0][i]=self.up[i]
    
                # adjust minimum position if neseccary
                if new_particle[0][i] < self.low[i]:
                    new_particle[0][i]=self.low[i]
            
            elif self.datatype[i].strip() == 'int':
                move_cond=(random.random() < self._sigmoid(new_particle[1][i])) * 1    #returns binary 0/1              
                if move_cond: #perturb this parameter
                    
                    if int(self.low[i]) == int(self.up[i]):
                        new_particle[0][i] = int(self.low[i])
                    else:
                        # make a list of possiblities after excluding the current value to enforce mutation
                        choices=list(range(self.low[i],self.up[i]+1))
                        choices.remove(new_particle[0][i])
                        # randint is NOT used here since it could re-draw the same integer value, choice is used instead
                        new_particle[0][i] = random.choice(choices)
            else:
                raise Exception('The particle position in PSO cannot be modified, either int/float data types are allowed')
                     
                     
        return new_particle

    def _sigmoid(self, x):
        """
        Helper method for the sigmoid function
        Input:
            x (scalar or numpy.ndarray): input attribute(s)
        Returns:
            scalar or numpy.ndarray: output sigmoid computation
        """
        return 1 / (1 + np.exp(-x))

    def gen_object(self, inp):
        """
        Worker for pool process, just to run the fitness function
        """
        return self.fit(inp[0])

    def select(self, pop, k=1):
        """
        Reorder the swarm and select the best `k` particles from it
        Input:
            -pop (dict): swarm of particles 
            -k (int): number of particles to survive [ k < len(pop) ]
        Returns:
            -best_dict (dict): dictionary of the best k particles in the swarm
        """
        pop=list(pop.items())
        pop.sort(key=lambda e: e[1][2], reverse=True)
        sorted_dict=dict(pop[:k])
        #Next block creates a new dict where keys are reset to 0 ... k in order to avoid unordered keys after sort
        best_dict=defaultdict(list)
        index=0
        for key in sorted_dict:
            for j in range (5): 
                #5 refers to the properties of each particle in order
                #0: current pos, 1: speed, 2: current fitness, 3: best local pos, 4: best local fitness
                best_dict[index].append(sorted_dict[key][j])
            index+=1
        
        sorted_dict.clear()
        return best_dict

    def GenSwarm(self, swm):
        """
        Generate the new swarm (offspring) based on the old swarm, 
        by looping and updating all particles
        Input:
            swm (dict): current swarm 
        Return:
            offspring (dict): new updated swarm
        """
        offspring = defaultdict(list)
        for i in range(len(swm)):
            
            offspring[i] = self.UpdateParticle(particle=swm[i], local_pos=self.local_pos[i], local_fit=self.local_fit[i])
            offspring[i][2] = 0  #this fitness item is set to zero since it is not evaluated yet by self.fit
    
        return offspring

    def evolute(self, swarm, ngen, local_pos, local_fit, swm_best=None, 
                exstep=None, exsteps=None, mu=None, caseids=None, verbose=0):
        """
        This is the PSO evolutionary algorithm.
        
        Inputs:
            swarm (dict): initial swarm population (position, speed, fitness for all particles)
            ngen (int): number of generations to evolute
            local_pos (list of lists): best local position of each particle (same as initial position for standalone PSO)
            local_fit (list): best local fitness of each particle (same as initial fitness for standalone PSO)
            -swm_best (list of 2): option to overwrite swarm best position/fitness externally 
                                   (ONLY NEEDED FOR PESA) --> default is to keep the observed data for swarm
                                    swm_best[0] --> initial position of the swarm (list)
                                    swm_best[1] --> initial fitness of the swarm  (float)
            -exstep (int), exsteps (int): current and total number of steps to anneal w 
                                          only used when speed_mech=`timew`. 
                                          (ONLY NEEDED FOR PESA)
            -mu: number of particles to survive for next generation (ONLY USED FOR PESA).
                 by default all particles survive
            -caseids (list of strings): to correspond each evaluated particle with caseid
            -verbose (bool): print statistics to screen
            
        Returns:
            population (dict): last swarm population (position, speed, fitness for all particles)
            swm_pos (list):  best swarm position
            swm_fit (float): best swarm fitness
        """
        
        #------------------------
        # Parameters
        #------------------------
        self.local_pos=local_pos
        self.local_fit=local_fit
        # This option is for testing only to make the swarm biased to the PESA memory 
        biased_swarm=False
        
        if not mu:
            # this is only if PSO is used in PESA,
            #if standalone keep the whole swarm without cutting it (mu=n_particles)
            mu=self.npar  
                         
        #enforce swm_best from external if this variable is activated
        # the values will overwrite the ones defined in the class
        if swm_best:
            self.swm_pos=copy.deepcopy(swm_best[0])
            self.swm_fit=swm_best[1]
        
        if caseids:
            assert len(caseids) == len(swarm) * ngen, \
            'caseids length ({}) is not equal to total number of cases to run: swarm_size ({}) * ngen ({})'. format(len(caseids),len(swarm) ,ngen)
            case_idx=0
        #-----------------------------
        # Begin the evolution process
        #-----------------------------
        for gen in range(1, ngen + 1):
            
            #--biased swarm overwrite
            if biased_swarm:
                # This option will force the swarm to follow the memory data, more biased
                best_ind=local_fit.index(max(local_fit))
                if local_fit[best_ind] > self.swm_fit:
                    self.swm_fit = local_fit[best_ind]
                    self.swm_pos=copy.deepcopy(local_pos[best_ind])
                    
            #--Vary the particles and generate new offspring/swarm
            offspring = self.GenSwarm(swm=swarm)
            
            #***************************
            #Parallel: Evaluate the particles 
            # with multiprocessign Pool
            #***************************
            if self.ncores > 1:
                t0=time.time()
                core_list=[]
                for key in offspring:
                    core_list.append([offspring[key][0],caseids[case_idx]])
                    case_idx+=1
                
                #initialize a pool
                p=MyPool(self.ncores)
                fitness = p.map(self.gen_object, core_list)
                p.close(); p.join()

                #with joblib.Parallel(n_jobs=self.ncores) as parallel:
                #    fitness=parallel(joblib.delayed(self.gen_object)(item) for item in core_list)
                
                self.partime=time.time()-t0
                #print('PSO:', self.partime)
                
                #append fitness data to the swarm dict
                for k in range(len(offspring)):
                    offspring[k][2] = fitness[k]
                    
                #check and update local best
                for par in range (len(fitness)):
                    if fitness[par] > self.local_fit[par]:
                        self.local_pos[par]= copy.deepcopy(offspring[par][0])
                        self.local_fit[par]= fitness[par]
                        #check and update global/swarm best
                        if fitness[par] > self.swm_fit:
                            self.swm_fit = fitness[par]
                            self.swm_pos=copy.deepcopy(offspring[par][0])
            #***************************
            #Serial: no Pool
            #***************************
            else: 
                for par in range(len(offspring)):
                    if caseids:
                        fitness=self.fit(offspring[par][0])
                        case_idx+=1
                    else:
                        fitness=self.fit(offspring[par][0])
                    offspring[par][2]=fitness
            
                    #check and update local best
                    if fitness > self.local_fit[par]:
                        self.local_pos[par]= copy.deepcopy(offspring[par][0])
                        self.local_fit[par]= fitness
                        #check and update global/swarm best
                        if fitness > self.swm_fit:
                            self.swm_fit = fitness
                            self.swm_pos=copy.deepcopy(offspring[par][0])
                
                self.partime=0
                            
            if self.speed_mech=='timew':
                if exstep and exsteps:
                    # this is a block when PESA is used, current global step MUST be provided
                    step=exstep
                    totsteps=exsteps
                else:
                    # this is a block for standalone PSO, current step can be inferred from the loop
                    step=gen*self.npar
                    totsteps=ngen*self.npar
                
                self.w = self.wmax - (self.wmax-self.wmin)*step/totsteps
                #print('timew', self.w)
            swarm = copy.deepcopy(offspring)            
            
            # Print data
            if verbose:
                mean_speed=[np.mean(swarm[i][1]) for i in swarm]
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                print('PSO step {}/{}, C1={}, C2={}, W={}, Particles={}'.format(gen*self.npar, ngen*self.npar, np.round(self.c1,2), np.round(self.c2,2), np.round(self.w,2), self.npar))
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                print('Statistics for generation {}'.format(gen))
                print('Best Swarm Fitness:', np.round(self.swm_fit,2))
                print('Best Swarm Position:', np.round(self.swm_pos,2))
                print('Max Speed:', np.round(np.max(mean_speed),3))
                print('Min Speed:', np.round(np.min(mean_speed),3))
                print('Average Speed:', np.round(np.mean(mean_speed),3))
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        
        # Append local data to the final swarm population, items [3] and [4] in the dictionary
        for par in range(len(swarm)):
            swarm[par].append(self.local_pos[par])
            swarm[par].append(self.local_fit[par])
        
        #Select and order the last population 
        population=copy.deepcopy(self.select(pop=swarm, k=mu))
        return population, self.swm_pos, self.swm_fit, self.partime