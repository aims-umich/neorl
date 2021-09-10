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
#Created on Mon Jun 15 19:37:04 2020
#@author: Majdi Radaideh

import random
import numpy as np
from collections import defaultdict
import copy
import time
import joblib
from neorl.evolu.discrete import mutate_discrete, encode_grid_to_discrete, decode_discrete_to_grid

class PSO:
    """
    Parallel Particle Swarm Optimisaion (PSO) module
	 
    :param mode: (str) problem type, either "min" for minimization problem or "max" for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param fit: (function) the fitness function 
    :param npar: (int) number of particles in the swarm
    :param c1: (float) cognitive speed constant 
    :param c2: (float) social speed constant 
    :param speed_mech: (str) type of speed mechanism to update particle velocity, choose between ``constric``, ``timew``, ``globw``.			
    :param ncores: (int) number of parallel processors
    :param seed: (int) random seed for sampling
    """
    def __init__ (self, mode, bounds, fit, npar=50, c1=2.05, c2=2.05, speed_mech='constric', ncores=1, seed=None):  

        if seed:
            random.seed(seed)
            self.seed=seed
        
        self.bounds=bounds
        self.npar=npar
        
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
              
        self.ncores=ncores
        self.speed_mech=speed_mech
        self.c1=c1
        self.c2=c2
        self.size=len(bounds)
        
        self.v0=0.1 # factor to intialize the speed
        
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
        
        self.low = np.array([self.bounds[item][1] for item in self.bounds])
        self.up = np.array([self.bounds[item][2] for item in self.bounds])
                        
    def GenParticle(self, bounds):
        #"""
        #Particle generator
        #Input: 
        #    -bounds (dict): input paramter type and lower/upper bounds in dictionary form
        #Returns: 
        #    -particle (list): particle position
        #    -speed (list): particle speed
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
        particle=list(content)
        speed = list(self.v0*np.array(content))
        return particle, speed

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
                    
        fitness = self.fit(x)
        return fitness
    
    def InitSwarm(self, x0=None):
        #"""
        #Swarm intializer 
        #Inputs:
        #    -warmup (int): number of individuals to create and evaluate initially
        #Returns 
        #    -pop (dict): initial swarm in a dictionary form, looks like this:
        #        
        #    pop={particle key: [particle position, particle velocity, particle fitness]}
        #    pop={0: [[1,2,3,4,5], [0.1,0.2,0.3,0.4,0.5], 1.2], 
        #         ... 
        #         99: [[1.1,2.1,3.1,4.1,5.1], [0.1,0.2,0.3,0.4,0.5], 5.2]}
        #   
        #"""
        #initialize the swarm and velocity and run them in parallel (these samples will be used to initialize the swarm)

            
        pop=defaultdict(list)
        # dict key runs from 0 to self.npar-1
        # index 0: individual, index 1: velocity, index 2: fitness 
        
        #Establish the swarm
        if x0:
            print('The first particle provided by the user:', x0[0])
            print('The last particle provided by the user:', x0[-1])
            for i in range(len(x0)):
                pop[i].append(x0[i])
                speed = list(self.v0*np.array(x0[i]))
                pop[i].append(speed)
        else:
            for i in range (self.npar):
                particle, speed=self.GenParticle(self.bounds)
                pop[i].append(particle)
                pop[i].append(speed)
        
        #Evaluate the swarm
        if self.ncores > 1:  #evaluate swarm in parallel
            core_list=[]
            for particle in pop:
                core_list.append(pop[particle][0])

            with joblib.Parallel(n_jobs=self.ncores) as parallel:
                fitness=parallel(joblib.delayed(self.fit_worker)(item) for item in core_list)
                
            [pop[particle].append(fitness[particle]) for particle in range(len(pop))]
        
        else: #evaluate swarm in series
            for particle in pop:
                fitness=self.fit_worker(pop[particle][0])
                pop[particle].append(fitness)
                
        #Setup the local position and fitness for PSO calculations
        local_pos=[]
        local_fit=[]
        for particle in pop:
            local_pos.append(pop[particle][0])
            local_fit.append(pop[particle][2])
        
        return pop, local_pos, local_fit  #return final pop dictionary with particle, velocity, and fitness
    
    def UpdateParticle(self, particle, local_pos, local_fit):
        #"""
        #Function that updates the particle speed and position based on the 
        #best local positon of the particle and best global position of the swarm
        #The function works with both int or float variables
        #Input: 
        #    particle (list of lists):
        #            particle[0] (list) = current position
        #            particle[1] (list) = speed 
        #            particle[2] (list) = current fit 
        #    local_pos (list): best local position observed for this particle
        #
        #Return:
        #    new_particle (list of lists): modified particle with same structure as `particle` 
        #"""  
        
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
        #"""
        #Helper method for the sigmoid function
        #Input:
        #    x (scalar or numpy.ndarray): input attribute(s)
        #Returns:
        #    scalar or numpy.ndarray: output sigmoid computation
        #"""
        return 1 / (1 + np.exp(-x))

    def select(self, pop, k=1):
        #"""
        #Reorder the swarm and select the best `k` particles from it
        #Input:
        #    -pop (dict): swarm of particles 
        #    -k (int): number of particles to survive [ k < len(pop) ]
        #Returns:
        #    -best_dict (dict): dictionary of the best k particles in the swarm
        #"""
        pop=list(pop.items())
        pop.sort(key=lambda e: e[1][2], reverse=True)
        sorted_dict=dict(pop[:k])
        #Next block creates a new dict where keys are reset to 0 ... k in order to avoid unordered keys after sort
        best_dict=defaultdict(list)
        index=0
        for key in sorted_dict:
            for j in range (3): 
                #5 refers to the properties of each particle in order
                #0: current pos, 1: speed, 2: current fitness, 3: best local pos, 4: best local fitness
                best_dict[index].append(sorted_dict[key][j])
            index+=1
        
        sorted_dict.clear()
        return best_dict

    def GenSwarm(self, swm):
        #"""
        #Generate the new swarm (offspring) based on the old swarm, 
        #by looping and updating all particles
        #Input:
        #    swm (dict): current swarm 
        #Return:
        #    offspring (dict): new updated swarm
        #"""
        offspring = defaultdict(list)
        for i in range(len(swm)):
            
            offspring[i] = self.UpdateParticle(particle=swm[i], local_pos=self.local_pos[i], local_fit=self.local_fit[i])
            offspring[i][2] = 0  #this fitness item is set to zero since it is not evaluated yet by the fitness
    
        return offspring

    def evolute(self, ngen, x0=None, verbose=True):
        """
        This function evolutes the PSO algorithm for number of generations.
        
        :param ngen: (int) number of generations to evolute
        :param x0: (list of lists) the initial position of the swarm particles
        :param verbose: (bool) print statistics to screen
        
        :return: (dict) dictionary containing major PSO search results
        """
            
        self.best_scores=[]
        if x0:
            #get the initial swarm position from the user, it has to be 
            #print('-- Using The Initial PSO Swarm from the User')
            assert len(x0) == self.npar, '--error: the length of x0 ({}) (initial swarm) must equal to number of particles ({})'.format(len(x0), self.npar)
            swarm, self.local_pos, self.local_fit=self.InitSwarm(x0=x0)
        else:
            #print('-- Using A Random Initial PSO Swarm')
            #generate the initial swarm internally, assign all variables
            swarm, self.local_pos, self.local_fit=self.InitSwarm()
        
        swm0=self.select(swarm, k=1)
        self.swm_pos=swm0[0][0]
        self.swm_fit=swm0[0][2]

        #-----------------------------
        # Begin the evolution process
        #-----------------------------
        for gen in range(1, ngen + 1):
                    
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
                    core_list.append(offspring[key][0])

                with joblib.Parallel(n_jobs=self.ncores) as parallel:
                    fitness=parallel(joblib.delayed(self.fit_worker)(item) for item in core_list)
                
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
                    
                    fitness=self.fit_worker(offspring[par][0])
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
                step=gen*self.npar
                totsteps=ngen*self.npar
                self.w = self.wmax - (self.wmax-self.wmin)*step/totsteps
                #print('timew', self.w)
            
            fit_best=np.max([offspring[item][2] for item in offspring])  #get the max fitness for this generation
            self.best_scores.append(fit_best)
            swarm = copy.deepcopy(offspring) 
            
            #--mir
            if self.mode=='min':
                self.swm_fit_correct=-self.swm_fit
            else:
                self.swm_fit_correct=self.swm_fit
            
            # Print data
            if verbose:
                mean_speed=[np.mean(swarm[i][1]) for i in swarm]
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                print('PSO step {}/{}, C1={}, C2={}, W={}, Particles={}, Ncores={}'.format(gen*self.npar, ngen*self.npar, np.round(self.c1,2), np.round(self.c2,2), np.round(self.w,2), self.npar, self.ncores))
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                print('Statistics for generation {}'.format(gen))
                print('Best Swarm Fitness:', np.round(self.swm_fit_correct,6))
                if self.grid_flag:
                    self.swm_decoded = decode_discrete_to_grid(self.swm_pos, self.orig_bounds, self.bounds_map)
                    print('Best Swarm Position:', self.swm_decoded,6)
                else:
                    print('Best Swarm Position:', self.swm_pos)             
                print('Max Speed:', np.round(np.max(mean_speed),3))
                print('Min Speed:', np.round(np.min(mean_speed),3))
                print('Average Speed:', np.round(np.mean(mean_speed),3))
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        #Select and order the last population 
        population=copy.deepcopy(self.select(pop=swarm, k=self.npar))
        
        # Append local data to the final swarm population, items [3] and [4] in the dictionary
        for par in range(len(swarm)):
            swarm[par].append(self.local_pos[par])
            swarm[par].append(self.local_fit[par])
        
        #mir-grid
        if self.grid_flag:
            self.swm_pos_correct=decode_discrete_to_grid(self.swm_pos,self.orig_bounds,self.bounds_map)
        else:
            self.swm_pos_correct=self.swm_pos
                
        if verbose:
            print('------------------------ PSO Summary --------------------------')
            print('Best fitness (y) found:', self.swm_fit_correct)
            print('Best individual (x) found:', self.swm_pos_correct)
            print('--------------------------------------------------------------')
    
        #--mir
        if self.mode=='min':
            self.best_scores=[-item for item in self.best_scores]
                
        return self.swm_pos_correct, self.swm_fit_correct, self.best_scores