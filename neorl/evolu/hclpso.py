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
#Created on Thu Dec  3 14:42:29 2020
#
#@author: Majdi
#"""

import random
import numpy as np
import joblib
from numpy import arange, dot, multiply, exp, ones, zeros, ceil
from neorl.evolu.discrete import mutate_discrete, encode_grid_to_discrete 
from neorl.evolu.discrete import decode_discrete_to_grid, encode_grid_indv_to_discrete
from neorl.utils.seeding import set_neorl_seed
from neorl.utils.tools import get_population, check_mixed_individual


class HCLPSO(object):
    """
    Heterogeneous comprehensive learning particle swarm optimization (HCLPSO)
    
    :param mode: (str) problem type, either ``min`` for minimization problem or ``max`` for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param fit: (function) the fitness function 
    :param g1: (int): number of particles in the exploration group
    :param g2: (int): number of particles in the exploitation group (total swarm size is ``g1 + g2``)
    :param int_transform: (str): method of handling int/discrete variables, choose from: ``nearest_int``, ``sigmoid``, ``minmax``.
    :param ncores: (int) number of parallel processors (must be ``<= g1+g2``)
    :param seed: (int) random seed for sampling
    """
    def __init__(self, mode, bounds, fit, g1=15, g2=25, int_transform='nearest_int', ncores=1, seed=None):
        
        set_neorl_seed(seed)
        
        assert ncores <= (g1+g2), '--error: ncores ({}) must be less than or equal total particles g1 + g2 ({})'.format(ncores, g1+g2)
        
        #--mir
        self.mode=mode
        if mode == 'min':
            self.fit=fit
        elif mode == 'max':
            def fitness_wrapper(*args, **kwargs):
                return -fit(*args, **kwargs) 
            self.fit=fitness_wrapper
        else:
            raise ValueError('--error: The mode entered by user is invalid, use either `min` or `max`')
          
        self.int_transform=int_transform
        if int_transform not in ["nearest_int", "sigmoid", "minmax"]:
            raise ValueError('--error: int_transform entered by user is invalid, must be `nearest_int`, `sigmoid`, or `minmax`')
            
        self.bounds=bounds
        self.ncores = ncores
        self.num_g1=g1
        self.num_g2=g2
        self.num_g=self.num_g1 + self.num_g2
        
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
        self.lb=np.array([self.bounds[item][1] for item in self.bounds])
        self.ub=np.array([self.bounds[item][2] for item in self.bounds])

    def init_sample(self, bounds):
    
        indv=[]
        for key in bounds:
            if bounds[key][0] == 'int':
                indv.append(random.randint(bounds[key][1], bounds[key][2]))
            elif bounds[key][0] == 'float':
                indv.append(random.uniform(bounds[key][1], bounds[key][2]))
            #elif bounds[key][0] == 'grid':
            #    indv.append(random.sample(bounds[key][1],1)[0])
            else:
                raise Exception ('unknown data type is given, either int, float, or grid are allowed for parameter bounds')   
        return np.array(indv)

    def eval_particles(self):
    
        #---------------------
        # Fitness calcs
        #---------------------
        core_lst=[]
        for case in range (0, self.Positions.shape[0]):
            core_lst.append(self.Positions[case, :])
    
        if self.ncores > 1:

            with joblib.Parallel(n_jobs=self.ncores) as parallel:
                fitness_lst=parallel(joblib.delayed(self.fit_worker)(item) for item in core_lst)
                
        else:
            fitness_lst=[]
            for item in core_lst:
                fitness_lst.append(self.fit_worker(item))
        
        return fitness_lst

    def select(self, pos, fit):
        
        best_fit=np.min(fit)
        min_idx=np.argmin(fit)
        best_pos=pos[min_idx,:].copy()
        
        return best_pos, best_fit 
        
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
        #This worker is for parallel calculations
        
        # Clip the salp with position outside the lower/upper bounds and return same position
        x=self.ensure_bounds(x)
        
        if self.grid_flag:
            #decode the individual back to the int/float/grid mixed space
            x=decode_discrete_to_grid(x,self.orig_bounds,self.bounds_map)
        
        # Calculate objective function for each search agent
        fitness = self.fit(x)
        
        return fitness
    
    def ensure_discrete(self, vec):
        #"""
        #to mutate a vector if discrete variables exist 
        #handy function to be used within alg phases

        #Params:
        #vec - salp position in vector/list form

        #Return:
        #vec - updated salp position vector with discrete values
        #"""
        
        for dim in range(self.dim):
            if self.var_type[dim] == 'int':
                vec[dim] = mutate_discrete(x_ij=vec[dim], 
                                               x_min=min(vec),
                                               x_max=max(vec),
                                               lb=self.lb[dim], 
                                               ub=self.ub[dim],
                                               alpha=self.a,
                                               method=self.int_transform,
                                               )
        
        return vec

    def UpdateParticles(self, a, b, friend_num, check_slope=False):
        
        #first particle index,
        #last particle index
        #number of particles to sample a friend
        
        for i in range(a,b):

            if check_slope:
                slope_cond=self.obj_func_slope[i] > 5
            else:
                slope_cond=True
                    
            if slope_cond:
                self.fri_best[i,:]=dot(i,ones((1,self.dim)))
                friend1=ceil(dot(friend_num,np.random.uniform(size=(self.dim)))) - 1
                friend2=ceil(dot(friend_num,np.random.uniform(size=(self.dim)))) - 1
                friend=multiply((self.pbest_val[friend1.astype(int)] < self.pbest_val[friend2.astype(int)]),friend1) \
                    + multiply((self.pbest_val[friend1.astype(int)] >= self.pbest_val[friend2.astype(int)]),friend2)
                toss=ceil(np.random.uniform(size=(self.dim)) - self.Pc[:,i].T)
                if np.all(toss == ones((self.dim))):
                    temp_index=np.random.choice(range(self.dim), self.dim, replace=False)
                    toss[temp_index[0]]=0
                self.fri_best[i,:]=multiply((1 - toss),friend) + multiply(toss,self.fri_best[i,:])
                for d in range(self.dim):
                    self.fri_best_pos[i,d]=self.pbest_pos[int(self.fri_best[i,d]),d]
                    
                if check_slope:
                    self.obj_func_slope[i]=0
            

    def evolute(self, ngen, x0=None, verbose=False):
        """
        This function evolutes the HCLPSO algorithm for a number of generations.
        
        :param ngen: (int) number of generations to evolute
        :param x0: (list of lists) initial position of the particles (must be of same size as ``g1 + g2``)
        :param verbose: (bool) print statistics to screen
        
        :return: (tuple) (best individual, best fitness, and dictionary containing major search results)
        """
        self.history = {'local_fitness':[], 'global_fitness':[], 'c1': [], 'c2': [], 'K': []}
        self.best_fitness=float("inf") 
        self.verbose=verbose
        self.Positions = np.zeros((self.num_g, self.dim))
        self.a=1
        if x0:
            assert len(x0) == self.num_g, '--error: the length of x0 ({}) MUST equal the number of particles in the group `g1+g2 `({})'.format(len(x0), self.num_g)
            for i in range(self.num_g):
                check_mixed_individual(x=x0[i], bounds=self.orig_bounds) #assert the type provided is consistent
                if self.grid_flag:
                    self.Positions[i,:] = encode_grid_indv_to_discrete(x0[i], bounds=self.orig_bounds, bounds_map=self.bounds_map)
                else:
                    self.Positions[i,:] = x0[i]
        else:

            for i in range(self.num_g):
                self.Positions[i,:]=self.init_sample(self.bounds)

        self.check_vel=np.zeros((self.num_g, ngen))
        j=np.linspace(0,1,self.num_g)
        j=np.dot(j,10)
        
        #learning probability Pc
        self.Pc=np.dot(ones((self.dim,1)),(0.0 + (multiply((0.25),(exp(j)- exp(j[0]))) / (exp(j[self.num_g-1]) - exp(j[0]))))[np.newaxis])
        self.Weight=0.99 - dot((arange(ngen)),0.79) / ngen  #inertia weight
        self.K=3 - dot((arange(ngen)),1.5) / ngen    # constriction coeff
        self.c1=2.5 - dot((arange(ngen)),2) / ngen   #cognitive coeff
        self.c2=0.5 + dot((arange(ngen)),2) / ngen    #social coeff
        
        # Initialization
        self.range_min=np.tile(self.lb,(self.num_g,1))
        self.range_max=np.tile(self.ub,(self.num_g,1))
        interval=self.range_max - self.range_min
        v_max=dot(interval,0.2)
        v_min=-v_max
        self.Positions=self.range_min + multiply(interval,np.random.uniform(size=(self.num_g,self.dim)))
        self.vel=v_min + multiply((v_max - v_min),np.random.uniform(size=(self.num_g,self.dim)))
        
        #ensure discrete mutation
        for iii in range(self.Positions.shape[0]):
            self.Positions[iii, :] = self.ensure_bounds(self.Positions[iii, :])
            self.Positions[iii, :] = self.ensure_discrete(self.Positions[iii, :])
                
        fitness0=self.eval_particles()
        self.gbest_pos, self.gbest_val = self.select(self.Positions, fitness0)
        
        self.pbest_pos=self.Positions.copy()
        self.pbest_val=np.array(fitness0)
        self.fri_best_pos=zeros((self.num_g,self.dim))
        self.fri_best=dot((arange(self.num_g)[np.newaxis]).T,ones((1,self.dim)))
        self.obj_func_slope=zeros((self.num_g))
        
        #Updating particles for group 1 (exploration)
        self.UpdateParticles(a=0, b=self.num_g1, friend_num=self.num_g1, check_slope=False)

        #Updating particles for group 2 (exploitation)
        self.UpdateParticles(a=self.num_g1, b=self.num_g, friend_num=self.num_g, check_slope=False)        
                       
        for k in range(ngen):
            
            self.a= 1 - k * ((1) / ngen)  #mir: a decreases linearly between 1 to 0, for discrete mutation
            
            #----------------------------
            #  Position/Veclocity Update
            #----------------------------
            
            #group 1 position estimate
            delta_g1=(multiply(multiply(self.K[k],np.random.uniform(size=(self.num_g1,self.dim))),(self.fri_best_pos[:self.num_g1,:] - self.Positions[:self.num_g1,:])))
            vel_g1=dot(self.Weight[k],self.vel[:self.num_g1,:]) + delta_g1
            vel_g1=(multiply((vel_g1 < v_min[:self.num_g1,:]),v_min[:self.num_g1,:]))  \
                    + (multiply((vel_g1 > v_max[:self.num_g1,:]),v_max[:self.num_g1,:])) \
                    + (multiply((np.logical_and((vel_g1 < v_max[:self.num_g1,:]),(vel_g1 > v_min[:self.num_g1,:]))),vel_g1))
            pos_g1=self.Positions[:self.num_g1,:] + vel_g1
            
            #group 2 position estimate
            gbest_pos_temp=np.tile(self.gbest_pos,(self.num_g2,1))
            delta_g2=(multiply(multiply(self.c1[k],np.random.uniform(size=(self.num_g2,self.dim))),(self.fri_best_pos[self.num_g1:,:] - self.Positions[self.num_g1:,:]))) \
                     + (multiply(multiply(self.c2[k],np.random.uniform(size=(self.num_g2,self.dim))),(gbest_pos_temp - self.Positions[self.num_g1:,:])))
            
            vel_g2=dot(self.Weight[k],self.vel[self.num_g1:,:]) + delta_g2
            vel_g2=(multiply((vel_g2 < v_min[self.num_g1:,:]),v_min[self.num_g1:,:])) \
                    + (multiply((vel_g2 > v_max[self.num_g1:,:]),v_max[self.num_g1:,:])) \
                    + (multiply((np.logical_and((vel_g2 < v_max[self.num_g1:,:]),(vel_g2 > v_min[self.num_g1:,:]))),vel_g2))
            pos_g2=self.Positions[self.num_g1:,:] + vel_g2
            
            #whole group concatenatation
            self.Positions=np.concatenate((pos_g1, pos_g2), axis=0)
            #ensure discrete mutation
            for iii in range(self.Positions.shape[0]):
                self.Positions[iii, :] = self.ensure_bounds(self.Positions[iii, :])
                self.Positions[iii, :] = self.ensure_discrete(self.Positions[iii, :])
            
            self.vel=np.concatenate((vel_g1, vel_g2), axis=0)
                    
            #----------------------
            #  Evaluate New Particles
            #----------------------
            fitness=self.eval_particles()
            for i, fits in enumerate(fitness):
                #save the best of the best!!!
                if fits < self.best_fitness:
                    self.best_fitness=fits
                    self.best_position=self.Positions[i, :].copy()

                if fits < self.pbest_val[i]:
                    self.pbest_pos[i,:]=self.Positions[i, :].copy()
                    self.pbest_val[i]=fits
                    self.obj_func_slope[i]=0
                else:
                    self.obj_func_slope[i]=self.obj_func_slope[i] + 1
                    
                if self.pbest_val[i] < self.gbest_val:
                    self.gbest_pos=self.pbest_pos[i,:].copy()
                    self.gbest_val=self.pbest_val[i]
            
            #Updating particles for group 1 (exploration)
            self.UpdateParticles(a=0, b=self.num_g1, friend_num=self.num_g1, check_slope=True)  #check slop is true
    
            #Updating particles for group 2 (exploitation)
            self.UpdateParticles(a=self.num_g1, b=self.num_g, friend_num=self.num_g-1, check_slope=True)  #check slop is true       
                
            #--mir
            if self.mode=='max':
                self.fitness_best_correct=-self.best_fitness
                self.local_fitness=-np.min(fitness)
            else:
                self.fitness_best_correct=self.best_fitness
                self.local_fitness=np.min(fitness)

            self.last_pop=self.Positions.copy()
            self.last_fit=np.array(fitness).copy()            
            self.history['local_fitness'].append(self.local_fitness)
            self.history['global_fitness'].append(self.fitness_best_correct)
            self.history['c1'].append(self.c1[k])
            self.history['c2'].append(self.c2[k])
            self.history['K'].append(self.K[k])
            
            # Print statistics
            if self.verbose:
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                print('HCLPSO step {}/{}, npar = g1+g2 ={}, Ncores={}'.format((k+1)*self.num_g, ngen*self.num_g, self.num_g, self.ncores))
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                print('Best Particle Fitness:', np.round(self.fitness_best_correct,6))
                if self.grid_flag:
                    self.particle_decoded = decode_discrete_to_grid(self.best_position, self.orig_bounds, self.bounds_map)
                    print('Best Particle Position:', self.particle_decoded)
                else:
                    print('Best Particle Position:', self.best_position)
                print('c1:', self.c1[k])
                print('c2:', self.c2[k])
                print('K:', self.K[k])
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        #mir-grid
        if self.grid_flag:
            self.hclpso_correct = decode_discrete_to_grid(self.best_position, self.orig_bounds, self.bounds_map)
        else:
            self.hclpso_correct = self.best_position                

        if self.mode=='max':
            self.last_fit=-self.last_fit
        
        #--mir return the last population for restart calculations
        if self.grid_flag:
            self.history['last_pop'] = get_population(self.last_pop, fits=self.last_fit, grid_flag=True, 
                                                     bounds=self.orig_bounds, bounds_map=self.bounds_map)
        else:
            self.history['last_pop'] = get_population(self.last_pop, fits=self.last_fit, grid_flag=False)
        
        if self.verbose:
            print('------------------------ HCLPSO Summary --------------------------')
            print('Best fitness (y) found:', self.fitness_best_correct)
            print('Best individual (x) found:', self.hclpso_correct)
            print('--------------------------------------------------------------')  
            
        return self.hclpso_correct, self.fitness_best_correct, self.history

