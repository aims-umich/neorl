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

# -*- encoding: utf-8 -*-
#'''
#@File    :   aco.py
#@Time    :   2021/07/30 16:10:59
#@Author  :   Xubo GU 
#@Email   :   guxubo@alumni.sjtu.edu.cn
#'''

#inspired from 
#https://towardsdatascience.com/introduction-to-global-optimization-algorithms-for-continuous-domain-functions-7ad9d01db055
#https://github.com/aidowu1/Py-Optimization-Algorithms/blob/master/AntColonyOptimization/Constants.py

import random
import numpy as np
import joblib

class ACO(object):
    """
    Ant Colony Optimization

    :param mode: (str) problem type, either "min" for minimization problem or "max" for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param fit: (function) the fitness function 
    :param nants: (int) number of total ants
    :param narchive: (int) size of archive of best ants (recommended ``narchive < nants``)
    :param Q: (float) diversification/intensification factor (see **Notes** below)
    :param Z: (float) deviation-distance ratio or pheromone evaporation rate, high Z leads to slow convergence (see **Notes** below).
    :param ncores: (int) number of parallel processors
    :param seed: (int) random seed for sampling
    """

    def __init__(self, mode, fit, bounds, nants=40, narchive=10,
                 Q=0.5, Z=1.0, ncores=1, seed=None):
        
        assert narchive <= nants, '--error: narchive must be less than or equal nants'
        self.seed=seed
        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)
            
        self.mode=mode
        if mode == 'min':
            self.fit=fit
        elif mode == 'max':
            def fitness_wrapper(*args, **kwargs):
                return -fit(*args, **kwargs)
            self.fit = fitness_wrapper
        else:
            raise ValueError('--error: The mode entered by user is invalid, \
                use either `min` or `max`')
            
        self.bounds = bounds
        self.npop = narchive
        self.nvars = len(bounds)
        self.nants = nants
        self.q = Q
        self.z = Z
        self.ncores = ncores
        
        self.__final_best_solution = None 
        self.__probs = None
        self.__new_pops = None   
        self.__random = np.random.RandomState(seed)
        self.pops_sorted = None

    def __computePdf(self) -> object:
        #"""
        #Computes the PDF values
        #"""
        points = np.array(range(self.npop), dtype=np.float)
        # Solution Weights
        w = 1/(np.sqrt(2*np.pi)*self.q*float(self.npop))*np.square(np.exp(-0.5*((points-1)/(self.q*float(self.npop)))))
        return w

    def __rouletteWheelSelection(self):
        #"""
        #Roulette wheel selection strategy for selecting the optimal Guassain Kernel
        #"""
        r = self.__random.rand()
        c = np.cumsum(np.reshape(self.__probs, (-1)))
        j = np.argwhere(r <= c)[0,0]
        return j

    def ensure_bounds(self, vec): # bounds check
        vec=vec.flatten()
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
        return np.array(vec_new)
                      
    def evolute(self, ngen, x0=None, verbose=0):
        """
        This function evolutes the ACO algorithm for number of generations
        
        :param ngen: (int) number of generations to evolute
        :param x0: (list of lists) the initial individuals of the population
        :param verbose: (bool) print statistics to screen
        
        :return: (tuple) (best position, best fitness, and list of fitness history)
        """
        if x0:
            assert len(x0) == self.nants, '--error: the length of x0 ({}) (initial population) must equal to number of ants ({})'.format(len(x0), self.nants)
            pops = Populations(self.nants, self.nvars, self.fit, self.bounds, ncores=self.ncores, x0=x0)
        else:
            pops = Populations(self.nants, self.nvars, self.fit, self.bounds, ncores=self.ncores, x0=None)
            
        fit_hist=[]
        self.ngen = ngen
        self.__best_solutions = [None]*self.ngen
        self.pops_sorted = pops.ant_populations     # length 10
        self.__final_best_solution = self.pops_sorted[0]
        self.__w = self.__computePdf()
        self.__probs = self.__w/np.sum(self.__w)
        self.__means = np.zeros((self.npop, self.nvars))
        self.__sigmas = np.zeros((self.npop, self.nvars))

        for iter in range(self.ngen):
            ## self.__constructNewPopulationSolution()
            # Means
            for _ in range(self.npop):
                self.__means[_, :] = np.reshape(self.pops_sorted[_].position, (1, -1))  

            # Standard Deviation
            for l_i in range(self.npop):
                d = 0.0
                for r_i in range(self.npop):
                    d += np.abs(self.__means[l_i, :] - self.__means[r_i, :])
                self.__sigmas[l_i, :] = (self.z * d) / (self.npop - 1) 
            
            self.__new_pops = Populations.createEmptyNewPopulations(self.nants, self.nvars)
            for i in range(self.nants):
                for j in range(self.nvars):
                    # Select Gaussian Kernel
                    k = self.__rouletteWheelSelection()
                    # Generate Gaussian Random Variable
                    self.__new_pops[i].position[j] = self.__means[k, j] + self.__sigmas[k, j] * self.__random.randn()
                # Apply Variable Bounds
                self.__new_pops[i].position=self.ensure_bounds(self.__new_pops[i].position)
                
            # Evaluation     
            if self.ncores > 1:
                with joblib.Parallel(n_jobs=self.ncores) as parallel:
                    temp_fitness=parallel(joblib.delayed(self.fit)(indv.position) for indv in self.__new_pops)
                for i in range(self.nants):
                    self.__new_pops[i].cost_function = temp_fitness[i]
            else:
                for i in range(self.nants):
                    self.__new_pops[i].cost_function = self.fit(self.__new_pops[i].position)
                
                    
            # Merge Main Population (Archive) and New Population (Samples)
            self.pops_sorted = self.pops_sorted + self.__new_pops
            # Sort Population
            self.pops_sorted = sorted(self.pops_sorted, key=lambda x: x.cost_function, reverse=False) # small -> large
            # Delete Extra Members
            self.pops_sorted = self.pops_sorted[:self.npop]
            # Update Best Solution Ever Found
            self.__final_best_solution = self.pops_sorted[0]
            # Store Best Cost
            self.__best_solutions[iter] = self.__final_best_solution
            
            #show the value wrt min/max
            if self.mode=='max':
                y_print = -float(self.__final_best_solution.cost_function)       
            else:
                y_print = float(self.__final_best_solution.cost_function)
            
            fit_hist.append(y_print)
            if verbose:
                print('************************************************************')
                print('ACOR step {}/{}, Ncores={}'.format(iter+1, self.ngen, self.ncores))
                print('************************************************************')
                print('Best fitness:', np.round(y_print,6))
                print('Best individual:', self.__final_best_solution.position.flatten())
                print('Archive mean individual:', self.__means.mean(axis=0))
                print('************************************************************')


        if verbose:
            print('------------------------ ACOR Summary --------------------------')
            print('Best fitness (y) found:', y_print)
            print('Best individual (x) found:', self.__best_solutions[-1].position.flatten())
            print('--------------------------------------------------------------')
        
        x_best=self.__best_solutions[-1].position.flatten()
        y_best=y_print
        return x_best, y_best, fit_hist
    
    @property
    def pops(self):
        #"""
        #Getter property of the 'self.__pops' attribute
        #"""
        return self.pops_sorted

    @property
    def new_pops(self):
        #"""
        #Getter property of the 'self.__new_pops' attribute
        #"""
        return self.__new_pops

    @property
    def final_best_solution(self):
        #"""
        #Getter property of the 'self.__best_solution' attribute
        #"""
        return self.__final_best_solution

    @property
    def best_solutions(self):
        #"""
        #Getter property of the 'self.__best_solutions' attribute
        #"""
        return self.__best_solutions

    @property
    def probs(self):
        #"""
        #Getter property of the 'self.__probs' attribute
        #"""
        return self.__probs

    @property
    def means(self):
        #"""
        #Getter property of the 'self.__meanss' attribute
        #"""
        return self.__means

    @property
    def sigmas(self):
        #"""
        #Getter property of the 'self.__sigmas' attribute
        #"""
        return self.__sigmas


class Population(object):
    #"""
    #Specifies an ACO Population
    #"""
    def __init__(self, position, cost_function) -> None:

        self.position = position
        self.cost_function = cost_function


class Populations(object):
    #"""
    #Specifies Populations of the Ant Colony i.e. the Archive Size
    #"""
    def __init__(self, n_pop, n_vars, fit, bounds, ncores, x0=None):
        #"""
        #Constructor
        #:param: n_pop: population size
        #:param: n_vars: number of variables
        #:param: cost_func_ cost function
        #:param: bounds: continious domain lower/upper bounds
        #"""
        self.fit=fit
        self.bounds = bounds
        self.npop = n_pop
        self.__pops = [None]*n_pop
        self.pops_sorted = None
        self.x0=x0
        self.ncores=ncores
        
    def __initializePopulation(self):
        #"""
        #Initializes the ant colony populations
        #"""
        position=[]
        for i in range(self.npop):
            
            if self.x0 is None:
                indv=[]
                for key in self.bounds:
                    if self.bounds[key][0] == 'int':
                        indv.append(random.randint(self.bounds[key][1], self.bounds[key][2]))
                    elif self.bounds[key][0] == 'float':
                        indv.append(random.uniform(self.bounds[key][1], self.bounds[key][2]))
                    elif self.bounds[key][0] == 'grid':
                        indv.append(random.sample(self.bounds[key][1],1)[0])
                position.append(np.array(indv))
            else:
                position.append(np.array(self.x0[i]))

        if self.ncores > 1:
            with joblib.Parallel(n_jobs=self.ncores) as parallel:
                cost=parallel(joblib.delayed(self.fit)(indv) for indv in position)                
        else:
            cost=[]
            for i in range(self.npop):
                cost.append(self.fit(position[i]))
                            
        for i in range(self.npop):
            self.__pops[i] = Population(position=position[i], cost_function=cost[i])

    @property
    def ant_populations(self):
        #"""
        #Getter property to retrieve the 'self.__pops' collection
        #"""
        self.__initializePopulation()        
        self.pops_sorted = sorted(self.__pops, key=lambda x: x.cost_function, reverse=False)
        return self.pops_sorted

    @staticmethod
    def createEmptyNewPopulations(n_ants: int, n_vars: int) -> object:
        #"""
        #Creates an empty collection of ant populations
        #:params: n_ants: number of ants
        #:params: n_vars: number of variables
        #"""
        position = np.zeros((n_vars, 1))
        empty_populations = [Population(position=position, cost_function=None) for _ in range(n_ants)]
        return empty_populations