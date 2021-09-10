# -*- coding: utf-8 -*-
#"""
#Created on Thu Dec  3 14:42:29 2020
#
#@author: Katelin Du
#"""

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

# Harris hawks optimization: Algorithm and applications
# Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris,
# Ibrahim Aljarah, Majdi Mafarja, Huiling Chen
# Future Generation Computer Systems,
# DOI: https://doi.org/10.1016/j.future.2019.02.028

import random
import numpy as np
import math
import joblib
from neorl.evolu.discrete import mutate_discrete, encode_grid_to_discrete, decode_discrete_to_grid
from neorl.hybrid.nhhocore.Latin import latin

class HHO(object):
    """
    Harris Hawks Optimizer

    :param mode: (str) problem type, either "min" for minimization problem or "max" for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param fit: (function) the fitness function
    :param nhawks: (int): number of the grey wolves in the group
    :param int_transform: (str): method of handling int/discrete variables, choose from: ``nearest_int``, ``sigmoid``, ``minmax``.
    :param ncores: (int) number of parallel processors (must be ``<= nhawks``)
    :param seed: (int) random seed for sampling
    """
    def __init__(self, mode, bounds, fit, nhawks, int_transform='nearest_int', ncores=1, seed=None):
        self.seed = seed
        if seed:
            random.seed(seed)
            np.random.seed(seed)

        assert mode == 'min' or mode == 'max', "Mode must be 'max' or 'min'."
        self.mode = mode
        self.fit = fit
        self.int_transform=int_transform
        self.ncores = ncores
        self.nhawks = nhawks
        self.dim = len(bounds)
        self.bounds = bounds

        #infer variable types
        self.var_type = np.array([bounds[item][0] for item in bounds])

        #mir-grid
        if "grid" in self.var_type:
            self.grid_flag=True
            self.orig_bounds=bounds  #keep original bounds for decoding
            print('--debug: grid parameter type is found in the space')
            self.bounds, self.bounds_map=encode_grid_to_discrete(self.bounds) #encoding grid to int
            #define var_types again by converting grid to int
            self.var_type = np.array([self.bounds[item][0] for item in self.bounds])
        else:
            self.grid_flag=False
            self.bounds = bounds

        self.lb = np.array([self.bounds[item][1] for item in self.bounds])
        self.ub = np.array([self.bounds[item][2] for item in self.bounds])

    def evolute(self, ngen, x0=None, verbose=True):
        """
        This function evolutes the HHO algorithm for number of generations.

        :param ngen: (int) number of generations to evolute
        :param x0: (list of lists) initial position of the hawks (must be of same size as ``nhawks``)
        :param verbose: (bool) print statistics to screen

        :return: (tuple) (best position, best fitness, and dictionary containing major search results)
        """
        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.history = {'local_fitness':[], 'global_fitness':[]}
        self.rabbit_energy = float("inf")
        self.rabbit_location = np.zeros(self.dim)
        self.verbose = verbose

        ##################################
        # Set initial locations of hawks #
        ##################################
        self.hawk_positions = np.zeros((self.nhawks, self.dim))
        if x0:
            assert len(x0) == self.nhawks, 'Length of x0 array MUST equal the number of hawks (self.nhawks).'
            self.hawk_positions = x0
        else:
            # self.hawk_positions = np.asarray([x * (self.ub - self.lb) + self.lb for x in np.random.uniform(0, 1, (self.nhawks, self.dim))])
            self.hawk_positions = self.init_sample()

        for t in range(ngen):
            self.a= 1 - t * ((1) / ngen)  #mir: a decreases linearly between 1 to 0, for discrete mutation
            ###########################
            # Evaluate hawk fitnesses #
            ###########################
            fitness_lst = self.eval_hawks()

            #######################################################################
            # Update rabbit energy and rabbit location based on best hawk fitness #
            #######################################################################
            for i, fitness in enumerate(fitness_lst):
                fitness = fitness if self.mode == 'min' else -fitness
                if fitness < self.rabbit_energy:
                    self.rabbit_energy = fitness
                    self.rabbit_location = self.hawk_positions[i, :].copy()

            #####################################################
            # Update best global and local fitnesses in history #
            #####################################################
            if self.mode=='max':
                self.best_global_fitness = -self.rabbit_energy
                self.best_local_fitness = -np.min(fitness_lst)
            else:
                self.best_global_fitness = self.rabbit_energy
                self.best_local_fitness = np.min(fitness)

            self.history['local_fitness'].append(self.best_local_fitness)
            self.history['global_fitness'].append(self.best_global_fitness)

            if self.verbose and t % self.nhawks: # change depending on how often message should be displayed
                print(f'HHO step {t*self.nhawks}/{ngen*self.nhawks}, nhawks={self.nhawks}, ncores={self.ncores}')
                print('Best global fitness:', np.round(self.best_global_fitness, 6))
                #mir-grid
                if self.grid_flag:
                    self.rabbit_decoded=decode_discrete_to_grid(self.rabbit_location,self.orig_bounds,self.bounds_map)
                    print('Best rabbit position:', self.rabbit_decoded)
                else:
                    print('Best rabbit position:', np.round(self.rabbit_location, 6))
                print()

            ################################
            # Update the location of hawks #
            ################################
            self.update_hawks(t, ngen, fitness_lst) # now self.hawk_positions is updated

            for hawk_i in range(self.nhawks):
                #mir: this bound check  line is needed to ensure that choices.remove option to work
                self.hawk_positions[hawk_i, :] = self.ensure_bounds(self.hawk_positions[hawk_i, :], self.bounds)
                for dim in range(self.dim):
                    if self.var_type[dim] == 'int':
                        self.hawk_positions[hawk_i, dim] = mutate_discrete(x_ij=self.hawk_positions[hawk_i, dim],
                                           x_min=self.hawk_positions[hawk_i, :].min(),
                                           x_max=self.hawk_positions[hawk_i, :].max(),
                                           lb=self.lb[dim],
                                           ub=self.ub[dim],
                                           alpha=self.a,
                                           method=self.int_transform)

        #mir-grid
        if self.grid_flag:
            self.rabbit_correct=decode_discrete_to_grid(self.rabbit_location,self.orig_bounds,self.bounds_map)
        else:
            self.rabbit_correct=self.rabbit_location

        if self.verbose:
            print('------------------------ HHO Summary --------------------------')
            print('Function:', self.fit.__name__)
            print('Best fitness (y) found:', self.best_global_fitness)
            print('Best individual (x) found:', self.rabbit_correct)
            print('-------------------------------------------------------------- \n \n')

        return self.rabbit_correct, self.best_global_fitness, self.history

    def ensure_bounds(self, vec, bounds):

        vec_new = []
        # cycle through each variable in vector
        for i, (key, val) in enumerate(bounds.items()):

            # variable exceedes the minimum boundary
            if vec[i] < bounds[key][1]:
                vec_new.append(bounds[key][1])

            # variable exceedes the maximum boundary
            if vec[i] > bounds[key][2]:
                vec_new.append(bounds[key][2])

            # the variable is fine
            if bounds[key][1] <= vec[i] <= bounds[key][2]:
                vec_new.append(vec[i])

        return vec_new

    # def init_sample(self):
    #     #"""
    #     #Initialize a hawk location.
    #
    #     #Return:
    #     #array - hawk position
    #     #"""
    #     hawk_pos = []
    #     for key in self.bounds:
    #         if self.bounds[key][0] == 'int':
    #             hawk_pos.append(random.randint(self.bounds[key][1], self.bounds[key][2]))
    #         elif self.bounds[key][0] == 'float':
    #             hawk_pos.append(random.uniform(self.bounds[key][1], self.bounds[key][2]))
    #         # elif self.bounds[key][0] == 'grid':
    #         #     hawk_pos.append(random.sample(self.bounds[key][1],1)[0])
    #         else:
    #             raise Exception ('unknown data type is given; either int, float, or grid are allowed for parameter bounds')
    #     return np.array(hawk_pos)

    def init_sample(self, num_hawks=None):
        if num_hawks is None:
            num_hawks = self.nhawks

        hawk_pos = latin(num_hawks, self.dim, self.lb, self.ub)
        for hawk in range(num_hawks):
            for d in range(self.dim):
                if self.var_type[d] == 'int':
                    hawk_pos[hawk][d] = round(hawk_pos[hawk][d])
        return np.array(hawk_pos)

    def eval_hawks(self):
        #"""
        #Evaluate fitness of hawks with parallel processing.

        #Return:
        #list - hawk fitnesses
        #"""
        #print(self.hawk_positions)
        if self.ncores > 1:
            with joblib.Parallel(n_jobs=self.ncores) as parallel:
                fitness_lst = parallel(joblib.delayed(self.fit_worker)(self.hawk_positions[i, :]) for i in range(self.nhawks))
        else:
            fitness_lst = []
            for i in range(self.nhawks):
                fitness_lst.append(self.fit_worker(self.hawk_positions[i, :]))
        return fitness_lst

    def update_hawks(self, cur_gen, ngen, fitness_lst):
        #"""
        #Update hawk positions according to HHO rules.

        #Params:
        #cur_gen - current generation of HHO
        #ngen - total number of generations in HHO
        #fitness_lst - list of hawk fitnesses in current generation

        #Return:
        #None
        #"""
        E1 = 2 * (1 - (cur_gen / ngen))  # factor to show the decreasing energy of rabbit
        for i in range(self.nhawks):

            E0 = 2 * random.random() - 1
            rabbit_escape_energy = E1 * E0  # escaping energy of rabbit: eq. (3)

            ##############################
            # Exploration phase: eq. (1) #
            ##############################
            if abs(rabbit_escape_energy) >= 1:
                self.exploration(i)

            ###############################################################
            # Exploitation phase: attacking the rabbit using 4 strategies #
            ###############################################################
            elif abs(rabbit_escape_energy) < 1:
                r = random.random() # probablity of each event

                ##########################################
                # Phase 1: surprise pounce (seven kills) #
                ##########################################
                if r >= 0.5:
                    self.surprise_pounce(i, rabbit_escape_energy)

                #############################################################
                # Phase 2: performing team rapid dives (leapfrog movements) #
                #############################################################
                elif r < 0.5:
                    self.team_rapid_dives(i, fitness_lst[i], rabbit_escape_energy)

    def exploration(self, hawk_index):
        #"""
        #Exploration phase based on two perching strategies:
        #    1. new hawk location based on other hawks' locations
        #    2. new hawk location generated randomly

        #Params:
        #hawk_index - index of hawk being updated

        #Return:
        #None
        #"""
        q = random.random()
        rand_hawk = math.floor(self.nhawks * random.random())
        rand_hawk_pos = self.hawk_positions[rand_hawk, :] # location of random hawk

        if q < 0.5: # perch based on other family members
            self.hawk_positions[hawk_index, :] = rand_hawk_pos \
                                                - random.random() * abs(rand_hawk_pos - 2 * random.random() * self.hawk_positions[hawk_index, :])
        else: # perch on a random tall tree (random site inside group's home range)
            self.hawk_positions[hawk_index, :] = (self.rabbit_location - self.hawk_positions.mean(0)) \
                                                - random.random() * ((self.ub - self.lb) * random.random() + self.lb)

    def surprise_pounce(self, hawk_index, escape_energy):
        #"""
        #Exploitation phase 1 based on two strategies:
        #    1. Hard besiege without team rapid dives
        #    2. Soft besiege without team rapid dives

        #Params:
        #hawk_index - index of hawk being updated
        #escape_energy - escape energy of the rabbit

        #Return:
        #None
        #"""

        ########################
        # Hard besiege eq. (6) #
        ########################
        if abs(escape_energy) < 0.5:
            self.hawk_positions[hawk_index, :] = (self.rabbit_location) \
                                                - escape_energy * abs(self.rabbit_location - self.hawk_positions[hawk_index, :])
        ########################
        # Soft besiege eq. (4) #
        ########################
        elif abs(escape_energy) >= 0.5:
            J = 2 * (1 - random.random())  # random jump strength of the rabbit; 1 < J < 2
            #mir: the index was "i" here, which should be hawk_index
            self.hawk_positions[hawk_index, :] = (self.rabbit_location - self.hawk_positions[hawk_index, :]) \
                                        - escape_energy * abs(J * self.rabbit_location - self.hawk_positions[hawk_index, :])

    def team_rapid_dives(self, hawk_index, hawk_fitness, escape_energy):
        #"""
        #Exploitation phase 2 based on two strategies:
        #    1. Hard besiege with team rapid dives
        #    2. Soft besiege with team rapid dives

        #Params:
        #hawk_index - index of hawk being updated
        #hawk_fitness - fitness of current hawk
        #escape_energy - escape energy of the rabbit

        #Return:
        #None
        #"""
        #########################
        # Soft besiege eq. (10) #
        #########################
        if abs(escape_energy) >= 0.5:
            J = 2 * (1 - random.random())
            Y = self.rabbit_location \
                - escape_energy * abs(J * self.rabbit_location - self.hawk_positions[hawk_index, :])

            #mir: uses a built-in function
            #(order is important for choices.remove(), first check bounds, then check discrete)
            Y=self.ensure_bounds(Y, self.bounds)
            Y=self.ensure_discrete(Y)

            if self.fit_worker(Y) < hawk_fitness:  # improved move?
                self.hawk_positions[hawk_index, :] = Y.copy()
            else:  # hawks perform self.levy-based short rapid dives around the rabbit
                Z = self.rabbit_location \
                    - escape_energy * abs(J * self.rabbit_location - self.hawk_positions[hawk_index, :]) \
                    + np.multiply(np.random.randn(self.dim), self.levy())

                #mir---
                #(order is important for choices.remove(), first check bounds, then check discrete)
                Z=self.ensure_bounds(Z, self.bounds)
                Z=self.ensure_discrete(Z)

                if self.fit_worker(Z) < hawk_fitness:
                    self.hawk_positions[hawk_index, :] = Z.copy()
        #########################
        # Hard besiege eq. (11) #
        #########################
        elif abs(escape_energy) < 0.5:
            J = 2 * (1 - random.random())
            Y = self.rabbit_location \
                - escape_energy * abs(J * self.rabbit_location - self.hawk_positions.mean(0))

            #---mir
            #(order is important for choices.remove(), first check bounds, then check discrete)
            Y=self.ensure_bounds(Y, self.bounds)
            Y=self.ensure_discrete(Y)

            if self.fit_worker(Y) < hawk_fitness:  # improved move?
                self.hawk_positions[hawk_index, :] = Y.copy()
            else:  # Perform levy-based short rapid dives around the rabbit
                Z = self.rabbit_location \
                    - escape_energy * abs(J * self.rabbit_location - self.hawk_positions.mean(0)) \
                    + np.multiply(np.random.randn(self.dim), self.levy())

                #--mir
                #(order is important for choices.remove(), first check bounds, then check discrete)
                Z=self.ensure_bounds(Z, self.bounds)
                Z=self.ensure_discrete(Z)

                if self.fit_worker(Z) < hawk_fitness:
                    self.hawk_positions[hawk_index, :] = Z.copy()


    def fit_worker(self, hawk_pos):
        #"""
        #Evaluates fitness of a hawk.

        #Params:
        #hawk_pos - hawk position vector

        #Return:
        #float - hawk fitness
        #"""

        #mir---
        hawk_pos=self.ensure_bounds(hawk_pos, self.bounds)

        #mir-grid
        if self.grid_flag:
            #decode the individual back to the int/float/grid mixed space
            hawk_pos=decode_discrete_to_grid(hawk_pos,self.orig_bounds,self.bounds_map)

        fitness = self.fit(hawk_pos)
        return fitness

    def ensure_discrete(self, vec):
        #"""
        #to mutate a vector if discrete variables exist
        #handy function to be used four times within HHO phases

        #Params:
        #vec - hawk position in vector/list form

        #Return:
        #vec - updated hawk position vector with discrete values
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

    def levy(self):
        #"""
        #Evaluates the levy flight (LF) function eq. (9).

        #Return:
        #float - result of levy flight function
        #"""
        beta = 1.5
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = 0.01 * np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        zz = np.power(np.absolute(v), (1 / beta))
        step = np.divide(u, zz)
        return step
