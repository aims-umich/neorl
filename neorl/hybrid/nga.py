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
#Created on Thu July 3 14:42:29 2021
#
#@author: Katelin Du
#"""

#Refactored and inspired from this paper:

#Huang, P., Wang, H., & Jin, Y. (2021). Offline data-driven evolutionary 
#optimization based on tri-training. Swarm and Evolutionary Computation, 60, 100800.

import numpy as np
import random
from neorl.hybrid.ngacore.RBFN import RBFN
from neorl.hybrid.ngacore.Latin import latin
from neorl.hybrid.ngacore.GA import GA


class NGA(object):
    """
    Neural Genetic Algorithm

    :param mode: (str) problem type, either "min" for minimization problem or "max" for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param fit: (function) the fitness function
    :param npop: (int) population size of genetic algorithms
    :param num_warmups: (int) number of warmup samples to train the surrogate which will be evaluated by the real fitness ``fit`` (if ``None``, ``num_warmups=20*len(bounds)``)
    :param hidden_shape: (int) number of hidden layers in the RBFN network (if ``None``, ``hidden_shape=int(sqrt(int(num_warmups/3)))``)
    :param kernel: (str) kernel type for the RBFN network (choose from ``gaussian``, ``reflect``, ``mul``, ``inmul``)
    :param ncores: (int) number of parallel processors (currently only ``ncores=1`` is supported)
    :param seed: (int) random seed for sampling
    """
    def __init__(self, mode, bounds, fit, npop, num_warmups=None, 
                 hidden_shape = None, kernel='gaussian', ncores=1, seed=None):
        self.seed = seed
        if seed:
            random.seed(seed)
            np.random.seed(seed)

        self.dimension = len(bounds)
        lb = np.array([bounds[item][1] for item in bounds])
        ub = np.array([bounds[item][2] for item in bounds])
        self.lower_bound = np.array(lb)
        self.upper_bound = np.array(ub)

        self.ga = GA(pop_size=npop, dimension=self.dimension, lower_bound=self.lower_bound, upper_bound=self.upper_bound)

        assert mode == 'min' or mode == 'max', "Mode must be 'max' or 'min'."
        assert ncores == 1, "NGA currently not supporting parallel processing."
        self.mode = mode
        self.fit = fit
        self.ncores = ncores
        self.pop_size = npop
        self.bounds = bounds

        self.num_warmups = num_warmups if num_warmups is not None else 20*self.dimension
        self.traindata = int(self.num_warmups/3)
        self.hidden_shape = hidden_shape if hidden_shape is not None else int(np.sqrt(self.traindata))
        self.kernel = kernel
        print('---NGA is running---')
        print('surrogate warmup size:', self.num_warmups)
        print('surrogate hidden_shape:', self.hidden_shape)
        print('surrogate kernel:', self.kernel)

    def evolute(self, ngen, verbose=True):
        """
        This function evolutes the NGA algorithm for number of generations.

        :param ngen: (int) number of generations to evolute
        :param verbose: (bool) print statistics to screen

        :return: (tuple) (list of best individuals, list of best fitnesses)
        """
        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)
        self.verbose = verbose

        self.history = {'local_fitness':[], 'best_individual':[]}

        ##########################################################
        # Generate warmup samples and create generation 0 models #
        ##########################################################
        self.warmup_samples = self.init_Population(pop_num=self.num_warmups)
        self.warmup_fitnesses = self.eval_samples(self.warmup_samples)

        self.train_warmups() # models fitted and saved in self.models

        ##########
        # Run GA #
        ##########
        self.ga.init_Population()
        for t in range(ngen):
            self.updatemodel()
            self.ga.crossover(self.ga.pc)
            self.ga.mutation(self.ga.pm)
            self.ga.pop = np.unique(self.ga.pop, axis=0)
            pred_sum = 0
            for i in range(3):
                pred = self.models[i].predict(self.ga.pop)
                pred_sum += pred
            fit_values = (pred_sum/3).reshape((len(self.ga.pop), 1))
            self.ga.selection(fit_values)
            self.resetmodel() # models are back to using warmup samples only

            #####################################################
            # Update best global and local fitnesses in history #
            #####################################################
            self.best_local_fitness = np.min(fit_values) if self.mode == 'min' else -np.min(fit_values)
            self.best_individual = self.ga.first[-1]

            self.history['local_fitness'].append(self.best_local_fitness)
            self.history['best_individual'].append(self.best_individual)

            if self.verbose and t % self.pop_size:
                print('************************************************************')
                print(f'NGA step {t*self.pop_size}/{ngen*self.pop_size}, pop_size={self.pop_size}, ncores={self.ncores}')
                print('************************************************************')
                print('Best surrogate fitness:', np.round(self.best_local_fitness, 6))
                print('Best surrogate individual:', np.round(self.best_individual, 6))
                print('************************************************************')
                print()

        if self.verbose:
            print('------------------------ NGA Summary --------------------------')
            print('Function:', self.fit.__name__)
            print('Final surrogate fitness (y) found:', self.best_local_fitness)
            print('Final surrogate individual (x) found:', self.best_individual)
            print('-------------------------------------------------------------- \n \n')

        return self.history['best_individual'], self.history['local_fitness'] 

    def init_Population(self, pop_num=None):
        # """
        # Initializes a population of size self.pop_size unless pop_num parameter is specified.
        #
        # Params:
        # pop_num - optional specification of population size to initiate
        #
        # Return:
        # array - sample vectors
        # """
        if pop_num is None:
            pop_num = self.pop_size
        return latin(pop_num, self.ga.chrom_length, self.lower_bound, self.upper_bound)

    def eval_samples(self, samples = None):
        # """
        # Evaluates samples using the fitness function. Evaluates self.ga.pop unless samples parameter
        # specified.
        #
        # Params:
        # samples - optional specification of samples to be evaluated
        #
        # Return:
        # array - evaluated fitnesses
        # """
        if samples is None:
            samples = self.ga.pop

        fitness_lst = []
        for sample in samples:
            if self.mode == 'min':
                fitness_lst.append(self.fit(sample))
            else:
                fitness_lst.append(-self.fit(sample))

        return np.array(fitness_lst)

    def train_warmups(self):
        # """
        # Performs tri-training on warmup samples. Produces self.models containing the three trained surrogate models.
        # """

        self.models = []
        for i in range(3):
            self.models.append(RBFN(input_shape=self.dimension, hidden_shape=self.hidden_shape, kernel=self.kernel))

        self.resetmodel()

    def resetmodel(self):
        # """
        # Resets surrogate models to their original state when only warmup samples were used for training.
        # """
        # shuffle and split data
        shuffledata = np.column_stack((self.warmup_fitnesses, self.warmup_samples))
        np.random.shuffle(shuffledata)
        self.warmup_samples = shuffledata[:, 1:]
        self.warmup_fitnesses = shuffledata[:, :1]
        self.warmup_x0 = self.warmup_samples[:self.traindata, ]
        self.warmup_y0 = self.warmup_fitnesses[:self.traindata, ]
        self.warmup_x1 = self.warmup_samples[self.traindata:2 * self.traindata, ]
        self.warmup_y1 = self.warmup_fitnesses[self.traindata:2 * self.traindata, ]
        self.warmup_x2 = self.warmup_samples[self.num_warmups - self.traindata:, ]
        self.warmup_y2 = self.warmup_fitnesses[self.num_warmups - self.traindata:, ]

        self.models[0].fit(self.warmup_x0, self.warmup_y0)
        self.models[1].fit(self.warmup_x1, self.warmup_y1)
        self.models[2].fit(self.warmup_x2, self.warmup_y2)

    def updatemodel(self):
        # """
        # Updates each model to include the most accurately predicted sample from the current generation.
        # """
        pre0 = self.models[0].predict(self.ga.pop)
        pre1 = self.models[1].predict(self.ga.pop)
        pre2 = self.models[2].predict(self.ga.pop)

        error = abs(pre1-pre2)
        seq = np.ravel(np.where(error == np.min(error)))[0]
        xtemp = np.row_stack((self.warmup_x0, self.ga.pop[seq]))
        ytemp = np.append(self.warmup_y0, (pre1[seq]+pre2[seq])/2)
        self.models[0].fit(xtemp, ytemp)

        error = abs(pre0-pre2)
        seq = np.ravel(np.where(error == np.min(error)))[0]
        xtemp = np.row_stack((self.warmup_x1, self.ga.pop[seq]))
        ytemp = np.append(self.warmup_y1, (pre0[seq]+pre2[seq])/2)
        self.models[1].fit(xtemp, ytemp)

        error = abs(pre0-pre1)
        seq = np.ravel(np.where(error == np.min(error)))[0]
        xtemp = np.row_stack((self.warmup_x2, self.ga.pop[seq]))
        ytemp = np.append(self.warmup_y2, (pre0[seq]+pre1[seq])/2)
        self.models[2].fit(xtemp, ytemp)