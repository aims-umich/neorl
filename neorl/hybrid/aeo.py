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

#TODO: for nonuniform weights need to incorporate fitness values at the member level

import numpy as np

class Population:
    # Class to store information and functionality related to a single population
    # in the AEO algorithm. Should be characterized by an evolution strategy passed
    # as one of the optimization classes from the other algorithms in NEORL.
    def __init__(self, strategy, init_pop, conv = None):
        # strategy should be one of the optimization objects containing an "evolute" method
        # init_pop needs to match the population given to the strategy object initially
        # conv is a function which takes ngen and returns number of evaluations
        self.conv = conv
        self.strategy = strategy
        self.members = [strategy.init_sample(strategy.bounds) for i in range(init_pop)]

        self.fitlog = []

    @property
    def fitness(self):
        return self.fitlog[-1]

    def evolute(self, ngen):
        #perform evolution and store relevant information
        self.fitlog.append(self.strategy.evolute(ngen, x0 = self.members)[1])
        self.members = self.strategy.Positions.tolist()

    #TODO: method to export members, return them and remove from list
    #TODO: method to reviece members, update them in members
    #TODO: calc strength method, no need to normalize, may need some scaling parametrs

class AEO(object):
    """
    Animorphoc Ensemble Optimizer

    :param mode: (str) problem type, either "min" for minimization problem or "max" for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param fit: (function) the fitness function
    :param optimizers: (list) list of optimizer instances to be included in the ensemble
    :param ncores: (int) number of parallel processors
    :param seed: (int) random seed for sampling
    """
    def __init__(self, mode, bounds, fit, optimizers, ncores = 1, seed = None):

        if not (seed is None):
            random.seed(seed)
            np.random.seed(seed)

        self.mode=mode
        if mode == 'min':
            self.fit=fit
        elif mode == 'max':
            def fitness_wrapper(*args, **kwargs):
                return -fit(*args, **kwargs) 
            self.fit=fitness_wrapper
        else:
            raise ValueError('--error: The mode entered by user is invalid, use either `min` or `max`')

        self.optimizers = optimizers

        self.bounds = bounds
        self.ncores = ncores

        #infer variable types 
        self.var_type = np.array([bounds[item][0] for item in bounds])

        self.dim = len(bounds)
        self.lb=[self.bounds[item][1] for item in self.bounds]
        self.ub=[self.bounds[item][2] for item in self.bounds]

        #check that all optimizers have options that match
        self.ensure_consistency()

    def ensure_consistency(self):
        #loop through all optimizers and make sure all options are set to be the same
        gen_warning = ', check that options of all optimizers are the same as AEO'
        for o in self.optimizers:
            assert self.mode == o.mode,'%s has incorrect optimization mode'%o + gen_warning
            assert self.bounds == o.bounds,'%s has incorrect bounds'%o + gen_warning
            try:
                assert self.fit(self.lb) == o.fit(self.lb)
                assert self.fit(self.ub) == o.fit(self.ub)
                inner_test = [np.random.uniform(self.lb[i], self.ub[i]) for i in range(len(self.ub))]
                assert self.fit(inner_test) == o.fit(inner_test)
            except:
                raise Exception('i%s has incorrect fitness function'%o + gen_warning)

        #TODO: Initialize populations
    #TODO: Set up evolute method
    #TODO: Set up migration method with 3 phases and markov matrix calculation




