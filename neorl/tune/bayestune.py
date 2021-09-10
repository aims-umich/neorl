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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#"""
#Created on Mon Jun 29 15:36:46 2020
#
#@author: alyssawang
#"""

import inspect
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Scikit-optimise
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args

class BAYESTUNE:
    """
    A module for Bayesian search for hyperparameter tuning
    
    :param param_grid: (dict) the type and range of each hyperparameter in a dictionary form (types are ``int/discrete`` or ``float/continuous`` or ``grid/categorical``). Example: {'x1': [[40, 50, 60, 100], 'grid'], 'x2': [[0.2, 0.8], 'float'], 'x3': [['blend', 'cx2point'], 'grid'], 'x4': [[20, 80], 'int']}
    :param fit: (function) the self-defined fitness function that includes the hyperparameters as input and algorithm score as output
    :param mode: (str) problem type, either "min" for minimization problem or "max" for maximization. Default: Bayesian tuner is set to minimize an objective
    :param ncases: (int) number of random hyperparameter cases to generate per core, ``ncases >= 11`` (see **Notes** for an important remark) 
    :param seed: (int) random seed for sampling reproducibility
    """
    def __init__(self, param_grid, fit, mode='min', ncases=50, seed=None):
        self.mode=mode
        assert self.mode in ['min', 'max'], '--error: The mode entered by user is invalid, use either `min` or `max`'
        self.param_grid=param_grid
        self.fit=fit
        self.ncases=ncases
        self.seed=seed
        if self.ncases < 11:
            print('--warning: ncases={} < 11 is given by the user, but ncases must be more than 11, reset ncases to 11'.format(self.ncases))
            self.ncases = 11
            
        self.full_grid()
        
    def get_func_args(self, f):
        #this function returns the argument names of the input function "f"
        return inspect.getfullargspec(f)[0]
    
#    def plot_results(self):
#        plot_convergence(self.search_result)

    def full_grid(self):
        #This function parses the param_grid variable from the user and sets up the 
        #parameter space for Bayesian search
        
        self.param_types=[self.param_grid[item][0] for item in self.param_grid]
        self.param_lst=[]
        for i, item in enumerate(self.param_grid):
            if self.param_types[i] in ['grid', 'categorical']:
                self.param_lst.append(self.param_grid[item][1])
            else:
                self.param_lst.append(self.param_grid[item][1:])
        
        self.param_names=[item for item in self.param_grid]
        self.dimensions=[]
        self.func_args=self.get_func_args(self.fit)
        
        for types, vals, names in zip(self.param_types, self.param_lst, self.param_names):
            if types in ['int', 'discrete']:
                lb=vals[0]
                ub=vals[1]
                self.dimensions.append(Integer(low=lb, high=ub, name=names))
            elif types in ['float', 'continuous']:
                lb=vals[0]
                ub=vals[1]
                self.dimensions.append(Real(low=lb, high=ub, name=names))
            elif types in ['grid', 'categorical']: 
                real_grid=vals
                self.dimensions.append(Categorical(categories=tuple(real_grid),  name=names))
            else:
                raise Exception('--error: the param types must be one of int/discrete or float/continuous or grid/categorical, this type is not avaiable: `{}`'.format(types))
                
    def worker(self,x):
        #This function setup a case worker to pass to the Parallel pool
        
        if self.mode=='min':
            @use_named_args(dimensions=self.dimensions)
            def fitness_wrapper(*args, **kwargs):
                return self.fit(*args, **kwargs)             
        else:
            @use_named_args(dimensions=self.dimensions)
            def fitness_wrapper(*args, **kwargs):
                return -self.fit(*args, **kwargs) 
        
        if self.seed:
            core_seed=self.seed + x
        else:
            core_seed=None

        search_result = gp_minimize(func=fitness_wrapper,
                                    dimensions=self.dimensions,
                                    acq_func='EI', # Expected Improvement.
                                    n_calls=self.ncases,
                                    random_state=core_seed, verbose=self.verbose)
        
        return search_result.x_iters, list(search_result.func_vals)
    
    def plot_results(self, pngname=None):
        if self.mode=='max':
            plt.plot(pd.DataFrame.cummax(self.bayesres['score']), '-og')
            plt.ylabel('Max score so far')
        else:
            plt.plot(pd.DataFrame.cummin(self.bayesres['score']), '-og')
            plt.ylabel('Min score so far')
            
        plt.xlabel('Iteration')
        plt.grid()
        if pngname is not None:
            plt.savefig(str(pngname)+'.png', dpi=200, format='png')
        plt.show()
        
    def tune(self, ncores=1, csvname=None, verbose=True):
        """
        This function starts the tuning process with specified number of processors
    
        :param nthreads: (int) number of parallel threads (see the **Notes** section below for an important note about parallel execution)
        :param csvname: (str) the name of the csv file name to save the tuning results (useful for expensive cases as the csv file is updated directly after the case is done)
        :param verbose: (bool) whether to print updates to the screen or not
        """
        self.ncores=ncores
        self.csvlogger=csvname
        self.verbose=verbose

        if self.verbose:
            print('***************************************************************')
            print('****************Bayesian Search is Running*********************')
            print('***************************************************************')
            
            if self.ncores > 1:
                print('--- Running in parallel with {} threads and {} cases per threads'.format(self.ncores, self.ncases))
                print('--- Total number of executed cases is {}*{}={} cases'.format(self.ncores,self.ncases,self.ncores*self.ncases))
   
        if self.ncores > 1:
            
            with joblib.Parallel(n_jobs=self.ncores) as parallel:
                x_vals, func_vals=zip(*parallel(joblib.delayed(self.worker)(core+1) for core in range(self.ncores)))
            
            #flatten the x-lists for all cores
            x_vals_flatten=[]
            for lists in x_vals:
                for item in lists:
                    x_vals_flatten.append(item)
            
            #flatten the y results from all cores 
            func_vals_flatten = [item for sublist in func_vals for item in sublist]

            assert len(func_vals_flatten) == len(x_vals_flatten), '--error: the length of func_vals_flatten and x_vals_flatten in parallel Bayesian search must be equal'
            self.bayesres=pd.DataFrame(x_vals_flatten, columns = self.func_args)
            
            self.bayesres['score'] = np.array(func_vals_flatten) if self.mode=='min' else -np.array(func_vals_flatten)
         
        else:
            
            if self.mode=='min':
                @use_named_args(dimensions=self.dimensions)
                def fitness_wrapper(*args, **kwargs):
                    return self.fit(*args, **kwargs)             
            else:
                @use_named_args(dimensions=self.dimensions)
                def fitness_wrapper(*args, **kwargs):
                    return -self.fit(*args, **kwargs) 
            
            #Single core search
            self.search_result = gp_minimize(func=fitness_wrapper,
                                            dimensions=self.dimensions,
                                            acq_func='EI', # Expected Improvement.
                                            n_calls=self.ncases,
                                            random_state=self.seed, verbose=self.verbose)

            self.bayesres = pd.DataFrame(self.search_result.x_iters, columns = self.func_args)
            self.bayesres['score'] = self.search_result.func_vals if self.mode=='min' else -self.search_result.func_vals

        self.bayesres.index+=1
        
        if self.csvlogger:
            self.bayesres.index.name='id'
            self.bayesres.to_csv(self.csvlogger)
                
        return self.bayesres