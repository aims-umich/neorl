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
#Created on Wed Mar  4 11:51:22 2020
#
#@author: majdi
#"""

import random
import numpy as np
import pandas as pd
from multiprocessing import Pool
import joblib
import csv

class RANDTUNE:
    """
    A module for random search for hyperparameter tuning

    :param param_grid: (dict) the type and range of each hyperparameter in a dictionary form (types are ``int/discrete`` or ``float/continuous`` or ``grid/categorical``). Example: {'x1': [[40, 50, 60, 100], 'grid'], 'x2': [[0.2, 0.8], 'float'], 'x3': [['blend', 'cx2point'], 'grid'], 'x4': [[20, 80], 'int']}
    :param fit: (function) the self-defined fitness function that includes the hyperparameters as input and algorithm score as output
    :param ncases: (int) number of random hyperparameter cases to generate 
    :param seed: (int) random seed for sampling reproducibility
    """
    def __init__(self, param_grid, fit, ncases=50, seed=None):
        self.param_grid=param_grid
        self.fit=fit
        self.ncases=ncases
        self.seed=seed
        self.full_grid()

    def full_grid(self):
        #This function builds the full multi-dimensional grid
        if self.seed:
            random.seed(self.seed)
        
        #self.param_lst=[self.param_grid[item][0] for item in self.param_grid]
        #self.param_types=[self.param_grid[item][1] for item in self.param_grid]
        #self.param_names=[item for item in self.param_grid]
        self.param_types=[self.param_grid[item][0] for item in self.param_grid]
        self.param_lst=[]
        for i, item in enumerate(self.param_grid):
            if self.param_types[i] in ['grid', 'categorical']:
                self.param_lst.append(self.param_grid[item][1])
            else:
                self.param_lst.append(self.param_grid[item][1:])
        
        self.param_names=[item for item in self.param_grid]
        
        self.hyperparameter_cases=[]
        
        for _ in range(self.ncases):
            sample=[]
            for types, vals in zip(self.param_types, self.param_lst):
                if types in ['int', 'discrete']:
                    lb=vals[0]
                    ub=vals[1]
                    sample.append(random.randint(lb, ub))
                elif types in ['float', 'continuous']:
                    lb=vals[0]
                    ub=vals[1]
                    sample.append(random.uniform(lb, ub))
                elif types in ['grid', 'categorical']: 
                    real_grid=vals
                    sample.append(random.sample(real_grid,1)[0])
                else:
                    raise Exception('--error: the param types must be one of int/discrete or float/continuous or grid/categorical, this type is not avaiable: `{}`'.format(types))
                
            self.hyperparameter_cases.append(tuple(sample))
                         
    def worker(self,x):
        #This function setup a case object to pass to the Parallel pool

        caseid=x[0]
        param_vals=x[1]
        
        #form the dictionary for this case
        case_dict={}
        case_dict['id']=caseid
        assert len(param_vals) == len(self.param_names), '--error: it seems the length of the param_names ({}) and param_values ({}) are not equal, cannot proceed'.format(len(self.param_names), len(x))
        for name, val in zip(self.param_names, param_vals):
            case_dict[name]=val
        
        try:
            obj=self.fit(*param_vals)
            case_dict['score']=obj
            if self.verbose:
                print('-------------------------------------------------------------------------------------------')
                print('TUNE Case {}/{} is completed'.format(caseid, len(self.hyperparameter_cases), case_dict))
                print(case_dict)
                print('-------------------------------------------------------------------------------------------')
            
            if self.csvlogger:
                with open (self.csvlogger, 'a') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')
                    csvwriter.writerow([case_dict[item] for item in case_dict])
            
            return obj
        
        except:
            print('--error: case {} failed during execution'.format(caseid))
            print('--error: {} failed'.format(case_dict))
            
            return 'case{}:failed'.format(caseid)
        
        
    def tune(self, ncores=1, csvname=None, verbose=True):
        """
        This function starts the tuning process with specified number of processors
    
        :param ncores: (int) number of parallel processors (see the **Notes** section below for an important note about parallel execution)
        :param csvname: (str) the name of the csv file name to save the tuning results (useful for expensive cases as the csv file is updated directly after the case is done)
        :param verbose: (bool) whether to print updates to the screen or not
        """
        self.ncores=ncores
        self.csvlogger=csvname
        self.verbose=verbose

        if self.verbose:
            print('***************************************************************')
            print('****************Random Search is Running*************************')
            print('***************************************************************')
            
            if self.ncores > 1:
                print('--- Running in parallel with {} cores'.format(self.ncores))
                
        if self.csvlogger:
            headers=['id']  + self.param_names + ['score']
            with open (self.csvlogger, 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')
                csvwriter.writerow(headers)
                
        core_lst=[]
        for i in range (len(self.hyperparameter_cases)):
            core_lst.append([i+1, self.hyperparameter_cases[i]])
        
        if self.ncores > 1:
            #p=Pool(self.ncores)
            #results = p.map(self.worker, core_lst)
            #p.close()
            #p.join()
            
            with joblib.Parallel(n_jobs=self.ncores) as parallel:
                results=parallel(joblib.delayed(self.worker)(item) for item in core_lst)
                
        else:
            results=[]
            for item in core_lst:
                results.append(self.worker(item))

        gridres = pd.DataFrame(self.hyperparameter_cases, columns=self.param_names)
        gridres.index += 1
        gridres['score'] = results
        #gridres = gridres.sort_values(['score'], axis='index', ascending=False)     
        
        return gridres