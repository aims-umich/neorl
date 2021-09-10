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
#Created on Tue Feb 25 14:42:24 2020
#
#@author: Majdi Radaideh
#"""

#Implementation of RL-informed DE (ACKTR-DE)
#Based on papers:

#Radaideh, M. I., & Shirvan, K. (2021). Rule-based reinforcement learning 
#methodology to inform evolutionary algorithms for constrained optimization 
#of engineering applications. Knowledge-Based Systems, 217, 106836.

import warnings
warnings.filterwarnings("ignore")
import random
import pandas as pd
import numpy as np
from neorl.evolu.discrete import mutate_discrete, encode_grid_to_discrete, decode_discrete_to_grid
from neorl import DE
from neorl import ACKTR, MlpPolicy, RLLogger

def encode_grid_individual_to_discrete(individual, bounds):
    
    new_indv=[]
    for i, key in enumerate(bounds):
        if bounds[key][0]=='grid':
            int_val=bounds[key][1].index(individual[i])
            new_indv.append(int_val)
        else:
            new_indv.append(individual[i])
    
    return new_indv

class ACKDE(object):
    """
    A ACKTR-informed DE Neuroevolution module 
    
    :param mode: (str) problem type, either ``min`` for minimization problem or ``max`` for maximization
    :param fit: (function) the fitness function to be used with DE
    :param env: (NEORL environment or Gym environment) The environment to learn with ACKTR, either use NEORL method ``CreateEnvironment`` (see **below**) or construct your custom Gym environment.  
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param npop: (int): population size of DE
    :param npop_rl: (int): number of RL/ACKTR individuals to use in DE population (``npop_rl < npop``)
    :param init_pop_rl: (bool) flag to initialize DE population with ACKTR individuals
    :param hyperparam: (dict) dictionary of DE hyperparameters (``F``, ``CR``) 
                              and ACKTR hyperparameters (``n_steps``, ``gamma``, ``learning_rate``, ``ent_coef``, ``vf_coef``, ``vf_fisher_coef``, ``kfac_clip``, ``max_grad_norm``, ``lr_schedule``)
    :param seed: (int) random seed for sampling
    """
    def __init__ (self, mode, fit, env, bounds, npop=60, npop_rl=6, 
                  init_pop_rl=True, hyperparam={}, seed=None):    
        
        self.seed = seed
        if seed:
            random.seed(seed)
            np.random.seed(seed)
            
        assert npop_rl < npop, '--error: the size of RL individuals `npop_rl` MUST be less than `npop`'
        self.mode=mode
        self.bounds=bounds
        self.fit=fit
        self.env=env
        self.npop=npop
        self.npop_rl=npop_rl
        self.init_pop_rl=init_pop_rl
        
        #--mir
        self.mode=mode

        #infer variable types 
        self.var_type = np.array([bounds[item][0] for item in bounds])
        self.bounds=bounds
        
        #mir-grid
        if "grid" in self.var_type:
            self.grid_flag=True
        else:
            self.grid_flag=False
            
        self.dim = len(bounds)
        self.var_names=[item for item in self.bounds]
            
        self.hyperparam = hyperparam
        #ACKTR hyperparameters
        self.n_steps = hyperparam['n_steps'] if 'n_steps' in hyperparam else 20
        self.gamma = hyperparam['gamma'] if 'gamma' in hyperparam else 0.99
        self.ent_coef = hyperparam['ent_coef'] if 'ent_coef' in hyperparam else 0.01
        self.learning_rate = hyperparam['learning_rate'] if 'learning_rate' in hyperparam else 0.25
        self.vf_coef = hyperparam['vf_coef'] if 'vf_coef' in hyperparam else 0.25
        self.vf_fisher_coef = hyperparam['vf_fisher_coef'] if 'vf_fisher_coef' in hyperparam else 1.0
        self.max_grad_norm = hyperparam['max_grad_norm'] if 'max_grad_norm' in hyperparam else 0.5
        self.kfac_clip = hyperparam['kfac_clip'] if 'kfac_clip' in hyperparam else 0.001
        self.lr_schedule = hyperparam['lr_schedule'] if 'lr_schedule' in hyperparam else 'linear'
        #DE hyperparameters
        self.F = hyperparam['F'] if 'F' in hyperparam else 0.5
        self.CR = hyperparam['CR'] if 'CR' in hyperparam else 0.3
        
        #will be activated after using `learn` method
        self.ACKTR_RUN_FLAG=False
        
    def learn(self, total_timesteps, rl_filter=100, verbose=False):
        """
        This function starts the learning of ACKTR algorithm for number of timesteps to create individuals for evolutionary search
        
        :param total_timesteps: (int) number of timesteps to run
        :param rl_filter: (int) number of top individuals to keep from the full RL search
        :param verbose: (bool) print statistics to screen
        
        :return: (dataframe) dataframe of individuals/fitness sorted from best to worst
        """        
        self.ACKTR_RUN_FLAG=True

        print('---------------------------------------------------------------------------------')
        print('------------------------------- ACKTR-DE is Running -------------------------------')
        print('---------------------------------------------------------------------------------')
        
        try:
            ncores=len(self.env.get_attr('mode'))
            print('Paralell RL is running with {} cores'.format(ncores))
            self.env.mode=self.env.get_attr('mode')[0]
        except:
            try:
                self.env.mode  #
                ncores=1
                print('Serial RL is running with {} core'.format(ncores))
            except:
                self.env.mode = 'max'      # or some other default value.
        
        print('--warning: Problem mode defined in the RL enviroment is', self.env.mode)
        print('--warning: Problem mode defined in the ACKDE class is', self.mode)
        if self.env.mode == self.mode:
            print('--warning: Both problem modes match')
        else:
            raise ValueError('The two problem modes do not match, alg terminates')                
        print('------------------------------- Part I: ACKTR is collecting data -------------------------------')
        cb=RLLogger(check_freq=1)
        acktr = ACKTR(MlpPolicy, env=self.env, 
                   n_steps=self.n_steps, 
                   gamma=self.gamma,
                   ent_coef=self.ent_coef,
                   vf_coef=self.vf_coef,
                   vf_fisher_coef=self.vf_fisher_coef,
                   max_grad_norm=self.max_grad_norm,
                   kfac_clip=self.kfac_clip,
                   lr_schedule=self.lr_schedule,
                   seed=self.seed,
                   verbose=verbose)  #run ACKTR
        acktr.learn(total_timesteps=total_timesteps, callback=cb) 
        
        rl_data=pd.DataFrame(cb.x_hist, columns=self.var_names)  #get the RL invidiuals
        assert len(cb.x_hist) == len(cb.r_hist), '--error: the length of reward hist ({}) and individual list ({}) must be the same, evolutionary run cannot continue'.format(len(cb.r_hist), len(cb.x_hist))
        rl_data["score"]=cb.r_hist    #append thier fitness/score as new column
        
        #sort the dataframe to filter the best
        if self.mode == 'min':
            self.sorted_df=rl_data.sort_values(['score'], axis='index', ascending=True)  
        else:
            self.sorted_df=rl_data.sort_values(['score'], axis='index', ascending=False)
        
        #check the shape of RL data before filtering the top `rl_filter`
        if self.sorted_df.shape[0] < rl_filter:
            print('--warning: the number of samples collected by RL ({}) is less than rl_filter ({}), so all samples are passed to DE'.format(self.sorted_df.shape[0], rl_filter))
            self.data=self.sorted_df.values[:,:-1]   #get rid of the score column
        else:
            self.data=self.sorted_df.values[:rl_filter,:-1]  #get rid of the score column
        
        if verbose:
            print('--Top 10 individuals found by the RL search')
            print(self.sorted_df.head(10))
        
        #decode the data before using it with DE
        if self.grid_flag:
            for i in range(self.data.shape[0]):
                self.data[i,:]=encode_grid_individual_to_discrete(self.data[i,:], bounds=self.bounds)
        
        return self.sorted_df
    
    def evolute(self, ngen, ncores=1, verbose=False):
        """
        This function evolutes the DE algorithm for number of generations with guidance from RL individuals.
        
        :param ngen: (int) number of generations to evolute
        :param ncores: (int) number of parallel processors to use with DE 
        :param verbose: (bool) print statistics to screen
        
        :return: (dict) dictionary containing major ACKTR-DE search results
        """
        print('------------------------------- Part II: DE is running and informed by ACKTR -------------------------------')
        
        if not self.ACKTR_RUN_FLAG:
            raise Exception('--error: The user is attempting to run DE before ACKTR, please use .learn first to leverage ACKTR, then use .evolute')
        
        rl_kwargs={'npop_rl': self.npop_rl, 'init_pop_rl': self.init_pop_rl, 'RLdata': self.data}
        if self.init_pop_rl:
            x0=[]
            for i in range(self.npop):
                idx=random.randint(0,self.data.shape[0]-1)
                x0.append(list(self.data[idx,:]))  
        else:
            x0=None
        
        de=DE(mode=self.mode, bounds=self.bounds, fit=self.fit, npop=self.npop, F=self.F, 
              CR=self.CR, ncores=ncores, int_transform='nearest_int', seed=self.seed, **rl_kwargs)
        x_best, y_best, es_hist=de.evolute(ngen=ngen, x0=x0, verbose=verbose)

        print('************************* ACKTR-DE Summary *************************')
        print('Best fitness (y) found:', x_best)
        print('Best individual (x) found:', y_best)
        print('******************************************************************')
            
        return x_best, y_best, es_hist