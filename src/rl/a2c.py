#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:09:27 2020

@author: majdi
"""

import matplotlib
matplotlib.use('Agg')
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0, './src/utils')

# External dependencies
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from test_policy import evaluate_policy
#from stable_baselines.common.callbacks import BaseCallback
#from src.utils.neorlcalls import SavePlotCallback

#from logger import plot_print, calc_cumavg
# import input parameters from the user 
from ParamList import InputParam
        
for env in list(gym.envs.registry.env_specs):
      if 'casmo6x6' in env:
          print("Remove {} from registry".format(env))
          del gym.envs.registry.env_specs[env]

class A2CAgent(InputParam):
    def __init__ (self, inp, callback):
        self.inp=inp    
        self.callback=callback             
        self.mode=self.inp.a2c_dict['mode'][0]
            
    def make_env(self, env_id, rank, seed=0):
        #Utility function for multiprocessed env.
    
        #:param env_id: (str) the environment ID
        #:param num_env: (int) the number of environments you wish to have in subprocesses
        #:param seed: (int) the inital seed for RNG
        #:param rank: (int) index of the subprocess
        
        def _init():
            env = gym.make(env_id, casename=self.inp.a2c_dict['casename'][0])
            env.seed(seed + rank)
            return env
        set_global_seeds(seed)
        return _init
    
    def build (self):
        
        # Create the vectorized environment
        if self.inp.a2c_dict['ncores'][0] > 1:
            self.env = SubprocVecEnv([self.make_env(self.inp.gen_dict['env'][0], i) for i in range(self.inp.a2c_dict['ncores'][0])])
        else:
            self.env = gym.make(self.inp.gen_dict['env'][0], casename=self.inp.a2c_dict['casename'][0])
        

        if self.mode == 'train':
        #tensorboard --logdir=logs --host localhost --port 8088
            model = A2C(MlpPolicy, self.env,
                        n_steps=self.inp.a2c_dict['n_steps'][0],
                        gamma=self.inp.a2c_dict['gamma'][0], 
                        learning_rate=self.inp.a2c_dict['learning_rate'][0], 
                        vf_coef=self.inp.a2c_dict['vf_coef'][0], 
                        max_grad_norm=self.inp.a2c_dict['max_grad_norm'][0], 
                        ent_coef=self.inp.a2c_dict['ent_coef'][0], 
                        alpha=self.inp.a2c_dict['alpha'][0], 
                        epsilon=self.inp.a2c_dict['epsilon'][0], 
                        lr_schedule=self.inp.a2c_dict['lr_schedule'][0],
                        verbose=1)
            model.learn(total_timesteps=self.inp.a2c_dict['time_steps'][0], callback=self.callback)
            model.save('./master_log/'+self.inp.a2c_dict['casename'][0]+'_model_last.pkl')
        
        if self.mode=='continue':
            
            model = A2C.load(self.inp.a2c_dict['model_load_path'][0], env=self.env)
            model.learn(total_timesteps=self.inp.a2c_dict['time_steps'][0], callback=self.callback)
            model.save('./master_log/'+self.inp.a2c_dict['casename'][0]+'_lastmodel.pkl')

            
        if self.mode=='test':
            
            print('debug: a2c is running in test mode, single core is used to test the policy')
            env = gym.make(self.inp.gen_dict['env'][0], casename=self.inp.a2c_dict['casename'][0])
            model = A2C.load(self.inp.a2c_dict['model_load_path'][0])
            evaluate_policy(model, env, log_dir='./master_log/a2c', 
                            n_eval_episodes=self.inp.a2c_dict["n_eval_episodes"][0], render=self.inp.a2c_dict["render"][0], 
                            video_record=self.inp.a2c_dict["video_record"][0], fps=self.inp.a2c_dict["fps"][0])