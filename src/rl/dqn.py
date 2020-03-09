#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:29:00 2020

@author: majdi
"""


import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# External dependencies
import gym
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from stable_baselines.common.vec_env import VecVideoRecorder
from test_policy import evaluate_policy


# import input parameters from the user 
from ParamList import InputParam

class DQNAgent(InputParam):
    def __init__ (self, inp, callback):
        self.inp=inp    
        self.callback=callback 
        self.mode=self.inp.dqn_dict['mode'][0]
        self.env = gym.make(self.inp.gen_dict['env'][0], casename=self.inp.dqn_dict['casename'][0])
        
    def build (self):
        
        if self.mode == 'train':
        #tensorboard --logdir=logs --host localhost --port 8088
            model = DQN(MlpPolicy, self.env,
                        gamma=self.inp.dqn_dict['gamma'][0], 
                        learning_rate=self.inp.dqn_dict['learning_rate'][0], 
                        buffer_size=self.inp.dqn_dict['buffer_size'][0], 
                        exploration_fraction=self.inp.dqn_dict['exploration_fraction'][0], 
                        exploration_final_eps=self.inp.dqn_dict['exploration_final_eps'][0], 
                        learning_starts=self.inp.dqn_dict['learning_starts'][0], 
                        batch_size=self.inp.dqn_dict['batch_size'][0], 
                        target_network_update_freq=self.inp.dqn_dict['target_network_update_freq'][0],
                        exploration_initial_eps=self.inp.dqn_dict['exploration_initial_eps'][0],
                        train_freq=self.inp.dqn_dict['train_freq'][0],
                        double_q=self.inp.dqn_dict['double_q'][0],
                        verbose=2)
            model.learn(total_timesteps=self.inp.dqn_dict['time_steps'][0], callback=self.callback)
            model.save('./master_log/'+self.inp.dqn_dict['casename'][0]+'_lastmodel.pkl')
        
        if self.mode=='continue':
            
            model = DQN.load(self.inp.dqn_dict['model_load_path'][0], env=self.env)
            model.learn(total_timesteps=self.inp.dqn_dict['time_steps'][0], callback=self.callback)
            model.save('./master_log/'+self.inp.dqn_dict['casename'][0]+'_lastmodel.pkl')
            
        if self.mode=='test':
            print('debug: dqn is running in test mode, single core is used to test the policy')
            env = gym.make(self.inp.gen_dict['env'][0], casename=self.inp.dqn_dict['casename'][0])
            model = DQN.load(self.inp.dqn_dict['model_load_path'][0])
            evaluate_policy(model, env, log_dir='./master_log/dqn', 
                            n_eval_episodes=self.inp.dqn_dict["n_eval_episodes"][0], render=self.inp.dqn_dict["render"][0], 
                            video_record=self.inp.dqn_dict["video_record"][0], fps=self.inp.dqn_dict["fps"][0])            
    
            
            
# if __name__ =='__main__':
#      inp=InputParam()
#      dqn=DQNAgent(inp)
#      dqn.build()