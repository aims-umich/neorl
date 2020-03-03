#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:09:27 2020

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
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv, VecVideoRecorder

# import input parameters from the user 
from ParamList import InputParam

for env in list(gym.envs.registry.env_specs):
      if 'casmo6x6' in env:
          print("Remove {} from registry".format(env))
          del gym.envs.registry.env_specs[env]


class PPOAgent(InputParam):
    def __init__ (self, inp):
        self.inp = inp                 
        self.mode=self.inp.ppo_dict['mode'][0]
            
    def make_env(self, env_id, rank, seed=0):
        """
        Utility function for multiprocessed env.
    
        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environments you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            env = gym.make(env_id, casename=self.inp.ppo_dict['casename'][0])
            env.seed(seed + rank)
            return env
        set_global_seeds(seed)
        return _init
        
    def build (self):
        
        # if __name__ =='__main__':
        
        # Create the vectorized environment
        if self.inp.ppo_dict['ncores'][0] > 1:
            self.env = SubprocVecEnv([self.make_env(self.inp.gen_dict['env'][0], i) for i in range(self.inp.ppo_dict['ncores'][0])])
        else:
            self.env = gym.make(self.inp.gen_dict['env'][0], casename=self.inp.ppo_dict['casename'][0])
        
        #print('majdi')
        if self.mode == 'train':
        #tensorboard --logdir=logs --host localhost --port 8088
            model = PPO2(MlpPolicy, self.env,
                        n_steps=self.inp.ppo_dict['n_steps'][0],
                        gamma=self.inp.ppo_dict['gamma'][0], 
                        learning_rate=self.inp.ppo_dict['learning_rate'][0], 
                        vf_coef=self.inp.ppo_dict['vf_coef'][0], 
                        max_grad_norm=self.inp.ppo_dict['max_grad_norm'][0], 
                        lam=self.inp.ppo_dict['lam'][0], 
                        nminibatches=self.inp.ppo_dict['nminibatches'][0], 
                        noptepochs=self.inp.ppo_dict['noptepochs'][0], 
                        cliprange=self.inp.ppo_dict['cliprange'][0],
                        verbose=1)
            model.learn(total_timesteps=self.inp.ppo_dict['time_steps'][0], callback=None)
            model.save("last_model.pkl")
            model.save('./master_log/'+self.inp.ppo_dict['casename'][0]+'_model_last.pkl')
        
        if self.mode=='continue':
            
            model = PPO2.load(self.inp.ppo_dict['model_load_path'][0], env=self.env)
            model.learn(total_timesteps=self.inp.ppo_dict['time_steps'][0], callback=None)
            model.save('./master_log/'+self.inp.ppo_dict['casename'][0]+'_model_last.pkl')
            
        if self.mode=='test':
     
            model = PPO2.load(self.inp.ppo_dict['model_load_path'][0])
            #mean_reward, n_steps = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    
            self.env.reset()
            obs=self.env.reset()
            test_eps=10
            for i in range (test_eps):
                for j in range (21):
                    action, _states = model.predict(obs)
                    obs, rewards, dones, info = self.env.step(action)
                    # print(obs, rewards, dones, action)
                    if (j==20):
                        self.env.render()
                    if dones:
                        self.env.reset()
                        
                        
# if __name__ =='__main__':
#     inp=InputParam()
#     ppo=PPOAgent(inp)
#     ppo.build()
