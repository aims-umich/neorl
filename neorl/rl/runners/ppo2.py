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
"""
Created on Tue Feb 25 12:09:27 2020

@author: majdi
"""

import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")
# External dependencies
import gym
from neorl.rl.baselines.shared.policies import MlpPolicy
from neorl.rl.baselines.ppo2.ppo2 import PPO2
from neorl.rl.baselines.shared import set_global_seeds
from neorl.rl.baselines.shared.vec_env import SubprocVecEnv
#from neorl.utils.vec_env import SubprocVecEnv
from neorl.utils.test_policy import evaluate_policy
# import input parameters from the user 
from neorl.parsers.PARSER import InputChecker
import time

class PPOAgent(InputChecker):
    def __init__ (self, inp, callback):
        """
        Input: 
            inp: is a dictionary of validated user input {"ncores": 8, "env": 6x6, ...}
            callback: a class of callback built from stable-baselines to allow intervening during training 
                      to process data and save models
        """
        # self.inp.gen_dict ---> GENERAL CARD: env, env_data, nactions, ...etc
        # self.inp.ppo_dict ---> PPO CARD 
        self.inp=inp      # the full user input dictionary
        self.callback=callback                   
        self.mode=self.inp.ppo_dict['mode'][0]
        self.log_dir=self.inp.gen_dict['log_dir']
        set_global_seeds(3)
            
    def make_env(self, env_id, rank, seed=0):
        """
        This function makes multiprocessed/parallel envs based on gym.make with specific seeds
        env_id: (str) the environment ID
        num_env: (int) the number of environments you wish to have in subprocesses
        seed: (int) the inital seed for RNG
        rank: (int) index of the subprocess
        
        Returns: _init, which is a gym enviroment with specific seed
        """
        def _init():
            env = gym.make(env_id, casename=self.inp.ppo_dict['casename'][0], exepath=self.inp.gen_dict['exepath'][0], 
                           log_dir=self.log_dir, env_data=self.inp.gen_dict['env_data'][0], env_seed=seed+rank)
            env.seed(seed+rank)
            return env
        set_global_seeds(seed)
        return _init
        
    def build (self):
        """
        This function builds the PPO agent based on the selected mode and runs it according to:
            1- Initializes the env
            2- If mode train is selected, train the model from scratch, learn, and save.
            3- If mode continue is selected, provide a path for pretrained model, load the model, learn, and save.
            4- If mode test is selected, provide a path fror pretrained model, load the model and test. 
        """
        # Create the vectorized environment
        if self.inp.ppo_dict['ncores'][0] > 1:
            self.env = SubprocVecEnv([self.make_env(self.inp.gen_dict['env'][0], i) for i in range(self.inp.ppo_dict['ncores'][0])], daemon=self.inp.gen_dict['daemon'][0])
        else:
            self.env = gym.make(self.inp.gen_dict['env'][0], casename=self.inp.ppo_dict['casename'][0], log_dir=self.log_dir,
                                exepath=self.inp.gen_dict['exepath'][0], env_data=self.inp.gen_dict['env_data'][0], env_seed=1)
        
        
        #tensorboard activation (if used)
        #to view tensorboard type
        #tensorboard --logdir=./log_dir/{self.casename}_tensorlog 
        if self.inp.ppo_dict['tensorboard'][0]:
            tensorboard_log=self.log_dir+'{}_tensorlog'.format(self.inp.ppo_dict['casename'][0])
        else:
            tensorboard_log=None
        
        
        if self.mode == 'train':
            # Train from scratch, initialize the model and then learn, and save the last model.
            # Callbacks are used if provided 
            model = PPO2(MlpPolicy, self.env,
                        n_steps=self.inp.ppo_dict['n_steps'][0],
                        gamma=self.inp.ppo_dict['gamma'][0], 
                        learning_rate=self.inp.ppo_dict['learning_rate'][0], 
                        ent_coef=self.inp.ppo_dict['ent_coef'][0],
                        vf_coef=self.inp.ppo_dict['vf_coef'][0], 
                        max_grad_norm=self.inp.ppo_dict['max_grad_norm'][0], 
                        lam=self.inp.ppo_dict['lam'][0], 
                        nminibatches=self.inp.ppo_dict['nminibatches'][0], 
                        noptepochs=self.inp.ppo_dict['noptepochs'][0], 
                        cliprange=self.inp.ppo_dict['cliprange'][0],
                        verbose=1,seed=3)
            model.learn(total_timesteps=self.inp.ppo_dict['time_steps'][0], callback=self.callback)
            model.save(self.log_dir+self.inp.ppo_dict['casename'][0]+'_lastmodel.pkl')
            
        if self.mode=='continue':
            # load, contine learning, and save last model  
            model = PPO2.load(self.inp.ppo_dict['model_load_path'][0], env=self.env,
                        n_steps=self.inp.ppo_dict['n_steps'][0],
                        gamma=self.inp.ppo_dict['gamma'][0], 
                        learning_rate=self.inp.ppo_dict['learning_rate'][0], 
                        ent_coef=self.inp.ppo_dict['ent_coef'][0],
                        vf_coef=self.inp.ppo_dict['vf_coef'][0], 
                        max_grad_norm=self.inp.ppo_dict['max_grad_norm'][0], 
                        lam=self.inp.ppo_dict['lam'][0], 
                        nminibatches=self.inp.ppo_dict['nminibatches'][0], 
                        noptepochs=self.inp.ppo_dict['noptepochs'][0], 
                        cliprange=self.inp.ppo_dict['cliprange'][0],
                        verbose=1,seed=3)
            model.learn(total_timesteps=self.inp.ppo_dict['time_steps'][0], callback=self.callback)
            model.save(self.log_dir+self.inp.ppo_dict['casename'][0]+'_lastmodel.pkl')
            
        if self.mode=='test':
            # load and test the agent. Env is recreated since test mode only works in single core
            print('debug: ppo is running in test mode, single core is used to test the policy')
            env = gym.make(self.inp.gen_dict['env'][0], log_dir=self.log_dir, casename=self.inp.ppo_dict['casename'][0], 
                           exepath=self.inp.gen_dict['exepath'][0], env_data=self.inp.gen_dict['env_data'][0], env_seed=1)
            model = PPO2.load(self.inp.ppo_dict['model_load_path'][0])
            evaluate_policy(model, env, log_dir=self.log_dir+'ppo', 
                            n_eval_episodes=self.inp.ppo_dict["n_eval_episodes"][0], render=self.inp.ppo_dict["render"][0])
                        
                        
# if __name__ =='__main__':
#     inp=InputParam()
#     ppo=PPOAgent(inp)
#     ppo.build()

