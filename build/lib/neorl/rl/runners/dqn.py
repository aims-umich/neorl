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
Created on Tue Feb 25 10:29:00 2020

@author: majdi
"""


import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")

#import os, sys
#print(os.getcwd())

#rl_path = os.path.dirname(os.path.realpath(__file__))
#sys.path.append(os.path.join(rl_path,'baselines'))
#print(sys.path)

#print(rl_path)


# External dependencies
import gym
from neorl.rl.baselines.deepq.policies import MlpPolicy
from neorl.rl.baselines.deepq.dqn import DQN
from neorl.utils.test_policy import evaluate_policy
# import input parameters from the user 
from neorl.parsers.PARSER import InputChecker


class DQNAgent(InputChecker):
    def __init__ (self, inp, callback):
        """
        Input: 
            inp: is a dictionary of validated user input {"ncores": 8, "env": 6x6, ...}
            callback: a class of callback built from stable-baselines to allow intervening during training 
                      to process data and save models
        """
        # self.inp.gen_dict ---> GENERAL CARD: env, env_data, nactions, ...etc
        # self.inp.dqn_dict ---> DQN CARD 
        self.inp=inp    # the full user input dictionary
        self.callback=callback 
        self.mode=self.inp.dqn_dict['mode'][0]
        self.log_dir=self.inp.gen_dict['log_dir']
        self.env = gym.make(self.inp.gen_dict['env'][0], casename=self.inp.dqn_dict['casename'][0], 
                            log_dir=self.log_dir, exepath=self.inp.gen_dict['exepath'][0], env_data=self.inp.gen_dict['env_data'][0], env_seed=1)
        
    def build (self):
        """
        This function builds the DQN agent (ONLY 1 Env/Core is supported) based on the selected mode and runs it according to:
            1- Initializes the env
            2- If mode train is selected, train the model from scratch, learn, and save.
            3- If mode continue is selected, provide a path for pretrained model, load the model, learn, and save.
            4- If mode test is selected, provide a path fror pretrained model, load the model and test. 
        """
        #tensorboard activation (if used)
        #to view tensorboard type
        #tensorboard --logdir=./log_dir/{self.casename}_tensorlog 
        if self.inp.dqn_dict['tensorboard'][0]:
            tensorboard_log=self.log_dir+'{}_tensorlog'.format(self.inp.dqn_dict['casename'][0])
        else:
            tensorboard_log=None
            
        
        if self.mode == 'train':
            # Train from scratch, initialize the model and then learn, and save the last model.
            # Callbacks are used if provided 
            model = DQN(MlpPolicy, self.env,
                        gamma=self.inp.dqn_dict['gamma'][0], 
                        learning_rate=self.inp.dqn_dict['learning_rate'][0], 
                        buffer_size=self.inp.dqn_dict['buffer_size'][0], 
                        exploration_fraction=self.inp.dqn_dict['exploration_fraction'][0], 
                        eps_final=self.inp.dqn_dict['eps_final'][0], 
                        learning_starts=self.inp.dqn_dict['learning_starts'][0], 
                        batch_size=self.inp.dqn_dict['batch_size'][0], 
                        target_network_update_freq=self.inp.dqn_dict['target_network_update_freq'][0],
                        eps_init=self.inp.dqn_dict['eps_init'][0],
                        train_freq=self.inp.dqn_dict['train_freq'][0],
                        prioritized_replay=self.inp.dqn_dict['prioritized_replay'][0],
                        verbose=2, seed=1)
            model.learn(total_timesteps=self.inp.dqn_dict['time_steps'][0], callback=self.callback)
            model.save(self.log_dir+self.inp.dqn_dict['casename'][0]+'_lastmodel.pkl')
        
        if self.mode=='continue':
            # load, contine learning, and save last model
            model = DQN.load(self.inp.dqn_dict['model_load_path'][0], env=self.env,
                        gamma=self.inp.dqn_dict['gamma'][0], 
                        learning_rate=self.inp.dqn_dict['learning_rate'][0], 
                        buffer_size=self.inp.dqn_dict['buffer_size'][0], 
                        exploration_fraction=self.inp.dqn_dict['exploration_fraction'][0], 
                        eps_final=self.inp.dqn_dict['eps_final'][0], 
                        learning_starts=self.inp.dqn_dict['learning_starts'][0], 
                        batch_size=self.inp.dqn_dict['batch_size'][0], 
                        target_network_update_freq=self.inp.dqn_dict['target_network_update_freq'][0],
                        eps_init=self.inp.dqn_dict['eps_init'][0],
                        train_freq=self.inp.dqn_dict['train_freq'][0],
                        prioritized_replay=self.inp.dqn_dict['prioritized_replay'][0],
                        verbose=2, seed=1)

            model.learn(total_timesteps=self.inp.dqn_dict['time_steps'][0], callback=self.callback)
            model.save(self.log_dir+self.inp.dqn_dict['casename'][0]+'_lastmodel.pkl')
            
        if self.mode=='test':
            # load and test the agent. Env is recreated since test mode only works in single core
            print('debug: dqn is running in test mode, single core is used to test the policy')
            env = gym.make(self.inp.gen_dict['env'][0], casename=self.inp.dqn_dict['casename'][0],  log_dir=self.log_dir, 
                           exepath=self.inp.gen_dict['exepath'][0], env_data=self.inp.gen_dict['env_data'][0], env_seed=1)
            model = DQN.load(self.inp.dqn_dict['model_load_path'][0])
            evaluate_policy(model, env, log_dir=self.log_dir+'dqn', 
                            n_eval_episodes=self.inp.dqn_dict["n_eval_episodes"][0], render=self.inp.dqn_dict["render"][0])            
    
            
            
# if __name__ =='__main__':
#      inp=InputParam()
#      dqn=DQNAgent(inp)
#      dqn.build()
