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


# import input parameters from the user 
from ParamList import InputParam

def calc_ma(data, N):
    cumsum, moving_aves = [0], []
    
    for i, x in enumerate(data, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    return moving_aves


def calc_cumavg(data, N):
    cum_aves = []

    epochs=int(len(data)/N)
    index=0
    for i in range(epochs):
        cum_aves.append(np.mean(data[index:index+N]))
        index+=N
        
    return cum_aves


epochs_rewards=[]

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % (eps_per_epo * 21) == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-eps_per_epo:])
            print('-----------------------------------------------------------------------------------------')
            print(x[-1], 'timesteps')
            # print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save('best_model.pkl')

            epochs_rewards.append(mean_reward)
            print('Epoch {:d} is ended, Best mean reward: {:.2f}, Current mean reward: {:.2f}'.format(int(n_steps / (eps_per_epo * 21)+1), best_mean_reward, mean_reward))
            print('-----------------------------------------------------------------------------------------')
            
            if (n_steps + 1) > warmup_epochs * eps_per_epo * 21 * 2 :
                avg_loss=pd.read_csv('loss_logger.txt', index_col=False).values[:,2]
                plt.figure(figsize=(5,6))
                plt.subplot(211)
                plt.plot(calc_ma(y,eps_per_epo),'b')
                plt.xlabel('Episode'); plt.ylabel('Average Reward')
                
                plt.subplot(212)
                plt.plot(calc_ma(avg_loss,eps_per_epo),'g')
                plt.xlabel('Episode'); plt.ylabel('Average Loss')
                plt.tight_layout()
                plt.savefig('moving_metrics.png', format='png', dpi=200)
                plt.close()
                
                plt.figure(figsize=(5, 6))
                plt.subplot(211)
                plt.plot(calc_cumavg(y,eps_per_epo),'-ob')
                plt.xlabel('Epoch'); plt.ylabel('Reward')
                
                plt.subplot(212)
                plt.plot(calc_cumavg(avg_loss,eps_per_epo),'-og')
                plt.xlabel('Epoch'); plt.ylabel('Loss')
                plt.tight_layout()
                plt.savefig('epoch_metrics.png', format='png', dpi=200)   
                plt.close()
            
    n_steps += 1
    return True

class DQNAgent(InputParam):
    def __init__ (self, inp):
        self.inp=inp
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
            model.learn(total_timesteps=self.inp.dqn_dict['time_steps'][0], callback=None)
            model.save('./master_log/'+self.inp.dqn_dict['casename'][0]+'_model_last.pkl')
        
        if self.mode=='continue':
            
            model = DQN.load(self.inp.dqn_dict['model_load_path'][0], env=self.env)
            model.learn(total_timesteps=self.inp.dqn_dict['time_steps'][0], callback=None)
            model.save('./master_log/'+self.inp.dqn_dict['casename'][0]+'_model_last.pkl')
            
        if self.mode=='test':
     
            model = DQN.load(self.inp.dqn_dict['model_load_path'][0], env=self.env)
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
#      inp=InputParam()
#      dqn=DQNAgent(inp)
#      dqn.build()