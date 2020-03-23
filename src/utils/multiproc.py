#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 09:33:45 2020

@author: majdi
"""
import os
#import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# import input parameters from the user 
from src.parsers.PARSER import InputChecker
from src.rl.dqn import DQNAgent
from src.rl.ppo2 import PPOAgent
from src.rl.a2c import A2CAgent
from multiprocessing import Process
from src.evolu.ga import GAAgent
from src.evolu.sa import SAAgent
from src.utils.neorlcalls import SavePlotCallback
from stable_baselines.common.callbacks import BaseCallback


class MultiProc (InputChecker):
    
     def __init__ (self, inp):
    
         self.inp=inp
         os.environ["KMP_WARNINGS"] = "FALSE"
         #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
     
     def dqn_proc(self):
         dqn_callback=SavePlotCallback(check_freq=self.inp.dqn_dict["check_freq"][0], avg_step=self.inp.dqn_dict["avg_episodes"][0], 
                                       log_dir='./master_log/'+self.inp.dqn_dict["casename"][0], plot_mode=self.inp.gen_dict["plot_mode"][0],
                                       total_timesteps=self.inp.dqn_dict["time_steps"][0], basecall=BaseCallback())
         dqn=DQNAgent(self.inp, dqn_callback)
         dqn.build()
         
         return

     def ppo_proc(self):
         ppo_callback=SavePlotCallback(check_freq=self.inp.ppo_dict["check_freq"][0], avg_step=self.inp.ppo_dict["avg_episodes"][0], 
                                       log_dir='./master_log/'+self.inp.ppo_dict["casename"][0], plot_mode=self.inp.gen_dict["plot_mode"][0],
                                       total_timesteps=self.inp.ppo_dict["time_steps"][0], basecall=BaseCallback())
         ppo=PPOAgent(self.inp, ppo_callback)
         ppo.build()
         
         return

     def a2c_proc(self):
         a2c_callback=SavePlotCallback(check_freq=self.inp.a2c_dict["check_freq"][0], avg_step=self.inp.a2c_dict["avg_episodes"][0], 
                                       log_dir='./master_log/'+self.inp.a2c_dict["casename"][0], plot_mode=self.inp.gen_dict["plot_mode"][0],
                                       total_timesteps=self.inp.a2c_dict["time_steps"][0], basecall=BaseCallback())
         a2c=A2CAgent(self.inp, a2c_callback)
         a2c.build()
         
         return

     def ga_proc(self):
         #check_freq is set by default to every generation for GA, so it does not have any effect in the callback 
         #total_timesteps is set by default to all generations for GA,  so it does not have any effect in the callback 
         ga_callback=SavePlotCallback(check_freq=self.inp.ga_dict["pop"][0], avg_step=self.inp.ga_dict["pop"][0], 
                               log_dir='./master_log/'+self.inp.ga_dict["casename"][0], plot_mode=self.inp.gen_dict["plot_mode"][0],
                               total_timesteps=self.inp.ga_dict["ngen"][0], basecall=BaseCallback())
         ga=GAAgent(self.inp, ga_callback)
         ga.build()
         
         return
     
     def sa_proc(self):
         #avg_step is set by default to every check_freq, so it does not have any effect in the callback 
         #total_timesteps is set by default to all generations for SA,  so it does not have any effect in the callback 
         sa_callback=SavePlotCallback(check_freq=self.inp.sa_dict["check_freq"][0], avg_step=self.inp.sa_dict["check_freq"][0], 
                               log_dir='./master_log/'+self.inp.sa_dict["casename"][0], plot_mode=self.inp.gen_dict["plot_mode"][0],
                               total_timesteps=self.inp.sa_dict["steps"][0], basecall=BaseCallback())
         sa=SAAgent(self.inp, sa_callback)
         sa.build()
         
         return

     def run_all(self):
        
        # setup all processes
        if self.inp.dqn_dict['flag'][0]:
            dqn_task = Process(name='dqn', target=self.dqn_proc)
            
        if self.inp.ppo_dict['flag'][0]:
            ppo_task = Process(name='ppo', target=self.ppo_proc)

        if self.inp.a2c_dict['flag'][0]:
            a2c_task = Process(name='a2c', target=self.a2c_proc)       
            
        if self.inp.ga_dict['flag'][0]:
            ga_task = Process(name='ga', target=self.ga_proc)   

        if self.inp.sa_dict['flag'][0]:
            sa_task = Process(name='sa', target=self.sa_proc)
            
        # start running processes
        if self.inp.dqn_dict['flag'][0]:
            dqn_task.start()
            print('--- DQN is running on {cores} core(s)'.format(cores=self.inp.dqn_dict["ncores"][0]))
            
        if self.inp.ppo_dict['flag'][0]:
            ppo_task.start()
            print('--- PPO is running on {cores} core(s)'.format(cores=self.inp.ppo_dict["ncores"][0]))

        if self.inp.a2c_dict['flag'][0]:
            a2c_task.start()
            print('--- A2C is running on {cores} core(s)'.format(cores=self.inp.a2c_dict["ncores"][0]))

        if self.inp.ga_dict['flag'][0]:
            ga_task.start()
            print('--- GA is running on {cores} core(s)'.format(cores=self.inp.ga_dict["ncores"][0]))
            
        if self.inp.sa_dict['flag'][0]:
            sa_task.start()
            print('--- SA is running on {cores} core(s)'.format(cores=self.inp.sa_dict["ncores"][0]))
        
        print('------------------------------------------------------------------------------')
            
# if __name__ == '__main__':
#     inp=InputParam()
#     master=MultiProc(inp)
#     master.run_all()
    
    
