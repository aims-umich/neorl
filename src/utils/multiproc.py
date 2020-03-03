#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 09:33:45 2020

@author: majdi
"""

# import input parameters from the user 
from ParamList import InputParam
from dqn import DQNAgent
from ppo2 import PPOAgent
from a2c import A2CAgent
from multiprocessing import Process
from ga import GAAgent

class MultiProc (InputParam):
    
     def __init__ (self, inp):
    
         self.inp=inp
     
     def dqn_proc(self):
         dqn=DQNAgent(self.inp)
         dqn.build()
         
         return

     def ppo_proc(self):
         dqn=PPOAgent(self.inp)
         dqn.build()
         
         return

     def a2c_proc(self):
         dqn=A2CAgent(self.inp)
         dqn.build()
         
         return

     def ga_proc(self):
         ga=GAAgent(self.inp)
         ga.build()
         
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
            
# if __name__ == '__main__':
#     inp=InputParam()
#     master=MultiProc(inp)
#     master.run_all()
    
    