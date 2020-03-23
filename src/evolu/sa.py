#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:42:24 2020

@author: majdi
"""


import random
import gym
import pandas as pd
import numpy as np
import multiprocessing
from deap import algorithms, base, creator, tools
from src.parsers.PARSER import InputChecker
import copy, math, random, sys, time


class SAAgent(InputChecker):
    
    def __init__ (self, inp, callback):    
    
        self.inp= inp 
        self.env = gym.make(self.inp.gen_dict['env'][0], casename='sa', exepath=self.inp.gen_dict['exepath'][0])
        self.log_dir='./master_log/'+self.inp.sa_dict["casename"][0]
        self.callback=callback
        
        self.Tmax=self.inp.sa_dict['Tmax'][0]
        self.Tmin=self.inp.sa_dict['Tmin'][0]
        self.steps=self.inp.sa_dict['steps'][0]
        self.swap=self.inp.sa_dict['swap'][0]
        self.check_freq=self.inp.sa_dict['check_freq'][0]
        self.lbound=self.inp.sa_dict['lbound'][0]
        self.ubound=self.inp.sa_dict['ubound'][0]
        
        if self.inp.sa_dict['initstate'][0] == None:
            self.x=np.random.randint(self.lbound, self.ubound+1, self.inp.gen_dict['xsize'][0])
        else:
            self.x=self.inp.sa_dict['initstate'][0]
            
        self.fit=self.env.fit
        self.x_index=list(range(self.inp.gen_dict['xsize'][0])) # list of all indices of input space

    def singleswap(self) :
        # this function swaps two indices in the input vector, sutiable for input shuffling
        
        dual_index=random.sample(self.x_index,2)
        index1, index2  = dual_index[0], dual_index[1]
        
        if self.lbound == None and self.ubound == None:
            self.x[index1], self.x[index2] = self.x[index2], self.x[index1]
            print('majdi')
        elif len(self.lbound) == 1 and len(self.ubound) == 1:
            sample=np.random.randint(self.lbound, self.ubound+1)
            while self.x[index1] == sample:
                sample=np.random.randint(self.lbound, self.ubound+1)
                print('--while loop')
                print(self.x[index1])
                print(index1)
                print(sample)
                
            self.x[index1] = sample
        
        return
     
    def dualswap(self) :
        # this function swaps two indices in the input vector, sutiable for input shuffling
        
        dual_index=random.sample(self.x_index,2)
        index1, index2  = dual_index[0], dual_index[1]
        
        if self.lbound == None and self.ubound == None:
            self.x[index1], self.x[index2] = self.x[index2], self.x[index1]
        elif len(self.lbound) == 1 and len(self.ubound) == 1:
            sample1, sample2 = np.random.randint(self.lbound, self.ubound+1), np.random.randint(self.lbound, self.ubound+1)
            while self.x[index1] == sample1 or self.x[index2] == sample2:
                sample1, sample2 = np.random.randint(self.lbound, self.ubound+1), np.random.randint(self.lbound, self.ubound+1)
            self.x[index1], self.x[index2] = sample1, sample2
        
        return

    
    def quadswap(self) :
        # this function swaps four indices in the input vector, sutiable for shuffling
        
        quad_index=random.sample(self.x_index,4)
        index1, index2, index3, index4 = quad_index[0], quad_index[1], quad_index[2], quad_index[3] 
        
        if self.lbound == None and self.ubound == None:
            self.x[index1], self.x[index2], self.x[index3], self.x[index4] = self.x[index2], self.x[index1], self.x[index4], self.x[index3]
        elif len(self.lbound) == 1 and len(self.ubound) == 1:
            sample1, sample2 = np.random.randint(self.lbound, self.ubound+1), np.random.randint(self.lbound, self.ubound+1)
            sample3, sample4 = np.random.randint(self.lbound, self.ubound+1), np.random.randint(self.lbound, self.ubound+1)
            while self.x[index1] == sample1 or self.x[index2] == sample2:
                sample1, sample2 = np.random.randint(self.lbound, self.ubound+1), np.random.randint(self.lbound, self.ubound+1)
            while self.x[index3] == sample3 or self.x[index4] == sample4:
                sample3, sample4 = np.random.randint(self.lbound, self.ubound+1), np.random.randint(self.lbound, self.ubound+1)
            self.x[index1], self.x[index2] = sample1, sample2
            self.x[index3], self.x[index4] = sample3, sample4
                                              
        return 

    def fullswap(self) :
        # this function randomly shuffels all data in the input vector
        if self.lbound == None and self.ubound == None:
             self.x=random.sample( self.x,len( self.x))
        elif len(self.lbound) == 1 and len(self.ubound) == 1:
            self.x=np.random.randint(self.lbound, self.ubound+1,len(self.x))
            
        return
        
    def build(self):
        

        Tfac = -np.log(float(self.Tmax) / self.Tmin)
        T = self.Tmax
        energy = self.fit(self.x)[0] #use[0] as fit returns tuple for GA/DEAP
        
        x_prev = copy.deepcopy(self.x)
        x_best = copy.deepcopy(self.x)
        energy_prev = energy
        energy_best = energy
        
        if self.swap == 'singleswap':
            self.move = self.singleswap
        elif self.swap == 'dualswap':
            self.move = self.dualswap
        elif self.swap == 'quadswap':
            self.move = self.quadswap
        elif self.swap == 'fullswap':
            self.move = self.fullswap
        else:
            raise Exception ('--error: the swap mode for SA is not defined')
        
        trials, accepts, improves = 0, 0, 0        
        # Attempt moves to new states
        step=0
        fit_lst=[]
        while step <= self.steps:
            step += 1
            T = self.Tmax * np.exp( Tfac * step / self.steps)
            self.move()
            energy = self.fit(self.x)[0]  #use[0] as fit returns tuple for GA/DEAP
            fit_lst.append(energy)
            
            dE = energy - energy_prev
            trials += 1
            
            #print('step=', step)
            #print('energy=', energy)
            #print('state=', self.x)
            
            if dE < 0.0 and np.exp(-dE/T) > random.random():
                # Restore previous state
                self.x = copy.deepcopy(x_prev)
                energy = energy_prev
            
            else:
                # Accept new state and compare to best state
                accepts += 1
                if dE > 0.0:
                    improves += 1
                x_prev = copy.deepcopy(self.x)
                energy_prev = energy
                if energy > energy_best:
                    x_best = copy.deepcopy(self.x)
                    energy_best = energy
                    #print('Best Reward= ', energy_best)
              
            if step % self.check_freq == 0:
                
                accept_rate=accepts/trials*100
                improve_rate=improves/trials*100
                
                out_data=pd.read_csv(self.log_dir+'_out.csv')
                inp_data=pd.read_csv(self.log_dir+'_inp.csv')
                sorted_out=out_data.sort_values(by=['reward'],ascending=False)   
                sorted_inp=inp_data.sort_values(by=['reward'],ascending=False)   
                
                #------------
                # plot progress 
                #------------
                self.callback.plot_progress()
                
                #------------
                # print summary 
                #------------
                with open (self.log_dir + '_summary.txt', 'a') as fin:
                    fin.write('*****************************************************\n')
                    fin.write('Summary data for annealing step {}/{} \n'.format(step, self.steps))
                    fin.write('*****************************************************\n')
                    fin.write('Current Temperature/Max Temperature: {}/{} \n'.format(int(T), self.Tmax))
                    fin.write('Accept Rate: {}% \n'.format(accept_rate))
                    fin.write('Improvement Rate: {}% \n'.format(improve_rate))
                    fin.write ('--------------------------------------------------------------------------------------\n')
                    fin.write('Max Reward: {0:.2f} \n'.format(np.max(fit_lst)))
                    fin.write('Mean Reward: {0:.2f} \n'.format(np.mean(fit_lst)))
                    fin.write('Std Reward: {0:.2f} \n'.format(np.std(fit_lst)))
                    fin.write('Min Reward: {0:.2f} \n'.format(np.min(fit_lst)))
                    fin.write ('--------------------------------------------------------------------------------------\n')
                    fin.write ('Best output for this annealing step \n')
                    fin.write(sorted_out.iloc[0,:].to_string())
                    fin.write('\n')
                    fin.write ('-------------------------------------------------------------------------------------- \n')
                    fin.write ('Best corresponding input for this annealing step \n')
                    fin.write(sorted_inp.iloc[0,:].to_string())
                    fin.write('\n')
                    fin.write ('-------------------------------------------------------------------------------------- \n')
                    fin.write('\n\n')
                
                trials, accepts, improves = 0, 0, 0
                fit_lst=[]