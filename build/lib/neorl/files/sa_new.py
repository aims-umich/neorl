#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:42:24 2020

@author: majdi
"""

import warnings
warnings.filterwarnings("ignore")
import random
import gym
import pandas as pd
import numpy as np
from multiprocessing import Pool
from deap import algorithms, base, creator, tools
from neorl.parsers.PARSER import InputChecker
import copy, math, random, sys, time

class SAAgent(InputChecker):
    
    def __init__ (self, inp, callback):    
        
        
        self.inp= inp 
        self.env = gym.make(self.inp.gen_dict['env'][0], casename='sa', exepath=self.inp.gen_dict['exepath'][0])
        
        #np.random.seed(10)
        self.log_dir=self.inp.gen_dict['log_dir']+self.inp.sa_dict["casename"][0]
        self.callback=callback
        
        self.Tmax=self.inp.sa_dict['Tmax'][0]
        self.Tmin=self.inp.sa_dict['Tmin'][0]
        self.steps=self.inp.sa_dict['steps'][0]
        self.swap=self.inp.sa_dict['swap'][0]
        self.check_freq=self.inp.sa_dict['check_freq'][0]
        self.lbound=self.inp.sa_dict['lbound'][0]
        self.ubound=self.inp.sa_dict['ubound'][0]
        self.cooling=self.inp.sa_dict['cooling'][0]
        self.ncores=self.inp.sa_dict['ncores'][0]
        self.fit=self.env.fit
        self.x_index=list(range(self.inp.gen_dict['xsize'][0])) # list of all indices of input space
        self.core_seed=list(range(self.ncores)) # list of all indices of input space

    def singleswap(self,x) :
        # this function swaps two indices in the input vector, sutiable for input shuffling
        
        dual_index=random.sample(self.x_index, 2)
        index1, index2  = dual_index[0], dual_index[1]
        
        if self.lbound == None and self.ubound == None:
            x[index1], x[index2] = x[index2], x[index1]
        elif len(self.lbound) == 1 and len(self.ubound) == 1:
            sample=np.random.randint(self.lbound, self.ubound+1)
            while x[index1] == sample:
                sample=np.random.randint(self.lbound, self.ubound+1)
                
            x[index1] = sample
        
        return x
     
    def dualswap(self, x) :
        # this function swaps two indices in the input vector, sutiable for input shuffling
        
        dual_index=random.sample(self.x_index, 2)
        index1, index2  = dual_index[0], dual_index[1]
        
        print(index1,index2)
        if self.lbound == None and self.ubound == None:
            x[index1], x[index2] = x[index2], x[index1]
        elif len(self.lbound) == 1 and len(self.ubound) == 1:
            sample1, sample2 = np.random.randint(self.lbound, self.ubound+1), np.random.randint(self.lbound, self.ubound+1)
            while x[index1] == sample1 or x[index2] == sample2:
                sample1, sample2 = np.random.randint(self.lbound, self.ubound+1), np.random.randint(self.lbound, self.ubound+1)
            x[index1], x[index2] = sample1, sample2
        
        return x

    
    def quadswap(self, x) :
        # this function swaps four indices in the input vector, sutiable for shuffling
        
        quad_index=random.sample(self.x_index, 4)
        index1, index2, index3, index4 = quad_index[0], quad_index[1], quad_index[2], quad_index[3] 
        
        if self.lbound == None and self.ubound == None:
            x[index1], x[index2], x[index3], x[index4] = x[index2], x[index1], x[index4], x[index3]
        elif len(self.lbound) == 1 and len(self.ubound) == 1:
            sample1, sample2 = np.random.randint(self.lbound, self.ubound+1), np.random.randint(self.lbound, self.ubound+1)
            sample3, sample4 = np.random.randint(self.lbound, self.ubound+1), np.random.randint(self.lbound, self.ubound+1)
            while x[index1] == sample1 or x[index2] == sample2:
                sample1, sample2 = np.random.randint(self.lbound, self.ubound+1), np.random.randint(self.lbound, self.ubound+1)
            while x[index3] == sample3 or x[index4] == sample4:
                sample3, sample4 = np.random.randint(self.lbound, self.ubound+1), np.random.randint(self.lbound, self.ubound+1)
            x[index1], x[index2] = sample1, sample2
            x[index3], x[index4] = sample3, sample4
                                              
        return x

    def fullswap(self, x) :
        # this function randomly shuffels all data in the input vector
        if self.lbound == None and self.ubound == None:
             x=random.sample(x, len(x))
        elif len(self.lbound) == 1 and len(self.ubound) == 1:
            x=np.random.randint(self.lbound, self.ubound+1,len(x))
            
        return x
    
    def chain_object (self,inp):
        
        x=inp[0]
        step=inp[1]
        core_seed=inp[2]
        
        if self.initflag:
            #random.seed(core_seed)
            if self.inp.sa_dict['initstate'][0] == None:
                if self.lbound == None and self.lbound == None:
                    random.seed(core_seed)
                    x=random.sample(range(0,self.inp.gen_dict['xsize'][0]),self.inp.gen_dict['xsize'][0])
                else:
                    np.random.seed(core_seed)
                    x=np.random.randint(self.lbound, self.ubound+1, self.inp.gen_dict['xsize'][0])
            else:
                x=self.inp.sa_dict['initstate'][0] 
        
            reward = self.fit(x)[0] #use[0] as fit returns tuple for GA/DEAP
            T=self.Tmax
            xnew=copy.deepcopy(x)
        
        else:
            #random.seed(core_seed)
            if self.cooling=='fast':
                Tfac = -np.log(float(self.Tmax) / self.Tmin)
                T = self.Tmax * np.exp( Tfac * step / self.steps)
            elif self.cooling=='boltzmann':
                T = self.Tmax / np.log(step + 1)
            elif self.cooling=='cauchy':
                T = self.Tmax / (step + 1)
            else:
                raise Exception ('--error: unknown cooling mode is entered, fast, boltzmann, or cauchy are ONLY allowed')
                
            xnew=copy.deepcopy(self.move(x))
            #SA is programmed to minimize but we are maximizing rewards 
            # so we will be minimizing -reward, which is essentially maximizing reward
            reward=self.fit(xnew)[0] #use[0] as fit returns tuple for GA/DEAP
        
        return xnew,-reward, T # negative sign is used for reward to convert reward (max) to energy (min)

    def build(self):
        
        #random.seed(10)
        
        T = self.Tmax
        
        if self.swap == 'singleswap':
            self.move = self.singleswap
        elif self.swap == 'dualswap':
            self.move = self.dualswap
        elif self.swap == 'quadswap':
            self.move = self.quadswap
        elif self.swap == 'fullswap':
            self.move = self.fullswap
        else:
            raise Exception ('--error: the swap mode for SA is not defined, use singleswap, dualswap, quadswap, or fullswap')
        
        self.initflag=True
        inp_list=[]
        for i in range(self.ncores):
            inp_list.append([0,0,i])
        # Pool to get initial guesses for the SA chains
        pinit=Pool(self.ncores)
        results = pinit.map(self.chain_object, inp_list)
        pinit.close(); pinit.join()
        
        print(results)
        min_index=[y[1] for y in results].index(min([y[1] for y in results]))
        
        #print(max_index)
        # Initialize the master memory with best result so far
        x_prev = copy.deepcopy(results[min_index][0])
        x_best = copy.deepcopy(results[min_index][0])
        self.x=copy.deepcopy(results[min_index][0])
        energy_prev = results[min_index][1] 
        energy_best = results[min_index][1]
        
        #print(x_prev,x_best)
        #print(energy_prev,energy_best)
        
        trials, accepts, improves = 0, 0, 0        
        # Attempt moves to new states
        step=0
        self.initflag=False
        fit_lst=[]
        for step in range (1,self.steps+1,self.ncores):
            
            #print('current:', self.x)
            print('previos:', x_prev)
            
            
            inp_list=[]
            for j in range(self.ncores):
                inp_list.append([self.x,step,j])

            p=Pool(self.ncores)
            results = p.map(self.chain_object, inp_list)
            p.close(); p.join()
            
            #print(results)
            min_index=[y[1] for y in results].index(min([y[1] for y in results]))
            
            #print(min_index)
            
            self.x=copy.deepcopy(results[min_index][0])
            self.energy=results[min_index][1]
            T=results[min_index][2]
            fit_lst.append(self.energy)
            print('T=',T)
            
            dE = self.energy - energy_prev
            #print(dE)
            trials += 1
            
            #np.random.seed(1)
            alpha=np.random.random()
            #print('alpha=',alpha)
            if dE > 0.0 and np.exp(-dE/T) < alpha:
                # Restore previous state
                self.x = copy.deepcopy(x_prev)
                self.energy = energy_prev
            else:
                # Accept new state and compare to best state
                accepts += 1
                print('accept')
                if dE < 0.0:
                    improves += 1
                    print('improve')
                
                x_prev = copy.deepcopy(self.x)
                energy_prev = self.energy
                if self.energy < energy_best:
                    x_best = copy.deepcopy(self.x)
                    energy_best = self.energy
              
            if step % self.check_freq == 0 or step==self.steps:
                
                accept_rate=np.round(accepts/trials*100,2)
                improve_rate=np.round(improves/trials*100,2)
                
                out_data=pd.read_csv(self.log_dir+'_out.csv')
                inp_data=pd.read_csv(self.log_dir+'_inp.csv')
                sorted_out=out_data.sort_values(by=['reward'],ascending=False)   
                sorted_inp=inp_data.sort_values(by=['reward'],ascending=False)   
                
                #------------
                # plot progress 
                #------------
                self.callback.plot_progress('Annealing Step')
                
                #------------
                # print summary 
                #------------
                print('*****************************************************')
                print('Annealing step={}/{}'.format(step,self.steps))
                print('Current Temperature/Max Temperature: {}/{}'.format(int(T), self.Tmax))
                print('Accept Rate: {}%'.format(accept_rate))
                print('Improvement Rate: {}% '.format(improve_rate))
                print('Best Reward So Far: {}'.format(-np.round(energy_best,3)))
                print('Best Solution So Far: {}'.format(x_best))
                print('*****************************************************')
                
                with open (self.log_dir + '_summary.txt', 'a') as fin:
                    fin.write('*****************************************************\n')
                    fin.write('Summary data for annealing step {}/{} \n'.format(step, self.steps))
                    fin.write('*****************************************************\n')
                    fin.write('Current Temperature/Max Temperature: {}/{} \n'.format(int(T), self.Tmax))
                    fin.write('Accept Rate: {}% \n'.format(accept_rate))
                    fin.write('Improvement Rate: {}% \n'.format(improve_rate))
                    fin.write ('--------------------------------------------------------------------------------------\n')
                    fin.write ('Statistics for THIS annealing step \n')
                    fin.write('Max Reward: {0:.2f} \n'.format(np.max(fit_lst)))
                    fin.write('Mean Reward: {0:.2f} \n'.format(np.mean(fit_lst)))
                    fin.write('Std Reward: {0:.2f} \n'.format(np.std(fit_lst)))
                    fin.write('Min Reward: {0:.2f} \n'.format(np.min(fit_lst)))
                    fin.write ('--------------------------------------------------------------------------------------\n')
                    fin.write ('Best output for ALL annealing steps so far \n')
                    fin.write(sorted_out.iloc[0,:].to_string())
                    fin.write('\n')
                    fin.write ('-------------------------------------------------------------------------------------- \n')
                    fin.write ('Best corresponding input for ALL annealing steps so far \n')
                    fin.write(sorted_inp.iloc[0,:].to_string())
                    fin.write('\n')
                    fin.write ('-------------------------------------------------------------------------------------- \n')
                    fin.write('\n\n')
                
                trials, accepts, improves = 0, 0, 0
                fit_lst=[]