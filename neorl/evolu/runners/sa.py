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
#"""
#Created on Tue Feb 25 14:42:24 2020
#
#@author: majdi
#"""

import warnings
warnings.filterwarnings("ignore")
import random
import gym
import pandas as pd
import numpy as np
import multiprocessing
from neorl.parsers.PARSER import InputChecker
import copy, math, random, sys, time, os

class SAAgent(InputChecker):
    
    def __init__ (self, inp, callback):    
        
        """
        Input: 
        inp: is a dictionary of validated user input {"ncores": 8, "env": 6x6, ...}
        callback: a class of callback built from stable-baselines to allow intervening during training 
                  to process data and save models
        """
        self.inp= inp 
        self.env = gym.make(self.inp.gen_dict['env'][0], casename=self.inp.sa_dict['casename'][0], exepath=self.inp.gen_dict['exepath'][0], 
                            log_dir=self.inp.gen_dict['log_dir'], env_data=self.inp.gen_dict['env_data'][0])
        self.fit=self.env.fit
        
        np.random.seed(10)
        self.log_dir=self.inp.gen_dict['log_dir']+self.inp.sa_dict["casename"][0]
        self.callback=callback
        
        #--------------------------------
        # Infer SA user paramters from the sa_dict
        #--------------------------------
        self.Tmax=self.inp.sa_dict['Tmax'][0]
        self.Tmin=self.inp.sa_dict['Tmin'][0]
        self.steps=self.inp.sa_dict['steps'][0]
        self.swap=self.inp.sa_dict['swap'][0]
        self.check_freq=self.inp.sa_dict['check_freq'][0]
        self.lbound=self.inp.sa_dict['lbound'][0]
        self.ubound=self.inp.sa_dict['ubound'][0]
        self.lbound=[int(item) for item in self.lbound]
        self.ubound=[int(item) for item in self.ubound]
        self.ubound=[item+1 for item in self.ubound]  # adding +1 to the ubound for np.random.randint --> samples between [lbound, ubound)
        self.indpb=self.inp.sa_dict['indpb'][0]
        self.cooling=self.inp.sa_dict['cooling'][0]
        self.x_index=list(range(self.inp.gen_dict['xsize'][0])) # list of all indices of input space
        
        #--------------------------------
        # KBS mode (via a CSV dataset)
        #--------------------------------
        if self.swap=='kbs':
            print('--debug: A path to KBS dataset is provided for SA')
            self.indpb_kbs=self.inp.sa_dict['indpb_kbs'][0]
            self.kbs_path=self.inp.sa_dict['kbs_path'][0]
            if not os.path.exists (self.kbs_path):
                raise Exception ('a kbs mode is used for SA, but the path to dataset ({}) does not exist'.format(self.kbs_path))
            print('--debug: Loading the dataset from {} ...'.format(self.kbs_path))
            self.kbs_data=pd.read_csv(self.kbs_path).values
            self.ind_list=list(range(0,self.kbs_data.shape[0]))
        
        self.kbs_usage=0
        self.real_good=0
        
    def singleswap(self) :
        #np.random.seed(10)
        # this function swaps two indices in the input vector, sutiable for input shuffling
        
        dual_index=random.sample(self.x_index,2)
        index1, index2  = dual_index[0], dual_index[1]
        
        if self.lbound == None and self.ubound == None:   #swap only
            self.x[index1], self.x[index2] = self.x[index2], self.x[index1]
        else: 
            
            if len(self.lbound) == 1 and len(self.ubound) == 1:
                sample=np.random.randint(self.lbound, self.ubound)
            else:
                sample=np.random.randint(self.lbound[index1], self.ubound[index1])
                
            while self.x[index1] == sample:  # Resample if no change exists
                if len(self.lbound) == 1 and len(self.ubound) == 1:
                    sample=np.random.randint(self.lbound, self.ubound)
                else:
                    sample=np.random.randint(self.lbound[index1], self.ubound[index1])
                
            self.x[index1] = sample            
        
        return
     
    def dualswap(self) :
        #np.random.seed(10)
        # this function swaps two indices in the input vector, sutiable for input shuffling
        
        dual_index=random.sample(self.x_index,2)
        index1, index2  = dual_index[0], dual_index[1]
        
        if self.lbound == None and self.ubound == None:
            self.x[index1], self.x[index2] = self.x[index2], self.x[index1]
        else:
            if len(self.lbound) == 1 and len(self.ubound) == 1:
                sample1, sample2 = np.random.randint(self.lbound, self.ubound), np.random.randint(self.lbound, self.ubound)
            else:
                sample1, sample2 = np.random.randint(self.lbound[index1], self.ubound[index1]), np.random.randint(self.lbound[index2], self.ubound[index2])
                
            while self.x[index1] == sample1 or self.x[index2] == sample2:
                if len(self.lbound) == 1 and len(self.ubound) == 1:
                    sample1, sample2 = np.random.randint(self.lbound, self.ubound), np.random.randint(self.lbound, self.ubound)
                else:
                    sample1, sample2 = np.random.randint(self.lbound[index1], self.ubound[index1]), np.random.randint(self.lbound[index2], self.ubound[index2])

            self.x[index1], self.x[index2] = sample1, sample2

        return

    
    def quadswap(self) :
        #np.random.seed(10)
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
        #np.random.seed(10)
        # this function randomly shuffels all data in the input vector
        if self.lbound == None and self.ubound == None:
             self.x=random.sample(self.x,len(self.x))
        elif len(self.lbound) == 1 and len(self.ubound) == 1:
            self.x=np.random.randint(self.lbound, self.ubound,len(self.x))
        else:
            for i in range(len(self.x)):
                if random.random() < self.indpb:
                    self.x[i]=np.random.randint(self.lbound[i], self.ubound[i])
        return
    
    def kbs(self):
        
        try:
            if random.random() < self.indpb_kbs:
                
                ind=random.sample(self.ind_list,1)[0]
                self.ind_list.remove(ind)  # remove the new location from the list
                if not self.ind_list:
                    print("--- KBS Indices are reinitialized, so repititive samples would appear")
                    self.ind_list=list(range(0,self.kbs_data.shape[0]))
                    
                self.x=copy.deepcopy(self.kbs_data[ind,:])
                self.kbs_usage+=1
            else:
                if self.lbound == None and self.ubound == None:
                     self.x=random.sample(self.x,len(self.x))
                elif len(self.lbound) == 1 and len(self.ubound) == 1:
                    #self.x=np.random.randint(self.lbound, self.ubound,len(self.x))
                    for i in range(len(self.x)):
                        if random.random() < self.indpb:
                            self.x[i]=np.random.randint(self.lbound, self.ubound)
                else:
                    for i in range(len(self.x)):
                        if random.random() < self.indpb:
                            self.x[i]=np.random.randint(self.lbound[i], self.ubound[i])
        except:
            raise Exception ('--error: the kbs block cannot be resolved')
            
        return
        
    def build(self):

        """
        This function builds a SA chain and execute it
        """
        random.seed(10)
        
        # initialize x0 
        if self.inp.sa_dict['initstate'][0] == None:
            if self.lbound == None and self.lbound == None:
                self.x=random.sample(range(0,self.inp.gen_dict['xsize'][0]),self.inp.gen_dict['xsize'][0])
            elif len(self.lbound) == 1 and len(self.ubound) == 1:
                self.x=np.random.randint(self.lbound, self.ubound, self.inp.gen_dict['xsize'][0])
            else:
                self.x=np.random.randint(self.lbound, self.ubound)
        else:
            self.x=self.inp.sa_dict['initstate'][0]
        
        # initialize Temperature
        T = self.Tmax
        
        #---------------------------------------------------------------------
        #SA is programmed to minimize but we are maximizing rewards in neorl 
        #so we will be minimizing energy= -reward, which is same as maximizing reward
        #---------------------------------------------------------------------
        # initialize energy 
        energy = -self.fit(self.x)[0] #use[0] as fit returns tuple for GA/DEAP
        # set x/xbest and energy/energy_best to x0, energy0
        x_prev = copy.deepcopy(self.x)
        x_best = copy.deepcopy(self.x)
        energy_prev = energy
        energy_best = energy
        
        # decide the movment mode 
        if self.swap == 'singleswap':
            self.move = self.singleswap
        elif self.swap == 'dualswap':
            self.move = self.dualswap
        elif self.swap == 'quadswap':
            self.move = self.quadswap
        elif self.swap == 'fullswap':
            self.move = self.fullswap
        elif self.swap == 'kbs':
            self.move = self.kbs
        else:
            raise Exception ('--error: the swap mode for SA is not defined, use singleswap, dualswap, quadswap, fullswap, or kbs')
        
        trials, accepts, improves = 0, 0, 0        
        # Attempt moves to new states
        step=0
        fit_lst=[]
        
        #---------------------------------------------
        # Start SA chain 
        #--------------------------------------------
        while step <= self.steps:

            step += 1
            
            # Decide cooling schedule
            if self.cooling=='fast':
                Tfac = -np.log(float(self.Tmax) / self.Tmin)
                T = self.Tmax * np.exp( Tfac * step / self.steps)
            elif self.cooling=='boltzmann':
                T = self.Tmax / np.log(step + 1)
            elif self.cooling=='cauchy':
                T = self.Tmax / (step + 1)
            else:
                raise Exception ('--error: unknown cooling mode is entered, fast, boltzmann, or cauchy are ONLY allowed')
            
            # Perturb input
            self.move()
            
            #---------------------------------------------------------------------
            #SA is programmed to minimize but we are maximizing rewards in neorl 
            #so we will be minimizing energy= -reward, which is same as maximizing reward
            #---------------------------------------------------------------------
            reward=self.fit(self.x)[0] #use[0] as the fit function returns tuple for GA/DEAP
            fit_lst.append(reward)
            energy =-reward  
            
            dE = energy - energy_prev
            trials += 1
            
            #-----------------------------------
            # Accept/Reject
            #-----------------------------------
            alpha=random.random()
            if dE > 0.0 and np.exp(-dE/T) < alpha:
                # Restore previous state if no improvment and accpetance criterion is not satisfied
                self.x = copy.deepcopy(x_prev)
                energy = energy_prev
            else:
                # Accept new state and compare to best state
                accepts += 1
                if dE < 0.0:
                    improves += 1
                x_prev = copy.deepcopy(self.x)
                energy_prev = energy
                if energy < energy_best:
                    x_best = copy.deepcopy(self.x)
                    energy_best = energy
            
            #-----------------------------------
            # Logging and progress check
            #-----------------------------------
            if step % self.check_freq == 0 or step==self.steps:
                
                #print('step=', step, 'kbs=', self.kbs_usage, 'real_good=', self.real_good)
                
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