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
# import input parameters from the user
from neorl.parsers.PARSER import InputChecker
from neorl.evolu.crossover import cxES2point, mutES, select
from collections import defaultdict

import multiprocessing
import multiprocessing.pool

import os, csv
import copy

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

class GAAgent(InputChecker):
    
    def __init__ (self, inp, callback):    
    
        """
        Input: 
        inp: is a dictionary of validated user input {"ncores": 8, "env": 6x6, ...}
        callback: a class of callback built from stable-baselines to allow intervening during training 
                  to process data and save models
        """
        random.seed(42)
        self.inp= inp
        self.env = gym.make(self.inp.gen_dict['env'][0], casename=self.inp.ga_dict['casename'][0], exepath=self.inp.gen_dict['exepath'][0], 
                            log_dir=self.inp.gen_dict['log_dir'], env_data=self.inp.gen_dict['env_data'][0])
        self.log_dir=self.inp.gen_dict['log_dir']+self.inp.ga_dict["casename"][0]
        self.callback=callback
        
        # Infer GA user paramters from the ga_dict
        self.check_freq=self.inp.ga_dict["check_freq"][0]
        self.ncores=self.inp.ga_dict["ncores"][0]
        self.kbs_path=self.inp.ga_dict["kbs_path"][0]
        
        self.npop=self.inp.ga_dict["pop"][0]
        self.smax=self.inp.ga_dict["smax"][0]
        self.smin=self.inp.ga_dict["smin"][0]
        self.cxpb=self.inp.ga_dict["cxpb"][0]
        self.mutpb=self.inp.ga_dict["mutpb"][0]
        self.ngen=self.inp.ga_dict["ngen"][0]
        self.mu=self.inp.ga_dict["pop"][0]
        if self.inp.ga_dict["lambda"][0]:
            self.lambda_=self.inp.ga_dict["lambda"][0]
        else:
            self.lambda_=self.mu
        
        #-------------------------------------
        # Initialize lower/upper bound 
        #-------------------------------------
        self.lbound=self.inp.ga_dict['lbound'][0]
        self.ubound=self.inp.ga_dict['ubound'][0]
        self.datatype=self.inp.ga_dict['type'][0]
        assert len(self.lbound) == len(self.ubound), '--error: the lbound ({}) and ubounds ({}) must have same length'.format(len(self.lbound),len(self.ubound))
        assert len(self.lbound) == len(self.datatype), '--error: the lbound/ubound ({}) and type ({}) must have same length'.format(len(self.lbound),len(self.datatype))
        if len(self.lbound) > 1 or len(self.ubound) > 1:
            self.lbound=[item for item in self.lbound]
            self.ubound=[item for item in self.ubound]
            self.datatype=[str(item) for item in self.datatype]
        
        
        #-------------------------------------
        # KBS Mode (via a CSV dataset)
        # This option is valid if solutions are provided by RL
        # This is activated if kbs_path is provided 
        #-------------------------------------
        if self.kbs_path:
            print('--debug: A path to KBS dataset is provided for GA')
            if not os.path.exists (self.kbs_path):
                raise Exception ('--error: a kbs mode is used for GA, but the path to dataset ({}) does not exist'.format(self.kbs_path))
            kbs_frac=self.inp.ga_dict["kbs_frac"][0]
            print('--debug: Loading the dataset from {} ...'.format(self.kbs_path))
            self.kbs_data=pd.read_csv(self.kbs_path).values
            self.kbs_pop=int(kbs_frac*self.lambda_) # number of kbs individuals to contribute from total population
            self.kbs_append=False
            
    def gen_object(self, inp):
        """
        This is a worker for multiprocess Pool
        Inputs:
            inp (list of lists): contains data for each core [[ind1, caseid1], ...,  [indN, caseidN]]
        Returns:
            fitness value (float)
        """
        return self.env.fit(inp)

    def GenES(self, lb, datatype, smin, smax, ub=None):
        """
        Individual generator
        Inputs:
            -lb,ub,datatype (list): input paramter lower/upper bounds in dictionary form
        Returns 
            -ind (list): an individual vector with values sampled from bounds
            -strategy (list): the strategy vector with values between smin and smax
        """
        size = len(lb) # size of individual 
        content=[]
        
        if self.kbs_path:
            assert self.kbs_data.shape[0] > self.lambda_, '--error: the size of the KBS dataset ({}) is not enough to cover initial population lambda ({})'.format(self.kbs_data.shape[0],self.lambda_)
            index=random.randint(0,self.kbs_data.shape[0]-1)
            ind=list(self.kbs_data[index,:])
        else:
            for i in range (len(lb)):
                if datatype[i] == 'int':
                    content.append(random.randint(lb[i], ub[i]))
                elif datatype[i] == 'float':
                    content.append(random.uniform(lb[i], ub[i]))
                elif datatype[i] == 'grid':
                    content.append(random.sample(lb[i],1)[0])
                else:
                    raise Exception ('unknown data type is given, either int, float, or grid are allowed for parameter bounds')
            
            ind=list(content)
        strategy = [random.uniform(smin,smax) for _ in range(size)]
        return ind, strategy

    def init_pop(self, warmup):
        """
        Population intializer 
        Inputs:
            -warmup (int): number of individuals to create and evaluate initially
        Returns 
            -pop (dict): initial population in a dictionary form, looks like this
            
        """
        #initialize the population and strategy and run them in parallel (these samples will be used to initialize the memory)
        pop=defaultdict(list)
        # dict key runs from 0 to warmup-1
        # index 0: individual, index 1: strategy, index 2: fitness 
        # Example: 
        """
        pop={key: [ind, strategy, fitness]}
        pop={0: [[1,2,3,4,5], [0.1,0.2,0.3,0.4,0.5], 1.2], 
             ... 
             99: [[1.1,2.1,3.1,4.1,5.1], [0.1,0.2,0.3,0.4,0.5], 5.2]}
        """
        for i in range (warmup):
            #caseid='es_gen{}_ind{}'.format(0,i+1)  #caseid are only for logging purposes to distinguish sample source
            data=self.GenES(lb=self.lbound, ub=self.ubound, datatype=self.datatype, smax=self.smax, smin=self.smin)
            pop[i].append(data[0])
            pop[i].append(data[1])
        
        if self.ncores > 1:  #evaluate warmup in parallel
            core_list=[]
            for key in pop:
                core_list.append(pop[key][0])
            p=MyPool(self.ncores)
            fitness = p.map(self.gen_object, core_list)
            p.close()
            p.join()
            
            [pop[ind].append(fitness[ind]) for ind in range(len(pop))]
        
        else: #evaluate warmup in series
            for key in pop:
                #caseid='es_gen{}_ind{}'.format(0,key+1)
                fitness=self.env.fit(pop[key][0])
                pop[key].append(fitness)
        
        return pop  #return final pop dictionary with ind, strategy, and fitness

    def GenOffspring(self, pop):
        """
        
        This function generates the offspring by applying crossover, mutation **or** reproduction. 
        The sum of both probabilities self.cxpb and self.mutpb must be in [0,1]
        The reproduction probability is 1 - cxpb - mutpb
        The new offspring goes for fitness evaluation
        
        Inputs:
            pop (dict): population in dictionary structure
        Returns:
            offspring (dict): new modified population in dictionary structure    
        """
        pop_indices=list(range(0,len(pop)))
        offspring = defaultdict(list)
        for i in range(self.lambda_):
            alpha = random.random()
            #------------------------------
            # Crossover
            #------------------------------
            if alpha < self.cxpb:            
                index1, index2 = random.sample(pop_indices,2)
                ind1, ind2, strat1, strat2 = cxES2point(ind1=list(pop[index1][0]),ind2=list(pop[index2][0]), 
                                                     strat1=list(pop[index1][1]),strat2=list(pop[index2][1]))
                offspring[i].append(ind1)
                offspring[i].append(strat1)
                #print('crossover is done for sample {} between {} and {}'.format(i,index1,index2))
            #------------------------------
            # Mutation
            #------------------------------
            elif alpha < (self.cxpb + self.mutpb):  # Apply mutation
                index = random.choice(pop_indices)
                ind, strat=mutES(ind=list(pop[index][0]), strat=list(pop[index][1]), lb=self.lbound, ub=self.ubound, datatype=self.datatype, smax=self.smax, smin=self.smin)
                offspring[i].append(ind)
                offspring[i].append(strat)
                #print('mutation is done for sample {} based on {}'.format(i,index))
            #------------------------------
            # Reproduction from population
            #------------------------------
            else:                         
                index=random.choice(pop_indices)
                offspring[i].append(pop[index][0])
                offspring[i].append(pop[index][1])
                #print('reproduction is done for sample {} based on {}'.format(i,index))
    
        return offspring

    def evolute(self, population, ngen, verbose=0):
        """This is the :math:`(\mu~,~\lambda)` evolutionary algorithm.
    
        Inputs:
            population (dict): initial population in dictionary structure
            ngen (int): number of generations to evolute
            verbose (bool): print summary to screen
        Returns:
            population (dict): final population after running all generations (ngen)
        """
        
        # Begin the evolution process
        for gen in range(1, ngen + 1):
            
            # Vary the population and generate new offspring
            offspring = self.GenOffspring(pop=population)
            
            # Evaluate the individuals with an invalid fitness with multiprocessign Pool
            # create and run the Pool
            if self.ncores > 1:
                core_list=[]
                for key in offspring:
                    core_list.append(offspring[key][0])
                #initialize a pool
                print('---------- Start the real generations------')
                p=MyPool(self.ncores)
                fitness = p.map(self.gen_object, core_list)
                p.close(); p.join()
                
                [offspring[ind].append(fitness[ind]) for ind in range(len(offspring))]
            else: #serial calcs
                for ind in range(len(offspring)):
                    fitness=self.env.fit(offspring[ind][0])
                    offspring[ind].append(fitness)
                                
            # Select the next generation population
            #print([(offspring[item][0],offspring[item][2]) for item in offspring])
            #print('-------------------------------')
            population = copy.deepcopy(select(pop=offspring, k=self.mu))
            #print([(population[item][0],population[item][2]) for item in population])
            #print('-------------------------------')        
            #print([(population[item][0],population[item][2]) for item in population])
            #print('-------------------------------')
                            
            #------------
            # Monitor Progress
            #------------
            if  gen % self.check_freq == 0 or gen == self.ngen:
                
                out_data=pd.read_csv(self.log_dir+'_out.csv')
                inp_data=pd.read_csv(self.log_dir+'_inp.csv')
                sorted_out=out_data.sort_values(by=['reward'],ascending=False)   
                sorted_inp=inp_data.sort_values(by=['reward'],ascending=False)  
                inds, rwd=[population[i][0] for i in population], [population[i][2] for i in population]
                mean_strategy=[np.mean(population[i][1]) for i in population]
                #------------
                # plot progress 
                #------------
                self.callback.plot_progress('Generation')
                
                #------------
                # print summary 
                #------------
                with open (self.log_dir + '_summary.txt', 'a') as fin:
                    fin.write('*****************************************************\n')
                    fin.write('Summary data for generation {}/{} \n'.format(gen, self.ngen))
                    fin.write('*****************************************************\n')
                    fin.write('Max Reward: {0:.2f} \n'.format(np.max(rwd)))
                    fin.write('Mean Reward: {0:.2f} \n'.format(np.mean(rwd)))
                    fin.write('Std Reward: {0:.2f} \n'.format(np.std(rwd)))
                    fin.write('Min Reward: {0:.2f} \n'.format(np.min(rwd)))
                    
                    fin.write ('--------------------------------------------------------------------------------------\n')
                    fin.write ('Best output for this generation \n')
                    fin.write(sorted_out.iloc[0,:].to_string())
                    fin.write('\n')
                    fin.write ('-------------------------------------------------------------------------------------- \n')
                    fin.write ('Best corresponding input for this generation \n')
                    fin.write(sorted_inp.iloc[0,:].to_string())
                    fin.write('\n')
                    fin.write ('-------------------------------------------------------------------------------------- \n')
                    fin.write('\n\n')

                print('############################################################')
                print('ES generation {}/{}, CX={}, MUT={}, MU={}, LAMBDA={}'.format(gen,ngen, np.round(self.cxpb,2), np.round(self.mutpb,2), self.mu, self.lambda_))
                print('############################################################')
                print('Statistics for generation {}'.format(gen))
                print('Best Reward:', np.round(np.max(rwd),2))
                print('Mean Reward:', np.round(np.mean(rwd),2))
                print('Max Strategy:', np.round(np.max(mean_strategy),3))
                print('Min Strategy:', np.round(np.min(mean_strategy),3))
                print('Average Strategy:', np.round(np.mean(mean_strategy),3))
                print('############################################################')

            #------------
            # Update the population with KBS 
            #------------
            #print([(population[item][0],population[item][2]) for item in population])            
            if self.kbs_path:
                kbs_pop_indices=random.sample(range(self.kbs_data.shape[0]),self.kbs_pop)
                last_index=list(population.keys())
                last_index=last_index[-1]
                for i in range (len(kbs_pop_indices)):
                    ind_vec=list(self.kbs_data[kbs_pop_indices[i],:])
                    
                    if self.kbs_append: #append KBS individuals to the population
                        strategy = [random.uniform(self.smin,self.smax) for _ in range(len(self.lbound))]
                        kbs_ind=[ind_vec,strategy,0]
                        population[last_index+(i+1)]=kbs_ind
                    else: #replace the worst individuals in pop with KBS individuals
                        #population[last_index-i-1]=kbs_ind
                        #print(last_index-i)
                        population[last_index-i][0]=ind_vec
                        #print(population[last_index-i][0])
                    
        return population
        
    def build(self):
        
        """
        This function builds a GA module based on the passed user parameters
        """
        
        random.seed(42)
        #self.env.reset()
        #-------------------------------------
        # The KBS block 
        #-------------------------------------
        pop0=self.init_pop(warmup=self.lambda_)
        self.evolute(population=pop0, ngen=self.ngen, verbose=0)