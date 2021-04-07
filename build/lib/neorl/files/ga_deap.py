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
from deap import algorithms, base, creator, tools
# import input parameters from the user
from neorl.parsers.PARSER import InputChecker
import multiprocessing
import os, csv

class GAAgent(InputChecker):
    
    def __init__ (self, inp, callback):    
    
        """
        Input: 
        inp: is a dictionary of validated user input {"ncores": 8, "env": 6x6, ...}
        callback: a class of callback built from stable-baselines to allow intervening during training 
                  to process data and save models
        """
        self.inp= inp
        self.env = gym.make(self.inp.gen_dict['env'][0], casename=self.inp.ga_dict['casename'][0], exepath=self.inp.gen_dict['exepath'][0], 
                            log_dir=self.inp.gen_dict['log_dir'], env_data=self.inp.gen_dict['env_data'][0])
        self.log_dir=self.inp.gen_dict['log_dir']+self.inp.ga_dict["casename"][0]
        self.callback=callback
        
        # Infer GA user paramters from the ga_dict
        self.check_freq=self.inp.ga_dict["check_freq"][0]
        self.ncores=self.inp.ga_dict["ncores"][0]
        self.kbs_path=self.inp.ga_dict["kbs_path"][0]
        self.pop=self.inp.ga_dict["pop"][0]
        
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
            self.kbs_pop=int(kbs_frac*self.pop) # number of kbs individuals to contribute from total population
        
    def build(self):
        
        """
        This function builds a GA module based on the passed user parameters
        """
        
        random.seed(42)
        self.env.reset()
        
        #-------------------------------------
        # Initialize lower/upper bound 
        #-------------------------------------
        self.lbound=self.inp.ga_dict['lbound'][0]
        self.ubound=self.inp.ga_dict['ubound'][0]
        if len(self.lbound) > 1 or len(self.ubound) > 1:
            self.lbound=[int(item) for item in self.lbound]
            self.ubound=[int(item) for item in self.ubound]
        
        #-------------------------------------
        # Create the GA engine in DEAP
        #-------------------------------------
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        
        #-------------------------------------
        # Problem solution mode
        #-------------------------------------
        # Mode 1: shuffle, for shuffling or permutation withOUT repeatition
        if self.inp.ga_dict["mode"][0] == 'shuffle': #for simulate3
            toolbox.register("indices", random.sample, range(self.inp.gen_dict["nactions"][0]), self.inp.gen_dict["nactions"][0])
            toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
            toolbox.register("mutate", tools.mutShuffleIndexes, indpb=self.inp.ga_dict["indpb"][0])
            toolbox.register("mate", tools.cxOrdered)
        
        # Mode 2: assign, for assigning or permutation with repeatition
        elif self.inp.ga_dict["mode"][0] == 'assign': #for casmo4
            
            if len(self.lbound) > 1 and len(self.ubound) > 1: 
                #-----------------------------
                functions = []
                for i in range(len(self.lbound)):
                    def fun(idx=i):
                        return random.randint(self.lbound[idx], self.ubound[idx])
                    functions.append(fun)
                
                toolbox.register("individual", tools.initCycle, creator.Individual, functions, n=1)
                #-----------------------------
            else:
                toolbox.register("attr_int", random.randint, self.lbound, self.ubound)
                toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, self.inp.gen_dict["xsize"][0])
                
            toolbox.register("mutate", tools.mutUniformInt, low=self.lbound, up=self.ubound, indpb=self.inp.ga_dict["indpb"][0])
            toolbox.register("mate", tools.cxTwoPoint)
        else:
            raise Exception ('--error: the GA mode is not supported')
        
        #-------------------------------------
        # The KBS block 
        #-------------------------------------
        if self.kbs_path:
            def initPopulation(pcls, ind_init, filename, indices):
                content=[]
                content=[self.kbs_data[i,:] for i in indices]
                return pcls(ind_init(c) for c in content)
            
            try:
                toolbox.register("population", initPopulation, list, creator.Individual, self.kbs_path, range(0,self.pop))
                population = toolbox.population()
            except:
                raise Exception ('--error: the size of the external csv dataset is not enough to cover the initial population')
            
        else:
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            population = toolbox.population(n=self.pop)

        #-------------------------------------
        # The KBS block 
        #-------------------------------------
        toolbox.register("evaluate", self.env.fit)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        if self.ncores > 1:
            pool = multiprocessing.Pool(processes=self.ncores)
            toolbox.register("map", pool.map)
        
        NGEN=self.inp.ga_dict["ngen"][0]
        
        #--------------------------------------------------------------------------------
        # Function to mix the population with external kbs dataset (i.e. elite childs) 
        #--------------------------------------------------------------------------------
        def KbsMixPop(pcls, ind_init, population, indices):
            
            for i in range (1,len(kbs_pop_indices)+1):
                population[-i]=list(self.kbs_data[kbs_pop_indices[i-1],:])
            assert len(population) == self.pop
            
            return pcls(ind_init(c) for c in population)
        #--------------------------------------------------------------------------------
        
        #--------------------------------------------------------------------------------
        # Evolution
        #--------------------------------------------------------------------------------
        
        for gen in range(1,NGEN+1):
            
            #------------
            # If KBS exists
            #------------
            if self.kbs_path and gen > 1:
                kbs_pop_indices=random.sample(range(self.kbs_data.shape[0]),self.kbs_pop)
                toolbox.register("population", KbsMixPop, list, creator.Individual, population, kbs_pop_indices)
                population = toolbox.population()
            
            fit_lst=[]
            # apply crossover and mutation on the current mutation
            offspring = algorithms.varAnd(population, toolbox, cxpb=self.inp.ga_dict["cxpb"][0], mutpb=self.inp.ga_dict["mutpb"][0])  
            # create the offspring based on fitness values
            fits = toolbox.map(toolbox.evaluate, offspring)

            for fit, ind in zip(fits, offspring):    # get fitness data
                ind.fitness.values = fit
                fit_lst.append(fit)
                
            population = toolbox.select(offspring, k=len(population))  # create the new population based on best fitness
            
            #------------
            # Monitor Progress
            #------------
            if  gen % self.check_freq == 0 or gen == NGEN:
                
                out_data=pd.read_csv(self.log_dir+'_out.csv')
                inp_data=pd.read_csv(self.log_dir+'_inp.csv')
                sorted_out=out_data.sort_values(by=['reward'],ascending=False)   
                sorted_inp=inp_data.sort_values(by=['reward'],ascending=False)  
                #------------
                # plot progress 
                #------------
                self.callback.plot_progress('Generation')
                
                #------------
                # print summary 
                #------------
                with open (self.log_dir + '_summary.txt', 'a') as fin:
                    fin.write('*****************************************************\n')
                    fin.write('Summary data for generation {}/{} \n'.format(gen, NGEN))
                    fin.write('*****************************************************\n')
                    fin.write('Max Reward: {0:.2f} \n'.format(np.max(fit_lst)))
                    fin.write('Mean Reward: {0:.2f} \n'.format(np.mean(fit_lst)))
                    fin.write('Std Reward: {0:.2f} \n'.format(np.std(fit_lst)))
                    fin.write('Min Reward: {0:.2f} \n'.format(np.min(fit_lst)))
                    
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

        if self.ncores > 1:
            pool.close()
            

#        def search_element(target, element,flag):
#            if flag == 0: # --- matrix
#                for row in range(len(target)):
#                    try:
#                        column = target[row].index(element)
#                    except ValueError:
#                        continue
#                    return(row, column)
#            elif flag == 1: # --- vector
#                try:
#                    column = target.index(element)
#                except ValueError:
#                    print('Error not found')
#                return(column)
#            return 0