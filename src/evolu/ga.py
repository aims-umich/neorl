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
# import input parameters from the user
from src.parsers.PARSER import InputChecker

class GAAgent(InputChecker):
    
    def __init__ (self, inp):    
    
        self.inp= inp 
        self.env = gym.make(self.inp.gen_dict['env'][0], casename='ga')
        self.log_dir='./master_log/'+self.inp.ga_dict["casename"][0]
        
    def build(self):
    
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        #-----------------------------
    #        functions = []
    #        for i in range(len(self.lb)):
    #            def fun(_idx=i):
    #                return random.uniform(self.lb[_idx], self.ub[_idx])
    #            functions.append(fun)
            
    #        toolbox.register("individual", tools.initCycle, creator.Individual, functions, n=1)
        #-----------------------------
        
        
        #ncores=8
        #pool = multiprocessing.Pool(ncores)
        #toolbox.register("map", pool.map)
        
        toolbox.register("attr_bool", random.randint, 1, self.inp.gen_dict["nactions"][0])
        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
            toolbox.attr_bool, self.inp.gen_dict["xsize"][0])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        toolbox.register("evaluate", self.env.fit)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=self.inp.ga_dict["indpb"][0])
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        population = toolbox.population(n=self.inp.ga_dict["pop"][0])
        
        NGEN=self.inp.ga_dict["ngen"][0]
        for gen in range(NGEN):
            fit_lst=[]
            #print('gen= ' + str(gen))
            offspring = algorithms.varAnd(population, toolbox, cxpb=self.inp.ga_dict["cxpb"][0], mutpb=self.inp.ga_dict["mutpb"][0])
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
                fit_lst.append(fit)
            population = toolbox.select(offspring, k=len(population))
            #best_sol=tools.selBest(population, k=1)
            #print('Best Solution Generation ' + str(gen) + ':', max(fit_lst))
            #print(best_sol[0])
            
            out_data=pd.read_csv(self.log_dir+'_out.csv')
            inp_data=pd.read_csv(self.log_dir+'_inp.csv')
            sorted_out=out_data.sort_values(by=['reward'],ascending=False)   
            sorted_inp=inp_data.sort_values(by=['reward'],ascending=False)   
            
            with open (self.log_dir + '_summary.txt', 'a') as fin:
                fin.write('*****************************************************\n')
                fin.write('Summary data for generation {}/{} \n'.format(gen+1, NGEN))
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
            
        #top10 = tools.selBest(population, k=10)
        #print(top10)
