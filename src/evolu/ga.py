#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:42:24 2020

@author: majdi
"""


import random
import gym
import multiprocessing
from deap import algorithms, base, creator, tools
# import input parameters from the user
from ParamList import InputParam

class GAAgent(InputParam):
    
    def __init__ (self, inp):    
    
        self.inp= inp 
        self.env = gym.make(self.inp.gen_dict['env'][0], casename='ga')
        
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
            print('gen= ' + str(gen))
            offspring = algorithms.varAnd(population, toolbox, cxpb=self.inp.ga_dict["cxpb"][0], mutpb=self.inp.ga_dict["mutpb"][0])
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
                fit_lst.append(fit)
            population = toolbox.select(offspring, k=len(population))
            best_sol=tools.selBest(population, k=1)
            print('Best Solution Generation ' + str(gen) + ':', max(fit_lst))
            print(best_sol[0])
        top10 = tools.selBest(population, k=10)
        print(top10)
