#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:42:24 2020

@author: Katelin and Majdi
"""

import warnings
warnings.filterwarnings("ignore")
import random
import numpy as np
import math

def mutate_discrete(x_ij, x_min, x_max, lb, ub, alpha, method):
    #"""
    #Changes float to discrete integer.

    #Params:
    #x_ij: a scalar to be converted to discrete
    #x_min/x_max: - minimum/maximum value of the individual vector 
    #lb/ub: lower/upper bounds allowed for x_ij
    #alpha: probability to mutate a discrete (used for sigmoid/minmax)
    #method - out of {'nearest_int', 'sigmoid', 'tanh', 'minmax'}

    #Return:
    #int - new discrete value
    #"""
    
    if method == 'nearest_int':
        to_ret = int(x_ij)

    if method == 'sigmoid':
        rand = random.random()
        sig_a=0 #support for sigmoid min
        sig_b=2 #support for sigmoid max
        norm = (sig_b-sig_a)*(x_ij - x_min) / (x_max - x_min) + sig_a  
        sig = 1 / (1 + math.exp(norm))
        if rand < sig and rand < alpha:
            choices = list(range(lb, ub+1)) # each <class 'int'>
            choices.remove(int(x_ij))
            to_ret = random.choice(choices)
            #print('--I am mutating')
        else:
            #print('--I am NOT mutating')
            to_ret = int(x_ij)

    if method == 'minmax':
        
        rand = random.random()
        norm = (x_ij + x_min) / (abs(x_min) + x_max)
        if norm >= 0.5 and rand < alpha:
            choices = list(range(lb, ub+1))
            choices.remove(int(x_ij))
            to_ret = random.choice(choices)
            #print('--I am mutating')
        else:
            to_ret = int(x_ij)
            #print('--I am NOT mutating')

    # make sure to_ret is in bound
    if to_ret < lb:
        to_ret = lb
    if to_ret > ub:
        to_ret = ub

    return to_ret

bounds={
'cxpb': ['float', 0.1, 0.9],             
'mu':   ['int', 30, 60],                 
'alpha':['grid', (0.1, 0.2, 0.3, 0.4)],    
'cxmode':['grid', ('blend', 'cx2point')]}


def encode_grid_to_discrete(bounds):
    
    bounds_new={}
    bounds_map={}
    
    for item in bounds:
        if bounds[item][0] == 'grid':
            bounds_new[item]=['int', 0, len(bounds[item][1])-1]
            
            bounds_map[item]={}
            for i in range(len(bounds[item][1])):
                bounds_map[item][i] = bounds[item][1][i]
        else:
            bounds_new[item]=bounds[item]
        
    return bounds_new, bounds_map


bounds_new, bounds_map=encode_grid_to_discrete(bounds)

def decode_discrete_to_grid(individual, bounds, bounds_map):
    new_indv=[]
    for i, key in enumerate(bounds):
        if bounds[key][0]=='grid':
            #print(bounds_map[key][i])
            index=individual[i]
            new_indv.append(bounds_map[key][index])
        else:
            new_indv.append(individual[i])
    
    return new_indv   

x=[0.2, 45, 2, 1]
0