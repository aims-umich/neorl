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

#Thanks to https://github.com/DEAP/deap, which inspired the content of this script

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
import numpy as np
from collections import defaultdict

######################################
# ES Crossovers                      #
######################################

def cxESBlend(ind1, ind2, strat1, strat2, alpha=0.1):
    """Executes a blend crossover on both, the individual and the strategy. The
    individuals shall be a :term:`sequence` and must have a :term:`sequence`
    :attr:`strategy` attribute. Adjustment of the minimal strategy shall be done
    after the call to this function, consider using a decorator.
    :param ind1: The first evolution strategy participating in the crossover.
    :param ind2: The second evolution strategy participating in the crossover.
    :param alpha: Extent of the interval in which the new values can be drawn
                  for each attribute on both side of the parents' attributes.
    :returns: A tuple of two evolution strategies.
    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    
    for i, (x1, s1, x2, s2) in enumerate(zip(ind1, strat1, ind2, strat2)):
        # Blend the individuals
        gamma = (1. + 2. * alpha) * random.random() - alpha
        ind1[i] = (1. - gamma) * x1 + gamma * x2
        ind2[i] = gamma * x1 + (1. - gamma) * x2
        # Blend the strategies
        gamma = (1. + 2. * alpha) * random.random() - alpha
        strat1[i] = (1. - gamma) * s1 + gamma * s2
        strat2[i] = gamma * s1 + (1. - gamma) * s2

    return ind1, ind2, strat1, strat2
    
def cxES2point(ind1, ind2, strat1, strat2):
    """Executes a classical two points crossover on both the individuals and their
    strategy. The individuals /strategies should be a list. The crossover points for the
    individual and the strategy are the same.
    
    Inputs:
        -ind1 (list): The first individual participating in the crossover.
        -ind2 (list): The second individual participating in the crossover.
        -strat1 (list): The first evolution strategy participating in the crossover.
        -strat2 (list): The second evolution strategy participating in the crossover.
    Returns:
        The new ind1, ind2, strat1, strat2 after crossover in list form
    """
    size = min(len(ind1), len(ind2))

    pt1 = random.randint(1, size)
    pt2 = random.randint(1, size - 1)
    if pt2 >= pt1:
        pt2 += 1
    else:  # Swap the two cx points
        pt1, pt2 = pt2, pt1

    ind1[pt1:pt2], ind2[pt1:pt2] = ind2[pt1:pt2], ind1[pt1:pt2]
    strat1[pt1:pt2], strat2[pt1:pt2] = strat2[pt1:pt2], strat1[pt1:pt2]
    
    return ind1, ind2, strat1, strat2


######################################
# GA Crossovers                      #
######################################

def select(pop, k=1):
    """
    Select function sorts the population from max to min based on fitness and select k best
    Inputs:
        pop (dict): population in dictionary structure
        k (int): top k individuals are selected
    Returns:
        best_dict (dict): the new ordered dictionary with top k selected 
    """
    
    pop=list(pop.items())
    pop.sort(key=lambda e: e[1][2], reverse=True)
    sorted_dict=dict(pop[:k])
    
    #This block creates a new dict where keys are reset to 0 ... k in order to avoid unordered keys after sort
    best_dict=defaultdict(list)
    index=0
    for key in sorted_dict:
        best_dict[index].append(sorted_dict[key][0])
        best_dict[index].append(sorted_dict[key][1])
        best_dict[index].append(sorted_dict[key][2])
        index+=1
    
    sorted_dict.clear()
    return best_dict

def mutES(ind, strat, smin, smax, lb, ub, datatype):
    """Mutate an evolution strategy according to mixed Discrete/Continuous mutation rules
    attribute as described in [Li2013].
    The function mutates discrete/float variables according to their type as indicated in self.bounds
    .. Li, Rui, et al. "Mixed integer evolution strategies for parameter optimization." 
       Evolutionary computation 21.1 (2013): 29-64.
    Inputs:
        -ind (list): individual to be mutated.
        -strat (list): individual strategy to be mutated.
    Returns: 
        -ind (list): new individual after mutatation
        -strat (list): individual strategy after mutatation       
    """
    # Infer the datatype, lower/upper bounds from bounds for flexible usage 
#    lb=[]; ub=[]; datatype=[]
#    for key in self.bounds:
#        datatype.append(self.bounds[key][0])
#        lb.append(self.bounds[key][1])
#        ub.append(self.bounds[key][2])
        
    size = len(ind)
    tau=1/np.sqrt(2*size)
    tau_prime=1/np.sqrt(2*np.sqrt(size))
    
    for i in range(size):
        #--------------------------
        # Discrete ES Mutation 
        #--------------------------
        if datatype[i] == 'int':
            norm=random.gauss(0,1)
            # modify the ind strategy
            strat[i] = 1/(1+(1-strat[i])/strat[i]*np.exp(-tau*norm-tau_prime*random.gauss(0,1)))
            #make a transformation of strategy to ensure it is between smin,smax 
            y=(strat[i]-smin)/(smax-smin)
            if np.floor(y) % 2 == 0:
                y_prime=np.abs(y-np.floor(y))
            else:
                y_prime=1-np.abs(y-np.floor(y))
            strat[i] = smin + (smax-smin) * y_prime
            
            # check if this attribute is mutated based on the updated strategy
            if random.random() < strat[i]:
                # make a list of possiblities after excluding the current value to enforce mutation
                if int(lb[i]) == int(ub[i]):
                    ind[i] = int(lb[i])
                else:
                    choices=list(range(int(lb[i]),int(ub[i]+1)))
                    choices.remove(ind[i])
                    # randint is NOT used here since it could re-draw the same integer value, choice is used instead
                    ind[i] = random.choice(choices)
        
        #--------------------------
        # Continuous ES Mutation 
        #--------------------------
        elif datatype[i] == 'float':
            norm=random.gauss(0,1)
            strat[i] *= np.exp(tau*norm + tau_prime * random.gauss(0, 1)) #normal mutation of strategy
            ind[i] += strat[i] * random.gauss(0, 1) # update the individual position
            
            #check the new individual falls within lower/upper boundaries
            if ind[i] < lb[i]:
                ind[i] = lb[i]
            if ind[i] > ub[i]:
                ind[i] = ub[i]
        
        else:
            raise Exception ('ES mutation strategy works with either int/float datatypes, the type provided cannot be interpreted')
        
        
    return ind, strat