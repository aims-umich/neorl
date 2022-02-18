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
"""
Created on Tue Feb 25 14:42:24 2020

@author: majdi
"""

import warnings
warnings.filterwarnings("ignore")
import random
import numpy as np
from collections import defaultdict

#-------------------------------------
# GA Mutation                      
#-------------------------------------
def mutGA(ind, strat, smin, smax, lb, ub, datatype):
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
                    choices.remove(int(ind[i]))
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
	
#-------------------------------------
# ES Mutation                      
#-------------------------------------

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
                    choices.remove(int(ind[i]))
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