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
#Created on Wed Feb 26 09:33:45 2020
#
#@author: majdi
#"""

import pandas as pd
import numpy as np
from neorl.evolu.discrete import decode_discrete_to_grid

def is_integer_num(n):
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False

bounds = {}
x=[20.0, 0.1875, 59.0, 149.0, 'Hi', -4.0]
bounds['x1'] = ['int', 1, 20]
bounds['x2'] = ['grid', (0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625)]
bounds['x3'] = ['int', 10, 200]
bounds['x4'] = ['int', 10, 200]
bounds['x5'] = ['grid', ('Hi', 'Bye', 'New')]
bounds['x6'] = ['int', -5, 5]

def check_mixed_individual(x, bounds):
    #auxiliary function to assert the type of the individual passed to x0 as initial guess
    # to check if it fits the discrete, float, grid types.
    #x: an individual vector to check 
    #bounds: a typical NEORL dictionary containing the parameter space    

    for index, (key, value) in enumerate(bounds.items()):
        #print('Index:: ', index, ' :: ', key, '-', value)
        
        if value[0] == 'grid':
            assert x[index] in value[1], '--error: the value ({}) in individual ({}) provided to x0 does not belong to grid provided ({})'.format(x[index], x, value[1])
        
        if value[0] == 'int':
            assert is_integer_num(x[index]), '--error: the value ({}) in individual ({}) provided to x0 is not consistent with the type ({})'.format(x[index], x, value[0])
            assert value[1] <= x[index] <= value[2], '--error: the value ({}) in individual ({}) provided to x0 is not within the bounds [{}, {}]'.format(x[index], x, value[1], value[2])

        if value[0] == 'float':
            try:
                float(x[index])
            except:
                raise Exception ('--error: the value ({}) in individual ({}) provided to x0 is not consistent with the type ({})'.format(x[index], x, value[0]))
            assert value[1] <= x[index] <= value[2], '--error: the value ({}) in individual ({}) provided to x0 is not within the bounds [{}, {}]'.format(x[index], x, value[1], value[2]) 
   
def get_population(pop, fits=None, grid_flag=False, bounds=None, bounds_map=None):
    
    if isinstance(pop, dict):
        #either ES or PSO
        d=len(pop[0][0])
        npop=len(pop)
        df_pop=np.zeros((npop, d+1))   #additional column for fitness        
        for i, indv in enumerate(pop):
            df_pop[i,:d]=pop[indv][0]
            df_pop[i,-1]=pop[indv][2]
    
    elif isinstance(pop, list):   
        #DE mainly
        d=len(pop[0])
        npop=len(pop)
        if fits is not None:
            assert len(fits) == npop, '--error: the size of fits and pop are not equal, pop cannot be constructed'
        df_pop=np.zeros((npop, d+1))   #additional column for fitness        
        for i, indv in enumerate(pop):
            df_pop[i,:d]=indv
            df_pop[i,-1]=fits[i]

    elif type(pop).__module__ == 'numpy':   
        #GWO, HHO, MFO, WOA mainly
        npop, d=pop.shape
        if fits is not None:
            assert len(fits) == npop, '--error: the size of fits and pop are not equal, pop cannot be constructed'
        df_pop=np.c_[pop, np.array(fits)]
    else:
        raise ('--warning: population data structure type cannot be identified, the population cannot be reconstructed')
    
    
    try:    
        colnames=['var'+str(i) for i in range(1,d+1)] + ['fitness']
        rownames=['indv'+str(i) for i in range(1,npop+1)]
        df_pop=pd.DataFrame(df_pop, index=rownames, columns=colnames)
    except:
        df_pop=pd.DataFrame(np.zeros((5, 5)))   #return an empty dataframe
    
    if grid_flag:
        #convert the categorical value from the discrete space to its orignal grid space
        for k in range (df_pop.shape[0]):
            xx=list(df_pop.iloc[k,:-1].values)
            yy=decode_discrete_to_grid(xx, bounds, bounds_map)
            #print(yy)
            df_pop.iloc[k,:-1] = yy
        
    return df_pop