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
def get_population(pop, fits=None):
    
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
    
    return df_pop