# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 22:47:37 2021

@author: majdi
"""

#--------------------------------------------------------------------
# Paper: NEORL: A Framework for NeuroEvolution Optimization with RL
# Section: Script for section 3.4
# Contact: Majdi I. Radaideh (radaideh@mit.edu)
# Last update: 9/10/2021
#---------------------------------------------------------------------

import matplotlib
matplotlib.use('Agg')
from neorl import GWO, PESA, HHO, FNEAT
import matplotlib.pyplot as plt
import random
random.seed(1)
import time
import numpy as np
import pandas as pd

from neorl import PPO2
from neorl import MlpPolicy
from neorl import RLLogger
from neorl import CreateEnvironment
    
#Define the fitness function
def FIT(individual):
    """Sphere test objective function.
            F(x) = sum_{i=1}^d xi^2
            d=1,2,3,...
            Range: [-100,100]
            Minima: 0
    """
    time.sleep(1)
    y=sum(x**2 for x in individual)
    return y


if __name__=='__main__':    
    #Setup the parameter space (d=5)
    nx=5
    BOUNDS={}
    for i in range(1,nx+1):
        BOUNDS['x'+str(i)]=['float', -100, 100]


    npop=32
    ngen=20
    ncores_lst=[1,8,16,32]
    pesa_cores_lst=[1,24,48,96]
    pesa_npop=[32,32,32,32]
    pesa_ngen=[7,7,7,7]
    n_steps=[32,4,2,1]
    time_array=np.zeros((4,4))

    for i in range(len(ncores_lst)):
        
        t0=time.time()
        hho=HHO(mode='min', bounds=BOUNDS, fit=FIT, nhawks=npop, ncores=ncores_lst[i], seed=1)
        x_best, y_best, hho_hist=hho.evolute(ngen=ngen, verbose=1)
        time_array[i,0]=time.time()-t0
        
        #create an enviroment class
        env=CreateEnvironment(method='ppo', fit=FIT, ncores=ncores_lst[i],
                              bounds=BOUNDS, mode='min', episode_length=npop)
        
        #create a callback function to log data
        cb=RLLogger(check_freq=1, mode='min')
        
        t0=time.time()
        #create a RL object based on the env object
        ppo = PPO2(MlpPolicy, env=env, n_steps=n_steps[i], seed=1)
        #optimise the enviroment class
        ppo.learn(total_timesteps=ngen*npop, callback=cb)
        time_array[i,1]=time.time()-t0
        
        
        t0=time.time()
        gwo=GWO(mode='min', fit=FIT, bounds=BOUNDS, nwolves=npop, ncores=ncores_lst[i], seed=1)
        x_best, y_best, gwo_hist=gwo.evolute(ngen=ngen, verbose=1)
        time_array[i,2]=time.time()-t0

        x0=[[50,50,50,50,50] for i in range(npop)]  #initial guess
        t0=time.time()
        pesa=PESA(mode='min', bounds=BOUNDS, fit=FIT, npop=pesa_npop[i], mu=int(pesa_npop[i]/2), alpha_init=0.2,
                  alpha_end=1.0, alpha_backdoor=0.1, ncores=pesa_cores_lst[i])
        x_best, y_best, pesa_hist=pesa.evolute(ngen=pesa_ngen[i], x0=x0, verbose=1)
        time_array[i,3]=time.time()-t0
            
    time_df=pd.DataFrame(time_array, columns=['HHO', 'PPO', 'GWO', 'PESA'], index=['1', '8', '16', '32'])
    time_df.to_csv('parallel_time.csv')
    print(time_df)
    
    axes=time_df.plot.bar(rot=0)
    axes.set_xlabel('Number of Processors')
    axes.set_ylabel('Total Computing Time (s)')
    axes.legend(ncol=2)
    axes.figure.savefig('parallel.png',format='png', dpi=300, bbox_inches="tight")