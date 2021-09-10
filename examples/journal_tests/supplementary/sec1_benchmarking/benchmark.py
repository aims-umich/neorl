#--------------------------------------------------------------------
# Paper: NEORL: A Framework for NeuroEvolution Optimization with RL
# Section: Script for supplementary materials section 1
# Contact: Majdi I. Radaideh (radaideh@mit.edu)
# Last update: 9/10/2021
#---------------------------------------------------------------------

import matplotlib
matplotlib.use('Agg')
import numpy as np
from neorl.benchmarks.classic import sphere, cigar, ackley, bohachevsky, brown, zakharov, salomon, levy
from neorl import JAYA, PSO, DE, SSA, GWO, MFO, BAT, PESA2, WOA, HHO
import pandas as pd
import joblib

#All functions have global minima of zero

def method_worker(inp):
    #inp=[func, func_min, func_max, seed]
    
    FIT=inp[0]
    func_min=inp[1]
    func_max=inp[2]
    seed=inp[3]
    results=np.zeros((10))
    
    BOUNDS={}
    for i in range(nx):
        BOUNDS['x'+str(i+1)]=['float', func_min, func_max]
            
    #---------------------------------
    #JAYA
    #---------------------------------
    jaya=JAYA(mode='min', bounds=BOUNDS, fit=FIT, npop=npop, ncores=1, seed=seed)
    x_jaya, y_jaya, jaya_hist=jaya.evolute(ngen=ngen, verbose=0)
    results[0]=y_jaya
    #---------------------------------
    # PSO
    #---------------------------------
    pso=PSO(mode='min', bounds=BOUNDS, fit=FIT, c1=2.05, c2=2.1, npar=npop, speed_mech='constric', ncores=1, seed=1)
    pso_x, pso_y, pso_hist=pso.evolute(ngen=ngen, verbose=0)
    results[1]=pso_y
    #---------------------------------
    # DE
    #---------------------------------
    de=DE(mode='min', bounds=BOUNDS, fit=FIT, npop=npop, F=0.5, CR=0.7, ncores=1, seed=seed)
    de_x, de_y, de_hist=de.evolute(ngen=ngen, verbose=0)
    results[2]=de_y
    #---------------------------------
    # SSA
    #---------------------------------
    ssa=SSA(mode='min', bounds=BOUNDS, fit=FIT, nsalps=npop, c1=None, ncores=1, seed=seed)
    ssa_x, ssa_y, ssa_hist=ssa.evolute(ngen=ngen, verbose=0)
    results[3]=ssa_y
    #---------------------------------
    # BAT
    #---------------------------------
    bat=BAT(mode='min', bounds=BOUNDS, fit=FIT, nbats=npop, fmin = 0 , fmax = 1, A=0.5, r0=0.5, levy = True, seed = 1, ncores=1)
    bat_x, bat_y, bat_hist=bat.evolute(ngen=ngen, verbose=0)
    results[4]=bat_y
    #---------------------------------
    # GWO
    #---------------------------------
    gwo=GWO(mode='min', fit=FIT, bounds=BOUNDS, nwolves=npop, ncores=1, seed=seed)
    gwo_x, gwo_y, gwo_hist=gwo.evolute(ngen=ngen, verbose=0)
    results[5]=gwo_y
    #---------------------------------
    # MFO
    #---------------------------------
    mfo=MFO(mode='min', bounds=BOUNDS, fit=FIT, nmoths=npop, b = 0.2, ncores=1, seed=seed)
    mfo_x, mfo_y, mfo_hist=mfo.evolute(ngen=ngen, verbose=0)
    results[6]=mfo_y
    #---------------------------------
    # PESA2
    #---------------------------------
    pesa2=PESA2(mode='min', bounds=BOUNDS, fit=FIT, npop=npop, nwolves=5, ncores=1, seed=seed)
    pesa2_x, pesa2_y, pesa2_hist=pesa2.evolute(ngen=ngen, replay_every=1, verbose=0)    
    results[7]=pesa2_y
    #---------------------------------
    # WOA
    #---------------------------------
    woa=WOA(mode='min', bounds=BOUNDS, fit=FIT, nwhales=npop, a0=1.5, b=1, ncores=1, seed=seed)
    woa_x, woa_y, woa_hist=woa.evolute(ngen=ngen, verbose=0)
    results[8]=woa_y
    #---------------------------------
    # HHO
    #---------------------------------
    hho=HHO(mode='min', bounds=BOUNDS, fit=FIT, nhawks=npop, ncores=1, seed=seed)
    hho_x, hho_y, hho_hist=hho.evolute(ngen=ngen, verbose=0)
    results[9]=hho_y
    
    return results

func_list=[sphere, cigar, ackley, bohachevsky, brown, zakharov, salomon, levy]
func_min=[-10, -100, -32, -100, -1, -5, -100, -10]
func_max=[10, 100, 32, 100, 4, 10, 100, 10]

n_rounds=5
methods=['JAYA', 'PSO', 'DE', 'SSA', 'BAT', 'GWO', 'MFO', 'PESA2', 'WOA', 'HHO']
colnames=['round'+str(i+1) for i in range(n_rounds)]
nx=10
npop=60
ngen=100
seed_list=[1,10,100,1000,10000]
best_res=np.zeros((len(func_list)))
ncores=1    

core_lst=[]
for idx,item in enumerate(func_list):
    for k in range (n_rounds):
        core_lst.append((item, func_min[idx], func_max[idx], seed_list[k]))

if ncores > 1:
    with joblib.Parallel(n_jobs=ncores) as parallel:
        fitness=parallel(joblib.delayed(method_worker)(item) for item in core_lst)
else:
    fitness=[]
    for i,item in enumerate(core_lst):
        fitness.append(method_worker(item))
        print('task {}/{} is done'.format(i+1,len(core_lst)))

index=0
results=np.zeros((10,n_rounds))
best_res=np.zeros((len(func_list)))
fun_index=1
for f in range (1,len(func_list)*n_rounds+1):
    results[:,index]=fitness[f-1]
    #print(index, f)
    index+=1
    if f % n_rounds == 0:
        #print(results)
        df=pd.DataFrame(results, columns=colnames, index=methods)
        df.to_csv('{}.csv'.format(func_list[fun_index-1].__name__),index=True) 
        index=0
        #print('Function {} is done'.format(idx+1))
        best_res[fun_index-1]=np.min(results)
        fun_index+=1
        results=np.zeros((10,n_rounds))
   
best_df=pd.DataFrame(best_res)
index_lst=['sphere', 'cigar', 'ackley', 'bohachevsky', 'brown', 'zakharov', 'salomon', 'levy']
best_df.index=index_lst
best_df.to_csv('best_classic.csv',index=True) 

    
