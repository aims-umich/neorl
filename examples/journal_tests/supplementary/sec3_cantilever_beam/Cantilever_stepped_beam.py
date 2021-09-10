# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 15:24:52 2021

@author: Devin Seyler
"""

#--------------------------------------------------------------------
# Paper: NEORL: A Framework for NeuroEvolution Optimization with RL
# Section: Script for supplementary materials section 3
# Contact: Majdi I. Radaideh (radaideh@mit.edu)
# Last update: 9/10/2021
#---------------------------------------------------------------------

#---------------------------------
# Import packages
#---------------------------------
import numpy as np
from math import cos, pi, exp, e, sqrt
import matplotlib.pyplot as plt
from neorl import DE, SSA, GWO, PESA2

#---------------------------------
# Fitness function
#---------------------------------
def CSB(individual):
    """Cantilever Stepped Beam
    individual[i = 0 - 4] are beam widths
    individual[i = 5 - 9] are beam heights
    """
    
    check=all([item >= BOUNDS['x'+str(i+1)][1] for i,item in enumerate(individual)]) \
          and all([item <= BOUNDS['x'+str(i+1)][2] for i,item in enumerate(individual)])
    if not check:
        raise Exception ('--error check fails')
    
    P = 50000
    E = 2 * 10**7
    l = 100
    g = np.zeros(11)
    g[0] = 600*P/(individual[4] * individual[9]**2) - 14000
    g[1] = 6*P*(2*l)/(individual[3] * individual[8]**2) - 14000
    g[2] = 6*P*(3*l)/(individual[2] * individual[7]**2) - 14000
    g[3] = 6*P*(4*l)/(individual[1] * individual[6]**2) - 14000
    g[4] = 6*P*(5*l)/(individual[0] * individual[5]**2) - 14000
    g[5] = 12*P*l**3/(3*E) * (1/(individual[4] * individual[9]**3) + 7/(individual[3] * individual[8]**3) + 19/(individual[2] * individual[7]**3) + 37/(individual[1] * individual[6]**3) + 61/(individual[0] * individual[5]**3)) - 2.7
    g[6] = individual[9]/individual[4] - 20
    g[7] = individual[8]/individual[3] - 20
    g[8] = individual[7]/individual[2] - 20
    g[9] = individual[6]/individual[1] - 20
    g[10] = individual[5]/individual[0] - 20

    g_round=np.round(g,6)
    w1=2000
    w2=2000
        
    phi=sum(max(item,0) for item in g_round)
    viol=sum(float(num) > 0 for num in g_round)

    V = 0
    for i in range(5):
        V += individual[i] * individual[i+5] * 100

    return V + w1*phi + w2*viol

#---------------------------------
# Parameter space
#---------------------------------
nx=10
BOUNDS={}
for i in range(1, 6):
    BOUNDS['x'+str(i)]=['float', 1, 5]
for i in range(6, 11):
    BOUNDS['x'+str(i)]=['float', 30, 65]

#---------------------------------
# DE
#---------------------------------
de=DE(mode='min', bounds=BOUNDS, fit=CSB, npop=50, F=0.5, CR=0.7, ncores=1, seed=1)
de_x, de_y, de_hist=de.evolute(ngen=300, verbose=1)
assert CSB(de_x) == de_y

#---------------------------------
# SSA
#---------------------------------
ssa=SSA(mode='min', bounds=BOUNDS, fit=CSB, nsalps=50, c1=None, ncores=1, seed=1)
ssa_x, ssa_y, ssa_hist=ssa.evolute(ngen=300, verbose=1)
assert CSB(ssa_x) == ssa_y

#---------------------------------
# GWO
#---------------------------------
gwo=GWO(mode='min', fit=CSB, bounds=BOUNDS, nwolves=50, ncores=1, seed=1)
gwo_x, gwo_y, gwo_hist=gwo.evolute(ngen=300, verbose=1)
assert CSB(gwo_x) == gwo_y

#---------------------------------
# PESA2
#---------------------------------
pesa2=PESA2(mode='min', bounds=BOUNDS, fit=CSB, npop=50, nwolves=5, ncores=1, seed=1)
pesa2_x, pesa2_y, pesa2_hist=pesa2.evolute(ngen=600, replay_every=2, verbose=1)
assert CSB(pesa2_x) == pesa2_y

#---------------------------------
# Plot
#---------------------------------
plt.figure()
plt.plot(de_hist, '-', label = 'DE')
plt.plot(ssa_hist['global_fitness'], '--', label = 'SSA')
plt.plot(gwo_hist['fitness'], '-.', label = 'GWO')
plt.plot(pesa2_hist, ':', label = 'PESA2')
plt.legend()
plt.xlabel('Generation', fontsize = 12)
plt.ylabel('Fitness', fontsize = 12)
plt.ylim(60000,150000)
print('de: ', de_x, de_y)
print('ssa: ', ssa_x, ssa_y)
print('gwo: ', gwo_x , gwo_y)
print('pesa2: ', pesa2_x , pesa2_y)
plt.savefig('CSB_fitness.png',format='png', dpi=300, bbox_inches="tight")