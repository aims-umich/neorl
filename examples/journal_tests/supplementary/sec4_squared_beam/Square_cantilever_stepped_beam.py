# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 17:05:54 2021

@author: Devin Seyler
"""

#--------------------------------------------------------------------
# Paper: NEORL: A Framework for NeuroEvolution Optimization with RL
# Section: Script for supplementary materials section 4
# Contact: Majdi I. Radaideh (radaideh@mit.edu)
# Last update: 9/10/2021
#---------------------------------------------------------------------

#---------------------------------
# Import packages
#---------------------------------
import numpy as np
from math import cos, pi, exp, e, sqrt
import matplotlib.pyplot as plt
from neorl import DE, GWO, BAT

#---------------------------------
# Fitness function
#---------------------------------
def CSB_square(individual):
    """Square Cantilever Stepped Beam
    individual[i = 0 - 4] are beam heights and widths
    """
    
    check=all([item >= BOUNDS['x'+str(i+1)][1] for i,item in enumerate(individual)]) \
          and all([item <= BOUNDS['x'+str(i+1)][2] for i,item in enumerate(individual)])
    if not check:
        raise Exception ('--error check fails')
    
    g = 61/individual[0]**3 + 37/individual[1]**3 + 19/individual[2]**3 + 7/individual[3]**3 + 1/individual[4]**3 - 1

    g_round=np.round(g,6)
    w1=1000
        
    #phi=max(g_round,0)
    if g_round > 0:
        phi = 1
    else:
        phi = 0

    V = 0.0624*(np.sum(individual)) 

    return V + w1*phi

#---------------------------------
# Parameter space
#---------------------------------
nx=5
BOUNDS={}
for i in range(1, 6):
    BOUNDS['x'+str(i)]=['float', 0.01, 100]

#---------------------------------
# DE
#---------------------------------
de=DE(mode='min', bounds=BOUNDS, fit=CSB_square, npop=50, F=0.5, CR=0.7, ncores=1, seed=1)
de_x, de_y, de_hist=de.evolute(ngen=200, verbose=1)
assert CSB_square(de_x) == de_y

#---------------------------------
# BAT
#---------------------------------
bat=BAT(mode='min', bounds=BOUNDS, fit=CSB_square, nbats=50, fmin = 0 , fmax = 1, A=0.5, r0=0.5, levy = True, seed = 1, ncores=1)
bat_x, bat_y, bat_hist=bat.evolute(ngen=200, verbose=1)
assert CSB_square(bat_x) == bat_y

#---------------------------------
# GWO
#---------------------------------
gwo=GWO(mode='min', bounds=BOUNDS, fit=CSB_square, nwolves=50, ncores=1, seed=1)
gwo_x, gwo_y, gwo_hist=gwo.evolute(ngen=200, verbose=1)
assert CSB_square(gwo_x) == gwo_y

#---------------------------------
# Plot
#---------------------------------
plt.figure()
plt.plot(de_hist, label = 'DE')
plt.plot(bat_hist['global_fitness'], '--', label = 'BAT')
plt.plot(gwo_hist['fitness'], '-.', label = 'GWO')
plt.legend(fontsize = 12)
plt.xlabel('Generation', fontsize = 12)
plt.ylabel('Fitness', fontsize = 12)
#plt.ylim(0,150000)
print('de: ', de_x, de_y)
print('bat: ', bat_x , bat_y)
print('gwo: ', gwo_x , gwo_y)
plt.savefig('CSB_square_fitness.png',format='png', dpi=300, bbox_inches="tight")