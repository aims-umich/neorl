#----------------------------------------------------------------
#                ANS Winter Meeting 2021
# Paper: NEORL: An Open-source Python Framework for 
#               Optimization with Machine Learning Neuroevolution
# Section: Script for Case 1 (Ackley Function)
# Contact: Majdi I. Radaideh (radaideh@mit.edu)
# Last update: 7/7/2021
#----------------------------------------------------------------

#---------------------------------
# Import packages
#---------------------------------
import numpy as np
import matplotlib.pyplot as plt
from neorl import PSO, GWO, WOA
from math import exp, sqrt, cos, pi

#---------------------------------
# Fitness function
#---------------------------------
def ACKLEY(individual):
    #Ackley objective function.
    d = len(individual)
    f=20 - 20 * exp(-0.2*sqrt(1.0/d * sum(x**2 for x in individual))) \
            + exp(1) - exp(1.0/d * sum(cos(2*pi*x) for x in individual))
    return f

#---------------------------------
# Parameter Space
#---------------------------------
d=20
space={}
for i in range(1,d+1):
    space['x'+str(i)]=['float', -32, 32]

#---------------------------------
# PSO
#---------------------------------
pso=PSO(mode='min', bounds=space, fit=ACKLEY, npar=50, 
        c1=2.05, c2=2.1, speed_mech='constric', seed=1)
x_pso, y_pso, pso_hist=pso.evolute(ngen=150, verbose=0)

#---------------------------------
# GWO
#---------------------------------
gwo=GWO(mode='min', fit=ACKLEY, bounds=space, nwolves=50, ncores=1, seed=1)
x_gwo, y_gwo, gwo_hist=gwo.evolute(ngen=150, verbose=0)

#---------------------------------
# WOA
#---------------------------------
woa=WOA(mode='min', bounds=space, fit=ACKLEY, nwhales=50, a0=1.5, b=1, ncores=1, seed=1)
x_woa, y_woa, woa_hist=woa.evolute(ngen=150, verbose=0)

#---------------------------------
# Plot
#---------------------------------
#Plot fitness for both methods
plt.figure()
plt.plot(np.array(pso_hist), label='PSO')           
plt.plot(np.array(gwo_hist['fitness']), '--', label='GWO')            
plt.plot(np.array(woa_hist['global_fitness']), '-.', label='WOA')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.savefig('ex1_ans21_fitness.png',format='png', dpi=300, bbox_inches="tight")
plt.show()

#---------------------------------
# Comparison
#---------------------------------
print('---Best PSO Results---')
print('Best x:', np.round(x_pso,4))
print('Best y:', y_pso)
print('---Best GWO Results---')
print('Best x:', np.round(x_gwo,4))
print('Best y:', y_gwo)
print('---Best WOA Results---')
print('Best x:', np.round(x_woa,4))
print('Best y:', y_woa)
print()