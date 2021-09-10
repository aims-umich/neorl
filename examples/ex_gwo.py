# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 22:47:37 2021

@author: majdi
"""

from neorl import GWO
import matplotlib.pyplot as plt
    
#Define the fitness function
def FIT(individual):
    """Sphere test objective function.
            F(x) = sum_{i=1}^d xi^2
            d=1,2,3,...
            Range: [-100,100]
            Minima: 0
    """
    y=sum(x**2 for x in individual)
    return y
    
#Setup the parameter space (d=5)
nx=5
BOUNDS={}
for i in range(1,nx+1):
    BOUNDS['x'+str(i)]=['float', -100, 100]

nwolves=5
gwo=GWO(mode='min', fit=FIT, bounds=BOUNDS, nwolves=nwolves, ncores=4, seed=1)
x_best, y_best, gwo_hist=gwo.evolute(ngen=100, verbose=1)

#-----
#or with fixed initial guess for all wolves (uncomment below)
#-----
#x0=[[-90, -85, -80, 70, 90] for i in range(nwolves)]
#x_best, y_best, gwo_hist=gwo.evolute(ngen=100, x0=x0)

plt.figure()
plt.plot(gwo_hist['alpha_wolf'], label='alpha_wolf')
plt.plot(gwo_hist['beta_wolf'], label='beta_wolf')
plt.plot(gwo_hist['delta_wolf'], label='delta_wolf')
plt.plot(gwo_hist['fitness'], label='best')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.show()