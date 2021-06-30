#---------------------------------
# Import packages
#---------------------------------
import numpy as np
from math import cos, pi, exp, e, sqrt
import matplotlib.pyplot as plt
from neorl import BAT, GWO, MFO

#---------------------------------
# Fitness function
#---------------------------------
def TBT(individual):
    """Three-bar truss Design
    """
    x1 = individual[0]
    x2 = individual[1]
    
    y = (2*sqrt(2)*x1 + x2) * 100
    
    #Constraints
    if x1 <= 0:
    	g = [1,1,1]
    else:
    	g1 = (sqrt(2)*x1+x2)/(sqrt(2)*x1**2 + 2*x1*x2) * 2 - 2
    	g2 = x2/(sqrt(2)*x1**2 + 2*x1*x2) * 2 - 2
    	g3 = 1/(x1 + sqrt(2)*x2) * 2 - 2
    	g = [g1,g2,g3]
    
    g_round=np.round(np.array(g),6)
    w1=100
    w2=100
    
    phi=sum(max(item,0) for item in g_round)
    viol=sum(float(num) > 0 for num in g_round)
    
    return y + w1*phi + w2*viol
#---------------------------------
# Parameter space
#---------------------------------
nx = 2
BOUNDS = {}
for i in range(1, nx+1):
    BOUNDS['x'+str(i)]=['float', 0, 1]


#---------------------------------
# BAT
#---------------------------------
bat=BAT(mode='min', bounds=BOUNDS, fit=TBT, nbats=10, fmin = 0 , fmax = 1, A=0.5, r0=0.5, levy = True, seed = 1, ncores=1)
bat_x, bat_y, bat_hist=bat.evolute(ngen=100, verbose=1)

#---------------------------------
# GWO
#---------------------------------
gwo=GWO(mode='min', fit=TBT, bounds=BOUNDS, nwolves=10, ncores=1, seed=1)
gwo_x, gwo_y, gwo_hist=gwo.evolute(ngen=100, verbose=1)

#---------------------------------
# MFO
#---------------------------------
mfo=MFO(mode='min', bounds=BOUNDS, fit=TBT, nmoths=10, b = 0.2, ncores=1, seed=1)
mfo_x, mfo_y, mfo_hist=mfo.evolute(ngen=100, verbose=1)

#---------------------------------
# Plot
#---------------------------------
plt.figure()
plt.plot(bat_hist['global_fitness'], label = 'BAT')
plt.plot(gwo_hist['fitness'], label = 'GWO')
plt.plot(mfo_hist['global_fitness'], label = 'MFO')
plt.legend()
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.savefig('TBT_fitness.png',format='png', dpi=300, bbox_inches="tight")