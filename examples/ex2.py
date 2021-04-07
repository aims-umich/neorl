#---------------------------------
# Import packages
#---------------------------------
import numpy as np
import matplotlib.pyplot as plt
from neorl import PSO, DE, XNES
from math import exp, sqrt, cos, pi
np.random.seed(50)

#---------------------------------
# Fitness function
#---------------------------------
def ACKLEY(individual):
    #Ackley objective function.
    d = len(individual)
    f=20 - 20 * exp(-0.2*sqrt(1.0/d * sum(x**2 for x in individual))) \
            + exp(1) - exp(1.0/d * sum(cos(2*pi*x) for x in individual))
    return -f   #-1 to convert to maximization problem

#---------------------------------
# Parameter Space
#---------------------------------
#Setup the parameter space (d=8)
d=8
lb=-32
ub=32
BOUNDS={}
for i in range(1,d+1):
    BOUNDS['x'+str(i)]=['float', lb, ub]

#---------------------------------
# PSO
#---------------------------------
pso=PSO(bounds=BOUNDS, fit=ACKLEY, npar=60, c1=2.05, c2=2.1, speed_mech='constric', seed=1)
x_best, y_best, pso_hist=pso.evolute(ngen=120, verbose=0)

#---------------------------------
# DE
#---------------------------------
de=DE(bounds=BOUNDS, fit=ACKLEY, npop=60, F=0.5, CR=0.7, ncores=1, seed=1)
x_best, y_best, de_hist=de.evolute(ngen=120, verbose=0)

#---------------------------------
# NES
#---------------------------------
x0=[-18]*d
amat = np.eye(len(x0))
xnes = XNES(ACKLEY, x0, amat, npop=80, bounds=BOUNDS, use_adasam=True, eta_bmat=0.04, eta_sigma=0.1, patience=9999, verbose=0, ncores=1)
x_best, y_best, nes_hist=xnes.evolute(120)

#---------------------------------
# Plot
#---------------------------------
#Plot fitness for both methods
plt.figure()
plt.plot(-np.array(pso_hist), label='PSO')             #multiply by -1 to covert back to a min problem
plt.plot(-np.array(de_hist), label='DE')               #multiply by -1 to covert back to a min problem
plt.plot(-np.array(nes_hist['fitness']), label='NES')  #multiply by -1 to covert back to a min problem
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.savefig('ex2_fitness.png',format='png', dpi=300, bbox_inches="tight")
plt.show()