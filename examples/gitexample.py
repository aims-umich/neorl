# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:59:41 2021

@author: majdi
"""

#---------------------------------
# Import packages
#---------------------------------
import numpy as np
import matplotlib.pyplot as plt
from neorl import PSO, DE, XNES, ES
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

print(np.round(x_best,4))
#---------------------------------
# DE
#---------------------------------
de=DE(bounds=BOUNDS, fit=ACKLEY, npop=60, F=0.5, CR=0.7, ncores=1, seed=1)
x_best, y_best, de_hist=de.evolute(ngen=120, verbose=0)

print(np.round(x_best,4))

#---------------------------------
# NES
#---------------------------------
x0=[-18]*d
amat = np.eye(len(x0))
xnes = XNES(ACKLEY, x0, amat, npop=80, bounds=BOUNDS, use_adasam=True, eta_bmat=0.04, eta_sigma=0.1, patience=9999, verbose=0, ncores=1)
x_best, y_best, nes_hist=xnes.evolute(120)
print(np.round(x_best,4))

#---------------------------------
# ES
#---------------------------------
es=ES(bounds=BOUNDS, fit=ACKLEY, lambda_=100, mu=50, cxpb=0.7, cxmode='blend', ncores=1, seed=1)
x_best, y_best, es_hist=es.evolute(ngen=120, verbose=0)
print(np.round(x_best,4))


#---------------------------------
# GA
#---------------------------------
ga=GA(bounds=BOUNDS, fit=ACKLEY, lambda_=100, mu=50, cxpb=0.7, cxmode='blend', ncores=1, seed=1)
x_best, y_best, ga_hist=es.evolute(ngen=120, verbose=0)
print(np.round(x_best,4))

#---------------------------------
# Plot
#---------------------------------
#Plot fitness for both methods
plt.figure()
plt.plot(-np.array(pso_hist), label='PSO')             #multiply by -1 to covert back to a min problem
plt.plot(-np.array(de_hist), label='DE')               #multiply by -1 to covert back to a min problem
plt.plot(-np.array(nes_hist['fitness']), label='NES')  #multiply by -1 to covert back to a min problem
plt.plot(-np.array(es_hist), label='ES')  #multiply by -1 to covert back to a min problem
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.savefig('ex2_fitness.png',format='png', dpi=300, bbox_inches="tight")
plt.show()


from neorl import ES

#Define the fitness function
def FIT(individual):
        """Sphere test objective function.
                F(x) = sum_{i=1}^d xi^2
                d=1,2,3,...
                Range: [-100,100]
                Minima: 0
        """
        y=sum(x**2 for x in individual)
        return -y  #-1 to convert min to max problem

#Setup the parameter space (d=5)
nx=5
BOUNDS={}
for i in range(1,nx+1):
        BOUNDS['x'+str(i)]=['float', -100, 100]

ga=ES(bounds=BOUNDS, fit=FIT, lambda_=80, mu=40, mutpb=0.25,
     cxmode='blend', cxpb=0.7, ncores=1, seed=1)
x_best, y_best, es_hist=ga.evolute(ngen=100, verbose=0)