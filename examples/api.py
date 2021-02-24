"""
Created on Mon Jan 28 08:09:18 2019

@author: Majdi Radaideh
"""
    
import os, sys, warnings, random
warnings.filterwarnings("ignore")
import numpy as np
import gym
import time
import matplotlib.pyplot as plt
from neorl import DQN
from neorl import PPO2
from neorl import ACER
from neorl import A2C



#Define the fitness function
def FIT(individual):
    """Sphere test objective function.
        F(x) = sum_{i=1}^d xi^2
        d=1,2,3,...
        Range: [-100,100]
        Minima: 0
    """
    return -sum(x**2 for x in individual)

#Setup the parameter space
nx=5
BOUNDS={}
for i in range(1,nx+1):
    BOUNDS['x'+str(i)]=['float', -100, 100]
    
    
def GenParticle(bounds, n):
    x0=[]
    for i in range(n):
        content=[]
        for key in bounds:
            if bounds[key][0] == 'int':
                content.append(random.randint(bounds[key][1], bounds[key][2]))
            elif bounds[key][0] == 'float':
                content.append(random.uniform(bounds[key][1], bounds[key][2]))
            elif bounds[key][0] == 'grid':
                content.append(random.sample(bounds[key][1],1)[0])
            else:
                raise Exception ('unknown data type is given, either int, float, or grid are allowed for parameter bounds')   
        particle=list(content)
        x0.append(particle)
    return x0

#---------------------------------
# PSO
#---------------------------------
from neorl import PSO

NGEN=150              #number of generations
NPAR=60                #LAMBDA for ES, total length of each chains for SA, Swarm Size for PSO 
C1=2.05                #cognitive speed coeff (2.05 is typical value)
C2=2.05                 #social speed coeff (2.05 is typical value)
SPEED_MECH='constric'   #`constric`, `timew`, or `globw` --> how to modify particle speed
                        #Both LAMBDA and NPAR equal to NPOP, symmetry between methods
NCORES=1
x0=GenParticle(bounds=BOUNDS, n=60)
pso=PSO(bounds=BOUNDS, fit=FIT, npar=NPAR, ncores=NCORES, c1=C1, c2=C2, speed_mech=SPEED_MECH, seed=1)
swm_last_pop, swm_best_pos, swm_best_fit=pso.evolute(ngen=NGEN, x0=x0, verbose=0)

#---------------------------------
# SA
#---------------------------------
#To do: add special moves by user
from neorl import SA

TMAX=10000     #max annealing temperature for SA
CHI=0.15      #probablity to perturb each input attribute
COOLING='fast' #Cooling schedule (Fixed)
TMIN=1         #Minimum Temperature (Fixed)
STEPS=25*60
    
x0_sa=x0[0:5]
sa=SA(bounds=BOUNDS, fit=FIT, npop=60, ncores=1, chi=CHI, cooling=COOLING, Tmax=TMAX, Tmin=TMIN, seed=1)
x_best, E_best, stat=sa.anneal(ngen=100,verbose=0)

#---------------------------------
# DE
#---------------------------------
#To do: parallel DE
from neorl import DE

TMAX=10000     #max annealing temperature for SA
CHI=0.15      #probablity to perturb each input attribute
COOLING='fast' #Cooling schedule (Fixed)
TMIN=1         #Minimum Temperature (Fixed)
STEPS=25*60
    
de=DE(bounds=BOUNDS, fit=FIT, npop=60, mutate=0.5, recombination=0.7, ncores=1, seed=1)
x_best, y_best, de_hist=de.evolute(ngen=100, verbose=0)

#---------------------------------
# ES
#---------------------------------
#To do: check mu, lambda, mu + lambda
from neorl.evolu.es import ES

CXPB=0.6  #population crossover (0.4-0.8)
MUTPB=0.15   #population mutation (0.05-0.0.25)
INDPB=1.0 #ES attribute mutation (only used for cont. optimisation)
LAMBDA=60   #full population size before selection of MU (Fixed)
MU=30           # number of individuals to survive next generation in PSO/ES
SMIN = 1/nx #ES strategy min (Fixed)
SMAX = 0.5  #ES strategy max (Fixed)
    
NCORES=1
x0=GenParticle(bounds=BOUNDS, n=60)
es=ES(bounds=BOUNDS, fit=FIT, mu=MU, lambda_=LAMBDA, ncores=NCORES, cxmode='blend', alpha=0.1, cxpb=CXPB, mutpb=MUTPB, smin=SMIN, smax=SMAX)
pop_last=es.evolute(ngen=NGEN, verbose=0)

#---------------------------------
# NES
#---------------------------------
from neorl import XNES

mu = x0[0]
amat = np.eye(len(x0[0]))
eta_bmat=0.04  
eta_sigma=0.1
NPOP=40
# when adasam, use conservative eta
xnes = XNES(FIT, mu, amat, npop=NPOP, bounds=BOUNDS, use_adasam=True, eta_bmat=eta_bmat, eta_sigma=eta_sigma, patience=9999, verbose=0, ncores=1)
x_best, y_best, nes_hist=xnes.evolute(100)
#print(x_best, y_best)
plt.figure()
plt.plot(-np.array(de_hist), label='DE')
plt.plot(-np.array(nes_hist['fitness']), label='NES')
plt.xlabel('Step')
plt.ylabel('Fitness')
plt.legend()
plt.show()