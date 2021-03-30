# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:04:52 2021

@author: majdi
"""

#---------------------------------
# Import packages
#---------------------------------
from neorl.tune import GRIDTUNE
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
# DE
#---------------------------------
de=DE(bounds=BOUNDS, fit=ACKLEY, npop=60, F=0.5, CR=0.7, ncores=1, seed=1)
x_best, y_best, de_hist=de.evolute(ngen=120, verbose=0)