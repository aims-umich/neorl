# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:21:42 2021

@author: majdi
"""

#*********************************************
# NEORL Simple Example
# Using DE to minimize the sphere function  
#*********************************************
from neorl import DE

#--Define the fitness function
def FIT(individual):
    #Sphere test objective function. 
    y=sum(x**2 for x in individual)
    return -y  #-1 is used to convert min to max

#--Define parameter space (d=5)
nx=5
BOUNDS={}
for i in range(1,nx+1):
    BOUNDS['x'+str(i)]=['float', -100, 100]

#--Differential Evolution
de=DE(bounds=BOUNDS, fit=FIT, npop=60, CR=0.7, F=0.5, ncores=1, seed=1)
x_best, y_best, de_hist=de.evolute(ngen=100, verbose=0)

#--Optimal Results
print('best x:', x_best)
print('best y:', y_best)