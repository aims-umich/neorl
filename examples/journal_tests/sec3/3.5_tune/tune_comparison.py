#--------------------------------------------------------------------
# Paper: NEORL: A Framework for NeuroEvolution Optimization with RL
# Section: Script for section 3.5
# Contact: Majdi I. Radaideh (radaideh@mit.edu)
# Last update: 9/10/2021
#---------------------------------------------------------------------

from neorl.tune import GRIDTUNE, RANDTUNE, BAYESTUNE, ESTUNE
from neorl import ES
import matplotlib.pyplot as plt

#**********************************************************
# Part I: Original Problem Settings
#**********************************************************

#Define the fitness function (for original optimisation)
def sphere(individual):
    y=sum(x**2 for x in individual)
    return y

#*************************************************************
# Part II: Define fitness function for hyperparameter tuning
#*************************************************************
def tune_fit(cxpb, mutpb, alpha):

    #--setup the parameter space
    nx=100
    BOUNDS={}
    for i in range(1,nx+1):
        BOUNDS['x'+str(i)]=['float', -100, 100]

    #--setup the ES algorithm
    es=ES(mode='min', bounds=BOUNDS, fit=sphere, lambda_=80, mu=40, mutpb=mutpb, alpha=alpha,
         cxmode='blend', cxpb=cxpb, ncores=1, seed=1)

    #--Evolute the ES object and obtains y_best
    #--turn off verbose for less algorithm print-out when tuning
    x_best, y_best, es_hist=es.evolute(ngen=150, verbose=0)

    return y_best #returns the best score

#*************************************************************
# Grid Tuning
#*************************************************************
param_grid={
#def tune_fit(cxpb, mutpb, alpha):
'cxpb': [0.2, 0.4, 0.6, 0.7, 0.8],  #cxpb is first
'mutpb': [0.05, 0.1, 0.15, 0.2],  #mutpb is second
'alpha': [0.1, 0.2, 0.3, 0.4, 0.5]}  #alpha is third

#setup a grid tune object
gtune=GRIDTUNE(param_grid=param_grid, fit=tune_fit)
#view the generated cases before running them
print(gtune.hyperparameter_cases)
#tune the parameters with method .tune
gridres=gtune.tune(ncores=25, csvname='grid_tune.csv')
#gridres = gridres.sort_values(['score'], axis='index', ascending=True)
print(gridres)

#*************************************************************
# Random Tuning
#*************************************************************

param_grid={
'cxpb': ['float', 0.2, 0.8],             
'mutpb':   ['float', 0.05, 0.2],             
'alpha': ['float', 0.1, 0.5]} 

#setup a random tune object
rtune=RANDTUNE(param_grid=param_grid, fit=tune_fit, ncases=100, seed=1)
#view the generated cases before running them
print(rtune.hyperparameter_cases)
#tune the parameters with method .tune
randres=rtune.tune(ncores=25, csvname='rand_tune.csv')
#randres = randres.sort_values(['score'], axis='index', ascending=True)
print(randres)

#*************************************************************
# Evolutionary Tuning
#*************************************************************

def tune_fit2(x):
    
    cxpb=x[0] 
    mutpb=x[1]
    alpha=x[2]
    
    #--setup the parameter space
    nx=100
    BOUNDS={}
    for i in range(1,nx+1):
        BOUNDS['x'+str(i)]=['float', -100, 100]

    #--setup the ES algorithm
    es=ES(mode='min', bounds=BOUNDS, fit=sphere, lambda_=80, mu=40, mutpb=mutpb, alpha=alpha,
         cxmode='blend', cxpb=cxpb, ncores=1, seed=1)

    #--Evolute the ES object and obtains y_best
    #--turn off verbose for less algorithm print-out when tuning
    x_best, y_best, es_hist=es.evolute(ngen=150, verbose=0)

    return y_best #returns the best score

param_grid={
'cxpb': ['float', 0.2, 0.8],             
'mutpb':   ['float', 0.05, 0.2],             
'alpha': ['float', 0.1, 0.5]}  

#setup a evolutionary tune object
etune=ESTUNE(mode='min', param_grid=param_grid, fit=tune_fit2, ngen=10) #total cases is ngen * 10
#tune the parameters with method .tune
evolures=etune.tune(ncores=25, csvname='evolu_tune.csv', verbose=True)
#evolures = evolures.sort_values(['score'], axis='index', ascending=True) #rank the scores from min to max
print(evolures)

#*************************************************************
# Bayesian Tuning
#*************************************************************

param_grid={
'cxpb': ['float', 0.2, 0.8],            
'mutpb':   ['float', 0.05, 0.2],            
'alpha': ['float', 0.1, 0.5]}   

#setup a bayesian tune object
btune=BAYESTUNE(mode='min', param_grid=param_grid, fit=tune_fit, ncases=100)
#tune the parameters with method .tune
bayesres=btune.tune(ncores=1, csvname='bayes_tune.csv', verbose=True)
#bayesres = bayesres.sort_values(['score'], axis='index', ascending=True) 
print(bayesres)

n=10
g=[min(gridres['score'].iloc[i:i+n]) for i in range(0,len(gridres),n)]
r=[min(randres['score'].iloc[i:i+n]) for i in range(0,len(randres),n)]
b=[min(bayesres['score'].iloc[i:i+n]) for i in range(0,len(bayesres),n)]
e=evolures['score'].values

plt.figure()
plt.plot(g, '-s', label='Grid')
plt.plot(r, '--o', label='Random')
plt.plot(e, '-.d', label='Evolutionary')
plt.plot(b, '-^', label='Bayesian')
plt.legend()
plt.xlabel('Generation (10 Hyperparameter sets)')
plt.ylabel('Minimum Fitness')
plt.savefig('hyperparam.png',format='png', dpi=300, bbox_inches="tight")

