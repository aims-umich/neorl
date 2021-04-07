from neorl.tune import GRIDTUNE
from neorl import ES

#**********************************************************
# Part I: Original Problem Settings
#**********************************************************

#Define the fitness function (for original optimisation)
def sphere(individual):
    y=sum(x**2 for x in individual)
    return -y  #-1 to convert min to max problem

#*************************************************************
# Part II: Define fitness function for hyperparameter tuning
#*************************************************************
def tune_fit(cxpb, mutpb, alpha):

    #--setup the parameter space
    nx=5
    BOUNDS={}
    for i in range(1,nx+1):
        BOUNDS['x'+str(i)]=['float', -100, 100]

    #--setup the ES algorithm
    es=ES(bounds=BOUNDS, fit=sphere, lambda_=80, mu=40, mutpb=mutpb, alpha=alpha,
         cxmode='blend', cxpb=cxpb, ncores=1, seed=1)

    #--Evolute the ES object and obtains y_best
    #--turn off verbose for less algorithm print-out when tuning
    x_best, y_best, es_hist=es.evolute(ngen=100, verbose=0)

    return y_best #returns the best score

#*************************************************************
# Part III: Tuning
#*************************************************************
#Setup the parameter space
#VERY IMPORTANT: The order of these grids MUST be similar to their order in tune_fit
#see tune_fit
param_grid={
#def tune_fit(cxpb, mutpb, alpha):
'cxpb': [0.2, 0.4],  #cxpb is first
'mutpb': [0.05, 0.1],  #mutpb is second
'alpha': [0.1, 0.2, 0.3, 0.4]}  #alpha is third

#setup a grid tune object
gtune=GRIDTUNE(param_grid=param_grid, fit=tune_fit)
#view the generated cases before running them
print(gtune.hyperparameter_cases)
#tune the parameters with method .tune
gridres=gtune.tune(ncores=1, csvname='tune.csv')
print(gridres)   #the results are saved in dataframe and ranked from best to worst