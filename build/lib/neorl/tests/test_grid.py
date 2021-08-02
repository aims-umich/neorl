from neorl.tune import GRIDTUNE
from neorl import ES

def test_grid():
    
    #Define the fitness function (for original optimisation)
    def sphere(individual):
        y=sum(x**2 for x in individual)
        return y 
    
    def tune_fit(cxpb, mutpb, alpha):
    
        #--setup the parameter space
        nx=5
        BOUNDS={}
        for i in range(1,nx+1):
            BOUNDS['x'+str(i)]=['float', -100, 100]
    
        #--setup the ES algorithm
        es=ES(mode='min', bounds=BOUNDS, fit=sphere, lambda_=80, mu=40, mutpb=mutpb, alpha=alpha,
             cxmode='blend', cxpb=cxpb, ncores=1, seed=1)
    
        #--Evolute the ES object and obtains y_best
        #--turn off verbose for less algorithm print-out when tuning
        x_best, y_best, es_hist=es.evolute(ngen=100, verbose=0)
    
        return y_best #returns the best score
    
    param_grid={
    #def tune_fit(cxpb, mutpb, alpha):
    'cxpb': [0.2, 0.4],  #cxpb is first
    'mutpb': [0.05, 0.1],  #mutpb is second
    'alpha': [0.1, 0.2, 0.3, 0.4]}  #alpha is third
    
    gtune=GRIDTUNE(param_grid=param_grid, fit=tune_fit)
    gridres=gtune.tune(ncores=1)
    print(gridres)

test_grid()