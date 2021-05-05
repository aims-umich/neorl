from neorl.tune import BAYESTUNE
from neorl import ES

def test_bayes():
    
    #Define the fitness function (for original optimisation)
    def sphere(individual):
            y=sum(x**2 for x in individual)
            return y
    
    def tune_fit(cxpb, mu, alpha, cxmode):
    
            #--setup the parameter space
            nx=5
            BOUNDS={}
            for i in range(1,nx+1):
                    BOUNDS['x'+str(i)]=['float', -100, 100]
    
            #--setup the ES algorithm
            es=ES(mode='min', bounds=BOUNDS, fit=sphere, lambda_=80, mu=mu, mutpb=0.1, alpha=alpha,
                     cxmode=cxmode, cxpb=cxpb, ncores=1, seed=1)
    
            #--Evolute the ES object and obtains y_best
            #--turn off verbose for less algorithm print-out when tuning
            x_best, y_best, es_hist=es.evolute(ngen=100, verbose=0)
    
            return y_best #returns the best score

    param_grid={
    #def tune_fit(cxpb, mu, alpha, cxmode):
    'cxpb': [[0.1, 0.9],'float'],             #cxpb is first (low=0.1, high=0.8, type=float/continuous)
    'mu':   [[30, 60],'int'],                 #mu is second (low=30, high=60, type=int/discrete)
    'alpha':[[0.1, 0.2, 0.3, 0.4],'grid'],    #alpha is third (grid with fixed values, type=grid/categorical)
    'cxmode':[['blend', 'cx2point'],'grid']}  #cxmode is fourth (grid with fixed values, type=grid/categorical)
    
    #setup a bayesian tune object
    btune=BAYESTUNE(param_grid=param_grid, fit=tune_fit, ncases=15)
    bayesres=btune.tune(nthreads=1, verbose=True)
    print(bayesres)   #the results are saved in dataframe and ranked from best to worst

test_bayes()