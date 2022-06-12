from neorl import NSGAII, NSGAIII
from neorl.benchmarks.dtlz import DTLZ2   #Multi-objective benchmark

def test_nsga():

    #parameters
    NOBJ = 3   #number of objectives to optimize
    nx=12
    lambda_ = 92
    problem = DTLZ2(n_var=nx, n_obj=NOBJ)   #adapted and taken from pymop package
    dtlz2 = problem.evaluate                #objective function
    
    #Setup the parameter space
    BOUNDS={}
    for i in range(1,nx+1):
        BOUNDS['x'+str(i)]=['float', 0, 1]
    
    #setup and evolute NSGA-II
    nsgaii=NSGAII(mode='min', bounds=BOUNDS, fit=dtlz2, lambda_=lambda_, mutpb=0.1,
         cxmode='blend', cxpb=0.8, sorting = 'log',ncores=1,seed=1)
    x_best2, y_best2, es_hist2=nsgaii.evolute(ngen=10, verbose=1)

    nsgaiii=NSGAIII(mode='min', bounds=BOUNDS, fit=dtlz2, lambda_=lambda_, mutpb=0.1,
         cxmode='blend', cxpb=0.8, ncores=1, p = nx ,sorting = 'log',seed=1)
    x_best, y_best, es_hist=nsgaiii.evolute(ngen=10, verbose=1)

test_nsga()