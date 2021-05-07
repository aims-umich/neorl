import numpy as np
import neorl.benchmarks.cec17 as functions    #import all cec17 functions
from neorl import DE

def test_benchmarks2():
    
    reduced_func=functions.all_functions[:10] #keep only the first 10 functions
    nx = 2 #set dimension
    BOUNDS={}
    for i in range(1,nx+1):
        BOUNDS['x'+str(i)]=['float', -100, 100]
    
    for FIT in reduced_func:
        #setup and evolute PSO
        de=DE(mode='min', bounds=BOUNDS, fit=FIT, npop=60, F=0.5, 
              CR=0.7, ncores=1, seed=1)
        x_best, y_best, de_hist=de.evolute(ngen=100, verbose=0)
        opt=float(FIT.__name__.strip('f'))*100
        print('Function: {}, x-DE={}, y-DE={}, y-Optimal={}'.format(FIT.__name__, 
                                                                 np.round(x_best,2), 
                                                                 np.round(y_best,2), 
                                                                 opt))
        
test_benchmarks2()