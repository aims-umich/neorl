import numpy as np
import neorl.benchmarks.cec17 as functions    #import all cec17 functions
import neorl.benchmarks.classic as classics   #import all classical functions
from neorl.benchmarks.classic import ackley, levy, bohachevsky  #import specific functions
from neorl.benchmarks.cec17 import f3, f10, f21  #import cec17 specific functions 
from neorl.benchmarks import bench_2dplot   #import the built-in plotter

def test_benchmarks():
    d1 = 2 #set dimension for classical functions
    d2 = 10 #set dimension for cec functions (choose between 2, 10, 20, 30, 50 or 100)
    print('------------------------------------------------------')
    print('Classical Functions')
    print('------------------------------------------------------')
    
    for f in classics.all_functions:
        sample = np.random.uniform(low=0, high=10, size=d1)
        y = f(sample)
        print('Function: {}, x={}, y={}'.format(f.__name__, np.round(sample,2), np.round(y,2)))
    
    print('------------------------------------------------------')
    print('CEC2017 Functions')
    print('------------------------------------------------------')
    for f in functions.all_functions:
        sample = np.random.uniform(low=-10, high=10, size=d2)
        y = f(sample)
        print('Function: {}, x={}, y={}'.format(f.__name__, np.round(sample,2), np.round(y,2)))

test_benchmarks()