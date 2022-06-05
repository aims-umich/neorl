# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 14:28:17 2022

@author: Majdi Radaideh
"""
from neorl import NHHO
import math, random
import sys

#################################
# Define Vessel Function 
#Mixed discrete/continuous/grid
#################################
def Vessel(individual):
    """
    Pressure vesssel design
    x1: thickness (d1)  --> discrete value multiple of 0.0625 in 
    x2: thickness of the heads (d2) ---> categorical value from a pre-defined grid
    x3: inner radius (r)  ---> cont. value between [10, 200]
    x4: length (L)  ---> cont. value between [10, 200]
    """
    
    x=individual.copy()
    x[0] *= 0.0625   #convert d1 to "in" 

    y = 0.6224*x[0]*x[2]*x[3]+1.7781*x[1]*x[2]**2+3.1661*x[0]**2*x[3]+19.84*x[0]**2*x[2];

    g1 = -x[0]+0.0193*x[2];
    g2 = -x[1]+0.00954*x[2];
    g3 = -math.pi*x[2]**2*x[3]-(4/3)*math.pi*x[2]**3 + 1296000;
    g4 = x[3]-240;
    g=[g1,g2,g3,g4]
    
    phi=sum(max(item,0) for item in g)
    eps=1e-5 #tolerance to escape the constraint region
    penality=1e7 #large penality to add if constraints are violated
    
    if phi > eps:  
        fitness=phi+penality
    else:
        fitness=y
    return fitness

def init_sample(bounds):
    #generate an individual from a bounds dictionary
    indv=[]
    for key in bounds:
        if bounds[key][0] == 'int':
            indv.append(random.randint(bounds[key][1], bounds[key][2]))
        elif bounds[key][0] == 'float':
            indv.append(random.uniform(bounds[key][1], bounds[key][2]))
        elif bounds[key][0] == 'grid':
            indv.append(random.sample(bounds[key][1],1)[0])
        else:
            raise Exception ('unknown data type is given, either int, float, or grid are allowed for parameter bounds')   
    return indv
    
try:
    ngen=int(sys.argv[1])  #get ngen as external argument for testing
except:
    ngen=50      #or use default ngen

for item in ['mixed', 'grid', 'float/int', 'float/grid', 'int/grid', 'float', 'int']:
    bounds = {}
    btype=item  #float, int, grid, float/int, float/grid, int/grid, mixed. 
    
    print(item, 'is running -----')
    if btype=='mixed':
        bounds['x1'] = ['int', 1, 99]
        bounds['x2'] = ['grid', (0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625)]
        bounds['x3'] = ['float', 10, 200]
        bounds['x4'] = ['float', 10, 200]
        bounds['x5'] = ['grid', ('Hi', 'Bye', 'New')]
        bounds['x6'] = ['int', -5, 5]
    
    elif btype=='int/grid':      
        bounds['x1'] = ['int', 1, 20]
        bounds['x2'] = ['grid', (0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625)]
        bounds['x3'] = ['int', 10, 200]
        bounds['x4'] = ['int', 10, 200]
        bounds['x5'] = ['grid', ('Hi', 'Bye', 'New')]
        bounds['x6'] = ['int', -5, 5]
    
    elif btype=='float/grid':      
        bounds['x1'] = ['float', 1, 20]
        bounds['x2'] = ['grid', (0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625)]
        bounds['x3'] = ['float', 10, 200]
        bounds['x4'] = ['float', 10, 200]
        bounds['x5'] = ['grid', ('Hi', 'Bye', 'New')]
        bounds['x6'] = ['float', -5, 5]
        
    elif btype=='float/int':      
        bounds['x1'] = ['int', 1, 20]
        bounds['x2'] = ['float', 1, 20]
        bounds['x3'] = ['int', 10, 200]
        bounds['x4'] = ['float', 10, 200]
        bounds['x5'] = ['float', -5, 5]
        bounds['x6'] = ['int', -5, 5]
    
    elif btype=='float':      
        bounds['x1'] = ['float', 1, 20]
        bounds['x2'] = ['float', 1, 20]
        bounds['x3'] = ['float', 10, 200]
        bounds['x4'] = ['float', 10, 200]
        bounds['x5'] = ['float', -5, 5]
        bounds['x6'] = ['float', -5, 5]
        
    elif btype=='int':      
        bounds['x1'] = ['int', 1, 20]
        bounds['x2'] = ['int', 1, 20]
        bounds['x3'] = ['int', 10, 200]
        bounds['x4'] = ['int', 10, 200]
        bounds['x5'] = ['int', -5, 5]
        bounds['x6'] = ['int', -5, 5]
        
    elif btype=='grid':      
        bounds['x1'] = ['grid', (0.0625, 0.125, 0.375, 0.4375, 0.5625, 0.625)]
        bounds['x2'] = ['grid', (0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625)]
        bounds['x3'] = ['grid', (1,2,3,4,5)]
        bounds['x4'] = ['grid', (32,64,128)]
        bounds['x5'] = ['grid', ('Hi', 'Bye', 'New')]
        bounds['x6'] = ['grid', ('Cat', 'Dog', 'Bird', 'Fish')]


    npop=20
    x0=[]
    for i in range(npop):
        x0.append(init_sample(bounds))
        
    ########################
    # Setup and evolute NHHO
    ######################## 

    nn_params = {}
    nn_params['num_nodes'] = [10, 5, 3]
    nn_params['learning_rate'] = 8e-4
    nn_params['epochs'] = 1
    nn_params['plot'] = False #will accelerate training
    nn_params['verbose'] = False #will accelerate training
    nn_params['save_models'] = False  #will accelerate training
    
    nhho = NHHO(mode='min', bounds=bounds, fit=Vessel, nhawks=npop, 
                nn_params=nn_params, ncores=3, seed=1)
    individuals, fitnesses = nhho.evolute(ngen=ngen, x0=x0, verbose=True)
    
    #make evaluation of the best individuals using the real fitness function
    real_fit=[Vessel(item) for item in individuals]
    
    #print the best individuals/fitness found
    min_index=real_fit.index(min(real_fit))
    print('------------------------ Final Summary --------------------------')
    print('Best real individual:', individuals[min_index])
    print('Best real fitness:', real_fit[min_index])
    print('-----------------------------------------------------------------')