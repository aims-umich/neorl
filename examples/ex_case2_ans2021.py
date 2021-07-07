from neorl import PESA
from neorl.tune import RANDTUNE
import math
import matplotlib.pyplot as plt
import random
random.seed(1)

#***************************************************************
# Part I: Define original fitness function (pressure vessel)
#***************************************************************
def Vessel(individual):
    """
    Pressure vesssel design
    x1: thickness (d1)  --> discrete value multiple of 0.0625 in 
    x2: thickness of the heads (d1) ---> discrete value multiple of 0.0625 in
    x3: inner radius (r)  ---> cont. value between [10, 200]
    x4: length (L)  ---> cont. value between [10, 200]
    """
    x=individual.copy()
    x[0] *= 0.0625   #convert d1 to "in" 
    x[1] *= 0.0625   #convert d2 to "in" 

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

#*************************************************************
# Part II: Define fitness function for hyperparameter tuning
#*************************************************************
def tune_fit(npop, frac, alpha_init, cxpb, mutpb):

    #--setup the PESA algorithm
    pesa=PESA(mode='min', bounds=bounds, fit=Vessel, npop=npop, mu=int(frac*npop), 
              alpha_init=alpha_init, alpha_end=1.0, alpha_backdoor=0.1, 
              cxpb=cxpb, mutpb=mutpb, c1=2.05, c2=2.05, seed=1)
    
    x_pesa, y_pesa, pesa_hist=pesa.evolute(ngen=300, x0=x0, verbose=False)

    return y_pesa #returns the best score

#*************************************************************
# Part III: Parameter Space
#*************************************************************
#--setup the parameter space
bounds = {}
bounds['x1'] = ['int', 1, 99]
bounds['x2'] = ['int', 1, 99]
bounds['x3'] = ['float', 10, 200]
bounds['x4'] = ['float', 10, 200]

#--Define initial guess
x0=[[random.randint(bounds['x1'][1], bounds['x1'][2]),
    random.randint(bounds['x2'][1], bounds['x2'][2]),
    random.uniform(bounds['x3'][1], bounds['x3'][2]),
    random.uniform(bounds['x4'][1], bounds['x4'][2])] for i in range(100)]

#*************************************************************
# Part IV: Tuning
#*************************************************************
#VERY IMPORTANT: The order of these parameters MUST be similar to their order in "tune_fit"
#The order is: npop, frac, alpha_init, cxpb, mutpb
param_grid={
'npop': [[50, 80],'int'],       
'frac': [(0.2, 0.3, 0.4, 0.5, 0.6),'grid'],       
'alpha_init':[[0.01, 0.2],'float'],                
'cxpb':[[0.5, 0.8],'float'],    
'mutpb':[[0.05, 0.2],'float']}

#setup a random tune object
rtune=RANDTUNE(param_grid=param_grid, fit=tune_fit, ncases=100, seed=1)
#tune the parameters with method .tune
randres=rtune.tune(ncores=30, csvname='tune.csv')
#sort the results from best to worst
sorted_res = randres.sort_values(['score'], axis='index', ascending=True)
#print the top 10 hyperparameter sets
print(sorted_res.head(10))

#*************************************************************
# Part V: Tuning
#*************************************************************
# Re-run PESA based upon the best hyperparameters
pesa=PESA(mode='min', bounds=bounds, fit=Vessel, 
          npop=sorted_res['npop'].iloc[0], 
          mu=int(sorted_res['frac'].iloc[0]*sorted_res['npop'].iloc[0]), 
          alpha_init=sorted_res['alpha_init'].iloc[0],
          alpha_end=1.0, 
          cxpb=sorted_res['cxpb'].iloc[0], 
          mutpb=sorted_res['mutpb'].iloc[0],
          c1=2.05, c2=2.05,
          alpha_backdoor=0.1, 
          seed=1)
x_pesa, y_pesa, pesa_hist=pesa.evolute(ngen=300, x0=x0, verbose=False)

#*************************************************************
# Part VI: Post-processing
#*************************************************************
#plot results
plt.figure()
plt.plot(pesa_hist, label='PESA')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.savefig('ex2_ans21_fitness.png',format='png', dpi=300, bbox_inches="tight")

#print best results
print('---Best PESA Results---')
print(x_pesa)
print(y_pesa)