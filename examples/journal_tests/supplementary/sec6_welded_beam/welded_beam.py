#---------------------------------
# Import packages
#---------------------------------
import numpy as np
np.random.seed(50)
import matplotlib.pyplot as plt
from math import sqrt
from neorl.tune import BAYESTUNE
from neorl import ES, JAYA, WOA, GWO, HHO

#**********************************************************
# Part I: Original Problem
#**********************************************************
#Define the fitness function (for the welded beam)
def BEAM(x):

    y = 1.10471*x[0]**2*x[1]+0.04811*x[2]*x[3]*(14.0+x[1])

    # parameters
    P = 6000; L = 14; E = 30e+6; G = 12e+6;
    t_max = 13600; s_max = 30000; d_max = 0.25;

    M = P*(L+x[1]/2)
    R = sqrt(0.25*(x[1]**2+(x[0]+x[2])**2))
    J = 2*(sqrt(2)*x[0]*x[1]*(x[1]**2/12+0.25*(x[0]+x[2])**2));
    P_c = (4.013*E/(6*L**2))*x[2]*x[3]**3*(1-0.25*x[2]*sqrt(E/G)/L);
    t1 = P/(sqrt(2)*x[0]*x[1]); t2 = M*R/J;
    t = sqrt(t1**2+t1*t2*x[1]/R+t2**2);
    s = 6*P*L/(x[3]*x[2]**2)
    d = 4*P*L**3/(E*x[3]*x[2]**3);
    # Constraints
    g1 = t-t_max; #done
    g2 = s-s_max; #done
    g3 = x[0]-x[3];
    g4 = 0.10471*x[0]**2+0.04811*x[2]*x[3]*(14.0+x[1])-5.0;
    g5 = 0.125-x[0];
    g6 = d-d_max;
    g7 = P-P_c; #done

    g=[g1,g2,g3,g4,g5,g6,g7]
    g_round=np.round(np.array(g),6)
    w1=100
    w2=100

    phi=sum(max(item,0) for item in g_round)
    viol=sum(float(num) > 0 for num in g_round)

    reward = (y + (w1*phi + w2*viol))

    return reward

#**********************************************************
# Part II: Setup parameter space
#**********************************************************
#--setup the parameter space for the welded beam
lb=[0.1, 0.1, 0.1, 0.1]
ub=[2.0, 10, 10, 2.0]
d2type=['float', 'float', 'float', 'float']
BOUNDS={}
nx=4
for i in range(nx):
    BOUNDS['x'+str(i+1)]=[d2type[i], lb[i], ub[i]]

#*************************************************************
# Part III: Define fitness function for hyperparameter tuning
#*************************************************************
def tune_fit(cxpb, mu, alpha, cxmode, mutpb):

    #--setup the ES algorithm
    es=ES(mode='min', bounds=BOUNDS, fit=BEAM, lambda_=50, mu=mu, mutpb=mutpb, alpha=alpha,
         cxmode=cxmode, cxpb=cxpb, ncores=1, seed=1)

    #--Evolute the ES object and obtains y_best
    #--turn off verbose for less algorithm print-out when tuning
    x_best, y_best, es_hist=es.evolute(ngen=100, verbose=0)

    return y_best #returns the best score

#*************************************************************
# Part IV: Tuning
#*************************************************************
#Setup the parameter space for Bayesian optimisation
#VERY IMPORTANT: The order of these parameters MUST be similar to their order in tune_fit
#see tune_fit
param_grid={
#def tune_fit(cxpb, mu, alpha, cxmode):
'cxpb': ['float', 0.1, 0.7],             #cxpb is first (low=0.1, high=0.8, type=float/continuous)
'mu':   ['int', 20, 30],                 #mu is second (low=30, high=60, type=int/discrete)
'alpha':['grid', [0.1, 0.2, 0.3, 0.4]],    #alpha is third (grid with fixed values, type=grid/categorical)
'cxmode':['grid', ['blend', 'cx2point']],
'mutpb': ['float', 0.05, 0.3]}  #cxmode is fourth (grid with fixed values, type=grid/categorical)

#setup a bayesian tune object
btune=BAYESTUNE(param_grid=param_grid, fit=tune_fit, ncases=30)
#tune the parameters with method .tune
bayesres=btune.tune(ncores=1, csvname='bayestune.csv', verbose=True)

print('----Top 10 hyperparameter sets----')
bayesres = bayesres[bayesres['score'] >= 1] #drop the cases with scores < 1 (violates the constraints)
bayesres = bayesres.sort_values(['score'], axis='index', ascending=True) #rank the scores from best (lowest) to worst (high)
print(bayesres.iloc[0:10,:])   #the results are saved in dataframe and ranked from best to worst

#*************************************************************
# Part V: Rerun ES with the best hyperparameter set
#*************************************************************
es=ES(mode='min', bounds=BOUNDS, fit=BEAM, lambda_=50, mu=bayesres['mu'].iloc[0],
      mutpb=bayesres['mutpb'].iloc[0], alpha=bayesres['alpha'].iloc[0],
      cxmode=bayesres['cxmode'].iloc[0], cxpb=bayesres['cxpb'].iloc[0],
      ncores=1, seed=1)

es_x, es_y, es_hist=es.evolute(ngen=200, verbose=0)

#*************************************************************
# Part VI: Run Other Algorithms without Tuning
#*************************************************************

#---------------------------------
# GWO
#---------------------------------
gwo=GWO(mode='min', bounds=BOUNDS, fit=BEAM, nwolves=50, ncores=1, seed=1)
gwo_x, gwo_y, gwo_hist=gwo.evolute(ngen=200, verbose=0)

#---------------------------------
# HHO
#---------------------------------
hho=HHO(mode='min', bounds=BOUNDS, fit=BEAM, nhawks=50, ncores=1, seed=1)
hho_x, hho_y, hho_hist=hho.evolute(ngen=200, verbose=0)

#---------------------------------
# JAYA
#---------------------------------
jaya=JAYA(mode='min', bounds=BOUNDS, fit=BEAM, npop=50, ncores=1, seed=1)
jaya_x, jaya_y, jaya_hist=jaya.evolute(ngen=200, verbose=0)

#---------------------------------
# WOA
#---------------------------------
woa=WOA(mode='min', bounds=BOUNDS, fit=BEAM, nwhales=50, a0=1.5, b=1, ncores=1, seed=1)
woa_x, woa_y, woa_hist=woa.evolute(ngen=200, verbose=0)

print('---- ES ----')
print('Best fitness (y) found:', es_y)
print('Best individual (x) found:', es_x)

print('---- GWO ----')
print('Best fitness (y) found:', gwo_y)
print('Best individual (x) found:', gwo_x)

print('---- HHO ----')
print('Best fitness (y) found:', hho_y)
print('Best individual (x) found:', hho_x)

print('---- JAYA ----')
print('Best fitness (y) found:', jaya_y)
print('Best individual (x) found:', jaya_x)

print('---- WOA ----')
print('Best fitness (y) found:', woa_y)
print('Best individual (x) found:', woa_x)


#---------------------------------
# Plot
#---------------------------------
#Plot fitness convergence
plt.figure()
plt.plot(np.array(es_hist), label='ES')
plt.plot(gwo_hist['fitness'], '--', label='GWO')
plt.plot(woa_hist['global_fitness'], '-.', label='WOA')
plt.plot(jaya_hist, ':', label='JAYA')
plt.plot(hho_hist['global_fitness'], '-o', markersize=2, label='HHO')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.savefig('beam_conv.png',format='png', dpi=300, bbox_inches="tight")
plt.show()