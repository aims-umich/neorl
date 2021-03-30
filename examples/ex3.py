


#---------------------------------
# Import packages
#---------------------------------
import numpy as np
import matplotlib.pyplot as plt
from neorl import PESA
from neorl.hybrid.pesacore.es import ESMod
from math import exp, sqrt, cos, pi
import csv, copy
from collections import defaultdict
import pandas as pd
np.random.seed(50)

#---------------------------------
# Fitness
#---------------------------------
def BEAM(x, return_g=False):

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
    #print(viol)
    #phi=sum(max(item,0) for item in g)
#    if phi > 1e-6:
#        reward=-phi-10
#    else:
#        reward=-y
    
    reward = -(y + (w1*phi + w2*viol))

    return reward

#---------------------------------
# Parameter Space
#---------------------------------
lb=[0.1, 0.1, 0.1, 0.1]
ub=[2.0, 10, 10, 2.0]
d2type=['float', 'float', 'float', 'float']
BOUNDS={}
nx=4
for i in range(nx):
    BOUNDS['x'+str(i+1)]=[d2type[i], lb[i], ub[i]]

#---------------------------------
# Initial Population
#---------------------------------
def start(PROBLEM, BOUNDS, FIT, WARMUP, extdata=None, savedata=False):
    """
    This function intializes population for all methods based on warmup samples
    Returns:
        pop0 (dict): initial population for PESA and ES
        swarm0 (dict): initial swarm for PSO 
        swm_pos (list), swm_fit (float): initial guess for swarm best position and fitness for PSO
        local_pos (list of lists), local_fit (list): initial guesses for local best position of each particle and their fitness for PSO
        x0 (list of lists), E0 (list): initial input vectors and their initial fitness for SA
    """
    NCORES=1
    MU=30
    LAMBDA=60
    NPOP=LAMBDA
    
    warm=ESMod(bounds=BOUNDS, fit=FIT, mu=MU, lambda_=LAMBDA, ncores=NCORES)
    pop0=warm.init_pop(warmup=WARMUP)  #initial population for ES
    #Save data to logger
    if savedata:
        inp_names=[] 
        [inp_names.append('x'+str(i)) for i in range(1,nx+1)]
        [inp_names.append('s'+str(i)) for i in range(1,nx+1)]
        inp_names.append('fit')
        with open (PROBLEM+'0.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')
            csvwriter.writerow(inp_names)
        pop0_csv=copy.deepcopy(pop0) 
        for item in pop0:
            data=pop0_csv[item][0]
            pop0_csv[item][1].append(pop0_csv[item][2])
            data.extend(pop0_csv[item][1])
            assert len(data) ==2*nx+1, 'length of warmup data is not equal to 2*nx+1'
            with open (PROBLEM+'0.csv', 'a') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')
                csvwriter.writerow(data)
    
    sorted_dict=dict(sorted(pop0.items(), key=lambda e: e[1][2], reverse=True)[:NCORES]) # sort the initial samples
    sorted_pso=dict(sorted(pop0.items(), key=lambda e: e[1][2], reverse=True)[:NPOP]) # sort the initial samples
    x0, E0=[sorted_dict[key][0] for key in sorted_dict], [sorted_dict[key][2] for key in sorted_dict] # initial guess for SA
    swarm0=defaultdict(list)
    index=0
    local_pos=[]
    local_fit=[]
    for key in sorted_pso:
        swarm0[index].append(sorted_pso[key][0])
        swarm0[index].append(list(0.1*np.array(sorted_pso[key][0])))
        swarm0[index].append(sorted_pso[key][2])
        local_pos.append(sorted_pso[key][0])
        local_fit.append(sorted_pso[key][2])
        index+=1
    
    swm_pos=swarm0[0][0]
    swm_fit=swarm0[0][2]
    
    return pop0, swarm0, swm_pos, swm_fit, local_pos, local_fit, x0, E0

pop0, swarm0, swm_pos, swm_fit, local_pos, local_fit, x0, E0=start(PROBLEM='hi', BOUNDS=BOUNDS, FIT=BEAM, 
                                                                   WARMUP=WARMUP, extdata=False, savedata=False)
        
#---------------------------------
# PESA
#---------------------------------

mu=[35]
cxpb=[0.7]
mutpb=[0.1]
phi=[2.05]
chi=[0.4]
npop=[50]
ngen=[250]
a0=[0.01]
       
#-----------------------
#PESA global parameters
#-----------------------
NGEN=250         #number of generations
NPOP=50          #LAMBDA for ES, total length of each chains for SA, Swarm Size for PSO 
STEPS=NGEN*NPOP  #Total number of steps to run for each method (Derived)
WARMUP=500       #warmup the replay memory with some samples
REPLAY_RATE=0.1  #FOR SA, replace random-walk with the `best` sample from memory
MU=35           # number of individuals to survive next generation in PSO/ES 
                 # and also equal to number of pop to join each generation from the memory
MEMORY_SIZE=STEPS*3+1000 #Max memory size (Derived)
#MEMORY_SIZE=6000
#--------------------
#Experience Replay
#--------------------
MODE='prior' #`uniform`, `greedy`, or `prior`
ALPHA0=0.01  #only needed for mode=prior
ALPHA1=1.0 #only needed for mode=prior
#--------------------
# ES HyperParameters
#--------------------
CXPB=0.7  #population crossover (0.4-0.8)
MUTPB=0.1   #population mutation (0.05-0.0.25)
INDPB=1.0 #ES attribute mutation (only used for cont. optimisation)
LAMBDA=NPOP #full population size before selection of MU (Fixed)
SMIN = 1/nx #ES strategy min (Fixed)
SMAX = 0.5  #ES strategy max (Fixed)
#--------------------
# PSO HyperParameters
#--------------------
C1=2.05                #cognitive speed coeff (2.05 is typical value)
C2=2.05                  #social speed coeff (2.05 is typical value)
SPEED_MECH='globw'   #`constric`, `timew`, or `globw` --> how to modify particle speed
NPAR=NPOP               #Swarm size which is equal to ES LAMBDA, 
                        #Both LAMBDA and NPAR equal to NPOP, symmetry between methods
#--------------------
# SA hyperparameters
#--------------------
TMAX=10000     #max annealing temperature for SA
CHI=0.4        #probablity to perturb each input attribute
COOLING='fast' #Cooling schedule (Fixed)
TMIN=1         #Minimum Temperature (Fixed)

NCORES=1
    
pesa=PESA(bounds=BOUNDS, fit=BEAM, ngen=NGEN, npop=NPOP, pop0=pop0, memory_size=MEMORY_SIZE, mode=MODE, 
          alpha0=ALPHA0, alpha1=ALPHA1, warmup=None, chi=CHI, replay_rate=REPLAY_RATE, Tmax=TMAX, 
          mu=MU, cxpb=CXPB, mutpb=MUTPB, c1=C1, c2=C2, speed_mech=SPEED_MECH, pso_flag=True, verbose=0)

xpesa_best, ypesa_best= pesa.evolute()


#---------------------------------
# Plot
#---------------------------------
#Plot fitness for both methods
plt.figure()
plt.plot(-np.array(pso_hist), label='PESA')            #multiply by -1 to covert back to a min problem
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.savefig('ex3_fitness.png',format='png', dpi=300, bbox_inches="tight")
plt.show()