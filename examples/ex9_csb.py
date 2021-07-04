# -*- coding: utf-8 -*-

import numpy as np
from neorl import GWO, HHO, MFO, SSA, JAYA, SA, DE, PSO, XNES, WOA, ES, PESA, PESA2

def CSB(individual):
    #--mir
    check=all([item >= bounds['x'+str(i+1)][1] for i,item in enumerate(individual)]) \
          and all([item <= bounds['x'+str(i+1)][2] for i,item in enumerate(individual)])
    if not check:
        raise Exception ('--error check fails')
        
    """Cantilever Stepped Beam
    individual[i = 0 - 4] are beam widths
    individual[i = 5 - 9] are beam heights
    """
        
    P = 50000
    E = 2 * 10**7
    l = 100
    g = np.zeros(11)
    g[0] = 600*P/(individual[4] * individual[9]**2) - 14000
    g[1] = 6*P*(2*l)/(individual[3] * individual[8]**2) - 14000
    g[2] = 6*P*(3*l)/(individual[2] * individual[7]**2) - 14000
    g[3] = 6*P*(4*l)/(individual[1] * individual[6]**2) - 14000
    g[4] = 6*P*(5*l)/(individual[0] * individual[5]**2) - 14000
    g[5] = 0 #P*l**3/(3*E) * (1/individual[4] + 7/individual[3] + 19/individual[2] + 37/individual[1] + 61/individual[0]) - 2.7
    g[6] = individual[9]/individual[4] - 20
    g[7] = individual[8]/individual[3] - 20
    g[8] = individual[7]/individual[2] - 20
    g[9] = individual[6]/individual[1] - 20
    g[10] = individual[5]/individual[0] - 20

    g_round=np.round(g,6)
    w1=1000
    w2=1000
        
    phi=sum(max(item,0) for item in g_round)
    viol=sum(float(num) > 0 for num in g_round)

    V = 0
    for i in range(5):
        V += individual[i] * individual[i+5] * 100
    #if phi > 1e-3:
    #    V += 1e9
    return V + w1*phi + w2*viol

nx=10
bounds={}
for i in range(1, 6):
    bounds['x'+str(i)]=['float', 1, 5]
for i in range(6, 11):
    bounds['x'+str(i)]=['float', 30, 65]

hho = HHO(mode='min', bounds=bounds, fit=CSB, nhawks=20, ncores=1, seed=1)
x_best, y_best, hist = hho.evolute(ngen=200, verbose=0)
assert CSB(x_best) == y_best

gwo = GWO(mode='min', bounds=bounds, fit=CSB, nwolves=60, ncores=1, seed=1)
x_best, y_best, hist=gwo.evolute(ngen=200, verbose=0)
assert CSB(x_best) == y_best

ssa = SSA(mode='min', bounds=bounds, fit=CSB, nsalps=60, ncores=1, seed=1)
x_best, y_best, hist=ssa.evolute(ngen=200, verbose=0)
assert CSB(x_best) == y_best

mfo = MFO(mode='min', bounds=bounds, fit=CSB, nmoths=60, ncores=1, seed=1)
x_best, y_best, hist=mfo.evolute(ngen=200, verbose=0)  
assert CSB(x_best) == y_best

jaya=JAYA(mode='min', bounds=bounds, fit=CSB, npop=60, ncores=1, seed=1)
x_best, y_best, hist=jaya.evolute(ngen=200, verbose=0)
assert CSB(x_best) == y_best

sa=SA(mode='min', bounds=bounds, fit=CSB, chain_size=10,chi=0.2, ncores=1, seed=1)
x_best, y_best, sa_hist=sa.evolute(ngen=100, verbose=1)
assert CSB(x_best) == y_best

de=DE(mode='min', bounds=bounds, fit=CSB, npop=60, F=0.5, CR=0.7, ncores=1, seed=1)
x_best, y_best, de_hist=de.evolute(ngen=100, verbose=1)
assert CSB(x_best) == y_best

pso=PSO(mode='min', bounds=bounds, fit=CSB, c1=2.05, c2=2.1, npar=50,
               speed_mech='constric', ncores=1, seed=1)
x_best, y_best, pso_hist=pso.evolute(ngen=100, verbose=1)
assert CSB(x_best) == y_best

xnes=XNES(mode='min', bounds=bounds, fit=CSB, npop=50, eta_mu=0.9,
          eta_sigma=0.25, adapt_sampling=True)
x_best, y_best, xnes_hist=xnes.evolute(ngen=300, verbose=1)
assert CSB(x_best) == y_best

es=ES(mode='min', bounds=bounds, fit=CSB, lambda_=80, mu=40, ncores=1, seed=1)
x_best, y_best, es_hist=es.evolute(ngen=100, verbose=1)
assert CSB(x_best) == y_best

woa=WOA(mode='min', bounds=bounds, fit=CSB, nwhales=100, a0=2.0, b=1, ncores=1, seed=1)
x_best, y_best, woa_hist=woa.evolute(ngen=200, verbose=1)
assert CSB(x_best) == y_best

pesa=PESA(mode='min', bounds=bounds, fit=CSB, npop=60, mu=30, alpha_init=0.1,
          alpha_end=1.0, cxpb=0.7, mutpb=0.2, alpha_backdoor=0.15, seed=1)
x_best, y_best, pesa_hist=pesa.evolute(ngen=100, verbose=False)
assert CSB(x_best) == y_best

pesa2=PESA2(mode='min', bounds=bounds, fit=CSB, npop=60, nwolves=5)
x_best, y_best, pesa2_hist=pesa2.evolute(ngen=50, replay_every=2, verbose=1)
assert CSB(x_best) == y_best
