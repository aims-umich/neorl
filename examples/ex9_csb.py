#************************************************************
#               Cantilever Stepped Beam
#************************************************************

#---------------------------------
# Import packages
#---------------------------------
import numpy as np
import matplotlib.pyplot as plt
from neorl import PSO, DE, SSA, GWO, MFO, BAT, PESA2

#---------------------------------
# Fitness function
#---------------------------------
def CSB(individual):
  """Cantilever Stepped Beam
  individual[i = 0 - 4] are beam widths
  individual[i = 5 - 9] are beam heights
  """

  check=all([item >= BOUNDS['x'+str(i+1)][1] for i,item in enumerate(individual)]) \
		and all([item <= BOUNDS['x'+str(i+1)][2] for i,item in enumerate(individual)])
  if not check:
	  raise Exception ('--error check fails')

  P = 50000
  E = 2 * 10**7
  l = 100
  g = np.zeros(11)
  g[0] = 600*P/(individual[4] * individual[9]**2) - 14000
  g[1] = 6*P*(2*l)/(individual[3] * individual[8]**2) - 14000
  g[2] = 6*P*(3*l)/(individual[2] * individual[7]**2) - 14000
  g[3] = 6*P*(4*l)/(individual[1] * individual[6]**2) - 14000
  g[4] = 6*P*(5*l)/(individual[0] * individual[5]**2) - 14000
  g[5] = 0
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
	  V += individual[i] * individual[i+5] * l
	  
  return V + w1*phi + w2*viol

#---------------------------------
# Parameter space
#---------------------------------
nx=10
BOUNDS={}
for i in range(1, 6):
  BOUNDS['x'+str(i)]=['float', 1, 5]
for i in range(6, 11):
  BOUNDS['x'+str(i)]=['float', 30, 65]

#---------------------------------
# PSO
#---------------------------------
pso=PSO(mode='min', bounds=BOUNDS, fit=CSB, c1=2.05, c2=2.1, npar=50, speed_mech='constric', ncores=1, seed=1)
pso_x, pso_y, pso_hist=pso.evolute(ngen=300, verbose=0)

#---------------------------------
# DE
#---------------------------------
de=DE(mode='min', bounds=BOUNDS, fit=CSB, npop=50, F=0.5, CR=0.7, ncores=1, seed=1)
de_x, de_y, de_hist=de.evolute(ngen=300, verbose=0)

#---------------------------------
# SSA
#---------------------------------
ssa=SSA(mode='min', bounds=BOUNDS, fit=CSB, nsalps=50, c1=None, ncores=1, seed=1)
ssa_x, ssa_y, ssa_hist=ssa.evolute(ngen=300, verbose=0)

#---------------------------------
# BAT
#---------------------------------
bat=BAT(mode='min', bounds=BOUNDS, fit=CSB, nbats=50, fmin = 0 , fmax = 1, A=0.5, r0=0.5, levy = True, seed = 1, ncores=1)
bat_x, bat_y, bat_hist=bat.evolute(ngen=300, verbose=0)

#---------------------------------
# GWO
#---------------------------------
gwo=GWO(mode='min', fit=CSB, bounds=BOUNDS, nwolves=50, ncores=1, seed=1)
gwo_x, gwo_y, gwo_hist=gwo.evolute(ngen=300, verbose=0)

#---------------------------------
# MFO
#---------------------------------
mfo=MFO(mode='min', bounds=BOUNDS, fit=CSB, nmoths=50, b = 0.2, ncores=1, seed=1)
mfo_x, mfo_y, mfo_hist=mfo.evolute(ngen=300, verbose=0)

#---------------------------------
# PESA2
#---------------------------------
pesa2=PESA2(mode='min', bounds=BOUNDS, fit=CSB, npop=50, nwolves=5, ncores=1, seed=1)
pesa2_x, pesa2_y, pesa2_hist=pesa2.evolute(ngen=600, replay_every=2, verbose=0)

#---------------------------------
# Plot
#---------------------------------
plt.figure()
plt.plot(pso_hist['global_fitness'], label = 'PSO')
plt.plot(de_hist['global_fitness'], label = 'DE')
plt.plot(ssa_hist['global_fitness'], label = 'SSA')
plt.plot(bat_hist['global_fitness'], label = 'BAT')
plt.plot(gwo_hist['fitness'], label = 'GWO')
plt.plot(mfo_hist['global_fitness'], label = 'MFO')
plt.plot(pesa2_hist, label = 'PESA2')
plt.legend()
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.ylim(0,150000)
plt.savefig('CSB_fitness.png',format='png', dpi=300, bbox_inches="tight")
plt.close()

#************************************************************
#              Square Cantilever Stepped Beam
#************************************************************

#---------------------------------
# Import packages
#---------------------------------
import numpy as np
from math import cos, pi, exp, e, sqrt
import matplotlib.pyplot as plt
from neorl import PSO, DE, SSA, GWO, MFO, BAT, PESA2

#---------------------------------
# Fitness function
#---------------------------------
def CSB_square(individual):
  """Square Cantilever Stepped Beam
  individual[i = 0 - 4] are beam heights and widths
  """

  check=all([item >= BOUNDS['x'+str(i+1)][1] for i,item in enumerate(individual)]) \
		and all([item <= BOUNDS['x'+str(i+1)][2] for i,item in enumerate(individual)])
  if not check:
	  raise Exception ('--error check fails')

  g = 61/individual[0]**3 + 37/individual[1]**3 + 19/individual[2]**3 + 7/individual[3]**3 + 1/individual[4]**3 - 1

  g_round=np.round(g,6)
  w1=1000
  #phi=max(g_round,0)
  if g_round > 0:
	  phi = 1
  else:
	  phi = 0

  V = 0.0624*(np.sum(individual)) 

  return V + w1*phi

#---------------------------------
# Parameter space
#---------------------------------
nx=5
BOUNDS={}
for i in range(1, 6):
  BOUNDS['x'+str(i)]=['float', 0.01, 100]
  
#---------------------------------
# PSO
#---------------------------------
pso=PSO(mode='min', bounds=BOUNDS, fit=CSB_square, c1=2.05, c2=2.1, npar=50, speed_mech='constric', ncores=1, seed=1)
pso_x, pso_y, pso_hist=pso.evolute(ngen=200, verbose=0)

#---------------------------------
# DE
#---------------------------------
de=DE(mode='min', bounds=BOUNDS, fit=CSB_square, npop=50, F=0.5, CR=0.7, ncores=1, seed=1)
de_x, de_y, de_hist=de.evolute(ngen=200, verbose=0)

#---------------------------------
# SSA
#---------------------------------
ssa=SSA(mode='min', bounds=BOUNDS, fit=CSB_square, nsalps=50, c1=0.05, ncores=1, seed=1)
ssa_x, ssa_y, ssa_hist=ssa.evolute(ngen=200, verbose=0)

#---------------------------------
# BAT
#---------------------------------
bat=BAT(mode='min', bounds=BOUNDS, fit=CSB_square, nbats=50, fmin = 0 , fmax = 1, A=0.5, r0=0.5, levy = True, seed = 1, ncores=1)
bat_x, bat_y, bat_hist=bat.evolute(ngen=200, verbose=0)

#---------------------------------
# GWO
#---------------------------------
gwo=GWO(mode='min', bounds=BOUNDS, fit=CSB_square, nwolves=50, ncores=1, seed=1)
gwo_x, gwo_y, gwo_hist=gwo.evolute(ngen=200, verbose=0)

#---------------------------------
# MFO
#---------------------------------
mfo=MFO(mode='min', bounds=BOUNDS, fit=CSB_square, nmoths=50, b = 0.2, ncores=1, seed=1)
mfo_x, mfo_y, mfo_hist=mfo.evolute(ngen=200, verbose=0)

#---------------------------------
# PESA2
#---------------------------------
pesa2=PESA2(mode='min', bounds=BOUNDS, fit=CSB_square, npop=50, nwolves=5, ncores=1, seed=1)
pesa2_x, pesa2_y, pesa2_hist=pesa2.evolute(ngen=400, replay_every=2, verbose=0)

#---------------------------------
# Plot
#---------------------------------
plt.figure()
plt.plot(pso_hist['global_fitness'], label = 'PSO')
plt.plot(de_hist['global_fitness'], label = 'DE')
plt.plot(ssa_hist['global_fitness'], label = 'SSA')
plt.plot(bat_hist['global_fitness'], label = 'BAT')
plt.plot(gwo_hist['fitness'], label = 'GWO')
plt.plot(mfo_hist['global_fitness'], label = 'MFO')
plt.plot(pesa2_hist, label = 'PESA2')
plt.legend()
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.savefig('CSB_square_fitness.png',format='png', dpi=300, bbox_inches="tight")
plt.close()