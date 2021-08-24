from neorl import ES, DE, GWO, PESA2, SSA, JAYA, HHO
from neorl.tune import RANDTUNE, GRIDTUNE
import math
import matplotlib.pyplot as plt

#################
# Set up bounds #
#################
bounds = {}
bounds['x1'] = ['float', 2.6, 3.6]
bounds['x2'] = ['float', 0.7, 0.8]
bounds['x3'] = ['float', 17, 28]
bounds['x4'] = ['float', 7.3, 8.3]
bounds['x5'] = ['float', 7.8, 8.3]
bounds['x6'] = ['float', 2.9, 3.9]
bounds['x7'] = ['float', 5.0, 5.5]

##################################
# Speed reducer fitness function #
##################################
def speedred(x):
    check=all([item >= bounds['x'+str(i+1)][1] for i,item in enumerate(x)]) \
          and all([item <= bounds['x'+str(i+1)][2] for i,item in enumerate(x)])
    if not check:
        raise Exception ('--error check fails')

    x[2] = int(x[2])     #or use round(x[2]) and see what is better

    y=0.7854*x[0]*x[1]**2*(3.3333*x[2]**2 + 14.9334*x[2] - 43.0934) \
    -1.508*x[0]*(x[5]**2+x[6]**2)  \
    +7.4777*(x[5]**3+x[6]**3)  \
    +0.7854*(x[3]*x[5]**2+x[4]*x[6]**2)

    g1=27/(x[0]*x[1]**2*x[2])-1
    g2=397.5/(x[0]*x[1]**2*x[2]**2)-1
    g3=1.93*x[3]**3/(x[1]*x[2]*x[5]**4)-1
    g4=1.93*x[4]**3/(x[1]*x[2]*x[6]**4)-1
    g5=1/(110*x[5]**3)*math.sqrt((745*x[3]/(x[1]*x[2]))**2 + 16.9e6) -1
    g6=1/(85*x[6]**3)*math.sqrt((745*x[4]/(x[1]*x[2]))**2 + 157.5e6) -1
    g7=x[1]*x[2]/40-1
    g8=5*x[1]/x[0]-1
    g9=x[0]/(12*x[1]) -1
    g10=(1.5*x[5]+1.9)/x[3]-1
    g11=(1.1*x[6]+1.9)/x[4]-1

    g=[g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11]

    phi=sum(max(item,0) for item in g)
    if phi > 1e-5:
        reward=phi+5500
    else:
        reward=y

    return reward

######################
# General parameters #
######################
seed=1
ngen=500

######
# ES #
######
print('Running ES')
es = ES(mode='min', bounds=bounds, fit=speedred, lambda_=60, mu=30, cxmode='blend', alpha=0.5, cxpb=0.8, mutpb=0.057, seed=seed)
x_es, y_es, es_hist = es.evolute(ngen=ngen, verbose=False)
assert speedred(x_es) == y_es

######
# DE #
######
print('Running DE')
de = DE(mode='min', bounds=bounds, fit=speedred, npop=50, F=0.5, CR=0.6, seed=seed)
x_de, y_de, de_hist = de.evolute(ngen=ngen, verbose=False)
assert speedred(x_de) == y_de

#######
# GWO #
#######
print('Running GWO')
gwo = GWO(mode='min', bounds=bounds, fit=speedred, nwolves=20, seed=seed)
x_gwo, y_gwo, gwo_hist = gwo.evolute(ngen=ngen, verbose=False)
gwo_hist=gwo_hist['fitness']
assert speedred(x_gwo) == y_gwo

#########
# PESA2 #
#########
print('Running PESA2')
pesa2 = PESA2(mode='min', bounds=bounds, fit=speedred, R_frac=0.5, memory_size=None, alpha_init=0.1, alpha_end=0.9, nwolves=6, npop=55, CR=0.75, F=0.5, nwhales=10, seed=seed)
x_pesa2, y_pesa2, pesa2_hist = pesa2.evolute(ngen=ngen, verbose=False)
assert speedred(x_pesa2) == y_pesa2

########
# JAYA #
########
print('Running JAYA')
jaya = JAYA(mode='min', bounds=bounds, fit=speedred, npop=50, seed=seed)
x_jaya, y_jaya, jaya_hist = jaya.evolute(ngen=ngen, verbose=False)
assert speedred(x_jaya) == y_jaya

############
# Graphing #
############
hists = [es_hist, de_hist, gwo_hist, pesa2_hist, jaya_hist]
labels = ['ES', 'DE', 'GWO', 'PESA2', 'JAYA']
plt.figure()
for i, hist in enumerate(hists):
    plt.plot(hist, label=labels[i])
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.legend()
plt.ylim([2990, 3500])
plt.savefig('speedred.png')

#####################
# Compare fitnesses #
#####################
print()
print('FITNESS COMPARISONS')
print('----------------------------------')
for i, hist in enumerate(hists):
    print(f'{labels[i]}: {hist[-1]}')
