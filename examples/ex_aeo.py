import matplotlib.pyplot as plt
from neorl import AEO

from neorl import DE
from neorl import ES
from neorl import GWO
from neorl import PSO
from neorl import WOA
from neorl import MFO
from neorl import SSA
from neorl import JAYA

#define the fitness function
def FIT(individual):
    """Sphere test objective function.
            F(x) = sum_{i=1}^d x_i^2
            d=1,2,3,...
            Range: [-100,100]
            Minima: 0
    """
    y = sum(x**2 for x in individual)
    
    return y

#Setup the parameter space (d=5)
nx=5
BOUNDS={}
for i in range(1,nx+1):
    BOUNDS['x'+str(i)]=['float', -100, 100]

#Define algorithms to be used in enembles
#   parameters not directly describing population size
#   are carried into the AEO algorithm. See de2 for an
#   example of this.
es  =  ES(mode='min', fit=FIT, bounds=BOUNDS)
gwo = GWO(mode='min', fit=FIT, bounds=BOUNDS)
woa = WOA(mode='min', fit=FIT, bounds=BOUNDS)
mfo = MFO(mode='min', fit=FIT, bounds=BOUNDS)
ssa = SSA(mode='min', fit=FIT, bounds=BOUNDS)
de1 =  DE(mode='min', fit=FIT, bounds=BOUNDS)
de2 =  DE(mode='min', fit=FIT, bounds=BOUNDS, F=0.5, CR=0.5)
pso = PSO(mode='min', fit=FIT, bounds=BOUNDS)
jaya=JAYA(mode='min', fit=FIT, bounds=BOUNDS)

ensemble = [es, gwo, woa, mfo, ssa, de1, de2, pso, jaya]

#initialize an intance of aeo
aeo = AEO(mode='min', fit=FIT, bounds=BOUNDS, optimizers=ensemble,
        gen_per_cycle=2)

#perform evolution
best_x, best_y, log = aeo.evolute(15)

print('Best x')
print(best_x)
print('Best y')
print(best_y)

plt.figure()
for p in log.coords['pop']:
    plt.plot(log.coords['cycle'], log['nmembers'].sel({'pop' : p}),
            label = p.values)

plt.xlabel("Cycle")
plt.ylabel("Number of Members")

plt.legend()
plt.show()