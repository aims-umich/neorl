from neorl import PESA
from neorl import PESA2

#Define the fitness function
def FIT(individual):
        y=sum(x**2 for x in individual)
        return y

#Define the fitness function
def FIT2(individual):
        y=sum(x**2 for x in individual)
        return y

#Setup the parameter space (d=5)
nx=5
BOUNDS={}
for i in range(1,nx+1):
        BOUNDS['x'+str(i)]=['float', -100, 100]
            
def test_pesa():

    npop=60
    pesa=PESA(mode='min', bounds=BOUNDS, fit=FIT, npop=npop, mu=40, alpha_init=0.2, 
              alpha_end=1.0, alpha_backdoor=0.1, ncores=9)
    x0=[[50,50,50,50,50] for i in range(npop)]  #initial guess
    x_best, y_best, pesa_hist=pesa.evolute(ngen=4, x0=x0, verbose=1)

test_pesa()

def test_pesa2():
    
    pesa2=PESA2(mode='min', bounds=BOUNDS, fit=FIT2, npop=50, nwolves=5, nwhales=5, ncores=9)
    x_best, y_best, pesa2_hist=pesa2.evolute(ngen=4, replay_every=2, verbose=2)

test_pesa2()