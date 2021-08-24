from neorl import ACO
import random

def test_aco():
    
    #Define the fitness function
    def FIT(individual):
        """Sphere test objective function.
                        F(x) = sum_{i=1}^d xi^2
                        d=1,2,3,...
                        Range: [-100,100]
                        Minima: 0
        """
        y=sum(x**2 for x in individual)
        return y
    
    #Setup the parameter space (d=5)
    nx=5
    BOUNDS={}
    for i in range(1,nx+1):
        BOUNDS['x'+str(i)]=['float', -100, 100]
    
    nants=40
    x0=[[random.uniform(-100,100)]*nx for item in range(nants)]
    acor = ACO(mode='min', fit=FIT, bounds=BOUNDS, nants=nants, narchive=10, 
               Q=0.5, Z=1, ncores=1, seed=1)
    x_best, y_best, acor_hist=acor.evolute(ngen=100, x0=x0, verbose=1)

test_aco()