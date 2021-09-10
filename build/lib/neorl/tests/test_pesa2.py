from neorl import PESA2

def test_pesa2():
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
    
    pesa2=PESA2(mode='min', bounds=BOUNDS, fit=FIT, npop=50, nwolves=5, nwhales=5, ncores=1)
    x_best, y_best, pesa2_hist=pesa2.evolute(ngen=50, replay_every=2, verbose=2)

test_pesa2()