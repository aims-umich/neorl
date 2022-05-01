from neorl import EDEV

def test_edev():
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
    
    #setup and evolute EDEV
    edev=EDEV(mode='min', bounds=BOUNDS, fit=FIT, npop=100, ncores=1, seed=1)
    x_best, y_best, edev_hist=edev.evolute(ngen=100, ng=10, verbose=1)

test_edev()