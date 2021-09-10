from neorl import HHO

def test_hho():
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
    
    #setup and evolute HHO
    hho=HHO(mode='min', bounds=BOUNDS, fit=FIT, nhawks=20, ncores=1, seed=1)
    x_best, y_best, hho_hist=hho.evolute(ngen=200, verbose=1)

test_hho()