from neorl import XNES


def test_xnes():
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
    
    xnes=XNES(mode='min', bounds=BOUNDS, fit=FIT, npop=50, eta_mu=0.9, 
              eta_sigma=0.25, adapt_sampling=True, ncores=1, seed=1)
    x_best, y_best, xnes_hist=xnes.evolute(ngen=100, x0=[25,25,25,25,25], verbose=1)
    
test_xnes()