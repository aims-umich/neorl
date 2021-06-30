from neorl import BAT

def test_bat():
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
    
    #setup and evolute BAT
    bat=BAT(mode='min', bounds=BOUNDS, fit=FIT, nbats=40, 
            fmin=0, fmax=1, A=1.0, r0=0.7,
            ncores=1, seed=1)
    x_best, y_best, bat_hist=bat.evolute(ngen=100, verbose=1)

test_bat()