from neorl import CS

def test_cs():
    #Define the fitness function
    def Sphere(individual):
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
    
    #setup and evolute CS
    cs = CS(mode = 'min', bounds = BOUNDS, fit = Sphere, ncuckoos = 40, pa = 0.25, seed=1)
    x_best, y_best, cs_hist=cs.evolute(ngen = 200, verbose=False)

test_cs()