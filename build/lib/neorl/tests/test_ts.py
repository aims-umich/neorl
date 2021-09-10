from neorl import TS

def test_ts():
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
    for i in range(nx):
        BOUNDS['x'+str(i)]=['int', -100, 100]
    
    #setup and evolute TS
    x0=[-25,50,100,-75,-100]  #initial guess
    ts=TS(mode = "min", bounds = BOUNDS, fit = Sphere, tabu_tenure=60, 
          penalization_weight = 0.8, swap_mode = "perturb", ncores=1, seed=1)
    x_best, y_best, ts_hist=ts.evolute(ngen = 700, x0=x0, verbose=0)

test_ts