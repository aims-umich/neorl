from neorl import PESA

def test_pesa():
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
    
    npop=60
    pesa=PESA(mode='min', bounds=BOUNDS, fit=FIT, npop=npop, mu=40, alpha_init=0.2, 
              alpha_end=1.0, alpha_backdoor=0.1, ncores=1)
    x0=[[50,50,50,50,50] for i in range(npop)]  #initial guess
    x_best, y_best, pesa_hist=pesa.evolute(ngen=50, x0=x0, verbose=1)

test_pesa()