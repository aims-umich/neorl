from neorl import NGA

def test_nga():
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
    
    nga = NGA(mode='min', bounds=BOUNDS, fit=FIT, npop=40, num_warmups=100, 
              hidden_shape=5, seed=1)
    individuals, surrogate_fit = nga.evolute(ngen=5, verbose=False)
    
    #make evaluation of the best individuals using the real fitness function
    real_fit=[FIT(item) for item in individuals]
    
    #print the best individuals/fitness found
    min_index=real_fit.index(min(real_fit))
    print('------------------------ Final Summary --------------------------')
    print('Best real individual:', individuals[min_index])
    print('Best real fitness:', real_fit[min_index])
    print('-----------------------------------------------------------------')

test_nga()