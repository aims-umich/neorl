from neorl import NHHO

def test_nhho():
    
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
    
    nn_params = {}
    nn_params['num_nodes'] = [20, 10]
    nn_params['learning_rate'] = 8e-4
    nn_params['epochs'] = 10
    nn_params['plot'] = False
    nn_params['verbose'] = False
    nn_params['save_models'] = False
    
    nhho = NHHO(mode='min', bounds=BOUNDS, fit=FIT, nhawks=20, nn_params=nn_params, seed=1)
    individuals, fitnesses = nhho.evolute(ngen=2, verbose=True)

test_nhho()