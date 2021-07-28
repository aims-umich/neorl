from neorl import RNEAT
import numpy as np 
from neorl import CreateEnvironment

def test_rneat():

    
    def Sphere(individual):
        """Sphere test objective function.
                F(x) = sum_{i=1}^d xi^2
                d=1,2,3,...
                Range: [-100,100]
                Minima: 0
        """
        return sum(x**2 for x in individual)
    
    nx=10
    bounds={}
    for i in range(1,nx+1):
            bounds['x'+str(i)]=['float', -100, 100]
    
    #create an enviroment class
    env=CreateEnvironment(method='neat', 
                          fit=Sphere, 
                          bounds=bounds, 
                          mode='min', 
                          episode_length=50)
    
    # modify your own NEAT config
    config = {
        'fitness_threshold': 1e5,
        'pop_size': 50,
        'num_hidden': 1,
        'activation_default': "random",
        'activation_mutate_rate': 0.1,
        'activation_options': 'tanh gauss relu',
        'initial_connection': 'partial_nodirect 0.5' # input-output not connect at first
        }
    
    # model config
    neats=RNEAT(env=env, config= config, ncores=1, seed=1)
    #some random guess (just one individual)
    x0 = np.random.random((10,))*50 
    x_best, y_best, neat_hist=neats.evolute(ngen=10, x0=x0, verbose=True, checkpoint_itv=None, startpoint=None) # 
    assert Sphere(x_best) == y_best
    print(x_best, y_best)
    
test_rneat()