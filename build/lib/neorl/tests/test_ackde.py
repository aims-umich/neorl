from neorl import ACKDE
from neorl import CreateEnvironment

def test_ackde():
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
    
    #create an enviroment class for RL/ACKTR
    env=CreateEnvironment(method='acktr', fit=Sphere, ncores=1,  
                          bounds=BOUNDS, mode='min', episode_length=50)
    
    #change hyperparameters of ACKTR/DE if you like (defaults should be good to start with)
    h={'F': 0.5,
       'CR': 0.3,
       'n_steps': 20,
       'learning_rate': 0.001}
    
    #Important: `mode` in CreateEnvironment and `mode` in ACKDE must be consistent
    #fit is needed to be passed again for DE, must be same as the one used in env
    ackde=ACKDE(mode='min', fit=Sphere, npop=60,
                env=env, npop_rl=6, init_pop_rl=False, 
                bounds=BOUNDS, hyperparam=h, seed=1)
    #first run RL for some timesteps
    rl=ackde.learn(total_timesteps=2000, verbose=True)
    #second run DE, which will use RL data for guidance
    ackde_x, ackde_y, ackde_hist=ackde.evolute(ngen=100, ncores=1, verbose=True) #ncores for DE
    
    return

test_ackde()