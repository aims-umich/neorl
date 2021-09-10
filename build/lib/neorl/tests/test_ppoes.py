from neorl import PPOES
from neorl import CreateEnvironment

def test_ppoes():
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
    
    #create an enviroment class for RL/PPO
    env=CreateEnvironment(method='ppo', fit=Sphere, ncores=1,  
                          bounds=BOUNDS, mode='min', episode_length=50)
    
    #change hyperparameters of PPO/ES if you like (defaults should be good to start with)
    h={'cxpb': 0.8,
       'mutpb': 0.2,
       'n_steps': 24,
       'lam': 1.0}
    
    #Important: `mode` in CreateEnvironment and `mode` in PPOES must be consistent
    #fit is needed to be passed again for ES, must be same as the one used in env
    ppoes=PPOES(mode='min', fit=Sphere, 
                env=env, npop_rl=4, init_pop_rl=True, 
                bounds=BOUNDS, hyperparam=h, seed=1)
    #first run RL for some timesteps
    rl=ppoes.learn(total_timesteps=2000, verbose=True)
    #second run ES, which will use RL data for guidance
    ppoes_x, ppoes_y, ppoes_hist=ppoes.evolute(ngen=20, ncores=1, verbose=True) #ncores for ES
    
    return

test_ppoes()