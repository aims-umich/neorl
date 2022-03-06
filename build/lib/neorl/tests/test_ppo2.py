from neorl import PPO2
from neorl import MlpPolicy
from neorl import RLLogger
from neorl import CreateEnvironment

#--------------------------------------------------------
# RL Optimisation
#--------------------------------------------------------
def test_ppo():
    def Sphere(individual):
            """Sphere test objective function.
                    F(x) = sum_{i=1}^d xi^2
                    d=1,2,3,...
                    Range: [-100,100]
                    Minima: 0
            """
            return sum(x**2 for x in individual)
    
    nx=5
    bounds={}
    for i in range(1,nx+1):
            bounds['x'+str(i)]=['float', -10, 10]
    
    #create an enviroment class
    env=CreateEnvironment(method='ppo', fit=Sphere, 
                          bounds=bounds, mode='min', episode_length=50)
    #create a callback function to log data
    cb=RLLogger(check_freq=1)
    #create an a2c object based on the env object
    ppo = PPO2(MlpPolicy, env=env, n_steps=12)
    #optimise the enviroment class
    ppo.learn(total_timesteps=2000, callback=cb)
    #print the best results
    print('--------------- PPO results ---------------')
    print('The best value of x found:', cb.xbest)
    print('The best value of y found:', cb.rbest)
    
    return 
	
test_ppo()