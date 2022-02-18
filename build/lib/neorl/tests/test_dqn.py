from neorl import DQN
from neorl import DQNPolicy
from neorl import RLLogger
from neorl import CreateEnvironment

#--------------------------------------------------------
# RL Optimisation
#--------------------------------------------------------
def test_dqn():
    def Sphere(individual):
            """Sphere test objective function.
                    F(x) = sum_{i=1}^d xi^2
                    d=1,2,3,...
                    Range: [-100,100]
                    Minima: 0
            """
            #print(individual)
            return sum(x**2 for x in individual)
    
    nx=5
    bounds={}
    for i in range(1,nx+1):
            bounds['x'+str(i)]=['int', -100, 100]
    
    #create an enviroment class
    env=CreateEnvironment(method='dqn', 
                          fit=Sphere, 
                          bounds=bounds, 
                          mode='min', 
                          episode_length=50)
    #create a callback function to log data
    cb=RLLogger(check_freq=1)
    #create an a2c object based on the env object
    dqn = DQN(DQNPolicy, env=env)
    #optimise the enviroment class
    dqn.learn(total_timesteps=2000, callback=cb)
    #print the best results
    print('--------------- DQN results ---------------')
    print('The best value of x found:', cb.xbest)
    print('The best value of y found:', cb.rbest)
    
    return

test_dqn()