from neorl import PPO2, ACKTR, A2C, ACER
from neorl import MlpPolicy
from neorl import RLLogger
from neorl import CreateEnvironment

def Sphere(individual):
        """Sphere test objective function.
                F(x) = sum_{i=1}^d xi^2
                d=1,2,3,...
                Range: [-100,100]
                Minima: 0
        """
        #print(individual)
        return sum(x**2 for x in individual)


#this __main__ block is needed for parallel RL to avoid freezing error of multiprocessing
if __name__=='__main__':
    ncores=10
    nx=5
    bounds={}
    for i in range(1,nx+1):
            bounds['x'+str(i)]=['int', -100, 100]
    #---------------------------------
    # PPO
    #---------------------------------
    env=CreateEnvironment(method='ppo', fit=Sphere, ncores=ncores, 
                      bounds=bounds, mode='min', episode_length=50)
    cb=RLLogger(check_freq=1)
    ppo = PPO2(MlpPolicy, env=env, n_steps=12, seed=1)
    ppo.learn(total_timesteps=1000, callback=cb)
    print('--------------- PPO results ---------------')
    print('The best value of x found:', cb.xbest)
    print('The best value of y found:', cb.rbest)

    #---------------------------------
    # ACKTR
    #---------------------------------  
    env=CreateEnvironment(method='acktr', fit=Sphere, ncores=ncores, 
                      bounds=bounds, mode='min', episode_length=50)
    acktr = ACKTR(MlpPolicy, env=env, n_steps=12, seed=1)
    acktr.learn(total_timesteps=1000, callback=cb)
    print('--------------- ACKTR results ---------------')
    print('The best value of x found:', cb.xbest)
    print('The best value of y found:', cb.rbest)

    #---------------------------------
    # A2C
    #---------------------------------   
    env=CreateEnvironment(method='a2c', fit=Sphere, ncores=ncores, 
                      bounds=bounds, mode='min', episode_length=50)
    a2c = A2C(MlpPolicy, env=env, n_steps=8, seed=1)
    a2c.learn(total_timesteps=1000, callback=cb)
    print('--------------- A2C results ---------------')
    print('The best value of x found:', cb.xbest)
    print('The best value of y found:', cb.rbest)
    
    #---------------------------------
    # ACER
    #---------------------------------
    nx=5
    bounds={}
    for i in range(1,nx+1):
            bounds['x'+str(i)]=['int', -10, 10]
            
    env=CreateEnvironment(method='acer', fit=Sphere, ncores=10, 
                      bounds=bounds, mode='min', episode_length=50)    
    cb=RLLogger(check_freq=1)
    acer = ACER(MlpPolicy, env=env, n_steps=25, q_coef=0.55, ent_coef=0.02, seed=1)
    acer.learn(total_timesteps=1000, callback=cb)
    print('--------------- ACER results ---------------')
    print('The best value of x found:', cb.xbest)
    print('The best value of y found:', cb.rbest)