from neorl import PPO2, A2C, ACKTR, ACER, DQN
from neorl import MlpPolicy
from neorl import DQNPolicy
from neorl import RLLogger
from neorl import CreateEnvironment
import math

def test_mixedrl():
    
    #---------------------------------
    # Fitness function
    #---------------------------------
    
    def Vessel(individual):
        """
        Pressure vesssel design
        x1: thickness (d1)  --> discrete value multiple of 0.0625 in 
        x2: thickness of the heads (d2) ---> categorical value from a pre-defined grid
        x3: inner radius (r)  ---> cont. value between [10, 200]
        x4: length (L)  ---> cont. value between [10, 200]
        """
    
        x=individual.copy()
        x[0] *= 0.0625   #convert d1 to "in" 
    
        y = 0.6224*x[0]*x[2]*x[3]+1.7781*x[1]*x[2]**2+3.1661*x[0]**2*x[3]+19.84*x[0]**2*x[2];
    
        g1 = -x[0]+0.0193*x[2];
        g2 = -x[1]+0.00954*x[2];
        g3 = -math.pi*x[2]**2*x[3]-(4/3)*math.pi*x[2]**3 + 1296000;
        g4 = x[3]-240;
        g=[g1,g2,g3,g4]
        
        phi=sum(max(item,0) for item in g)
        eps=1e-5 #tolerance to escape the constraint region
        penality=1e7 #large penality to add if constraints are violated
        
        if phi > eps:  
            fitness=phi+penality
        else:
            fitness=y
        return fitness
    
    #---------------------------------
    # Mixed int/float/grid space
    #---------------------------------
    bounds = {}
    bounds['x1'] = ['int', 1, 99]
    bounds['x2'] = ['grid', (0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625)]
    bounds['x3'] = ['float', 10, 200]
    bounds['x4'] = ['float', 10, 200]
        
    #---------------------------------
    # PPO
    #---------------------------------
    env=CreateEnvironment(method='ppo', fit=Vessel, bounds=bounds, mode='min', episode_length=50)
    cb=RLLogger(check_freq=1, mode='min')
    ppo = PPO2(policy=MlpPolicy, env=env, n_steps=20, seed=1)
    ppo.learn(total_timesteps=1000, callback=cb)
    print('--------------- PPO results ---------------')
    print('The best value of x found:', cb.xbest)
    print('The best value of y found:', cb.rbest)
    
    assert Vessel(cb.xbest) - cb.rbest < 1e-3
    
    #---------------------------------
    # A2C
    #---------------------------------
    cb=RLLogger(check_freq=1, mode='min')
    a2c = A2C(policy=MlpPolicy, env=env, n_steps=20, seed=1)
    a2c.learn(total_timesteps=1000, callback=cb)
    print('--------------- A2C results ---------------')
    print('The best value of x found:', cb.xbest)
    print('The best value of y found:', cb.rbest)
    assert Vessel(cb.xbest) - cb.rbest < 1e-3
    
    #---------------------------------
    # ACKTR
    #---------------------------------
    cb=RLLogger(check_freq=1, mode='min')
    acktr = ACKTR(policy=MlpPolicy, env=env, n_steps=20)
    acktr.learn(total_timesteps=1000, callback=cb)
    print('--------------- ACKTR results ---------------')
    print('The best value of x found:', cb.xbest)
    print('The best value of y found:', cb.rbest)
    assert Vessel(cb.xbest) - cb.rbest < 1e-3
    
    #---------------------------------
    # Mixed int/grid
    #---------------------------------
    bounds = {}
    bounds['x1'] = ['int', 1, 20]
    bounds['x2'] = ['grid', (0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625)]
    bounds['x3'] = ['int', 10, 200]
    bounds['x4'] = ['int', 10, 200]
    
    #---------------------------------
    # ACER
    #---------------------------------
    disc_env=CreateEnvironment(method='acer', fit=Vessel, mode='min', 
                               bounds=bounds, episode_length=50)
    cb=RLLogger(check_freq=1, mode='min')
    acer = ACER(MlpPolicy, env=disc_env, n_steps=25)
    acer.learn(total_timesteps=2000, callback=cb)
    print('--------------- ACER results ---------------')
    print('The best value of x found:', cb.xbest)
    print('The best value of y found:', cb.rbest)
    assert Vessel(cb.xbest) - cb.rbest < 1e-3
    
    #---------------------------------
    # DQN
    #---------------------------------
    cb=RLLogger(check_freq=1, mode='min')
    dqn = DQN(DQNPolicy, env=disc_env)
    dqn.learn(total_timesteps=1000, callback=cb)
    print('--------------- DQN results ---------------')
    print('The best value of x found:', cb.xbest)
    print('The best value of y found:', cb.rbest)
    assert Vessel(cb.xbest) - cb.rbest < 1e-3
    
    return

test_mixedrl()