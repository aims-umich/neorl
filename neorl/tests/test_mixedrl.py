from neorl import PPO2, A2C, ACKTR, ACER, DQN
from neorl import MlpPolicy
from neorl import DQNPolicy
from neorl import RLLogger
from neorl import CreateEnvironment
import math

def test_mixedrl():
    
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
    
    for item in ['float', 'int', 'grid', 'float/int', 'float/grid', 'int/grid', 'mixed']:
        bounds = {}
        t=60
        btype=item  #float, int, grid, float/int, float/grid, int/grid, mixed. 
        
        if btype=='mixed':
            bounds['x1'] = ['int', 1, 99]
            bounds['x2'] = ['grid', (0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625)]
            bounds['x3'] = ['float', 10, 200]
            bounds['x4'] = ['float', 10, 200]
            bounds['x5'] = ['grid', ('Hi', 'Bye', 'New')]
            bounds['x6'] = ['int', -5, 5]
        
        elif btype=='int/grid':      
            bounds['x1'] = ['int', 1, 20]
            bounds['x2'] = ['grid', (0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625)]
            bounds['x3'] = ['int', 10, 200]
            bounds['x4'] = ['int', 10, 200]
            bounds['x5'] = ['grid', ('Hi', 'Bye', 'New')]
            bounds['x6'] = ['int', -5, 5]
        
        elif btype=='float/grid':      
            bounds['x1'] = ['float', 1, 20]
            bounds['x2'] = ['grid', (0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625)]
            bounds['x3'] = ['float', 10, 200]
            bounds['x4'] = ['float', 10, 200]
            bounds['x5'] = ['grid', ('Hi', 'Bye', 'New')]
            bounds['x6'] = ['float', -5, 5]
            
        elif btype=='float/int':      
            bounds['x1'] = ['int', 1, 20]
            bounds['x2'] = ['float', 1, 20]
            bounds['x3'] = ['int', 10, 200]
            bounds['x4'] = ['float', 10, 200]
            bounds['x5'] = ['float', -5, 5]
            bounds['x6'] = ['int', -5, 5]
        
        elif btype=='float':      
            bounds['x1'] = ['float', 1, 20]
            bounds['x2'] = ['float', 1, 20]
            bounds['x3'] = ['float', 10, 200]
            bounds['x4'] = ['float', 10, 200]
            bounds['x5'] = ['float', -5, 5]
            bounds['x6'] = ['float', -5, 5]
            
        elif btype=='int':      
            bounds['x1'] = ['int', 1, 20]
            bounds['x2'] = ['int', 1, 20]
            bounds['x3'] = ['int', 10, 200]
            bounds['x4'] = ['int', 10, 200]
            bounds['x5'] = ['int', -5, 5]
            bounds['x6'] = ['int', -5, 5]
            
        elif btype=='grid':      
            bounds['x1'] = ['grid', (0.0625, 0.125, 0.375, 0.4375, 0.5625, 0.625)]
            bounds['x2'] = ['grid', (0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625)]
            bounds['x3'] = ['grid', (1,2,3,4,5)]
            bounds['x4'] = ['grid', (32,64,128)]
            bounds['x5'] = ['grid', ('Hi', 'Bye', 'New')]
            bounds['x6'] = ['grid', ('Cat', 'Dog', 'Bird', 'Fish')]
        
        #---------------------------------
        # PPO
        #---------------------------------
        env=CreateEnvironment(method='ppo', fit=Vessel, bounds=bounds, mode='min', episode_length=50)
        cb=RLLogger(check_freq=1)
        ppo = PPO2(policy=MlpPolicy, env=env, n_steps=4, seed=1)
        ppo.learn(total_timesteps=t, callback=cb)
        print('--------------- PPO results ---------------')
        print('The best value of x found:', cb.xbest)
        print('The best value of y found:', cb.rbest)
        
        assert Vessel(cb.xbest) - cb.rbest < 1
        
        #---------------------------------
        # A2C
        #---------------------------------
        cb=RLLogger(check_freq=1)
        a2c = A2C(policy=MlpPolicy, env=env, n_steps=4, seed=1)
        a2c.learn(total_timesteps=t, callback=cb)
        print('--------------- A2C results ---------------')
        print('The best value of x found:', cb.xbest)
        print('The best value of y found:', cb.rbest)
        assert Vessel(cb.xbest) - cb.rbest < 1
        
        #---------------------------------
        # ACKTR
        #---------------------------------
        cb=RLLogger(check_freq=1)
        acktr = ACKTR(policy=MlpPolicy, env=env, n_steps=4)
        acktr.learn(total_timesteps=t, callback=cb)
        print('--------------- ACKTR results ---------------')
        print('The best value of x found:', cb.xbest)
        print('The best value of y found:', cb.rbest)
        assert Vessel(cb.xbest) - cb.rbest < 1
        
        
        if btype in ['int', 'grid', 'int/grid']:
            #---------------------------------
            # ACER
            #---------------------------------
            disc_env=CreateEnvironment(method='acer', fit=Vessel, mode='min', 
                                       bounds=bounds, episode_length=50)
            cb=RLLogger(check_freq=1)
            acer = ACER(MlpPolicy, env=disc_env, n_steps=4)
            acer.learn(total_timesteps=t, callback=cb)
            print('--------------- ACER results ---------------')
            print('The best value of x found:', cb.xbest)
            print('The best value of y found:', cb.rbest)
            assert Vessel(cb.xbest) - cb.rbest < 1
            
            #---------------------------------
            # DQN
            #---------------------------------
            cb=RLLogger(check_freq=1)
            dqn = DQN(DQNPolicy, env=disc_env)
            dqn.learn(total_timesteps=t, callback=cb)
            print('--------------- DQN results ---------------')
            print('The best value of x found:', cb.xbest)
            print('The best value of y found:', cb.rbest)
            assert Vessel(cb.xbest) - cb.rbest < 1
    
    return

test_mixedrl()