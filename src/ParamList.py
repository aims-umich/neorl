
# List of neorl variables 

class InputParam():
    def __init__ (self):
    
        #---------------------------------    
        #Key: [Value, Category, Type]
        #---------------------------------   
            
        #Key: is reserved and must not be changed 
        #Value: is the value of the parameter 
        #Category: the block to which this parameter belongs: general, DQN, GA, parallel, etc. 
        #Type: data structure type: int, float, str, etc. 
        
        # General Category
        self.gen_dict={
        'env': ['casmo10x10:casmo10x10-v0','general','str'],                       # Evniorment Type
        'exepath': ['/home/majdi/...','general','str'],         # e.g. Casmo/Simulate/MCNP path  
        'maxcores': [16,'general','int'],                        # max number of cores to not exceed in parallel calculations
        'nactions': [19,'general', 'int'],
        'xsize':[51,'general','int'],
        #psutil.cpu_count(logical = True)
        #multiprocessing.cpu_count()
        }
        
        # DQN Category
        self.dqn_dict={
        'flag': [False, 'dqn', 'bool'],
        'casename': ['dqn', 'dqn', 'str'],
        'mode': ['train', 'dqn', 'str'],
        'ncores': [1,'dqn','int'],
        'gamma': [0.95, 'dqn', 'float'],
        'learning_rate': [1e-3, 'dqn', 'float'],
        'buffer_size': [50000, 'dqn', 'int'],
        'exploration_fraction': [0.1, 'dqn', 'float'],
        'exploration_final_eps': [0.04, 'dqn', 'float'],
        'exploration_initial_eps': [1.0, 'dqn', 'float'],
        'train_freq': [1, 'dqn', 'float'],
        'batch_size': [32, 'dqn', 'int'],
        'double_q': [True, 'dqn', 'bool'],
        'learning_starts': [10, 'dqn', 'int'],
        'target_network_update_freq': [1000, 'dqn', 'int'],
        'model_load_path': ['/home/majdi/','dqn', 'str'],
        'time_steps': [50000,'dqn','int']
        }
        
        # PPO Category
        self.ppo_dict={
        'flag': [False, 'ppo', 'bool'],
        'casename': ['ppo', 'ppo', 'str'],
        'mode': ['train', 'ppo', 'str'],
        'ncores': [5,'ppo','int'],
        'n_steps': [128,'ppo','int'],
        'gamma': [0.99, 'ppo', 'float'],
        'learning_rate': [0.00025, 'ppo', 'float'],
        'vf_coef': [0.5, 'ppo', 'float'],
        'max_grad_norm': [0.5, 'ppo', 'float'],
        'lam': [0.95, 'ppo', 'float'],
        'nminibatches': [4, 'ppo', 'int'],
        'noptepochs': [4, 'ppo', 'int'],
        'cliprange': [0.2, 'ppo', 'float'],
        'model_load_path': ['/home/majdi/','ppo', 'str'],
        'time_steps': [50000,'ppo','int']
        }
        
        
        self.a2c_dict={
        # Training parameters
        'flag': [True, 'a2c', 'bool'],
        'casename': ['a2c', 'a2c', 'str'],
        'mode': ['test', 'a2c', 'str'],
        'ncores': [3,'a2c','int'],
        'n_steps': [5,'a2c','int'],
        'gamma': [0.99, 'a2c', 'float'],
        'learning_rate': [0.0007, 'a2c', 'float'],
        'vf_coef': [0.25, 'a2c', 'float'],
        'max_grad_norm': [0.5, 'a2c', 'float'],
        'ent_coef': [0.01, 'a2c', 'float'],
        'alpha': [0.99, 'a2c', 'float'],
        'epsilon': [1e-5, 'a2c', 'float'],
        'lr_schedule': ['constant', 'a2c', 'str'],
        'time_steps': [50000,'a2c','int'],
        
        #Testing parameters
        'model_load_path': ['./base.pkl','a2c', 'str'],
        'render': ['video','a2c', 'str'],
        'video_length': [200,'a2c', 'int'],        
        }
    
        self.ga_dict={
        'flag': [False, 'ga', 'bool'],
        'casename': ['ga', 'ga', 'str'],
        'ncores': [1,'ga','int'],  # is not working
        'indpb': [0.05,'ga','float'],
        'cxpb': [0.99, 'ga', 'float'],
        'mutpb': [0.0007, 'ga', 'float'],
        'pop': [30, 'ga', 'int'],
        'ngen': [400, 'ga', 'int'],
        }
# if __name__ =='__main__':
#     data=InputParam()
