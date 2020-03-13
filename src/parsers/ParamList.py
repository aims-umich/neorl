
# List of all neorl variables 

class InputParam():
    def __init__ (self):
    
        """
        #---------------------------------    
        #Key: [Value, Category, Type]
        #---------------------------------   
            
        #Value: is the value of the parameter 
        #key: the type of the input: required, optional, required in special cases
        #Type: data structure type: int, float, str, etc. 
        
        # input keys: 
        r: required
        o: optional, defaults are available
        rs: required for special cases
        
        """
        
        # General Category
        self.gen_dict={
        'env': ['casmo10x10:casmo10x10-v0','r','str'],                       # Evniorment Type
        'exepath': ['/home/majdi/...','o','str'],         # e.g. Casmo/Simulate/MCNP path  
        'maxcores': [16,'o', 'int'],                        # max number of cores to not exceed in parallel calculations
        'nactions': [0,'r', 'int'],
        'xsize':[0,'r','int'],
        'xsize_plot':[0,'o','int'],
        'ysize':[0,'r','int']
        }
                
        # DQN Category
        self.dqn_dict={
        'time_steps': [50000,'r','int'],
        'flag': [False, 'o', 'bool'],
        'casename': ['dqn', 'o', 'str'],
        'mode': ['train', 'r', 'str'],
        'ncores': [1,'o','int'],
        'gamma': [0.99, 'o', 'float'],
        'learning_rate': [1e-3, 'o', 'float'],
        'buffer_size': [50000, 'o', 'int'],
        'exploration_fraction': [0.1, 'o', 'float'],
        'exploration_final_eps': [0.04, 'o', 'float'],
        'exploration_initial_eps': [1.0, 'o', 'float'],
        'train_freq': [1, 'o', 'float'],
        'batch_size': [32, 'o', 'int'],
        'double_q': [True, 'o', 'bool'],
        'learning_starts': [10, 'o', 'int'],
        'target_network_update_freq': [1000, 'o', 'int'],
        'model_load_path': ['/home/majdi/','rs', 'str'],
        
         # for plotting 
        'check_freq': [1000,'o', 'int'],
        'avg_episodes': [20,'o', 'int'],
        
        # only for testing
        'n_eval_episodes': [5,'rs', 'int'],
        'render': [False,'rs', 'bool'],
        'video_record': [False,'rs', 'bool'],
        'fps': [10, 'rs', 'int']
        
        }
        
        # PPO Category
        self.ppo_dict={
        'time_steps': [50000,'r','int'],
        'flag': [False, 'o', 'bool'],
        'casename': ['ppo', 'o', 'str'],
        'mode': ['train', 'r', 'str'],
        'ncores': [4,'o','int'],
        'n_steps': [128,'o','int'],
        'gamma': [0.99, 'o', 'float'],
        'learning_rate': [0.00025, 'o', 'float'],
        'vf_coef': [0.5, 'o', 'float'],
        'max_grad_norm': [0.5, 'o', 'float'],
        'lam': [0.95, 'o', 'float'],
        'nminibatches': [4, 'o', 'int'],
        'noptepochs': [4, 'o', 'int'],
        'cliprange': [0.2, 'o', 'float'],
        'model_load_path': ['/home/majdi/','rs', 'str'],
        'check_freq': [1000,'o', 'int'],
        'avg_episodes': [20,'o', 'int'],
        # only for testing
        'n_eval_episodes': [5,'o', 'int'],
        'render': [False,'o', 'bool'],
        'video_record': [False,'o', 'bool'],
        'fps': [10, 'o', 'int']
        }
        
        #A2C Category
        self.a2c_dict={
        'time_steps': [50000,'r','int'],
        'flag': [False, 'o', 'bool'],
        'casename': ['a2c', 'o', 'str'],
        'mode': ['train', 'r', 'str'],
        'ncores': [1,'o','int'],
        'n_steps': [5,'o','int'],
        'gamma': [0.99, 'o', 'float'],
        'learning_rate': [0.0007, 'o', 'float'],
        'vf_coef': [0.25, 'o', 'float'],
        'max_grad_norm': [0.5, 'o', 'float'],
        'ent_coef': [0.01, 'o', 'float'],
        'alpha': [0.99, 'o', 'float'],
        'epsilon': [1e-5, 'o', 'float'],
        'lr_schedule': ['constant', 'o', 'str'],
        'model_load_path': ['./base.pkl','rs', 'str'],
        'check_freq': [1000,'o', 'int'],
        'avg_episodes': [20,'o', 'int'],
        # only for testing
        'n_eval_episodes': [5,'o', 'int'],
        'render': [False,'o', 'bool'],
        'video_record': [False,'o', 'bool'],
        'fps': [10, 'o', 'int']
        
        }
    
        self.ga_dict={
        'flag': [False, 'o', 'bool'],
        'pop': [30, 'r', 'int'],
        'ngen': [400, 'r', 'int'],
        'casename': ['ga', 'o', 'str'],
        'ncores': [1,'o','int'],  # is not working
        'indpb': [0.05,'o','float'],
        'cxpb': [0.99, 'o', 'float'],
        'mutpb': [0.0007, 'o', 'float']
        }
# if __name__ =='__main__':
#     data=InputParam()
