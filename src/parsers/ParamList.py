
# List of all neorl variables 

class InputParam():
    def __init__ (self):
    
        """
        #---------------------------------    
        #Key: [Value, Category, Type]
        #---------------------------------   
            
        #Value: is the value of the parameter 
        #Key: the type of the input: required ('r'), optional ('o'), required in special cases ('rs')
        #Type: data structure type: int, float, str, etc. 
        
        # input keys: 
        r: required parameter, if not defined, error is raised, neorl terminates.
        o: optional parameter, if not defined, default value is used, warning is raised, execution continues.
        rs: required for special cases, special case is explained next to each parameter, if not defined and special case is violated, error is raised, neorl terminates. 
        
        """
        
        # General Card
        self.gen_dict={
        'neorl_mode': ['run','o','str'],
        'env': [None,'r','str'],             # Evniorment Type
        'exepath': [None,'o','str'],         # e.g. Casmo/Simulate/MCNP path  
        'maxcores': [16,'o', 'int'],         # max number of cores to not exceed in parallel calculations
        'nactions': [None,'r', 'int'],
        'xsize':[None,'r','int'],
        'xsize_plot':[None,'o','int'],
        'ysize':[None,'r','int'],
        'ynames':[['y'], 'o', 'strvec'],
        'xnames':[['x'], 'o', 'strvec'],
        'plot_mode':['subplot','o','str']
        }
                
        # DQN Card
        self.dqn_dict={
        'time_steps': [50000,'rs','int'],
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
        'model_load_path': [None,'rs', 'str'],
        
         # for plotting 
        'check_freq': [1000,'o', 'int'],
        'avg_episodes': [20,'o', 'int'],
        
        # only for testing
        'n_eval_episodes': [None,'rs', 'int'],
        'render': [False,'rs', 'bool'],
        'video_record': [False,'rs', 'bool'],
        'fps': [10, 'rs', 'int']
        
        }
        
        # PPO Card
        self.ppo_dict={
        'time_steps': [50000,'rs','int'],
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
        'model_load_path': [None,'rs', 'str'],
        'check_freq': [1000,'o', 'int'],
        'avg_episodes': [20,'o', 'int'],
        
        # only for testing
        'n_eval_episodes': [None,'rs', 'int'],
        'render': [False,'o', 'bool'],
        'video_record': [False,'o', 'bool'],
        'fps': [10, 'o', 'int']
        }
        
        #A2C Card
        self.a2c_dict={
        'time_steps': [50000,'rs','int'],
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
        'model_load_path': [None,'rs', 'str'],
        'check_freq': [1000,'o', 'int'],
        'avg_episodes': [20,'o', 'int'],
        
        # only for testing
        'n_eval_episodes': [None,'rs', 'int'],
        'render': [False,'o', 'bool'],
        'video_record': [False,'o', 'bool'],
        'fps': [10, 'o', 'int']
        
        }
    
        self.ga_dict={
        'flag': [False, 'o', 'bool'],
        'casename': ['ga', 'o', 'str'],
        'mode': [None, 'r', 'str'],
        'lbound': [None, 'o', 'vec'],
        'ubound': [None, 'o', 'vec'],
        'pop': [None, 'r', 'int'],
        'ngen': [None, 'r', 'int'],
        'ncores': [1,'o','int'],  # is not working
        'indpb': [0.05,'o','float'],
        'cxpb': [0.5, 'o', 'float'],
        'mutpb': [0.2, 'o', 'float']
        }

        self.sa_dict={
        'flag': [False, 'o', 'bool'],
        'casename': ['sa', 'o', 'str'],
        'ncores': [1,'o','int'],  # is not working
        'swap': ['dualswap', 'r', 'str'],
        'lbound': [None, 'o', 'vec'],
        'ubound': [None, 'o', 'vec'],
        'steps': [None, 'r', 'int'],
        'Tmin': [None, 'r', 'int'],
        'Tmax': [None,'r','int'],  # is not working
        'initstate': [None,'o','float'],
        'check_freq': [50, 'o', 'int']
        }
# if __name__ =='__main__':
#     data=InputParam()
