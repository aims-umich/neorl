#    This file is part of NEORL.

#    Copyright (c) 2021 Exelon Corporation and MIT Nuclear Science and Engineering
#    NEORL is free software: you can redistribute it and/or modify
#    it under the terms of the MIT LICENSE

#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

"""
Created on Tue Feb 25 12:09:27 2020

@author: Majdi Radaideh
"""

# List of all neorl variables 


class InputParam():
    def __init__ (self):
    
        """
        #---------------------------------    
        #Key: [Default value (if any), Category, Type]
        #---------------------------------   
            
        #Default Value: is the default value of optional parameter if not specified.
        #Key: the type of the input: required ('r'), optional ('o'), required in special cases ('rs')
        #Type: data structure type: int, float, str, vec, etc. 
        
        # input keys: 
        r: required parameter, if not defined, error is raised, neorl terminates.
        o: optional parameter, if not defined, default value is used, warning is raised, execution continues.
        rs: required for special cases, special case is explained next to each parameter, if not defined and special case is violated, error is raised, neorl terminates. 
        
        """
        
        #---------------------------------
        # General Card
        #---------------------------------
        self.gen_dict={
        'neorl_mode': ['run','o','str'],  # a 'developer' option for input parser testing, run or test
        'env': [None,'r','str'],          # pre-registered env name     
        'env_data': [None,'o','dict'],     # optional env data 
        'exepath': [None,'o','str'],      # see manual
        'maxcores': [16,'o', 'int'],      # or use -1 to infer maxcores
        'daemon': [True,'o', 'bool'],      # daemon option to allow running nested children in parallel for RL, set to False if yes
        'nactions': [None,'r', 'int'],    # action size to take by agent (self.action_space) 
        'xsize':[None,'r','int'],         # see manual
        'xsize_plot':[None,'o','int'],    # see manual 
        'ysize':[None,'r','int'],         # see manual
        'ynames':[['y'], 'o', 'strvec'],  # default names x1, x2, ..., x{xsize_plot}
        'xnames':[['x'], 'o', 'strvec'],  # default names y1, y2, ..., x{ysize}
        'plot_mode':['subplot','o','str'] #either subplot (recommended) or classic 
        }
        
        #---------------------------------
        # TUNE card 
        #---------------------------------
        self.tune_dict={
        'flag': [False, 'o', 'bool'],       #this will be activated if READ TUNE is found in input 
        'ncores': [1,'o','int'],            #number of parallel tuned cases to run    
        'mode': ['run','o','str'],  # tune mode: run or test, test will generate cases but not run them. 
        'method': [None,'r','str'],  # tune method: grid search, random search, genentic algorathim
        'n_last_episodes': [50,'o','int'],  # number of last episodes to average the reward and determine convergence
        'ncases': [100,'o','int'],  # number of last episodes to average the reward and determine convergence 
        'extfiles':[None, 'o', 'strvec']
        }
        
        #---------------------------------
        # DQN Card
        #---------------------------------
        self.dqn_dict={
        'mode': ['train', 'r', 'str'],     # either train, continue, or test     
        'time_steps': [50000,'rs','int'],  # required for mode=train/continue
        'flag': [False, 'o', 'bool'],      # this will be activated if READ DQN is found in input
        'casename': ['dqn', 'o', 'str'],   # prefix for logging and results
        'ncores': [1,'o','int'],                        # only 1 core supported
        'gamma': [0.99, 'o', 'float'],                  # see manual 
        'learning_rate': [1e-3, 'o', 'float'],          # see manual
        'buffer_size': [50000, 'o', 'int'],             # see manual
        'exploration_fraction': [0.1, 'o', 'float'],    # see manual
        'exploration_final_eps': [0.02, 'o', 'float'],  # see manual
        'exploration_initial_eps': [1.0, 'o', 'float'], # see manual
        'train_freq': [1, 'o', 'float'],                # see manual
        'batch_size': [32, 'o', 'int'],                 # see manual
        'double_q': [True, 'o', 'bool'],                # see manual
        'learning_starts': [1000, 'o', 'int'],          # see manual
        'target_network_update_freq': [1000, 'o', 'int'], # see manual
        
        #Priortized Replay Parameters 
        'prioritized_replay': [False, 'o', 'bool'], # whether to use priortized replay or not
        'prioritized_replay_alpha': [0.6, 'o', 'float'], # how much prioritization is used (0 is the uniform case)
        'prioritized_replay_beta0': [0.4, 'o', 'float'], # initial value of beta for prioritized replay buffer
        
        # for testing or continue modes 
        'model_load_path': [None,'rs', 'str'], # pre-trained model required only for mode=continue/test 
        
         # for plotting 
        'check_freq': [1000,'o', 'int'],     # see manual
        'avg_episodes': [20,'o', 'int'],     # see manual
        'tensorboard': [False,'o','bool'],   # tensorboard view option, see manual for usage 
        
        # only for testing
        'n_eval_episodes': [None,'rs', 'int'],   # required only for mode=test 
        'render': [False,'o', 'bool'],           # option to plot during test via self.render() 
        }
        
        #---------------------------------
        # PPO Card
        #---------------------------------
        self.ppo_dict={
        'mode': ['train', 'r', 'str'],     # either train, continue, or test
        'time_steps': [50000,'rs','int'],  # required only for mode=train/continue 
        'flag': [False, 'o', 'bool'],      # this will be activated if READ PPO is found in input
        'casename': ['ppo', 'o', 'str'],          # prefix for logging and results 
        'ncores': [1,'o','int'],                  # multiple cores are available
        'n_steps': [128,'o','int'],               # see manual
        'gamma': [0.99, 'o', 'float'],            # see manual
        'learning_rate': [0.00025, 'o', 'float'], # see manual 
        'ent_coef': [0.01, 'o', 'float'],         # see manual
        'vf_coef': [0.5, 'o', 'float'],           # see manual
        'max_grad_norm': [0.5, 'o', 'float'],     # see manual
        'lam': [0.95, 'o', 'float'],              # see manual
        'nminibatches': [4, 'o', 'int'],          # see manual
        'noptepochs': [4, 'o', 'int'],            # see manual
        'cliprange': [0.2, 'o', 'float'],         # see manual
        'cliprange_vf': [0.2, 'o', 'float'],      # see manual
        
        # for plotting 
        'check_freq': [1000,'o', 'int'],         # see manual
        'avg_episodes': [20,'o', 'int'],         # see manual
        'tensorboard': [False,'o','bool'],       # tensorboard view option, see manual for usage 
        
        # for testing or continue modes
        'model_load_path': [None,'rs', 'str'],   # required only for mode=continue/test
        
        # only for testing
        'n_eval_episodes': [None,'rs', 'int'],   # required only for mode=test 
        'render': [False,'o', 'bool'],           # option to plot during test via self.render() 
        }
        
        #---------------------------------
        #A2C Card
        #---------------------------------
        self.a2c_dict={
        'mode': ['train', 'r', 'str'],  # either train, continue, or test
        'time_steps': [50000,'rs','int'], # required only for mode=continue/test 
        'flag': [False, 'o', 'bool'],   # this will be activated if READ A2C is found in input
        'casename': ['a2c', 'o', 'str'],           # prefix for logging
        'ncores': [1,'o','int'],                   # multiple cores are available
        'n_steps': [5,'o','int'],                  # see manual
        'gamma': [0.99, 'o', 'float'],             # see manual
        'learning_rate': [0.0007, 'o', 'float'],   # see manual
        'vf_coef': [0.25, 'o', 'float'],           # see manual
        'max_grad_norm': [0.5, 'o', 'float'],      # see manual
        'ent_coef': [0.01, 'o', 'float'],          # see manual
        'momentum': [0.0, 'o', 'float'],          # see manual
        'alpha': [0.99, 'o', 'float'],             # see manual
        'epsilon': [1e-5, 'o', 'float'],           # see manual
        'lr_schedule': ['constant', 'o', 'str'],   # see manual
        
        'model_load_path': [None,'rs', 'str'], # required only for mode=continue/test
        
        # for plotting
        'check_freq': [1000,'o', 'int'],        # see manual
        'avg_episodes': [20,'o', 'int'],        # see manual
        'tensorboard': [False,'o','bool'],      # tensorboard view option, see manual for usage
        
        # only for testing
        'n_eval_episodes': [None,'rs', 'int'], # required only for mode=test 
        'render': [False,'o', 'bool'],         # option to plot during test via self.render()         
        }

        #---------------------------------
        #ACER Card
        #---------------------------------
        self.acer_dict={
        'mode': ['train', 'r', 'str'],  # either train, continue, or test
        'time_steps': [50000,'rs','int'], # required only for mode=continue/test 
        'flag': [False, 'o', 'bool'],   # this will be activated if READ A2C is found in input
        'casename': ['acer', 'o', 'str'],           # prefix for logging
        'ncores': [1,'o','int'],                   # multiple cores are available
        'n_steps': [20,'o','int'],                  # see manual
        'gamma': [0.99, 'o', 'float'],             # see manual
        'learning_rate': [0.0007, 'o', 'float'],   # see manual
        'q_coef': [0.5, 'o', 'float'],           # see manual
        'max_grad_norm': [10, 'o', 'float'],      # see manual
        'ent_coef': [0.01, 'o', 'float'],          # see manual
        'alpha': [0.99, 'o', 'float'],             # see manual
        'lr_schedule': ['linear', 'o', 'str'],   # see manual
        'rprop_alpha': [0.99, 'o', 'float'],           # see manual
        'rprop_epsilon': [1e-5, 'o', 'float'],           # see manual
        'buffer_size': [5000, 'o', 'int'],           # see manual
        'replay_ratio': [4, 'o', 'float'],           # see manual
        'replay_start': [1000, 'o', 'int'],           # see manual
        'correction_term': [10, 'o', 'float'],           # see manual
        'trust_region': [True, 'o', 'bool'],           # see manual
        'delta': [1, 'o', 'float'],           # see manual
        
        'model_load_path': [None,'rs', 'str'], # required only for mode=continue/test
        
        # for plotting
        'check_freq': [1000,'o', 'int'],        # see manual
        'avg_episodes': [20,'o', 'int'],        # see manual
        'tensorboard': [False,'o','bool'],      # tensorboard view option, see manual for usage
        
        # only for testing
        'n_eval_episodes': [None,'rs', 'int'], # required only for mode=test 
        'render': [False,'o', 'bool'],         # option to plot during test via self.render()         
        }
        
        #---------------------------------
        # GA card
        #---------------------------------
        self.ga_dict={
        'flag': [False, 'o', 'bool'],     # this will be activated if READ GA is found in input
        'mode': [None, 'r', 'str'],       # either assign (casmo) or shuffle (simulate3)
        'pop': [None, 'r', 'int'],        # number of population surviving from lambda per generation
        'lambda': [None, 'o', 'int'],     # number of total population created per generation
        'ngen': [None, 'r', 'int'],       # total number of generations 
        'casename': ['ga', 'o', 'str'],   # prefix for logging/results
        'lbound': [None, 'o', 'vec'],     # see manual for definition
        'ubound': [None, 'o', 'vec'],     # see manual for definition
        'type': [None, 'o', 'strvec'],     # see manual for definition
        'kbs_path': [None, 'o', 'str'],  # path to use kbs
        'kbs_frac': [0.1, 'o', 'float'],  # path to use kbs
        'ncores': [1,'o','int'],          # only 1 core
        'indpb': [0.05,'o','float'],      # independent prob for input attribute swap
        'cxpb': [0.5, 'o', 'float'],      # crossover prob
        'mutpb': [0.2, 'o', 'float'],      # mutation prob
        'smin': [0.01, 'o', 'float'],      # minimum strategy value in strategy vector
        'smax': [0.5, 'o', 'float'],      # maximum strategy value in strategy vector
        'check_freq': [1, 'o', 'int']    # update master log every check_freq (min is 1, max is ngen)
        }
        
        #---------------------------------
        # SA card
        #---------------------------------
        self.sa_dict={
        'flag': [False, 'o', 'bool'],     # this will be activated if READ SA is found in input
        'steps': [None, 'r', 'int'],      # total number of annealing steps (env calls)
        'casename': ['sa', 'o', 'str'],   # prefix for logging/results
        'ncores': [1,'o','int'],          # only 1 core
        'cooling': ['fast','o','str'],
        'swap': ['kbs', 'r', 'str'], # either singleswap, dualswap, quadswap, or fullswap, kbs
        'indpb_kbs': [0.5, 'o', 'float'], # prob to use kbs
        'kbs_path': [None, 'rs', 'str'], # path to kbs dataset
        'indpb': [0.05,'o','float'],      # independent prob for input attribute swap
        'lbound': [None, 'o', 'vec'],     # see manual for definition
        'ubound': [None, 'o', 'vec'],     # see manual for definition
        'type': [None, 'o', 'strvec'],     # see manual for definition
        'Tmin': [1, 'o', 'int'],       # Min T to end the annealing process 
        'Tmax': [25000,'o','int'],         # Max T to start the annealing process 
        'initstate': [None,'o','vec'],    # First state, random sample by default
        'check_freq': [50, 'o', 'int'],    # update master log every check_freq
        'avg_step': [50, 'o', 'int']      # update master log every check_freq (min is 1, max is steps)
        }
