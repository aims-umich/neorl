# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:54:11 2021

@author: majdi
"""

import stable_baselines
from stable_baselines.common.env_checker import check_env
from envs.hit6x6 import Casmo4Env
from neorl import PPO2
from neorl.rl.baselines.shared.policies import MlpPolicy
#from neorl.rl.baselines.ppo2.ppo2 import PPO2
#from neorl.rl.baselines.shared import set_global_seeds

env_data={
          #----------------
          #objectives
		  #----------------
          "kinf": 1.25,          #optimal value of kinf 
          "ppf": 1.35,           #max value of PPF
          "E": 2.3,              #optimal value of assembly average enrichment
          #----------------
          # options
          #----------------
          "lives": 4,            #number of assembly patterns to evaluate before terminating and restarting a new episode 
          "mode": "extdata",      #pick either casmo4 or extdata (if you have a precompiled dataset)
          "path": "data6x6.pkl",     #if mode is extdata, provide path to data library, if mode is casmo4, set path to casmo4e binary
          }

env=Casmo4Env(casename='ppo', log_dir='ppo_log/', env_data=env_data, env_seed=1)
# Instantiate the env
#check_env(Casmo4Env)

# Define and Train the agent
model = PPO2(policy=MlpPolicy, env=env).learn(total_timesteps=36000)