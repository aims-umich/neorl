# Copyright (c) 2021, NEORL authors.
# Licensed under the MIT license
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

logo="""

\t\t NEORL: NeuroEvolution Optimisation with Reinforcement Learning
\t\t\t ███╗   ██╗███████╗ ██████╗ ██████╗ ██╗     
\t\t\t ████╗  ██║██╔════╝██╔═══██╗██╔══██╗██║     
\t\t\t ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║     
\t\t\t ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║     
\t\t\t ██║ ╚████║███████╗╚██████╔╝██║  ██║███████╗
\t\t\t ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝                                                
Copyright © 2021 Exelon Corporation (https://www.exeloncorp.com/) in collaboration with 
             MIT Nuclear Science and Engineering (https://web.mit.edu/nse/)
                             All Rights Reserved

                       \n"""
                       
print(logo)

import os

from neorl.rl.baselines.a2c import A2C
from neorl.rl.baselines.acer import ACER
from neorl.rl.baselines.deepq import DQN
from neorl.rl.baselines.ppo2 import PPO2
from neorl.evolu.pso import PSO
from neorl.evolu.sa import SA
from neorl.evolu.de import DE
from neorl.evolu.xnes import XNES
from neorl.hybrid.pesa import PESA
from neorl.rl.baselines.shared.policies import MlpPolicy
from neorl.rl.baselines.deepq.policies import MlpPolicy as DQNPolicy

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), 'version.txt')
with open(version_file, 'r') as file_handler:
    __version__ = file_handler.read().strip()
