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

# Copyright (c) 2021, NEORL authors.
# Licensed under the MIT license
import warnings, os
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import tensorflow as tf
from tensorflow.python.util import deprecation
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
if type(tf.contrib) != type(tf): tf.contrib._warning = None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
deprecation._PRINT_DEPRECATION_WARNINGS = False

logo="""

\t    NEORL: NeuroEvolution Optimisation with Reinforcement Learning
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
                       

try:                    
    print(logo)
except:
    print(logo.encode('utf-8'))
    #print(logo.encode('ascii', 'ignore').decode('ascii'))
import os

from neorl.rl.baselines.a2c import A2C
from neorl.rl.baselines.acer import ACER
from neorl.rl.baselines.deepq import DQN
from neorl.rl.baselines.ppo2 import PPO2
from neorl.rl.baselines.acktr import ACKTR
from neorl.evolu.pso import PSO
from neorl.evolu.sa import SA
from neorl.evolu.bat import BAT
from neorl.evolu.de import DE
from neorl.evolu.xnes import XNES
from neorl.evolu.es import ES
from neorl.evolu.gwo import GWO
from neorl.evolu.ssa import SSA
from neorl.evolu.woa import WOA
from neorl.evolu.jaya import JAYA
from neorl.evolu.mfo import MFO
from neorl.evolu.hho import HHO
from neorl.hybrid.pesa import PESA
from neorl.hybrid.pesa2 import PESA2
from neorl.rl.baselines.shared.policies import MlpPolicy
from neorl.rl.baselines.deepq.policies import MlpPolicy as DQNPolicy
from neorl.utils.neorlcalls import RLLogger
from neorl.rl.make_env import CreateEnvironment
from neorl.hybrid.rneat import RNEAT
from neorl.hybrid.fneat import FNEAT
from neorl.hybrid.ppoes import PPOES
from neorl.hybrid.ackde import ACKDE
from neorl.evolu.aco import ACO
from neorl.evolu.cs import CS
from neorl.hybrid.nga import NGA
from neorl.hybrid.nhho import NHHO
from neorl.evolu.ts import TS