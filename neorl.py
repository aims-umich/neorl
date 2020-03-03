# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 08:09:18 2019

@author: Majdi Radaideh
"""


import numpy as np
import gym 
import subprocess 
import os
import time
import shutil 
import sys
import warnings
from deap import algorithms, base, creator, tools
warnings.filterwarnings("ignore")
sys.path.insert(0, './src/')
sys.path.insert(0, './src/rl')
sys.path.insert(0, './src/utils')
sys.path.insert(0, './src/evolu')

from ParamList import InputParam
from multiproc import MultiProc



print("""
             NEORL: Nuclear Engineering Optimisation with Reinforcement Learning
  
NNNNNNNN        NNNNNNNNEEEEEEEEEEEEEEEEEEEEEE     OOOOOOOOO     RRRRRRRRRRRRRRRRR   LLLLLLLLLLL             
N:::::::N       N::::::NE::::::::::::::::::::E   OO:::::::::OO   R::::::::::::::::R  L:::::::::L             
N::::::::N      N::::::NE::::::::::::::::::::E OO:::::::::::::OO R::::::RRRRRR:::::R L:::::::::L             
N:::::::::N     N::::::NEE::::::EEEEEEEEE::::EO:::::::OOO:::::::ORR:::::R     R:::::RLL:::::::LL             
N::::::::::N    N::::::N  E:::::E       EEEEEEO::::::O   O::::::O  R::::R     R:::::R  L:::::L               
N:::::::::::N   N::::::N  E:::::E             O:::::O     O:::::O  R::::R     R:::::R  L:::::L               
N:::::::N::::N  N::::::N  E::::::EEEEEEEEEE   O:::::O     O:::::O  R::::RRRRRR:::::R   L:::::L               
N::::::N N::::N N::::::N  E:::::::::::::::E   O:::::O     O:::::O  R:::::::::::::RR    L:::::L               
N::::::N  N::::N:::::::N  E:::::::::::::::E   O:::::O     O:::::O  R::::RRRRRR:::::R   L:::::L               
N::::::N   N:::::::::::N  E::::::EEEEEEEEEE   O:::::O     O:::::O  R::::R     R:::::R  L:::::L               
N::::::N    N::::::::::N  E:::::E             O:::::O     O:::::O  R::::R     R:::::R  L:::::L               
N::::::N     N:::::::::N  E:::::E       EEEEEEO::::::O   O::::::O  R::::R     R:::::R  L:::::L         LLLLLL
N::::::N      N::::::::NEE::::::EEEEEEEE:::::EO:::::::OOO:::::::ORR:::::R     R:::::RLL:::::::LLLLLLLLL:::::L
N::::::N       N:::::::NE::::::::::::::::::::E OO:::::::::::::OO R::::::R     R:::::RL::::::::::::::::::::::L
N::::::N        N::::::NE::::::::::::::::::::E   OO:::::::::OO   R::::::R     R:::::RL::::::::::::::::::::::L
NNNNNNNN         NNNNNNNEEEEEEEEEEEEEEEEEEEEEE     OOOOOOOOO     RRRRRRRR     RRRRRRRLLLLLLLLLLLLLLLLLLLLLLLL
                       \n""")
                       
print("All modules are imported sucessfully")


#if len(sys.argv) < 3:
#    raise Exception ("No input file after -i is passed, try --> python fuse.py -i NEORL_INPUT ")
#else:
#    print ("""---------------------------------------------------------------
#The input file \"%s\" is passed"
##---------------------------------------------------------------"""%(sys.argv[2]))
#ms_input=sys.argv[2]

# Uncomment these if you work directly from spyder!!!!

        
if __name__ == '__main__':
    
    # check if the log directory exists, move to old and create a new log
    
    if os.path.exists("./master_log/") and os.path.exists("./old_master_log/"):
        os.system('rm -Rf ./old_master_log/')
        os.rename('./master_log/','./old_master_log/')
        os.makedirs('./master_log/')
    elif os.path.exists("./master_log/"): 
        os.rename('./master_log/','./old_master_log/')
        os.makedirs('./master_log/')         
    else:
        os.makedirs('./master_log/') 
    
    inp=InputParam()
    master=MultiProc(inp)
    master.run_all()


