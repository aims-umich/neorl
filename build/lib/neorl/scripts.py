"""
Created on Mon Jan 28 08:09:18 2019

@author: Majdi Radaideh
"""
    
import os, sys, warnings, random
warnings.filterwarnings("ignore")
sys.path.insert(0, './neorl/')
sys.path.insert(0, './neorl/rl')
sys.path.insert(0, './neorl/rl/baselines')
sys.path.insert(0, './neorl/utils')
sys.path.insert(0, './neorl/evolu')
sys.path.insert(0, './neorl/parsers')
import numpy as np
import gym
import time

from neorl.parsers.PARSER import InputParser, InputChecker
from neorl.parsers.TuneChecker import TuneChecker
from neorl.parsers.ParamList import InputParam
from neorl.utils.multiproc import MultiProc
from neorl.tune.gridtune import GRIDTUNE
from neorl.tune.randtune import RANDTUNE
from neorl.tune.bayestune import BAYESTUNE
from neorl.tune.estune import ESTUNE
from neorl.utils.initfiles import initfiles
  
def main():
    
    
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
    
    if len(sys.argv) < 3:
        raise Exception ("--error: NO input file after neorl is passed, try --> neorl NEORL_INPUT")
    else:
        print ("--------------------------------------------------------------- ")
        print ("The input file {} is passed".format(sys.argv[2]))
        print ("--------------------------------------------------------------- ")
    input_file_path=sys.argv[2]
    
    print("--debug: All modules are imported sucessfully")
    print ("--------------------------------------------------------------- ")
    
    parser=InputParser(input_file_path)
    log_dir=parser.blocks()
    paramdict=InputParam()
    
    
    try:
        if parser.tune_flag:
            tune_block=TuneChecker(parsed=parser.tune_block,default=paramdict.tune_dict).tune_dict
            #print(tune_block)
            # Activate Hyperparameter tunning mode
            if tune_block['method'] == 'gridtune':
                tune=GRIDTUNE(inputfile=input_file_path,tuneblock=tune_block, logo=logo)  #Grid search mode
            elif tune_block['method'] == 'randtune':
                tune=RANDTUNE(inputfile=input_file_path,tuneblock=tune_block, logo=logo)  #Random search mode
            elif tune_block['method'] == 'gatune':
                tune=ESTUNE(inputfile=input_file_path,tuneblock=tune_block, logo=logo, tuneclass='gatune')  #GA search mode
            elif tune_block['method'] == 'bayestune': # Bayes optimization search mode
                tune=BAYESTUNE(inputfile=input_file_path, tuneblock=tune_block, logo=logo)
            elif tune_block['method'] == 'estune': # evolution strategy tuning
                tune=ESTUNE(inputfile=input_file_path,tuneblock=tune_block, logo=logo, tuneclass='estune')  #ES search mode
            else:
                raise Exception ('--error: parameter ({}) in card {} does not have the value ({}), check manual for available methods'.format('method', 'TUNE', tune_block['method']))
    
            tune.gen_cases()    #Generate the TUNE cases 
            # Run the the tune cases if run mode (default) is activated
            if 'mode' not in tune_block.keys():
                tune.run_cases()
            elif tune_block['mode']=='run':
                tune.run_cases()
            elif tune_block['mode']=='test':
                print ('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                print('--- All TUNE cases are generated, to run cases, change TUNE mode to (run)')
                print ('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            else:
                raise Exception ('TUNE mode given by the user is not allowed {}, either `run` or `test`'.format(tune_block['mode']))
                        
        else:
            # Regular mode 
            t0=time.time()
            inp=InputChecker(parser,paramdict,log_dir)
            inp.setup_input()
            initfiles(methods=inp.methods, nx=inp.gen_dict['xsize_plot'][0], ny=inp.gen_dict['ysize'][0], 
                      inp_headers= inp.gen_dict['xnames'][0], out_headers=inp.gen_dict['ynames'][0], 
                      log_dir=inp.gen_dict['log_dir'], logo=logo)  # Intialize the all loggers
            print('------------------------------------------------------------------------------')
            print('--debug: Input check is completed successfully, no major error is found')
            print('------------------------------------------------------------------------------')
            print('------------------------------------------------------------------------------')
            
            if inp.gen_dict['neorl_mode'][0] == 'run':
                engine=MultiProc(inp)
                engine.run_all()
            elif inp.gen_dict['neorl_mode'][0] == 'test':
                print('neorl_mode is set to check input only, the input has no errors')
            elif inp.gen_dict['neorl_mode'][0] == 'env_only':
                print('neorl_mode is set to run env_only, attempting to run the enviroment')
                env = gym.make(inp.gen_dict['env'][0], log_dir=inp.gen_dict['log_dir'],
                                    exepath=inp.gen_dict['exepath'][0], env_data=inp.gen_dict['env_data'][0])
            else:
                raise Exception ('neorl_mode is either check or run, the one given by the user is not defined')
            
            
    except KeyboardInterrupt:  # to escape neorl gracefull with ctrl + c 
            sys.exit()
        