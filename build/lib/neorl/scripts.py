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

#"""
#Created on Mon Jan 28 08:09:18 2019
#
#@author: Majdi Radaideh
#"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import os, sys, random
sys.path.insert(0, './neorl/')
sys.path.insert(0, './neorl/rl')
sys.path.insert(0, './neorl/rl/baselines')
sys.path.insert(0, './neorl/utils')
sys.path.insert(0, './neorl/evolu')
sys.path.insert(0, './neorl/parsers')
import numpy as np
import gym
import time
import argparse

import neorl
import pytest
from neorl.parsers.PARSER import InputParser, InputChecker
from neorl.parsers.TuneChecker import TuneChecker
from neorl.parsers.ParamList import InputParam
from neorl.utils.multiproc import MultiProc
from neorl.tune.runners.gridtune import GRIDTUNE
from neorl.tune.runners.randtune import RANDTUNE
from neorl.tune.runners.bayestune import BAYESTUNE
from neorl.tune.runners.estune import ESTUNE
from neorl.utils.initfiles import initfiles
from neorl.version import version


    
simple_example="""
#***************************************************************************
# NEORL Simple Example
# Using DE to minimize the sphere function 
#***************************************************************************
from neorl import DE

#--Define the fitness function
def FIT(individual):
    #Sphere test objective function. 
    y=sum(x**2 for x in individual)
    return -y  #-1 is used to convert min to max

#--Define parameter space (d=5)
nx=5
BOUNDS={}
for i in range(1,nx+1):
    BOUNDS['x'+str(i)]=['float', -100, 100]

#--Differential Evolution
de=DE(bounds=BOUNDS, fit=FIT, npop=60, CR=0.7, F=0.5, ncores=1, seed=1)
x_best, y_best, de_hist=de.evolute(ngen=100, verbose=0)

#--Optimal Results
print('best x:', x_best)
print('best y:', y_best)
#***************************************************************************
"""
  
def main():
    
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
    
    
    logo=logo.encode('utf-8')
                           
    __version__=version()
    path=os.path.dirname(neorl.__file__)
    parser = argparse.ArgumentParser(description='NEORL command line API parser')
    parser.add_argument('-i', '--input', required=False, help='Name of the input ASCII file, e.g. INPUT.inp, INPUT.dat (required arg)')
    parser.add_argument('-c', '--check', help="check input file syntax and exit")
    parser.add_argument('-t', '--test', action='store_true', help="run NEORL units tests")
    parser.add_argument('-e', '--example', action='store_true', help='print a simple NEORL script and exit')
    parser.add_argument('-v', '--version', action='version', version='NEORL-'+__version__)
    args = parser.parse_args()
    
    if args.example:
        print(simple_example)
        sys.exit()
        
    if args.test:
        pytest.main(["-x", path, "-v", "--continue-on-collection-errors", "--maxfail=10"])
        sys.exit(0)
        
    if args.input:
        input_file_path=args.input
    elif args.test:
        input_file_path=args.test
    else:
        parser.print_help()
        sys.exit(0)
    
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
            
            if args.test:
                print('***************************************************************************')
                print('NEORL was run on -t/--test mode successfully and the syntax seems correct')
                print('***************************************************************************')
                sys.exit()
            
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
        