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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:36:46 2020

@author: alyssawang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:51:22 2020

@author: majdi
"""

import numpy as np
import pandas as pd
import os
import random
import itertools
import sys, copy, shutil
import time
import pickle
import subprocess
from multiprocessing import Pool
from skopt import Optimizer

from skopt import gp_minimize, dump, load
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_convergence
from skopt.utils import use_named_args
import random
import numpy 
import matplotlib.pyplot as plt

class BAYESTUNE:
    
    """
    A class to parse neorl input template and construct cases for bayesian optimization for hyperparameters

    inputs: 
    The template input file
    Class object from PARSER.py, featuring user input for TUNE
    neorl logo
    """
    
    def __init__(self, inputfile, tuneblock, logo):
        self.logo=logo
        self.inputfile=inputfile
        self.tuneblock=tuneblock
        self.n_last_episodes=int(self.tuneblock["n_last_episodes"])
        self.ncores=int(self.tuneblock["ncores"])
        self.ncases=int(self.tuneblock["ncases"])
                
        #---------------------------------------
        # define bayesian optimization parameters
        #---------------------------------------
        self.caseindex=0 # for creating case directory purposes 
        
        #-------------------------------
        # construct results directory
        #-------------------------------
        if os.path.exists('./tunecases/'):
            shutil.rmtree('./tunecases/')
            os.makedirs('./tunecases/', exist_ok=True)
        else:
            os.makedirs('./tunecases/', exist_ok=True)
        self.csvlogger='tune.csv'
        self.tunesummary='tunesummary.txt'
        
        #---------------------------------
        # parse the input template
        #---------------------------------
        with open (self.inputfile, 'r') as input_file_text:
            self.template=input_file_text.readlines()
            
        first=0; last=0
        for i in range (len(self.template)):
            if ('READ TUNE' in self.template[i]):
                first=i
            if ('END TUNE' in self.template[i]):
                last=i
        if first == 0 and last==0:
            raise ('TUNE card cannot be found')
        
        del self.template[first : last+1]  
        self.template="".join(self.template)  
        
                     
    def tune_count(self):
        
        """
        1- This function uses self.tuneblock, parse it, infer all parameters to be tuned and thier distribution
        2- This function instantiates the search dimensions for the hyperparameters to be tuned
        """
        
        self.param_dict={}
        for item in self.tuneblock:
            if '{' in item and '}' in item and item[0] != '#':
                #-----------------------------------------------------
                # check the existence of the name in the template
                #-----------------------------------------------------
                if item not in self.template:
                    raise ValueError('parameter {} in TUNE block cannot be found in any other block, e.g. DQN, GA, PPO, etc.'.format(item)) 

                item_lst=self.tuneblock[item].split(",")
                item_lst=[item.strip() for item in item_lst] # get rid of white spaces in the splitted values
                #-------------------------------------------------------
                # check if a uniform distribution of floats is identified
                #-------------------------------------------------------
                try:
                    if "float" in item_lst:
                        item_lst[0]=float(item_lst[0])
                        item_lst[1]=float(item_lst[1])
                        print ('-- debug: parameter {} has uniform distribution of type --float-- between {} and {}'.format(item,item_lst[0],item_lst[1]))
                    elif "u" in item_lst:
                        item_lst[0]=float(item_lst[0])
                        item_lst[1]=float(item_lst[1])
                        print ('-- debug: parameter {} has uniform distribution of type --float-- between {} and {}'.format(item,item_lst[0],item_lst[1]))
                except:
                    raise Exception ('--error: TUNE cannot construct the user-given uniform distribution of --floats-- for {} according to (low, high, u) syntax'.format(item))
               
                #----------------------------------------------------
                # check if a random integer distribution is identified
                #----------------------------------------------------
                try:
                    if "int" in item_lst:
                        item_lst[0]=int(item_lst[0])
                        item_lst[1]=int(item_lst[1])
                        print ('-- debug: parameter {} has uniform distribution of type --int-- between {} and {}'.format(item,item_lst[0],item_lst[1]))
                    elif "randint" in item_lst:
                        item_lst[0]=int(item_lst[0])
                        item_lst[1]=int(item_lst[1])
                        print ('-- debug: parameter {} has uniform distribution of type --int-- between {} and {}'.format(item,item_lst[0],item_lst[1]))
                except:
                    raise Exception ('--error: TUNE cannot construct the user-given uniform distribution of --int-- for {} according to (low, high, u) syntax'.format(item))
        
               #-----------------------------------------------------
               # check if a grid is identified
               #-----------------------------------------------------
                try:
                    if "grid" in item_lst:
                        element_lst=[]
                        for element in item_lst:
                            # check if it is an integer
                            not_int=0
                            try:
                                element_lst.append(int(element.strip()))
                            except Exception:
                                not_int=1
                            
                            # else check if the elment is float
                            if not_int:
                                try:
                                    element_lst.append(float(element.strip()))
                                # else consider it a string
                                except Exception:
                                    element_lst.append(str(element.strip()))
                                    
                        item_lst=element_lst
                        print ('-- debug: parameter {} has grid type with values {}'.format(item,item_lst))
                except:
                    raise Exception ('--error: TUNE cannot construct the user-given grid for {} according to the comma-seperated syntax'.format(item))

                self.param_dict[item]=item_lst # Save the final parsed list for parameter {XXX}                    
                
        
        #------------------------------------------------------------------
        # Instantiate the search space dimensions for bayesian optimization
        #------------------------------------------------------------------
        #random.seed(64)
                
        self.dimensions = []            
        self.initialparams = [ [] for _ in range(self.ncores)]
          
        for key in list(self.param_dict.keys()):
            
            if ('float' in self.param_dict[key]) or ('u' in self.param_dict[key]):  
                # set dimension
                self.dimensions.append(Real(low=self.param_dict[key][0], high=self.param_dict[key][1]))
                # add to initial params
                for i in range(self.ncores):
                    self.initialparams[i].append(random.uniform(self.param_dict[key][0], self.param_dict[key][1]))
                
            if ('int' in self.param_dict[key]) or ('randint' in self.param_dict[key]):
                # set dimension
                self.dimensions.append(Integer(low=self.param_dict[key][0], high=self.param_dict[key][1]))
                # add to initial params
                for i in range(self.ncores):
                    self.initialparams[i].append(random.randint(self.param_dict[key][0], self.param_dict[key][1]))

            if 'grid' in self.param_dict[key]:
                self.real_grid=list(self.param_dict[key])
                self.real_grid.remove('grid') # get rid of the 'grid' to avoid sampling it
                # set dimension
                self.dimensions.append(Categorical(categories=self.real_grid))
                # add to initial params
                for i in range(self.ncores):
                    self.initialparams[i].append(random.sample(self.real_grid, 1)[0])
                
    def gen_cases(self, x=0):
        
        """
        This function infers neorl.py path 
        """
        
        self.tune_count()
        self.param_names=list(self.param_dict.keys())
        # Find neorl path
        #self.here=os.path.dirname(os.path.abspath(__file__))
        #self.neorl_path=self.here.replace('src/tune','neorl.py') #try to infer neorl.py internally to call neorl inside or neorl
        #self.python_path=self.here.replace('neorl/src/tune','anaconda3/bin/python3') #try to infer python3 path to call neorl inside or neorl
        self.neorl_path=sys.argv[0]
        self.python_path=sys.executable
        print('--debug: NEORLPATH=', self.neorl_path)
        print('--debug: PYTHONPATH=', self.python_path)

    def evalX(self, params):
        
        """
        This function evaluates objective function value, calls self.case_object(params) for evaluation
        Inputs: 
            - Single input point
        Outputs: 
            - Objective function value at the input point  
        """
        
        return self.case_object(params)
        
    def case_object(self,CASEPARAMS): 
        
        """
        This function sets up a case object for a single input point during bayesian optimization
        Inputs:
            - Single input point
        Outputs:
            - Negative mean reward for the input point (to be minimized to find maximal mean reward)
        """
        
        try:
            #--------------------------------------------------------------------------------------------------------------
            # Prepares directories and files for one case
            self.param_names=list(self.param_dict.keys())
            i = self.caseindex
            os.makedirs('./tunecases/case{}'.format(i+1), exist_ok=True)
            self.new_template=copy.deepcopy(self.template)
            for j in range (len(self.param_names)):
                self.new_template=self.new_template.replace(str(self.param_names[j]), str(CASEPARAMS[j]))
            
            filename='./tunecases/case{}/case{}.inp'.format(i+1, i+1)
            with open (filename, 'w') as fout:
                fout.writelines(self.new_template)
             
            # copy external files into the new directory, if extfiles card exists
            if 'extfiles' in self.tuneblock:
                if self.tuneblock['extfiles']:
                    print('--debug: external files are identified, copying them into each case directory')
                    for item in self.tuneblock['extfiles']:
                        os.system('cp -r {} ./tunecases/case{}/'.format(item, i+1))
            
            
            casenum = i+1
            print('--------------------------------------------------')
            print('Running TUNE Case {}/{}: {}'.format(i+1, self.ncases, CASEPARAMS))
            subprocess.call([self.python_path, self.neorl_path, '-i', 'case{}.inp'.format(casenum)], cwd='./tunecases/case{}/'.format(casenum))  # this exceutes neorl for this case.inp
            print('--------------------------------------------------')
            
            #--------------------------------------------------------------------------------------------------------------
            # Try to infer the _out.csv file in the directory since only one method is allowed
            csvfile=[f for f in os.listdir('./tunecases/case{}/case{}_log/'.format(casenum, casenum)) if f.endswith('_out.csv')]
            if len(csvfile) > 1:
                raise Exception ('multiple *_out.csv files can be found in the logger of TUNE, only one is allowed')
            #--------------------------------------------------------------------------------------------------------------
            
            reward_lst=pd.read_csv('./tunecases/case{}/case{}_log/{}'.format(casenum, casenum, csvfile[0]), usecols=['reward']).values
            mean_reward=np.mean(reward_lst[-self.n_last_episodes:])
            max_reward=np.max(reward_lst)
            
            with open (self.csvlogger, 'a') as fout:
                fout.write(str(casenum) +',')
                [fout.write(str(item) + ',') for item in CASEPARAMS]
                fout.write(str(mean_reward) + ',' + str(max_reward) + '\n')
            
            self.caseindex+=1 
            return -mean_reward
        
        except:
            print('--error: case{}.inp failed during execution'.format(casenum))
            
            return 'case{}.inp:failed'.format(casenum)
        
    
    def run_gp_minimize(self, initialx):
        
        """
        This function calls gp_minimize of a list of initial input points
        Inputs:
            - List of initial input points
        Outputs: 
            - The optimization result returned as a OptimizeResult scipy object 
        """
        
        self.caseindex=initialx[-1]
        initialx=initialx[:-1]
               
        return gp_minimize(func=self.evalX, 
                            dimensions=self.dimensions,
                            acq_func='EI',
                            n_calls=self.n_calls,
                            x0=initialx,
                            random_state=self.caseindex)
    
    def run_cases(self):
        
        """
        This function runs bayesian optimization and collects their stats
        """
        
        with open (self.csvlogger, 'w') as fout:
            fout.write('caseid, ')
            [fout.write(item + ',') for item in self.param_names]
            fout.write('mean_reward,max_reward\n')
        
        #random.seed(64)
        
        if self.ncores==1: 
            #------------------------------
            # running process over one core
            #------------------------------
            if self.ncases < 11:
                self.ncases = 11
            
            self.results = gp_minimize(func=self.evalX,
                                        dimensions=self.dimensions,
                                        acq_func='EI',
                                        n_calls=self.ncases,
                                        x0=self.initialparams)
        else:
            #---------------------------------------------
            # running process in parallel over multi cores
            #---------------------------------------------
            if int(self.ncases/self.ncores) < 11:
                self.n_calls = 11
            else:
                self.n_calls = int(self.ncases/self.ncores)
            index=0
            for params in self.initialparams:
                params.append(index)
                index+=self.n_calls
            
            p=Pool(self.ncores)
            self.results = p.map(self.run_gp_minimize, self.initialparams)
            p.close()
            p.join()
        
                    
        csvdata=pd.read_csv('tune.csv')
        asc_data=csvdata.sort_values(by=['caseid'],ascending=True)
        des_data=csvdata.sort_values(by=['mean_reward'],ascending=False)
        des_data2=csvdata.sort_values(by=['max_reward'],ascending=False)
        asc_data.to_csv('tune.csv', index=False)
    
        mean = np.mean(des_data.iloc[:,4:5])
        totalmean=mean.tolist()[0]
        try:
            failed_cases=len([print ('failed') for item in self.results["func_vals"] if isinstance(item, str)])
        except:
            failed_cases='NA'
        
        print ('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('Mean Rewards for all cases=', totalmean)
        print ('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print ('All TUNE CASES ARE COMPLETED')
        print ('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('--debug: Check tunesummary.txt file for best hyperparameters found')
        print('--debug: Check tune.csv file for complete csv logger of all cases results')
        print('--debug: Check tunecases directory for case-by-case detailed results')
        
        with open ('tunesummary.txt', 'w') as fout:
            
            #fout.write(self.logo)
            fout.write('*****************************************************\n')
            fout.write('Summary for the TUNE case \n')
            fout.write('*****************************************************\n')
            fout.write('Number of cases evaluated: {} \n'.format(self.ncases))
            fout.write('Number of failed cases: {} \n'.format(failed_cases))
            fout.write('Parameter names: {} \n'.format(self.param_names))
            fout.write('Parameter values: {} \n '.format(self.param_dict))
            fout.write ('--------------------------------------------------------------------------------------\n')
            if des_data.shape[0] < 20:
                top=des_data.shape[0]
                fout.write ('Top {} hyperparameter configurations ranked according to MEAN reward \n'.format(top))
                fout.write(des_data.iloc[:top].to_string(index=False))
            else:
                top=20
                fout.write ('Top {} hyperparameter configurations ranked according to MEAN reward \n'.format(top))
                fout.write(des_data.iloc[:top].to_string(index=False))
            fout.write ('\n')
            fout.write ('--------------------------------------------------------------------------------------\n')
            if des_data2.shape[0] < 20:
                top=des_data2.shape[0]
                fout.write ('Top {} hyperparameter configurations ranked according to MAX reward \n'.format(top))
                fout.write(des_data2.iloc[:top].to_string(index=False))
            else:
                top=20
                fout.write ('Top {} hyperparameter configurations ranked according to MAX reward \n'.format(top))
                fout.write(des_data2.iloc[:top].to_string(index=False))