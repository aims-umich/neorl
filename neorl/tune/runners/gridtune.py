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
#"""
#Created on Wed Mar  4 11:51:22 2020
#
#@author: majdi
#"""

import numpy as np
import pandas as pd
import os
import itertools
import sys, copy, shutil
import subprocess
from multiprocessing import Pool

class GRIDTUNE:
    """
    A class to parse neorl input template and construct cases for discrete hyperparameter optimisation

    inputs: 
    The template input file
    Class object from PARSER.py
    neorl logo
    """
    def __init__(self, inputfile, tuneblock, logo):
        self.logo=logo
        self.inputfile=inputfile
        self.tuneblock=tuneblock
        self.n_last_episodes=int(self.tuneblock["n_last_episodes"])
        self.ncores=int(self.tuneblock["ncores"])
                    
        # results directory
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
        This function uses self.tuneblock, parse it, infer all parameters to be tuned and thier distributions
        """
        
        self.param_lst=[]; self.param_names=[]
        for item in self.tuneblock:
            if '{' in item and '}' in item and item[0] != '#':
                element_lst=[]
                for element in self.tuneblock[item].split(","):
                    not_int=0
                    try:
                        element_lst.append(int(element.strip()))
                    except Exception:
                        not_int=1
                    
                    if not_int:
                        try:
                            element_lst.append(float(element.strip()))
                        except Exception:
                            element_lst.append(str(element.strip()))
                        
                        
                self.param_lst.append(element_lst)  #all tuned parameter values
                self.param_names.append(item)  #all tuned parameter names
        
                #check the existence of the name in the template:
                if item not in self.template:
                    raise ValueError('parameter {} in TUNE block cannot be found in any other block, e.g. DQN, GA, PPO, etc.'.format(item)) 
                
        #count all possible combinations      
        self.all_combine = list(itertools.product(*self.param_lst)) # * here helps passing list of lists to product function without need to know the size of parameters beforehand
        if len(self.all_combine) > 1e6:
            print ('Number of possible combinations for this tunning problem is huge ({})!!!'.format(int(len(self.all_combine))))
            answer=input ("Do you want to proceed? Enter yes or no: ")
            
            if answer == 'no':
                print('user asked to terminate the run')
                sys.exit()
            elif answer == 'yes':
                print('user decided to generate all cases')
            else:
                raise Exception ('please enter yes or no')
                
    def gen_cases(self):
        
        """
        1- This function prepares the directories and files for all cases
        2- It replaces the {XXX} with the sampled value
        """
        
        self.tune_count()
        
        
        for i in range (len(self.all_combine)):
            os.makedirs('./tunecases/case{}'.format(i+1), exist_ok=True)
            self.new_template=copy.deepcopy(self.template)
            for j in range (len(self.param_names)):
                self.new_template=self.new_template.replace(str(self.param_names[j]), str(self.all_combine[i][j]))
            
            filename='./tunecases/case{}/case{}.inp'.format(i+1, i+1)
            with open (filename, 'w') as fout:
                fout.writelines(self.new_template)
             
            # copy external files into the new directory, if extfiles card exists
            if 'extfiles' in self.tuneblock.keys():
                if self.tuneblock['extfiles']:
                    print('--debug: external files are identified, copying them into each case directory')
                    for item in self.tuneblock['extfiles']:
                        os.system('cp -r {} ./tunecases/case{}/'.format(item, i+1))
                
        # Find neorl path
        #self.here=os.path.dirname(os.path.abspath(__file__))
        #self.neorl_path=self.here.replace('src/tune','neorl.py') #try to infer neorl.py internally to call neorl inside or neorl
        #self.python_path=self.here.replace('neorl/src/tune','anaconda3/bin/python3') #try to infer python3 path to call neorl inside or neorl
        self.neorl_path=sys.argv[0]
        self.python_path=sys.executable
        print('--debug: NEORLPATH=', self.neorl_path)
        print('--debug: PYTHONPATH=', self.python_path)
         
    def case_object(self,x):
        
        """
        This function setup a case object to pass to the multiproc Pool
        """
        
        try:
            print('--------------------------------------------------')
            print('Running TUNE Case {}/{}: {}'.format(x, len(self.all_combine), self.all_combine[x-1]))
            subprocess.call([self.python_path, self.neorl_path, '-i', 'case{}.inp'.format(x)], cwd='./tunecases/case{}/'.format(x))
            print('--------------------------------------------------')
            
            #--------------------------------------------------------------------------------------------------------------
            # Try to infer the _out.csv file in the directory since only one method is allowed
            csvfile=[f for f in os.listdir('./tunecases/case{}/case{}_log/'.format(x, x)) if f.endswith('_out.csv')]
            if len(csvfile) > 1:
                raise Exception ('multiple *_out.csv files can be found in the logger of TUNE, only one is allowed')
            #--------------------------------------------------------------------------------------------------------------
            reward_lst=pd.read_csv('./tunecases/case{}/case{}_log/{}'.format(x, x, csvfile[0]), usecols=['reward']).values
            mean_reward=np.mean(reward_lst[-self.n_last_episodes:])
            max_reward=np.max(reward_lst)
            with open (self.csvlogger, 'a') as fout:
                fout.write(str(x) +',')
                [fout.write(str(item) + ',') for item in self.all_combine[x-1]]
                fout.write(str(mean_reward) + ',' + str(max_reward) + '\n')
            
            return mean_reward
        
        except:
            print('--error: case{}.inp failed during execution'.format(x))
            
            return 'case{}.inp:failed'.format(x)
        
        
    def run_cases(self):
        
        """
        This function calls multiproc Pool to run all cases, and collect their stats
        """
        
        with open (self.csvlogger, 'w') as fout:
            fout.write('caseid, ')
            [fout.write(item + ',') for item in self.param_names]
            fout.write('mean_reward,max_reward\n')

        
        p=Pool(self.ncores)
        results = p.map(self.case_object, range(1,len(self.all_combine)+1))
        p.close()
        p.join()
        
        csvdata=pd.read_csv('tune.csv')
        asc_data=csvdata.sort_values(by=['caseid'],ascending=True)
        des_data=csvdata.sort_values(by=['mean_reward'],ascending=False)
        des_data2=csvdata.sort_values(by=['max_reward'],ascending=False)
        asc_data.to_csv('tune.csv', index=False)
        
        try:
            failed_cases=len([print ('failed') for item in results if isinstance(item, str)])
        except:
            failed_cases='NA'
        
        print ('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print(results)
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
            fout.write('Number of cases evaluated: {} \n'.format(len(self.all_combine)))
            fout.write('Number of failed cases: {} \n'.format(failed_cases))
            fout.write('Parameter names: {} \n'.format(self.param_names))
            fout.write('Parameter values: {} \n '.format(self.param_lst))
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
        
