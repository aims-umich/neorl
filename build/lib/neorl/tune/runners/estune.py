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

import numpy as np
import pandas as pd
import os
import random
import math
from itertools import repeat
import itertools
import sys, copy, shutil
import subprocess
from multiprocessing.dummy import Pool
from collections import defaultdict
import copy

import random
import matplotlib.pyplot as plt

try: 
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

class ESTUNE:
    """
    A class to parse neorl input template and construct cases for evolution strategy (ES) hyperparameter optimisation

    inputs: 
    The template input file
    Class object from PARSER.py, featuring user input for TUNE
    neorl logo
    """

    def __init__(self, tuneclass, inputfile, tuneblock, logo):
        self.logo=logo
        self.inputfile=inputfile
        self.tuneblock=tuneblock
        self.n_last_episodes=int(self.tuneblock["n_last_episodes"])
        self.ncores=int(self.tuneblock["ncores"])
        self.ncases=int(self.tuneblock["ncases"])

        #---------------------------------------
        # define genetic algorithm parameters
        #---------------------------------------
        self.popsize=10
        if self.ncases < self.popsize:
            self.ngens=1
        else:
            self.ngens=int(self.ncases/self.popsize)
        self.MU=5
        if tuneclass == 'gatune': # ES/GA tune
            print("Performing semi-GA Tune")
            self.INDPB=0.1
        elif tuneclass == 'estune': # ES tune
            print("Performing ES Tune")
            self.INDPB=1.0
        else: # default setting is ES tune
            print("Performing ES Tune")
            self.INDPB=1.0
        self.CXPB=0.5
        self.MUTPB=0.2
        self.ETA=0.6
        self.SMAX=0.5
        self.paramvals=dict()
        self.paraminds=dict()
        self.datatypes=[]

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
        for i in range(len(self.template)):
            if ('READ TUNE' in self.template[i]):
                first=i
            if ('END TUNE' in self.template[i]):
                last=i
        if first == 0 and last ==0:
            raise ('TUNE card cannot be found')

        del self.template[first: last+1]
        self.template="".join(self.template)

    def tune_count(self):
        
        """
        1- This function uses self.tuneblock, parse it, infer all parameters to be tuned and thier distribution
        2- This function creates GA engine and instantiates the initial population for evolution algorithm
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
                        self.datatypes.append("float")
                        print ('-- debug: parameter {} has uniform distribution of type --float-- between {} and {}'.format(item,item_lst[0],item_lst[1]))
                    elif "u" in item_lst: 
                        item_lst[0]=float(item_lst[0])
                        item_lst[1]=float(item_lst[1])
                        self.datatypes.append("float")
                        print ('-- debug: parameter {} has uniform distribution of type --float-- between {} and {}'.format(item,item_lst[0],item_lst[1]))
                except:
                    raise Exception ('--error: TUNE cannot construct the user-given uniform distribution of --floats-- for {} according to (low, high, u) syntax'.format(item))
               
                #---------------------------------------------------
                # check if a random integer distribution is identified
                #---------------------------------------------------
                try:
                    if "int" in item_lst:
                        item_lst[0]=int(item_lst[0])
                        item_lst[1]=int(item_lst[1])
                        self.datatypes.append("int")
                        print ('-- debug: parameter {} has uniform distribution of type --int-- between {} and {}'.format(item,item_lst[0],item_lst[1]))
                    elif "randint" in item_lst:
                        item_lst[0]=int(item_lst[0])
                        item_lst[1]=int(item_lst[1])
                        self.datatypes.append("int")
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
                        self.datatypes.append("grid")
                        print ('-- debug: parameter {} has grid type with values {}'.format(item,item_lst))
                except:
                    raise Exception ('--error: TUNE cannot construct the user-given grid for {} according to the comma-seperated syntax'.format(item))

                self.param_dict[item]=item_lst # Save the final parsed list for parameter {XXX}            
        
        #-----------------------------------------------------
        # infer the bounds for strategy vector 
        #-----------------------------------------------------
        if len(self.param_dict.keys()) <= 10:
            self.SMIN=0.1
        else:
            self.SMIN=1/(len(self.param_dict.keys()))

    def gen_cases(self, x=0):
        """
        This function infers neorl.py path
        """
        self.tune_count()
        self.param_names=list(self.param_dict.keys())
        #-----------------------        
        # Infer neorl.py path
        #-----------------------
        # Find neorl path
        #self.here=os.path.dirname(os.path.abspath(__file__))
        #self.neorl_path=self.here.replace('src/tune','neorl.py') #try to infer neorl.py internally to call neorl inside or neorl
        #self.python_path=self.here.replace('neorl/src/tune','anaconda3/bin/python3') #try to infer python3 path to call neorl inside or neorl

        self.neorl_path=sys.argv[0]
        self.python_path=sys.executable
        print('--debug: NEORLPATH=', self.neorl_path)
        print('--debug: PYTHONPATH=', self.python_path)
        
    def GenES(self):
        """
        Individual generator:
        1- This function uses self.param_dict to obtain bounds for individual parameters
        Returns:
            -ind (list): an individual vector with values samples from inferred distribution 
            -strategy (list): the strategy vector with values between smin and smax     
        """   
        size=len(self.param_dict.keys()) # size of individual
        content=[]
        self.LOW=[] # Lower bounds for the parameters to be tuned
        self.UP=[] # Upper bounds for parameters to be tuned
        for key in list(self.param_dict.keys()):
            if 'int' in self.param_dict[key]:
                content.append(random.randint(self.param_dict[key][0], self.param_dict[key][1]))
            elif 'randint' in self.param_dict[key]:
                content.append(random.randint(self.param_dict[key][0], self.param_dict[key][1]))
            elif 'float' in self.param_dict[key]:
                content.append(random.uniform(self.param_dict[key][0], self.param_dict[key][1]))
            elif 'u' in self.param_dict[key]:
                content.append(random.uniform(self.param_dict[key][0], self.param_dict[key][1]))
            elif 'grid' in self.param_dict[key]:
                self.real_grid=list(self.param_dict[key])
                self.real_grid.remove('grid') # get rid of the 'grid' to avoid sampling it
                self.paramvals[key]=self.real_grid
                content.append(random.sample(self.real_grid, 1)[0])
                self.paraminds[len(content)-1]=key
            else:
                raise Exception('unknown data type is given, either int/randint, float/u, or grid are allowed for parameter distribution types')
            self.LOW.append(self.param_dict[key][0])
            self.UP.append(self.param_dict[key][1])
        ind=list(content)
        size = len(list(self.param_dict.keys()))
        strategy= [random.uniform(self.SMIN, self.SMAX) for _ in range(size)]
        return ind, strategy
    
    def init_pop(self):
        """
        Population initializer
        Returns:
            -pop (dict): initial population in a dictionary form  
        """
        # initialize the population and strategy and run them in parallel (these samples will be used to initialize the memory)
        pop=defaultdict(list)
        
        for i in range(self.popsize):
            #caseid='es_gen{}_ind{}'.format(0,i+1)
            data=self.GenES()
            pop[i].append(data[0])
            pop[i].append(data[1])
        
        if self.ncores > 1: # evaluate warmup in parallel
            core_list=[]
            for key in pop:
                caseid='ind{}'.format(key+1)
                core_list.append([pop[key][0], caseid])
            p=Pool(self.ncores)
            fitness=p.map(self.gen_object, core_list)
            p.close(); p.join()

            [pop[ind].append(fitness[ind]) for ind in range(len(pop))]
        
        else: # evaluate warmup in series
            for key in pop:
                caseid='ind{}'.format(key+1)
                fitness=self.fit(pop[key][0], caseid)
                pop[key].append(fitness)
        return pop # return final pop dictionary with ind, strategy, and fitness

    def fit(self, ind, caseid):
        """
        This function evaluates an individual's fitness
        Inputs:
            -ind (list): an individual whose fitness to evaluate
            -caseid (str): a string that specifies the given individual
        Returns: 
            -mean_reward (float): fitness value 
        """
        try:
            #---------------------------------------------
            # Prepares directories and files for one case
            # --------------------------------------------
            self.param_names=list(self.param_dict.keys())
            i = caseid[3:]

            os.makedirs('./tunecases/case{}'.format(i), exist_ok=True)
            self.new_template=copy.deepcopy(self.template)
            for j in range (len(self.param_names)):
                self.new_template=self.new_template.replace(str(self.param_names[j]), str(ind[j]))
            
            filename='./tunecases/case{}/case{}.inp'.format(i, i)
            with open (filename, 'w') as fout:
                fout.writelines(self.new_template)
                
            # copy external files into the new directory, if extfiles card exists
            if 'extfiles' in self.tuneblock.keys():
                if self.tuneblock['extfiles']:
                    print('--debug: external files are identified, copying them into each case directory')
                    for item in self.tuneblock['extfiles']:
                        os.system('cp -r {} ./tunecases/case{}/'.format(item, i))

            casenum = caseid[3:]
            print('--------------------------------------------------')
            print('Running TUNE Case {}/{}: {}'.format(casenum, self.ncases, ind))
            subprocess.call([self.python_path, self.neorl_path, '-i', 'case{}.inp'.format(casenum)], cwd='./tunecases/case{}/'.format(casenum))  # this exceutes neorl for this case.inp
            print('--------------------------------------------------')
            
            #--------------------------------------------------------------------------------------------------------------
            # Try to infer the _out.csv file in the directory since only one method is allowed
            csvfile=[f for f in os.listdir('./tunecases/case{}/case{}_log/'.format(casenum, casenum)) if f.endswith('_out.csv')]
            if len(csvfile) > 1:
                raise Exception ('multiple *_out.csv files can be found in the logger of TUNE, only one is allowed')
            #--------------------------------------------------------------------------------------------------------------
            reward_lst=pd.read_csv('./tunecases/case{}/case{}_log/{}'.format(casenum,casenum, csvfile[0]), usecols=['reward']).values
            mean_reward=np.mean(reward_lst[-self.n_last_episodes:])
            max_reward=np.max(reward_lst)
            
            with open (self.csvlogger, 'a') as fout:
                fout.write(str(casenum) +',')
                [fout.write(str(item) + ',') for item in ind]
                fout.write(str(mean_reward) + ',' + str(max_reward) + '\n')
                
            return mean_reward
        
        except:
            print('--error: case{}.inp failed during execution'.format(casenum))
            
            return 'case{}.inp:failed'.format(casenum)
            
    def gen_object(self, inp):
        """
        This is a worker for the multiprocess Pool 
        Inputs:
            -inp (list of lists): contains data for each core [[ind1, caseid1], ...,  [indN, caseidN]]
        Returns: 
            -fitness value (float)
        """
        return self.fit(inp[0], inp[1])
    
    def select(self, pop):
        """
        Selection function sorts the population from max to min based on fitness and selects the k best
        Inputs:
            -pop (dict): population in dictionary structure
            -k (int): top k individuals are selected
        Returns:
            -best_dict (dict): the new orded dictionary with top k selected
        """
        k=self.MU
        pop=list(pop.items())
        pop.sort(key=lambda e: e[1][2], reverse=True)
        sorted_dict=dict(pop[:k])

        # This block creates a new dict where keys are reset to 0 ... k in order to avoid unordered keys after sort 
        best_dict=defaultdict(list)
        index=0
        for key in sorted_dict:
            best_dict[index].append(sorted_dict[key][0])
            best_dict[index].append(sorted_dict[key][1])
            best_dict[index].append(sorted_dict[key][2])
            index+=1 

        sorted_dict.clear()
        return best_dict
    
    def cx(self, ind1, ind2, strat1, strat2):
        """
        Executes a classical two points crossover on both the individuals and their strategy. 
        The individuals/strategies should be a list. The crossover points for the individual and the 
        strategy are the same. 

        Inputs:
            -ind1 (list): The first individual participating in the crossover. 
            -ind2 (list): The second individual participating in the crossover.
            -strat1 (list): The first evolution strategy participating in the crossover. 
            -strat2 (list): The second evolution strategy 
        Returns:
            - The new ind1, ind2, strat1, strat2, after crossover in list form
        """
        
        #for item in ind1:
        #    print('individual 1', type(item))
        #for item in ind2:
        #    print('individual 2', type(item))
        #for item in strat1:
        #    print('strategy 1', type(item))
        #for item in strat2:
        #    print('strategy 2', type(item))
        
        size = min(len(ind1), len(ind2))

        pt1 = random.randint(1, size)
        pt2 = random.randint(1, size-1)
        if pt2 >= pt1:
            pt2 +=1
        else:
            pt1, pt2 = pt2, pt1
        
        ind1[pt1:pt2], ind2[pt1:pt2] = ind2[pt1:pt2], ind1[pt1:pt2]
        strat1[pt1:pt2], strat2[pt1:pt2] = strat2[pt1:pt2], strat1[pt1:pt2]

        return ind1, ind2, strat1, strat2 
    
    def mutES(self, ind, strat):
        """
        Mutate an evolution strategy according to mixed Discrete/Continuous mutation rules 
        Input:
            -ind (list): individual to be mutated
            -strat (list): individual strategy to be mutated 
        Returns:
            -ind (list): new individual after mutation
            -strat (list): individual strategy after mutation
        """
        size=len(ind)
        tau=1/np.sqrt(2*size)
        tau_prime=1/np.sqrt(2*np.sqrt(size))
        
        for i in range(size):
            # Grid distribution received
            if self.datatypes[i] == "grid":
            #if i in self.paraminds.keys():
                norm=random.gauss(0,1)
                # modify the ind strategy
                strat[i] = 1/(1+(1-strat[i])/strat[i]*np.exp(-tau*norm-tau_prime*random.gauss(0,1)))
                # make a transformation of strategy to ensure it is between smin, smax 
                y=(strat[i]-self.SMIN)/(self.SMAX-self.SMIN)
                if np.floor(y) % 2 == 0:
                    y_prime=np.abs(y-np.floor(y))
                else:
                    y_prime=1-np.abs(y-np.floor(y))
                strat[i] = self.SMIN + (self.SMAX-self.SMIN)*y_prime

                # check if this attribute is mutated based on the updated strategy
                if random.random() < strat[i]:
                    # make a list of possibilities after excluding the current value to enforce mutation
                    paramname=self.paraminds[i]
                    ind[i]=random.sample(self.paramvals[paramname], 1)[0]

            # Random integer distribution received
            elif self.datatypes[i] == "int":
                norm=random.gauss(0,1)
                # modify the ind strategy 
                strat[i] = 1/(1+(1-strat[i])/strat[i]*np.exp(-tau*norm-tau_prime*random.gauss(0,1)))
                # make a transformation of strategy to ensure it is between smin, smax 
                y=(strat[i]-self.SMIN)/(self.SMAX-self.SMIN)
                if np.floor(y) % 2 == 0:
                    y_prime=np.abs(y-np.floor(y))
                else:
                    y_prime=1-np.abs(y-np.floor(y))
                strat[i] = self.SMIN + (self.SMAX-self.SMIN)*y_prime

                # check if this attribute is mutated based on the updated strategy 
                #if random.random() < strat[i]:
                # make a list of possibilities after excluding the current value to enforce mutation
                choices=list(range(self.LOW[i], self.UP[i]+1))
                choices.remove(ind[i])
                ind[i] = random.choice(choices)

            # Uniform float distribution received
            elif self.datatypes[i] == "float":
                norm=random.gauss(0,1)
                if random.random() < self.INDPB: # this indicates whether ind/strategy to be mutated or not for this float variable
                    strat[i] *= np.exp(tau*norm + tau_prime * random.gauss(0,1)) # normal mutation strategy
                    ind[i] += strat[i] * random.gauss(0,1) # update the individual position
                
                # check if the new individual falls within lower/uppder boundaries
                if ind[i] < self.LOW[i]:
                    ind[i] = self.LOW[i]
                if ind[i] > self.UP[i]:
                    ind[i] = self.UP[i]
                
            else:
                raise Exception('ES mutation strategy works with int, float, or grid distributions, the type provided cannot be interpreted')
        
        return ind, strat
    
    def GenOffspring(self, pop):
        """
        This function generates the offspring by applying crossover, mutation, OR reproduction. 
        Inputs:
            -pop (dict): population in dictionary structure
        Returns:
            -offspring (dict): new modified population in dictionary structure
        """

        pop_indices=list(range(0,len(pop)))
        offspring=defaultdict(list)
        for i in range(self.popsize):
            alpha=random.random()
            #----------------------
            # Crossover
            #----------------------
            if alpha < self.CXPB:
                index1, index2=random.sample(pop_indices,2)
                ind1, ind2, strat1, strat2=self.cx(ind1=list(pop[index1][0]), ind2=list(pop[index2][0]),
                                                        strat1=list(pop[index1][1]), strat2=list(pop[index2][1]))
                offspring[i].append(ind1)
                offspring[i].append(strat1)
                #print('crossover is done for sample {} between {} and {}'.format(i,index1,index2))
            #----------------------
            # Mutation
            #----------------------
            elif alpha < self.CXPB + self.MUTPB:  # Apply mutation
                index = random.choice(pop_indices)
                
                ind, strat=self.mutES(ind=list(pop[index][0]), strat=list(pop[index][1]))
                offspring[i].append(ind)
                offspring[i].append(strat)
                #print('mutation is done for sample {} based on {}'.format(i,index))
            #------------------------------
            # Reproduction from population
            #------------------------------
            else:
                index=random.choice(pop_indices)
                offspring[i].append(pop[index][0])
                offspring[i].append(pop[index][1])
                #print('reproduction is done for sample {} based on {}'.format(i,index))
        return offspring 

    def run_cases(self):
        """
        This function runs the evolutioanry algorithm over self.ngens generations. 
        """
        #------------------------------
        # Begin the evolution process
        #------------------------------
        with open (self.csvlogger, 'w') as fout:
            fout.write('caseid, ')
            [fout.write(item + ',') for item in self.param_names]
            fout.write('mean_reward,max_reward\n')

        #print('PARAM dict', self.param_dict)
        #print('PARAM types', self.datatypes)
        self.population=self.init_pop()
        case_idx=0
        self.currentcase=self.popsize+1
        for gen in range(1, self.ngens): 
            case_idx=0
            caseids=['ind{}'.format(ind) for ind in range(self.currentcase, self.currentcase+self.popsize+1)]
            # Vary the population and generate new offspring
            offspring=self.GenOffspring(pop=self.population)

            # Evaluate the individuals with invalid fitness using multiprocessing Pool
            if self.ncores > 1:
                core_list=[]
                for key in offspring:
                    core_list.append([offspring[key][0], caseids[case_idx]])
                    case_idx+=1
                # initialize a pool
                p=Pool(self.ncores)
                fitness=p.map(self.gen_object, core_list)
                p.close(); p.join()

                [offspring[ind].append(fitness[ind]) for ind in range(len(offspring))]
            else:
                for ind in range(len(offspring)):
                    fitness=self.fit(offspring[ind][0], caseids[case_idx])
                    case_idx+=1
                    offspring[ind].append(fitness)
            
            self.currentcase+=self.popsize
            # Select the next generation population 
            self.population = copy.deepcopy(self.select(pop=offspring))
        

        csvdata=pd.read_csv('tune.csv')
        asc_data=csvdata.sort_values(by=['caseid'],ascending=True)
        des_data=csvdata.sort_values(by=['mean_reward'],ascending=False)
        des_data2=csvdata.sort_values(by=['max_reward'],ascending=False)
        asc_data.to_csv('tune.csv', index=False)

        mean = np.mean(des_data.iloc[:,4:5])
        totalmean=mean.tolist()[0]
        
        try:
            failed_cases=len([print ('failed') for item in self.population if isinstance(item, str)])
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