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
This class parses the master input file provided by the user
""" 
import multiprocessing, os, subprocess
import numpy as np
import json
from ast import literal_eval

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError # evil ValueError that doesn't tell you what the wrong value was

class InputParser():
    def __init__(self, input_file_path):
        
        #infer the casename from filename
        try:
            self.MainCaseName=input_file_path.split(".")[0]
            self.log_dir=self.MainCaseName+'_log/'
        except:
            self.log_dir=input_file_path+'_log/'
        #-----------------
        # method flags initialize
        #-----------------
        self.dqn_flag=False
        self.ga_flag=False
        self.a2c_flag=False
        self.acer_flag=False
        self.ppo_flag=False
        self.sa_flag=False
        self.tune_flag=False
        # add new block flag
        
        #-----------------
        #input blocks 
        #-----------------
        self.gen_block={}
        self.dqn_block={}
        self.ppo_block={}
        self.a2c_block={}
        self.acer_block={}
        self.ga_block={}
        self.sa_block={}
        self.tune_block={}
        # add new block 
        
        #------------------------------------------------------------------
        #Read the lines of the input file into the attribute input_file
        #Remove first line, empty lines and "\n" from every line
        #------------------------------------------------------------------
        #old way of parsing the input
#        with open(input_file_path) as input_file_text:
#            self.input_file = [parameter.replace("\n","").strip() for parameter in input_file_text.readlines() if not parameter.isspace() or parameter[0] != "#"]
        
        #-- Paul parser
        self.input_file = []
        input_file_text = open(input_file_path,'r', encoding='utf-8')
        while True:
            line1 = input_file_text.readline()
            if not line1:break
            else:
                if 'env_data' in line1:
                    line1 = line1[:line1.rfind("#")]
                    temp_data = line1.strip()
                    pairing1 = [elem for elem,x in enumerate(temp_data) if x == '{'] 
                    pairing2 = [elem for elem,x in enumerate(temp_data) if x == '}']
                    while len(pairing1) != len(pairing2):
                        line1 = input_file_text.readline()
                        line1 = line1[:line1.rfind("#")]
                        temp_data += line1.strip()
                        pairing1 = [elem for elem,x in enumerate(temp_data) if x == '{'] 
                        pairing2 = [elem for elem,x in enumerate(temp_data) if x == '}']
                    self.input_file.append(temp_data)
                else:
                    self.input_file.append(line1.strip())
        input_file_text.close()
        #print(self.input_file)
        # check first character in each line, if invalid, raise an error 
        for item in self.input_file:
            if item != '':
                if not item[0].isalpha() and item[0] not in ['#', '{']:
                    print('*******************************')
                    print('illegal character is identified')
                    print('*******************************')
                    print(item)
                    raise Exception('Only alphapet or # is allowed as first character per line')
                
        # remove empty strings 
        self.input_file=[item for item in self.input_file if item != '']
        # remove comments  #----
        self.input_file=[item for item in self.input_file if item[0] != '#']
        
        #------------------------------
        # check GENERAL card 
        #------------------------------
        try:
            gen_test = self.input_file[self.input_file.index("READ GENERAL") + 1 : self.input_file.index("END GENERAL")]
            print ('--debug: General block is identified')
        except:
            raise('ERR: The general block is required but not found in the input --> READ GENERAL ... END GENERAL is missing')
        
        
    def parse_card(self, card):
        """
        This is a function that gets the Block/Card name, and parse it in self.input_file!
        Returns: a parsed dictionary for the block GENERAL, PPO, etc.
        """
        
        block = self.input_file[self.input_file.index("READ {}".format(card)) + 1 : self.input_file.index("END {}".format(card))]
        block = [item for item in block if item[0] != "#"]   # dobule check if there is any comments
        block = [item.split("=") for item in block]
        block = [[element.strip() for element in item] for item in block] #strip all white spaces
        
        card_dict={}
        for item in block:
            #----------------------------------------
            # convert string dict to a real dict
            if 'env_data' in item:
                #print(item[1])
                try:
                    item[1] = json.loads(item[1])
                except:
                    try:
                        item[1] = literal_eval(item[1])
                    except:
                        raise ValueError ('--error: env_data dict cannot be converted to a python dict, the parser failed')                   
            
                #print(item)
            #----------------------------------------
            card_dict[item[0]] = item[1]  
        
        return card_dict
        
    def blocks (self):

        """
        This function moves block by block and uses parse_card to parse each block
        Returns: a parsed dictionary for each block and flag for the activated blocks
        """
        #Setting attributes for each block, move one by one, check if they are available
        self.gen_block=self.parse_card("GENERAL")
        flags=[]
        #-----------  
        if ("READ TUNE" in self.input_file and "END TUNE" in self.input_file):
            print ('--debug: TUNE block is identified')
            self.tune_block=self.parse_card("TUNE")
            self.tune_flag=True; 
            if 'method' not in self.tune_block.keys():
                raise Exception ('--error: parameter ({}) in card {} is required for neorl but it is not given in the input'.format('method', 'TUNE'))
            #self.tune_block['log_dir'] = self.log_dir  #add log_dir to tuneblock
        #-----------  
        if ("READ DQN" in self.input_file and "END DQN" in self.input_file):
            print ('--debug: DQN block is identified')
            self.dqn_block=self.parse_card("DQN")
            self.dqn_flag=True; flags.append(1)
        #-----------  
        if ("READ PPO" in self.input_file and "END PPO" in self.input_file):
            print ('--debug: PPO block is identified')
            self.ppo_block=self.parse_card("PPO")
            self.ppo_flag=True; flags.append(1)
        #-----------    
        if ("READ A2C" in self.input_file and "END A2C" in self.input_file):
            print ('--debug: A2C block is identified')
            self.a2c_block=self.parse_card("A2C")
            self.a2c_flag=True; flags.append(1)
        #-----------    
        if ("READ ACER" in self.input_file and "END ACER" in self.input_file):
            print ('--debug: ACER block is identified')
            self.acer_block=self.parse_card("ACER")
            self.acer_flag=True; flags.append(1)
        #-----------  
        if ("READ GA" in self.input_file and "END GA" in self.input_file):
            print ('--debug: GA block is identified')
            self.ga_block=self.parse_card("GA")
            self.ga_flag=True; flags.append(1)
        #-----------  
        if ("READ SA" in self.input_file and "END SA" in self.input_file):
            print ('--debug: SA block is identified')
            self.sa_block=self.parse_card("SA")
            self.sa_flag=True; flags.append(1)
        #-----------    
        if self.tune_flag:
            if len(flags) > 1:
                raise Exception ('only one algorithm block can be used with TUNE, user provided {} blocks'.format(len(flags)))
            elif len(flags) == 0:
                raise Exception ('user defined TUNE block but did not use any algorithm block with it, e.g. DQN, GA, SA, etc.')
        return self.log_dir

class InputChecker(InputParser):
    
    """
    This class checks the input for any errors/typos and then overwrites the default inputs by user ones
    Inputs:
        master_parser: are the parsed paramters from class InputParser
        master_paramdict: are the default values given from the class ParamList.py
    """ 
    def __init__(self, master_parser, master_paramdict, log_dir):
        self.paramdict=master_paramdict
        self.parser=master_parser
        self.log_dir=log_dir
        #-----------------------------------------------------------------
        #Initialize all dictionaries to the default ones in ParamList.py
        #-----------------------------------------------------------------
        self.gen_dict=self.paramdict.gen_dict
        self.dqn_dict=self.paramdict.dqn_dict
        self.a2c_dict=self.paramdict.a2c_dict
        self.acer_dict=self.paramdict.acer_dict
        self.ppo_dict=self.paramdict.ppo_dict
        self.ga_dict=self.paramdict.ga_dict
        self.sa_dict=self.paramdict.sa_dict
                    
    def check_input (self, parser, paramdict, card):
        """
        This function loops through any data list and check if data structure and types are correct
        Inputs: 
            parser: is a parsed dict from the user for any block 
            paramdict: is a default dict by neorl for any block
            card: is the block/card name
        
        Returns: this function does not return, but overwrites the default values in self.gen_dict, self.dqn_dict, ...
        """        
        for item in parser:
            if item not in paramdict:
                print('--error: {} is NOT found in neorl input variable names'.format(item))
                raise(ValueError)
                            
            try: 
                if paramdict[item][2] == "str":
                    parser[item] = str(parser[item]).strip()
                elif paramdict[item][2] == "int":
                    parser[item] = int(float(parser[item]))
                elif paramdict[item][2] == "float":
                    parser[item] = float(parser[item])
                elif paramdict[item][2] == "bool":
                    parser[item] = str_to_bool(parser[item])
                elif paramdict[item][2] == "strvec":
                    parser[item] = [str(element.strip()) for element in parser[item].split(",")]
                elif paramdict[item][2] == "vec":
                    parser[item] = np.array([float(element.strip()) for element in parser[item].split(",")])
            except:
                print('--error: the data structure for parameter {} in card {} must be {}, but something else is used'.format(item, card, paramdict[item][2]))
                raise(ValueError)
                
        for item in paramdict:
            if paramdict[item][1] == "r" and item not in parser.keys():
                raise Exception ('--error: parameter {} in card {} is required for neorl but it is not given in the input'.format(item, card))
                
            if paramdict[item][1] == "o" and item not in parser.keys():
                if item not in ['flag']:
                    print ('--warning: parameter {} in card {} is missed, Default is used ---> {}'.format(item,card, paramdict[item][0]))
            
            if paramdict[item][1] == "rs" and item not in parser.keys():
                
                if item == 'model_load_path' and parser['mode'][0] in ['test','continue']: #testing or continue learning without path to pre-trained model
                    raise Exception('--error: the user selected test mode in card {}, but no path to the pre-trained model is provided via model_load_path'.format(card))

                if item == 'time_steps' and parser['mode'][0] in ['train','continue']: #testing or continue learning without path to pre-trained model
                    raise Exception('--error: the user selected train or continue mode in card {}, but number of time_steps is not defined'.format(card))
                    
        # check the test 
        if card in ['DQN', 'PPO', 'A2C', 'ACER']:
            if parser['mode'] in ['train']:
                except_var=[i for i in ['model_load_path', 'n_eval_episodes', 'render'] if i in parser.keys()]
                if len(except_var) > 0:
                    raise Exception('--error: the following variables {} in card {} are NOT allowed for TRAINING mode'.format(except_var, card))
            
            if parser['mode'] in ['continue']:
                except_var=[i for i in ['n_eval_episodes', 'render'] if i in parser.keys()]
                if len(except_var) > 0:
                    raise Exception('--error: the following variables {} in card {} are NOT allowed for CONTINUAL mode'.format(except_var, card))

            if parser['mode'] in ['test']:
                test_allowed=['casename', 'mode', 'model_load_path', 'n_eval_episodes', 'render']
                except_var=[i for i in parser.keys() if i not in test_allowed]
                if len(except_var) > 0:
                    raise Exception('--error: the following variables {} in card {} are ONLY allowed for TEST mode'.format(test_allowed, card))
                   
        #check conditions in genral card
        if card in ['GENERAL']:
            if "xsize_plot" not in parser.keys():
                parser['xsize_plot'] = parser['xsize']
                print ('--warning: xsize_plot is set to equal to xsize ---> {}'.format(parser['xsize_plot']))
            
#            if "exepath" in parser.keys():
#                
#                print ('--debug: checking the exepath')
#                if os.path.isdir(parser['exepath']):
#                    raise Exception ('--error: the user provided a path for directory not to exefile --> {} --> not complete'.format(parser['exepath']))
#                execheck=os.system('which {}'.format(parser['exepath']))
#                if os.path.exists(parser['exepath']):
#                    print('--debug: User provided absolute directory and the binary file reported in {} exists'.format(parser['exepath']))
#                elif (execheck==0):
#                    exeinfer=subprocess.check_output(['which', str(parser['exepath'])])
#                    parser['exepath']=exeinfer.decode('utf-8').strip()
#                    print('--debug: neorl tried to infer the exepath via which and found {}'.format(parser['exepath']))
#                else:
#                    raise Exception ('--error: The binary file reported in {} cannot be found'.format(parser['exepath']))
#                    
            #------------------------------------
            #check availability of GYM env
            #------------------------------------
#            try: 
#                gym.make(parser['env'])
#                print('--debug: {} env is pre-registered in GYM'.format(parser['env']))
#            except:
#                print('--warning: {} env is NOT pre-registered in GYM'.format(parser['env']))
#                try:
#                    print('--warning: Attempting to register {} ...'.format(parser['env']))
#                except:
#                    raise Exception ('The enviroment failed to be registred in GYM, check the entry point to the env, or any syntax problems')
                
                    
    def setup_input(self):
        
        # check the strucutre and syntax first, then overwrite the default dictionary in paramdict.
        self.methods=[]
        self.used_cores=0
        self.check_input(self.parser.gen_block,self.gen_dict, 'GENERAL')
        maxcore_flag=False
        for item in self.parser.gen_block:
            self.gen_dict[item][0] = self.parser.gen_block[item]
            # General checks
            
            #----------------------------
            # check maxcores parameter
            #----------------------------
            
            if item == 'maxcores':
                print ('--debug: maxcores is given by user as {}'.format(self.parser.gen_block[item]))
                maxcore_flag=True
                self.max_cores=self.parser.gen_block[item]
                if self.parser.gen_block[item] <= 0:
                    self.parser.gen_block[item]=multiprocessing.cpu_count()
                    self.max_cores=self.parser.gen_block[item]
                    print ('--debug: user requested inference of maxcores for the machine which is {}'.format(self.max_cores))
        
        self.gen_dict['log_dir']=self.log_dir  # append the main log_dir as part of the gen_dict
        if not maxcore_flag:
            print('--warning: no limit on maxcores is provided by the user, so all specificed cores in the input will be used')
            
        #----------------------------
                
        if self.parser.dqn_flag:
            
            self.check_input(self.parser.dqn_block,self.dqn_dict, 'DQN')
            self.dqn_dict['flag'][0] = True
            for item in self.parser.dqn_block:
                self.dqn_dict[item][0] = self.parser.dqn_block[item]
            
            self.methods.append(self.dqn_dict['casename'][0])
            self.used_cores += self.dqn_dict["ncores"][0]
            

        if self.parser.ppo_flag:
            self.check_input(self.parser.ppo_block,self.ppo_dict, 'PPO')
            self.ppo_dict['flag'][0] = True
            for item in self.parser.ppo_block:
                self.ppo_dict[item][0] = self.parser.ppo_block[item]
            
            # adjust number of steps for parallel
            if self.ppo_dict["ncores"][0] > 1:
                if self.ppo_dict["check_freq"][0] % self.ppo_dict["ncores"][0] != 0:
                    mod=self.ppo_dict["check_freq"][0] % self.ppo_dict["ncores"][0]
                    print('warning: the check_freq parameter {} for ppo is not multiple of number of cores {}'.format(self.ppo_dict["check_freq"][0], self.ppo_dict["ncores"][0]))
                    self.ppo_dict["check_freq"][0] = (self.ppo_dict["check_freq"][0] + self.ppo_dict["ncores"][0] - mod)
                    assert (self.ppo_dict["check_freq"][0] % self.ppo_dict["ncores"][0]) == 0
                    print('warning: the check_freq parameter is adjusted to {} for ppo'.format(self.ppo_dict["check_freq"][0]))
                    
            
            self.methods.append(self.ppo_dict['casename'][0])
            self.used_cores += self.ppo_dict["ncores"][0]
            
        if self.parser.a2c_flag:
            self.check_input(self.parser.a2c_block,self.a2c_dict, 'A2C')
            self.a2c_dict['flag'][0] = True
            for item in self.parser.a2c_block:
                self.a2c_dict[item][0] = self.parser.a2c_block[item]
                            
            # adjust number of steps for parallel
            if self.a2c_dict["ncores"][0] > 1:
                if self.a2c_dict["check_freq"][0] % self.a2c_dict["ncores"][0] != 0:
                    mod=self.a2c_dict["check_freq"][0] % self.a2c_dict["ncores"][0]
                    print('warning: the check_freq parameter {} for a2c is not multiple of number of cores {}'.format(self.a2c_dict["check_freq"][0], self.a2c_dict["ncores"][0]))
                    self.a2c_dict["check_freq"][0] = (self.a2c_dict["check_freq"][0] + self.a2c_dict["ncores"][0] - mod)
                    assert (self.a2c_dict["check_freq"][0] % self.a2c_dict["ncores"][0]) == 0
                    print('warning: the check_freq parameter is adjusted to {} for a2c'.format(self.a2c_dict["check_freq"][0]))
                    
            self.methods.append(self.a2c_dict['casename'][0])
            self.used_cores += self.a2c_dict["ncores"][0]

        if self.parser.acer_flag:
            self.check_input(self.parser.acer_block,self.acer_dict, 'ACER')
            self.acer_dict['flag'][0] = True
            for item in self.parser.acer_block:
                self.acer_dict[item][0] = self.parser.acer_block[item]
                            
            # adjust number of steps for parallel
            if self.acer_dict["ncores"][0] > 1:
                if self.acer_dict["check_freq"][0] % self.acer_dict["ncores"][0] != 0:
                    mod=self.acer_dict["check_freq"][0] % self.acer_dict["ncores"][0]
                    print('warning: the check_freq parameter {} for acer is not multiple of number of cores {}'.format(self.acer_dict["check_freq"][0], self.acer_dict["ncores"][0]))
                    self.acer_dict["check_freq"][0] = (self.acer_dict["check_freq"][0] + self.acer_dict["ncores"][0] - mod)
                    assert (self.acer_dict["check_freq"][0] % self.acer_dict["ncores"][0]) == 0
                    print('warning: the check_freq parameter is adjusted to {} for acer'.format(self.acer_dict["check_freq"][0]))
                    
            self.methods.append(self.acer_dict['casename'][0])
            self.used_cores += self.acer_dict["ncores"][0]
            
        if self.parser.ga_flag:
            self.check_input(self.parser.ga_block,self.ga_dict, 'GA')
            self.ga_dict['flag'][0] = True
            for item in self.parser.ga_block:
                self.ga_dict[item][0] = self.parser.ga_block[item]
            
            self.methods.append(self.ga_dict['casename'][0])
            self.used_cores += self.ga_dict["ncores"][0]
                        
        if self.parser.sa_flag:
            self.check_input(self.parser.sa_block,self.sa_dict, 'SA')
            self.sa_dict['flag'][0] = True
            for item in self.parser.sa_block:
                self.sa_dict[item][0] = self.parser.sa_block[item]
            
            self.methods.append(self.sa_dict['casename'][0])
            self.used_cores += self.sa_dict["ncores"][0]            
        
        if maxcore_flag:
            assert self.used_cores <= self.max_cores, 'total number of cores assigned by the user ({}) are larger than the maxcores ({})'.format(self.used_cores, self.max_cores)
                