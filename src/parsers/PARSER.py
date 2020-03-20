"""
This class parses the master input file provided by the user
""" 
import psutil

class InputParser():
    def __init__(self, input_file_path):
        self.MainCaseName=input_file_path.split(".inp")[0]
        #Read the lines of the input file into the attribute input_file
        #Remove first line, empty lines and "\n" from every line
        
        # method flags 
        self.dqn_flag=False
        self.ga_flag=False
        self.a2c_flag=False
        self.ppo_flag=False
        
        
        #input blocks 
        self.gen_block={}
        self.dqn_block={}
        self.ppo_block={}
        self.a2c_block={}
        self.ga_block={}
        
        with open(input_file_path) as input_file_text:
            self.input_file = [parameter.replace("\n","").strip() for parameter in input_file_text.readlines() if not parameter.isspace() or parameter[0] != "%"]
                
        # remove empty strings 
        self.input_file=[item for item in self.input_file if item != '']
        # remove comments  #----
        self.input_file=[item for item in self.input_file if item[0] != '#']
        
        # check general card      
        try:
            gen_test = self.input_file[self.input_file.index("READ GENERAL") + 1 : self.input_file.index("END GENERAL")]
            print ('--debug: General block is identified')
        except:
            raise('ERR: The general block is required but not found in the input --> READ GENERAL ... END GENERAL is missing')
        
        
    def parse_card(self, card):
        
        block = self.input_file[self.input_file.index("READ {}".format(card)) + 1 : self.input_file.index("END {}".format(card))]
        block = [item for item in block if item[0] != "#"]   # dobule check if there is any comments
        block = [item.split("=") for item in block]
        block = [[element.strip() for element in item] for item in block] #strip all white spaces
        
        card_dict={}
        for item in block:
            card_dict[item[0]] = item[1]  
        
        return card_dict
        
    def blocks (self):
        #Setting attributes for each block
        self.gen_block=self.parse_card("GENERAL")
                
        if ("READ DQN" in self.input_file and "END DQN" in self.input_file):
            print ('--debug: DQN block is identified')
            self.dqn_block=self.parse_card("DQN")
            self.dqn_flag=True

        if ("READ PPO" in self.input_file and "END PPO" in self.input_file):
            print ('--debug: PPO block is identified')
            self.ppo_block=self.parse_card("PPO")
            self.ppo_flag=True
            
        if ("READ A2C" in self.input_file and "END A2C" in self.input_file):
            print ('--debug: A2C block is identified')
            self.a2c_block=self.parse_card("A2C")
            self.a2c_flag=True

        if ("READ GA" in self.input_file and "END GA" in self.input_file):
            print ('--debug: GA block is identified')
            self.ga_block=self.parse_card("GA")
            self.ga_flag=True
        
        return

"""
This checks the input for any errors/typos and then overwrites the default input provided by the developer
""" 

class InputChecker(InputParser):
    def __init__(self, master_parser, master_paramdict):
        self.paramdict=master_paramdict
        self.parser=master_parser
        #check that all data structures are correct
        
        self.gen_dict=self.paramdict.gen_dict
        self.dqn_dict=self.paramdict.dqn_dict
        self.a2c_dict=self.paramdict.a2c_dict
        self.ppo_dict=self.paramdict.ppo_dict
        self.ga_dict=self.paramdict.ga_dict
        
                    
    def check_input (self, parser, paramdict, card):
        # this function loops through any data list and check if data structure and types are correct
        
        for item in parser:
            if item not in paramdict:
                print('--error: {} is NOT found in neorl input variable names'.format(item))
                raise(ValueError)
                            
            try: 
                if paramdict[item][2] == "str":
                    parser[item] = str(parser[item]).strip()
                elif paramdict[item][2] == "int":
                    parser[item] = int(parser[item])
                elif paramdict[item][2] == "float":
                    parser[item] = float(parser[item])
                elif paramdict[item][2] == "bool":
                    parser[item] = bool(parser[item])
                elif paramdict[item][2] == "strvec":
                    parser[item] = [str(element.strip()) for element in parser[item].split(",")]
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
        if card in ['DQN', 'PPO', 'A2C']:
            if parser['mode'] in ['train']:
                except_var=[i for i in ['model_load_path', 'n_eval_episodes', 'video_record', 'render', 'fps'] if i in parser.keys()]
                if len(except_var) > 0:
                    raise Exception('--error: the following variables {} in card {} are NOT allowed for TRAINING mode'.format(except_var, card))
            
            if parser['mode'] in ['continue']:
                except_var=[i for i in ['n_eval_episodes', 'video_record', 'render', 'fps'] if i in parser.keys()]
                if len(except_var) > 0:
                    raise Exception('--error: the following variables {} in card {} are NOT allowed for CONTINUAL mode'.format(except_var, card))

            if parser['mode'] in ['test']:
                test_allowed=['casename', 'mode', 'model_load_path', 'n_eval_episodes', 'video_record', 'render', 'fps']
                except_var=[i for i in parser.keys() if i not in test_allowed]
                if len(except_var) > 0:
                    raise Exception('--error: the following variables {} in card {} are ONLY allowed for TEST mode'.format(test_allowed, card))
                   
        #check the xsize plot
        if card in ['GENERAL']:
            if "xsize_plot" not in parser.keys():
                parser['xsize_plot'] = parser['xsize']
                print ('--warning: xsize_plot is set to equal to xsize ---> {}'.format(parser['xsize_plot']))
               
               
                    
                
            
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
                    self.parser.gen_block[item]=psutil.cpu_count(logical = True)
                    self.max_cores=self.parser.gen_block[item]
                    print ('--debug: user requested inference of maxcores for the machine which is {}'.format(self.max_cores))
                    
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
                
        if self.parser.ga_flag:
            self.check_input(self.parser.ga_block,self.ga_dict, 'GA')
            self.ga_dict['flag'][0] = True
            for item in self.parser.ga_block:
                self.ga_dict[item][0] = self.parser.ga_block[item]
            
            self.methods.append(self.ga_dict['casename'][0])
            self.used_cores += self.ga_dict["ncores"][0]
        
        if maxcore_flag:
            assert self.used_cores <= self.max_cores, 'total number of cores assigned by the user ({}) are larger than the maxcores ({})'.format(self.used_cores, self.max_cores)
                
        #print(self.dqn_dict)
        #print(self.a2c_dict)
        #print(self.gen_dict)
                
#if __name__=='__main__':
#    input_file_path='../../test.inp'
#    pars=InputParser(input_file_path)
#    paramdict=InputParam()
#    inp=InputChecker(pars,paramdict).setup_input()
        