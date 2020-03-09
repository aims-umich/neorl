"""
This class parses the master input file provided by the user
""" 

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
        self.dqn_block=[]
        self.ppo_block=[]
        self.a2c_block=[]
        self.ga_block=[]
        
        with open(input_file_path) as input_file_text:
            self.input_file = [parameter.replace("\n","").strip() for parameter in input_file_text.readlines() if not parameter.isspace() or parameter[0] != "%"]
                
        # remove empty strings 
        self.input_file=[item for item in self.input_file if item != '']
        # remove comments  #----
        self.input_file=[item for item in self.input_file if item[0] != '#']
        
        #Setting attributes for each block
        try:
            self.gen_block = self.input_file[self.input_file.index("READ GENERAL") + 1 : self.input_file.index("END GENERAL")]
            print ('--debug: General block is identified')
        except:
            raise('ERR: The general block is required but not found in the input --> READ GENERAL ... END GENERAL is missing')
        
        self.gen_block=[item.split("=") for item in self.gen_block] # split the = sign
        self.gen_block = [[element.strip() for element in item] for item in self.gen_block] #strip all white spaces
                
        if ("READ DQN" in self.input_file and "END DQN" in self.input_file):
            print ('--debug: DQN block is identified')
            self.dqn_block = self.input_file[self.input_file.index("READ DQN") + 1 : self.input_file.index("END DQN")]
            self.dqn_block = [item for item in self.dqn_block if item[0] != "#"]   # dobule check if there is any comments
            self.dqn_block = [item.split("=") for item in self.dqn_block] # split by = sign
            self.dqn_block = [[element.strip() for element in item] for item in self.dqn_block] #strip all white spaces
            self.dqn_flag=True

        if ("READ PPO" in self.input_file and "END PPO" in self.input_file):
            print ('--debug: PPO block is identified')
            self.ppo_block = self.input_file[self.input_file.index("READ PPO") + 1 : self.input_file.index("END PPO")]
            self.ppo_block = [item for item in self.ppo_block if item[0] != "#"]   # dobule check if there is any comments
            self.ppo_block = [item.split("=") for item in self.ppo_block]
            self.ppo_block = [[element.strip() for element in item] for item in self.ppo_block] #strip all white spaces
            self.ppo_flag=True
            
        if ("READ A2C" in self.input_file and "END A2C" in self.input_file):
            print ('--debug: A2C block is identified')
            self.a2c_block = self.input_file[self.input_file.index("READ A2C") + 1 : self.input_file.index("END A2C")]
            self.a2c_block = [item for item in self.a2c_block if item[0] != "#"]   # dobule check if there is any comments
            self.a2c_block = [item.split("=") for item in self.a2c_block]
            self.a2c_block = [[element.strip() for element in item] for item in self.a2c_block] #strip all white spaces
            self.a2c_flag=True

        if ("READ GA" in self.input_file and "END GA" in self.input_file):
            print ('--debug: GA block is identified')
            self.ga_block = self.input_file[self.input_file.index("READ GA") + 1 : self.input_file.index("END GA")]
            self.ga_block = [item for item in self.ga_block if item[0] != "#"]   # dobule check if there is any comments
            self.ga_block = [item.split("=") for item in self.ga_block]
            self.ga_block = [[element.strip() for element in item] for item in self.ga_block] #strip all white spaces
            self.ga_flag=True

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
    
    def check_input (self, parser, paramdict):
        # this function loops through any data list and check if data structure and types are correct
        
        for item in parser:
            if item[0] not in paramdict:
                print('--error: {} is NOT found in neorl input syntax '.format(item[0]))
                raise(ValueError)
                
            #assert type(item[1]) == paramdict[item[0]][2], 'user input for {} is {}, but {} should be used'.format(item[1], type(item[1]), paramdict[item[0]][2])
            
            try: 
                if paramdict[item[0]][2] == "str":
                    item[1] = str(item[1]).strip()
                elif paramdict[item[0]][2] == "int":
                    item[1] = int(item[1])
                elif paramdict[item[0]][2] == "float":
                    item[1] = float(item[1])
                elif paramdict[item[0]][2] == "bool":
                    item[1] = bool(item[1])
            except:
                print('--error: the data structure for parameter {} must be {}, but something else is used'.format(item[0], paramdict[item[0]][2]))
                raise(ValueError)
        
        #print(self.gen_block, self.a2c_block, self.dqn_block)
            
            
    def setup_input(self):
        
        # check the strucutre and syntax first, then overwrite the default dictionary in paramdict.
        self.methods=[]
        for item in self.parser.gen_block:
            self.check_input(self.parser.gen_block,self.gen_dict)
            self.gen_dict[item[0].strip()][0] = item[1]
                
        if self.parser.dqn_flag:
            
            self.check_input(self.parser.dqn_block,self.dqn_dict)
            self.dqn_dict['flag'][0] = True
            for item in self.parser.dqn_block:
                self.dqn_dict[item[0].strip()][0] = item[1]
            
            self.methods.append(self.dqn_dict['casename'][0])

        if self.parser.ppo_flag:
            self.check_input(self.parser.ppo_block,self.ppo_dict)
            self.ppo_dict['flag'][0] = True
            for item in self.parser.ppo_block:
                self.ppo_dict[item[0].strip()][0] = item[1]
            
            # adjust number of steps for parallel
            if self.ppo_dict["ncores"][0] > 1:
                if self.ppo_dict["check_freq"][0] % self.ppo_dict["ncores"][0] != 0:
                    mod=self.ppo_dict["check_freq"][0] % self.ppo_dict["ncores"][0]
                    print('warning: the check_freq parameter {} for ppo is not multiple of number of cores {}'.format(self.ppo_dict["check_freq"][0], self.ppo_dict["ncores"][0]))
                    self.ppo_dict["check_freq"][0] = (self.ppo_dict["check_freq"][0] + self.ppo_dict["ncores"][0] - mod)
                    assert (self.ppo_dict["check_freq"][0] % self.ppo_dict["ncores"][0]) == 0
                    print('warning: the check_freq parameter is adjusted to {} for ppo'.format(self.ppo_dict["check_freq"][0]))
            
            self.methods.append(self.ppo_dict['casename'][0])
                
        if self.parser.a2c_flag:
            self.check_input(self.parser.a2c_block,self.a2c_dict)
            self.a2c_dict['flag'][0] = True
            for item in self.parser.a2c_block:
                self.a2c_dict[item[0].strip()][0] = item[1]
                
            self.methods.append(self.a2c_dict['casename'][0])
            
            # adjust number of steps for parallel
            if self.a2c_dict["ncores"][0] > 1:
                if self.a2c_dict["check_freq"][0] % self.a2c_dict["ncores"][0] != 0:
                    mod=self.a2c_dict["check_freq"][0] % self.a2c_dict["ncores"][0]
                    print('warning: the check_freq parameter {} for a2c is not multiple of number of cores {}'.format(self.a2c_dict["check_freq"][0], self.a2c_dict["ncores"][0]))
                    self.a2c_dict["check_freq"][0] = (self.a2c_dict["check_freq"][0] + self.a2c_dict["ncores"][0] - mod)
                    assert (self.a2c_dict["check_freq"][0] % self.a2c_dict["ncores"][0]) == 0
                    print('warning: the check_freq parameter is adjusted to {} for a2c'.format(self.a2c_dict["check_freq"][0]))
                
        if self.parser.ga_flag:
            self.check_input(self.parser.ga_block,self.ga_dict)
            self.ga_dict['flag'][0] = True
            for item in self.parser.ga_block:
                self.ga_dict[item[0].strip()][0] = item[1]
            
            self.methods.append(self.ga_dict['casename'][0])
                
        print('--debug: Input check is completed successfully, no major error is found')
                
        #print(self.dqn_dict)
        #print(self.a2c_dict)
        #print(self.gen_dict)
                
#if __name__=='__main__':
#    input_file_path='../../test.inp'
#    pars=InputParser(input_file_path)
#    paramdict=InputParam()
#    inp=InputChecker(pars,paramdict).setup_input()
        