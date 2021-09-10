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
This class parses the TUNE block and returns a checked dictionary containing all TUNE user
parameters
""" 
import numpy as np
import os

class TuneChecker():
    
    """
    This class checks the input for any errors/typos and then overwrites the default inputs by user ones
    Inputs:
        master_parser: are the parsed paramters from class InputParser
        master_paramdict: are the default values given from the class ParamList.py
    """ 
    def __init__(self, parsed, default):
        self.default=default
        self.parsed=parsed
        self.tune_dict={}
        for key in self.default:
            self.tune_dict[key] = self.default[key][0]
        
        self.tune_dict=self.check_input(self.parsed, self.default, 'TUNE')
                            
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
            
            if '{' in item or '}' in item:
                continue 
            
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
                    parser[item] = bool(parser[item])
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
            
        for item in parser:   #final checked dictionary   
            self.tune_dict[item] = parser[item]
        
        if 'extfiles' in parser.keys():
            for item in self.tune_dict['extfiles']:
                if not os.path.exists(item):
                    raise Exception('--error: User provided {} as external file/directory to be copied by TUNE, such file does not exist in the working directory'.format(item))
        
        self.tune_dict={k: v for k, v in self.tune_dict.items() if v is not None}

        self.tune_dict['flag'] = True
        
        return self.tune_dict

    
    
    