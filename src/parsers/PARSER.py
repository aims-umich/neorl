import numpy as np
from ParamList import VarNames
#--------------------------------------------------------------------------------------------------
# Class 1: The master class that has intilization of all global variables for all modules 
# It reads all inputs from the master input file, all other classes depend on this master class
#--------------------------------------------------------------------------------------------------     
        
class PARSER:
    def __init__(self, input_file_path):
        self.input_file_path=input_file_path
        self.CaseName=self.input_file_path.split(".inp")[0]
        self.MainOutName=self.CaseName + '.txt'
        #Read the lines of the input file into the attribute input_file
        #Remove first line, empty lines and "\n" from every line
        # code flags 
        self.ScaleFlag=0
        self.TraceFlag=0
        self.Relap5Flag=0
        self.BisonFlag=0
        self.UqFlag=0
        
        with open(input_file_path) as input_file_text:
            self.input_file = [parameter.replace("\n","").strip() for parameter in input_file_text.readlines() if not parameter.isspace() or parameter[0] != "%"]

        # Decide the Module 
        for item in self.input_file:
            if item[0] == "=":
                self.ModuleType=item.replace("=","")
            break
        
        #------------------------------------
        # a special exit block if ML is identified, other types of parsers are used to process ML input
        if (self.ModuleType=='ML'): 
            return 
        #-------------------------------------
        
        #Setting attributes for each block
        self.gen_block = self.input_file[self.input_file.index("READ GEN") + 1 : self.input_file.index("END GEN")]
        self.gen_block = [item for item in self.gen_block if item[0] != "%"]
        if ("READ SCALE" in self.input_file):
            self.scale_block = self.input_file[self.input_file.index("READ SCALE") + 1 : self.input_file.index("END SCALE")]
            self.scale_block = [item for item in self.scale_block if item[0] != "%"]
            self.ScaleFlag=1
        if ("READ TRACE" in self.input_file):
            self.trace_block = self.input_file[self.input_file.index("READ TRACE") + 1 : self.input_file.index("END TRACE")]
            self.trace_block = [item for item in self.trace_block if item[0] != "%"]
            self.TraceFlag=1
        if ("READ RELAP5" in self.input_file):
            self.r5_block = self.input_file[self.input_file.index("READ RELAP5") + 1 : self.input_file.index("END RELAP5")]
            self.r5_block = [item for item in self.r5_block if item[0] != "%"]
            self.Relap5Flag=1
        if ("READ BISON" in self.input_file):
            self.bison_block = self.input_file[self.input_file.index("READ BISON") + 1 : self.input_file.index("END BISON")]
            self.bison_block = [item for item in self.bison_block if item[0] != "%"]
            self.BisonFlag=1
        if ("READ UQ" in self.input_file):
            self.uq_block = self.input_file[self.input_file.index("READ UQ") + 1 : self.input_file.index("END UQ")]
            self.uq_block = [item for item in self.uq_block if item[0] != "%"]
            self.UqFlag=1
            
        #Read the lines of the parameter list file to know what parameters self.a has
        self.g = {}
        self.s = {}
        self.t = {}
        self.r5 = {}
        self.bs = {}
        
        #generating the parameters and distributions for the UQ block all else handled in NTuq
        if self.UqFlag == 1:
            self.uq = {}
            for i_line in self.uq_block:
                if i_line.split("=")[0] not in ["n_samples","perturb_xs","perturb_yield","perturb_decay"]:
                    self.uq[i_line.split("=")[0]] = i_line.split("=")[1].split(",")
            
        p_list=VarNames() # import the parameter list from /src/ParamList.py 
        for p_line in p_list:
                #finds name and type of all possible parameters
                name, card, typ = p_line[0], p_line[1], p_line[2]

                if (card in ['general','scale','trace','relap5','bison']):
                    #goes into a particular block and finds the value associated with that name
                    #then alias the attribute dictionary to the general dictionary "a"
                    if card == "general":
                        value = [i_line.split("=")[1] for i_line in self.gen_block if name in i_line]
                        a = self.g
                        
                    elif (card == "scale" and self.ScaleFlag==1):
                        value = [i_line.split("=")[1] for i_line in self.scale_block if name in i_line]
                        a = self.s
                        
                    elif (card == "trace" and self.TraceFlag==1):
                        value = [i_line.split("=")[1] for i_line in self.trace_block if name in i_line]
                        a = self.t
                    
                    elif (card == "relap5" and self.Relap5Flag==1):
                        value = [i_line.split("=")[1] for i_line in self.r5_block if name in i_line]
                        a = self.r5
                    
                    elif (card == "bison" and self.BisonFlag==1):
                        value = [i_line.split("=")[1] for i_line in self.bison_block if name in i_line]
                        a = self.bs
                    else:
                        continue   # skip the unactivated codes 
                        
                else:
                    raise Exception("extension is not correct for '{n}' in parameter_list.csv".format(n = name))
                
                #places value into the self.a dict with type conversion based on "typ"
                if not value:
                    a[name.strip()] = None
                elif typ == "str":
                    a[name.strip()] = str(value[0]).strip()
                elif typ == "int":
                    a[name.strip()] = int(value[0])
                elif typ == "float":
                    a[name.strip()] = float(value[0])
                elif typ == "vec":
                    a[name.strip()] = np.array([float(num.strip()) for num in value[0].split(",")])
                elif typ == "ivec":
                    a[name.strip()] = np.array([int(num.strip()) for num in value[0].split(",")])
                elif typ == "strvec":
                    a[name.strip()] = np.array([str(num.strip()) for num in value[0].split(",")])
                elif typ == "sp1":
                    # data type with "[]" characters used to define material ID with same properties 
                    if '[' in value[0] or ']' in value[0]:
                        parsed_list=[]
                        for num in value[0].split(","):
                            if ('[' in num or ']' in num):
                                num=num.strip('[ ]').split()
                                num=[int(i) for i in num]
                                parsed_list.append(num)
                            else:
                                parsed_list.append(int(num))
                                
                        a[name.strip()]=parsed_list
                    else:
                        a[name.strip()] = np.array([int(num.strip()) for num in value[0].split(",")])

        #Check to make sure vectors inside SCALE, TRACE, BISON, etc. match the others in that doctionary
#        if self.ScaleFlag==1:
#            for parameter, value in self.s.items():
#                if not np.isscalar(value):
#                    if not value.size == self.s["NodeID"].size:
#                        raise Exception("The size of {parm} in SCALE block does not match size of NodeID".format(parm = parameter))
#        if self.TraceFlag==1:
#            for parameter, value in self.t.items():
#                if not np.isscalar(value):
#                    if not value.size == self.t["NodeID"].size:
#                        raise Exception("The size of {parm} in TRACE block does not match size of NodeID".format(parm = parameter))
#        if self.Relap5Flag==1:
#            for parameter, value in self.r5.items():
#                if not np.isscalar(value):
#                    if not value.size == self.r5["NodeID"].size:
#                        raise Exception("The size of {parm} in RELAP5 block does not match size of NodeID".format(parm = parameter))
#                        
#        if self.BisonFlag==1:
#            for parameter, value in self.bs.items():
#                if not np.isscalar(value):
#                    if not value.size == self.bs["NodeID"].size:
#                        raise Exception("The size of {parm} in BISON block does not match size of NodeID".format(parm = parameter))
                        