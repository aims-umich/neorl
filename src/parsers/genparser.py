import ast
import numpy as np

def setv(**kwargs):
    """ This function returns a dictionary of the passed variables. This makes adding parameters for individual methods easier """
    return kwargs

def astcast(obj):
    """ This function identifies the type of variable passed and casts it accordingly """
    objtype = type(ast.literal_eval(obj))
    if(objtype == type(bool())): return bool(obj)
    if(objtype == type(str())): return str(eval(obj))
    if(objtype == type(int())): return int(obj)
    if(objtype == type(float())): return float(obj)
    if(objtype == type(list())): return list(eval(obj))
    if(objtype == type(tuple())): return tuple(eval(obj))
    if(objtype == type(dict())): return dict(eval(obj))

def defpar(list1):
    """ This function takes a list of all the lines passed in a block and returns a dictionary of parameters passed in that block """
    block_dict = dict()
    for line in range(0, len(list1)):
        if(len(list1[line].split('=', 1)) > 1 and list1[line].split('=', 1)[1].strip()[0] != '{'):
            try:
                block_dict[list1[line].split('=', 1)[0].strip()] = astcast(list1[line].split('=', 1)[1].strip())
            except:
                raise TypeError("The passed value datatype is either wrong or not supported")
        elif(len(list1[line].split('=', 1)) > 1 and list1[line].split('=', 1)[1].strip()[0] == '{' and len((list1[line].split('=',1)[1].strip()).split("=")) <2 ):
            try:
                block_dict[list1[line].split('=',1)[0].strip()] = astcast(list1[line].split('=',1)[1].strip())
            except:
                raise TypeError("The passed value datatype is either inaccurate or not supported")
        elif(len(list1[line].split('=', 1)) > 1 and list1[line].split('=', 1)[1].strip()[0] == '{' and len((list1[line].split('=', 1)[1].strip()).split("=")) > 1 ):
            try:
                block_dict[list1[line].split('=', 1)[0].strip()] = eval('setv('+ list1[line].split('=', 1)[1].strip()[1:len(list1[line].split('=', 1)[1].strip())-1] + ')')
            except:
                raise SyntaxError("For parameter dictionaries either pass a native python dictionary or use the following format:\n"+
                "{parameter1 = value1, parameter2 = value2, ...} using the correct datatype for values and no datatype for parameter names\n"+
                "eg. nn_h = {hidden_layer_sizes = (21,21), normalize_x = True, activation = 'relu'}")
    return(block_dict)

class genparser:
    def __init__(self, input_path):
        self.ModuleType = 'None'
        self.blocks = dict()
        with open(input_path) as input_file_text:
            self.input_file = [parameter.replace("\n","").strip() for parameter in input_file_text.readlines() if not parameter.isspace() or parameter[0] != "%"]
            while('' in self.input_file): self.input_file.remove('')

        if(self.input_file[0][0] == '='):
        #identifies type of input file and sets ModuleType Flag accordingly
            try:
                self.ModuleType = self.input_file[0][1:].strip()
            except:
                raise ValueError("Enter type of Module to be used")
        self.input_file = [item for item in self.input_file if item[0] != "%"]
        read_index = []; end_index = []
        for ind in range(0, len(self.input_file)):
            if(self.input_file[ind].strip()[0:4] == "READ"): read_index.append(ind) #identifies index of start of each block
            if(self.input_file[ind].strip()[0:3] == "END"): end_index.append(ind) #identifies index of end of each block
        for i in range(0, len(read_index)):
            if(end_index[i] - read_index[i] == 1): read_index[i] = -1; end_index[i] = -1 #sets index values of all empty blocks to negative (-1)
        for i in range(0, len(read_index)):
            if(read_index[i] >= 0):
                self.blocks[self.input_file[read_index[i]][4:].strip()] = defpar(self.input_file[read_index[i]+1 : end_index[i]]) #appends self.blocks dictionary with key = Block name and value = dictionary of passed parameters

        if(self.ModuleType == "ML"):
            from MLparser import MLparser
            CaseName=input_path.split(".inp")[0]
            obj = MLparser(self, CaseName)
            #if ModuleType is ML then MLparser is called
