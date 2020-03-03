import numpy as np
import os
import pandas as pd
#sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + "\src\ML-main")

class MLparser:
    def __init__(self, obj, CaseName):
        #rearranges the dictionary passed by genparser into a form that is supported by MLmodule
        self.maindict = dict()
#        try:
        self.maindict.update(obj.blocks['INPUT'])
        self.maindict.update(obj.blocks['OUTPUT'])
        if('x' in self.maindict):
            self.maindict['x']=pd.read_csv(self.maindict['x'])
            self.maindict['y']=pd.read_csv(self.maindict['y'])
        self.maindict['methods'] = list()
        
        for key in list(obj.blocks['METHODS'].keys()):
            obj.blocks['METHODS'][key.lower()] = obj.blocks['METHODS'].pop(key)
        if('svr' in list(obj.blocks['METHODS'].keys())): self.maindict['svr_h'] = obj.blocks['METHODS']['svr']; self.maindict['methods'].append('svr')
        if('nn' in list(obj.blocks['METHODS'].keys())): self.maindict['nn_h'] = obj.blocks['METHODS']['nn']; self.maindict['methods'].append('nn')
        if('lr' in list(obj.blocks['METHODS'].keys())): self.maindict['lr_h'] = obj.blocks['METHODS']['lr']; self.maindict['methods'].append('lr')
        if('gpr' in list(obj.blocks['METHODS'].keys())): self.maindict['gpr_h'] = obj.blocks['METHODS']['gpr']; self.maindict['methods'].append('gpr')
        if('rr' in list(obj.blocks['METHODS'].keys())): self.maindict['rr_h'] = obj.blocks['METHODS']['rr']; self.maindict['methods'].append('rr')
#        except:
#            raise SyntaxError("Supported Blocks in ML are: INPUT, METHODS, KWARGS, OUTPUT")

        print("Passed Parameters: ", {k:v for k,v in self.maindict.items() if k !='x' and k!='y'})
        if(obj.ModuleType == 'ML'):
            import MLmodule as ml
            ml_obj = ml.fitmodels()
            self.result_models = ml_obj.fit(input_dict = self.maindict, CaseName=CaseName)
            #calls the MLmodule.fitmodels.fit() function while passing the rearranged dictionary
