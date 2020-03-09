# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 08:09:18 2019

@author: Majdi Radaideh
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, './src/')
sys.path.insert(0, './src/rl')
sys.path.insert(0, './src/utils')
sys.path.insert(0, './src/evolu')
sys.path.insert(0, './src/parsers')

from PARSER import InputParser, InputChecker
from ParamList import InputParam
from multiproc import MultiProc


def init_files(methods, nx, ny):
    
    # check if the log directory exists, move to old and create a new log
    if os.path.exists("./master_log/") and os.path.exists("./old_master_log/"):
        os.system('rm -Rf ./old_master_log/')
        os.rename('./master_log/','./old_master_log/')
        os.makedirs('./master_log/')
    elif os.path.exists("./master_log/"): 
        os.rename('./master_log/','./old_master_log/')
        os.makedirs('./master_log/')         
    else:
        os.makedirs('./master_log/')
        
    
    inp_names=['caseid', 'reward'] 
    out_names=['caseid', 'reward']
    
    [inp_names.append('x'+str(i)) for i in range(1,nx+1)]
    [out_names.append('y'+str(i)) for i in range(1,ny+1)]
    
    
    for method in methods:
        
        with open ('./master_log/'+method+'_inp.csv','w') as fin:
            for i in range(len(inp_names)):
                if i==len(inp_names)-1:
                    fin.write(inp_names[i] +'\n')
                else:
                    fin.write(inp_names[i]+',')
                                
        with open ('./master_log/'+method+'_out.csv','w') as fin:
            for i in range(len(out_names)):
                if i==len(out_names)-1:
                    fin.write(out_names[i] + '\n')
                else:
                    fin.write(out_names[i]+',')
                    
        with open ('./master_log/'+method+'_summary.txt','w') as fin:
            fin.write('---------------------------------------------------\n')
            fin.write('Summary file for the {} method \n'.format(method))
            fin.write('---------------------------------------------------\n')
            fin.write(logo)
            
    print('--debug: All logging files are created')

#if len(sys.argv) < 3:
#    raise Exception ("No input file after -i is passed, try --> python fuse.py -i NEORL_INPUT ")
#else:
#    print ("""---------------------------------------------------------------
#The input file \"%s\" is passed"
##---------------------------------------------------------------"""%(sys.argv[2]))
#ms_input=sys.argv[2]

# Uncomment these if you work directly from spyder!!!!

        
if __name__ == '__main__':
    
    logo="""
                 NEORL: Nuclear Engineering Optimisation with Reinforcement Learning
      
    NNNNNNNN        NNNNNNNNEEEEEEEEEEEEEEEEEEEEEE     OOOOOOOOO     RRRRRRRRRRRRRRRRR   LLLLLLLLLLL             
    N:::::::N       N::::::NE::::::::::::::::::::E   OO:::::::::OO   R::::::::::::::::R  L:::::::::L             
    N::::::::N      N::::::NE::::::::::::::::::::E OO:::::::::::::OO R::::::RRRRRR:::::R L:::::::::L             
    N:::::::::N     N::::::NEE::::::EEEEEEEEE::::EO:::::::OOO:::::::ORR:::::R     R:::::RLL:::::::LL             
    N::::::::::N    N::::::N  E:::::E       EEEEEEO::::::O   O::::::O  R::::R     R:::::R  L:::::L               
    N:::::::::::N   N::::::N  E:::::E             O:::::O     O:::::O  R::::R     R:::::R  L:::::L               
    N:::::::N::::N  N::::::N  E::::::EEEEEEEEEE   O:::::O     O:::::O  R::::RRRRRR:::::R   L:::::L               
    N::::::N N::::N N::::::N  E:::::::::::::::E   O:::::O     O:::::O  R:::::::::::::RR    L:::::L               
    N::::::N  N::::N:::::::N  E:::::::::::::::E   O:::::O     O:::::O  R::::RRRRRR:::::R   L:::::L               
    N::::::N   N:::::::::::N  E::::::EEEEEEEEEE   O:::::O     O:::::O  R::::R     R:::::R  L:::::L               
    N::::::N    N::::::::::N  E:::::E             O:::::O     O:::::O  R::::R     R:::::R  L:::::L               
    N::::::N     N:::::::::N  E:::::E       EEEEEEO::::::O   O::::::O  R::::R     R:::::R  L:::::L         LLLLLL
    N::::::N      N::::::::NEE::::::EEEEEEEE:::::EO:::::::OOO:::::::ORR:::::R     R:::::RLL:::::::LLLLLLLLL:::::L
    N::::::N       N:::::::NE::::::::::::::::::::E OO:::::::::::::OO R::::::R     R:::::RL::::::::::::::::::::::L
    N::::::N        N::::::NE::::::::::::::::::::E   OO:::::::::OO   R::::::R     R:::::RL::::::::::::::::::::::L
    NNNNNNNN         NNNNNNNEEEEEEEEEEEEEEEEEEEEEE     OOOOOOOOO     RRRRRRRR     RRRRRRRLLLLLLLLLLLLLLLLLLLLLLLL
                           \n"""
                           
    print(logo)
    print("--debug: All modules are imported sucessfully")
        
    # Intialize the all loggers
    input_file_path='test'
    
    parser=InputParser(input_file_path)
    paramdict=InputParam()
    inp=InputChecker(parser,paramdict)
    inp.setup_input()
    init_files(inp.methods, inp.gen_dict['xsize_plot'][0], inp.gen_dict['ysize'][0]) 
    
    master=MultiProc(inp)
    master.run_all()


