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

from src.parsers.PARSER import InputParser, InputChecker
from src.parsers.ParamList import InputParam
from src.utils.multiproc import MultiProc


def init_files(methods, nx, ny, inp_headers, out_headers):
    
    
    if len(out_headers) >= 1 and out_headers[0] != 'y':
        assert ny == len(out_headers), 'number of outputs assigned in ysize ({}) is not equal to ynames ({})'.format(ny, len(out_headers))
    if len(inp_headers) >= 1 and inp_headers[0] != 'x':
        assert nx == len(inp_headers), 'number of inputs assigned in xsize_plot ({}) is not equal to xnames ({})'.format(nx, len(inp_headers))
        
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
    if len(inp_headers) == 1 and inp_headers[0]=='x':
        [inp_names.append('x'+str(i)) for i in range(1,nx+1)]
    else:
        [inp_names.append(i) for i in inp_headers]
        
    
    out_names=['caseid', 'reward']
    if len(out_headers) == 1 and out_headers[0]=='y':
        [out_names.append('y'+str(i)) for i in range(1,ny+1)]
    else:
        [out_names.append(i) for i in out_headers]
    
    #if (1):
    #    out_names=['caseid', 'reward', 'PPF', 'delta_h', 'boron', 'exposure', 'objective','feasible']
        
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
    
    if len(sys.argv) < 3:
        raise Exception ("NO input file after -i is passed, try --> python fuse.py -i FUSE_INPUT ")
    else:
        print ("--------------------------------------------------------------- ")
        print ("The input file {} is passed".format(sys.argv[2]))
        print ("--------------------------------------------------------------- ")
    input_file_path=sys.argv[2]
    
    # Uncomment these if you work directly from IDE (e.g. spyder)!!!!
    
    print("--debug: All modules are imported sucessfully")
    print ("--------------------------------------------------------------- ")
    #input_file_path='test'    
    
    parser=InputParser(input_file_path)
    parser.blocks()
    paramdict=InputParam()
    inp=InputChecker(parser,paramdict)
    inp.setup_input()
    init_files(inp.methods, inp.gen_dict['xsize_plot'][0], inp.gen_dict['ysize'][0], inp.gen_dict['xnames'][0], inp.gen_dict['ynames'][0])  # Intialize the all loggers

    print('------------------------------------------------------------------------------')
    print('--debug: Input check is completed successfully, no major error is found')
    print('------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------')
    
    master=MultiProc(inp)
    master.run_all()



