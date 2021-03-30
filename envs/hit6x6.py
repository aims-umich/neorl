# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 11:51:40 2019

@author: Majdi Radaideh
"""

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
from gym.spaces import Discrete, Box
import os
from gym.utils import seeding
import pickle, copy, json
import time
import subprocess

def get_string (filename,s1,s2):
        
    """
    This is an auxiliary postprocessing function to get data between two strings in a text file 
    (designed especially for this problem). You need to adjust for other purposes
    """
    data=[] 
    with open(filename,'r') as infile:
        copyy = False
        for line in infile:
            if line.strip() == s1 :
                copyy = True
            elif s2 in line.strip():
                copyy = False
            elif copyy:
                if len(line.split()) > 9:
                    data.append([float(k) for k in line.split()[6:]])
                else:
                    data.append([float(k) for k in line.split()[1:]])

    data = [x for x in data if x != []]
    
    return (data)

"""
    Implementation of a black-boxed environment for Casmo4 code for a BWR 6x6 assembly with 2 discrete actions
    
    Any RL enviroment consists of four major functions:
        1- initializer: to start the enviroment, define action space, state space 
            * For black-box codes, the initializer would be special as input files or templates 
            should be added. Hard-coded templates are used here without external input files.  
        2- Step: which recieves an action and observe the new state and reward
        3- reset: to reset the enviroment back to first state, when end of episode occurs
        4- render: to visualize the problem (optional)
    
    ***************
    Game Scenario
    ***************
    1- Intialize the assembly board to a uniform enrichment, and decide the next rod position to be visited (randomly) 
    2- Take an action at that position either LOW (1.87) or HIGH (2.53)
    3- Observe next state (next position to be visited), set reward to zero
    4- Take next action, and repeat the process until all 21 positions are visited. 
    5- Provide final reward as the inverse of the objective function, and start a new episode
    
    ***************
    Assembly Board
    ***************
    1
    2 3 
    4 5 6 
    7 8 9 10 
    11 12 13 14 15 
    16 17 18 19 20 21
    
    ***************
    Fixed Variable/Function Names for OpenAI Gym Data Strucutures
    ***************
    # variables
    self.action_space -->  within __init__
    self.observed_space  -->  within __init__
    # functions
    step
    reset
    render
    
    # The rest can be named as you like
    
"""

class Casmo4Env(gym.Env):
    
    def __init__ (self, casename='method', log_dir=None, exepath=None, env_data=None, env_seed=None):
        #---------------------------------------
        # NEORL main input data
        #---------------------------------------
        self.casename=casename
        self.log_dir=log_dir
        self.env_seed=env_seed
        
        os.system('rm -rf *.inp *.out *.cax *.log tmp* fort*')
        if env_data:
            # Get external data
            self.kinf0_thresh=env_data["kinf"]
            self.ppf_thresh=env_data["ppf"]
            self.avgenrich_thresh=env_data["E"]
            self.episode_length=env_data["lives"]
            self.mode=env_data["mode"]
            self.exepath=env_data["path"]
        else:
            # Use defaults
            self.kinf0_thresh=1.25
            self.ppf_thresh=1.35
            self.avgenrich_thresh=2.3
            self.episode_length=4
            self.mode='casmo4'
            self.exepath=exepath
        
        #---------------------------------------
        # Misc parameters for this assembly case
        #---------------------------------------
        if self.mode in ["casmo4"]:
            print('--running 6x6 via casmo4')
            self.exepath=self.check_exes()
        elif self.mode in ["extdata"]:
            print('--running 6x6 via external dataset')
            if not os.path.exists(self.exepath):
                print('--error: User provided {} as a path for the external library and it does not exist'.format(self.exepath))
            self.data = pickle.load(open(self.exepath, "rb"))
        else:
            raise Exception ('--error: either choose casmo4 or extdata for the card mode in env_data')
            
        self.numlocs=21 #see assembly board
        self.n=6   #6x6 
        self.fueltypes=2  #1.87 and 2.53
        bu=0 # BOL, no depletion c
        self.seeding=False
        self.epi=1
        self.fileindex=np.random.randint(1,1e6) # to track casmo input/output names
        self.file=self.casename+'_case'+str(self.fileindex)
        
        #*******************
        # Action/State Space
        #*******************
        # !!! These variables are very important, and their names have to be like this to work with Gym
        self.action_space = Discrete(self.fueltypes)  # Action space is discrete with two possibilities

        # By default the state space is a flattened version of self.numlocs*2 array 
        # first column has the encirchment value in each rod 
        # second column has vector of zeros except 1 at the position to be visited next
        self.observation_space = Box(low=0, high=self.fueltypes+1, shape=(self.numlocs * 2,), dtype=int)
            
        
        # Clean directory and reset enviroment when intialize the enviroment
        self.reset()
        self.clean()
        self.done = False # true if environement has reached terminal state, false otherwise
        
        #--------------------------------------------------------------------
        # These two templates will be used later in construcing Casmo inputs
        #--------------------------------------------------------------------
        
        self.front_cards="""TTL * BWR GE6x6
TFU=900 TMO=560 VOI=50
PDE 25 'KWL'
BWR 6 1.78 11.052 0.15 0.873 0.476 1.33/.2048 3.8928
PIN 1 0.612 0.625 0.7139
*PIN 2 1.17 1.24 /'MOD' 'BOX' //4
LPI
1
1 1
1 1 1
1 1 1 1
1 1 1 1 1
1 1 1 1 1 1 \n"""
 
        self.tail_cards="""LFU
1
2 3
4 5 6             
7 8 9 10      
11 12 13 14 15 
16 17 18 19 20 21
DEP -{bu}
STA
END""".format(bu=bu)
    #--------------------------------------------------------------------------
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]  
        
    def step(self, action):
        
        """
        VERY IMPORTANT 
        This function steps in the enviorment by executing RunCasmo4 and computes the reward based on the observed state
        Takes input as an action
        Retruns next state, rewards, termination_flag
        """
        
        action = action + 1 # action goes from 0 to num_colors-1 so we need to add one to get the actual color
        
        self.file=self.casename+'_case'+str(self.fileindex)
        
        #*********************************************************************************      
        # Sanity checks
        #*********************************************************************************
        if (action > self.fueltypes or action <= 0):
            print("Illegal action (fuel type): {} attempted.".format(action))
            raise('error')
            return [self.GetStateView(), -1, self.done, {}]

        if self.done:
            #print("Game is over, all trials are done")
            return [self.GetStateView(), 0, self.done, {}]
        #*********************************************************************************
        # Convert the integer data to float enrichment (this is done to fit Gym data structures)
        #*********************************************************************************
        if action==1:
            self.enrichvec[self.current_loc]=1.87
        elif action==2:
            self.enrichvec[self.current_loc]=2.53
        else:
            raise ("illegal action is given")
        #*********************************************************************************
        self.state[:,0] = copy.deepcopy(self.enrichvec)
        #-----------------------------------
        #execute casmo4 and compute rewards 
        #-----------------------------------
        
        if self.counter==self.numlocs:

            if self.mode in ["extdata"]:
                self.GetData(self.state[:,0])
            else:
                self.RunCasmo4()
        
            #self.compute_rewards()
            self.compute_rewards2()
            #self.compute_rewards3()
            self.epis_counter+=1
            
            #print('old',self.best_reward,'new',self.reward)
            if self.reward > self.best_reward: 
                #print('new reward is better')
                self.best_enrich=copy.deepcopy(self.enrichvec)
                self.best_reward=self.reward
                self.best_kinf0=self.kinf0
                self.best_avgenrich=self.avgenrich
                self.best_ppf=self.maxpppf
                self.bestname='hit6x6_'+'age'+str(self.env_seed+1)+'_{:05d}'.format(self.epi)
                
                if self.best_reward > 286.5:
                    self.render()
                
            #--------------------------------------------------
            #internal reset of enviroment back to the best
            #--------------------------------------------------
            self.reset_internal(self.best_enrich)
            
        else:
            self.reward=0
            self.GetNextLocation()
            self.counter += 1
        
        if self.epis_counter==self.episode_length: # check if episode is over
            self.monitor()
            self.done = True
            self.epi+=1
            
            #print(self.best_enrich)
            #print(self.fileindex,self.best_reward,self.best_kinf0, self.best_avgenrich, self.best_ppf)   
        #print(self.state[:,0])
        #*********************************************************************************
        self.fileindex+=1    #update file index for casmo4
        
        return [self.GetStateView(), self.reward, self.done, {}]    #these must be returned by the step function
    
    def render(self, mode='human'):
        
        """
        Render (optional): plot the assembly and the enrichments for each pin
        """
        uo2=self.ConstructAssembly(self.best_enrich)
        # Make the mesh plot
        plt.figure()
        plt.imshow(uo2); plt.set_cmap('summer'); plt.axis('off')
                
        # Loop over data dimensions and create text annotations.
        for i in range(uo2.shape[0]):
            for j in range(uo2.shape[1]):
                plt.text(j, i, uo2[i, j], ha="center", va="center", color="k", fontsize=13)
                

        if (self.best_reward > 0):
            plt.title('Reward='+ str(np.round(self.best_reward,3)) +'\n $k_{\infty}^0$=' + str(np.round(self.best_kinf0,5))+ ', PPF=' + str(np.round(self.best_ppf,3)) + ', E (wt%)=' + str(np.round(self.best_avgenrich,3)))
            #plt.title('Step '+str(self.counter)+'/'+str(self.numlocs)+ ', Reward='+ str(np.round(self.best_reward,2)) +'\n $k_{\infty}^0$=' + str(self.best_kinf0)+ ', PPF=' + str(np.round(self.best_ppf,3)) + ', U-235 wt%=' + str(np.round(self.best_avgenrich,3)))
        else:
            plt.title('Step '+str(self.counter)+'/'+str(self.numlocs)+ ', Reward=NA \n $k_{\infty}^0$= NA, PPF= NA, U-235 wt%=NA')
  
        plt.tight_layout()
        plt.savefig(self.log_dir+self.bestname+'.png',format='png', dpi=300, bbox_inches="tight")
        plt.show()
        
    def reset_internal(self, best_state=None):
        
        self.state = np.zeros((self.numlocs,2))  # initialize the assembly state
        
        if not best_state:
            self.enrichvec=[1.5]*self.numlocs  # this vector contains the enrichment in each rod, fixed to 1.5
        else:
            self.enrichvec=copy.deepcopy(best_state) #copy the best state if found from previous lives
        
        self.locs=list(range(0,self.numlocs))  # generate a list of locations from 0 to numlocs-1
        self.placevec=[0]*self.numlocs # intialize the pointer vector to zero
        
        if self.seeding:
            random.seed(1)
        self.current_loc=random.randint(0,self.numlocs-1) # sample random location 
        self.locs.remove(self.current_loc) # remove new location from the list
        self.placevec[self.current_loc]=1   # update the pointer 
        self.state[:,0]=copy.deepcopy(self.enrichvec) # update the state array
        self.state[:,1]=copy.deepcopy(self.placevec) # update the state array
        self.counter = 1  # time step counter
        
        self.clean()    # clean the directory
        
    def reset(self):
        """
        Important: Reset the enviroment after termination is reached
        """ 
        self.reset_internal()
        self.epis_counter=0
        self.best_enrich=copy.deepcopy(self.enrichvec)
        self.best_reward=0
        self.best_kinf0=0
        self.best_avgenrich=0
        self.best_ppf=0
        self.done = 0   # termination flag
        self.reward = 0
        self.creward=0

        return self.GetStateView()  # the reset function MUST return the initial state 
    
    def fit(self, x):
        """
        This objective function is special for GA, SA, PSO, and other classical optimistion methods
        It recieves x enrichmet as input and returns objective function as output
        """

        self.enrichvec=[]
        for i in range (len(x)):
            if x[i]==1:
                self.enrichvec.append(1.87)
            elif x[i]==2:
                self.enrichvec.append(2.53)
            else:
                raise Exception ("illegal value of input is sampled --> {}".format(x[i]))
        
        #self.fileindex=np.random.randint(1,1e7) # to track casmo input/output names
        #self.fileindex=int(caseid)
        self.file=np.random.randint(1,1e6) # to track casmo input/output names
        self.file=self.casename+'_case'+str(self.fileindex)
        self.reward=0
        #-----------------------------------
        #execute casmo4 and compute rewards 
        #-----------------------------------
        if self.mode in ["extdata"]:
            self.GetData(self.enrichvec)
        else:
            self.RunCasmo4()
            
        self.compute_rewards2()
        
        #-------------
        # FOR GA/SA, make best metrics as regular one, no difference 
        self.best_reward=self.reward; self.best_kinf0=self.kinf0
        self.best_avgenrich=self.avgenrich; self.best_ppf=self.maxpppf
        self.best_enrich=copy.deepcopy(self.enrichvec)
        #-------------
        
        self.monitor()
        self.clean()
        
        if (1):
          return self.reward,
        else:
          return self.reward

    def monitor(self):
        """
        This function is to dump data to csv file, print data to screen, needed only if you activate on-the-fly plotting/logging options. 
        """     
        
        # Y-output file logger
        with open (self.log_dir+self.casename+'_out.csv','a') as fin:
            fin.write(str(self.bestname) + ', ' + str(np.round(self.best_reward,3)) + ', '+ str(self.best_kinf0) + ', '+ str(self.best_avgenrich) + ', ' + str(self.best_ppf)  + '\n')
        
        # X-input file logger
        with open (self.log_dir+self.casename+'_inp.csv','a') as fin:
            fin.write(str(self.bestname) +','+ str(np.round(self.best_reward,3)) + ',')
            for j in range (len(self.best_enrich)):
                if j==len(self.best_enrich)-1:
                  fin.write(str(self.best_enrich[j])+'\n')
                else:
                  fin.write(str(self.best_enrich[j])+',')
            
                
        # Save good patterns 
        #if (self.reward > 6000):
        #    print('*************************************************************************')
        #    print('Reward:', np.round(self.reward,4))
        #    print('GOOD Pattern:', self.enrichvec)
        #    print('GOOD Pattern:', self.reward, self.kinf0, self.maxpppf, self.avgenrich)
            #self.render() 
        #    print('*************************************************************************')
            
        #Print to screen (debugging)
        print('Method:', self.casename)
        print('CaseID:', self.file)
        print('Reward:', np.round(self.best_reward,3))
        print('Enrich:', self.best_enrich)
        print('------------------------------------------------------------------------------')
        
    #*************************************************************
    #*************************************************************
    # Auxiliary Functions       
    #*************************************************************
    #*************************************************************
        
    def GetStateView(self):
        """
        returns the view the agent gets of the state, which is either identical to the the internal
        state view or a flattened view depending on the self.flatten paramater set during config
        BETTER TO RETURN STATE AS A VECTOR
        """

        return self.state.flatten()

    def clean(self):
        """
        This is an auxiliary function to clear the directory from any previous casmo inputs/outputs
        CASMO requires manual permission to overwrite files so always better to clean the directory first
        to avoid internal failure in python
        """
        os.system('rm -Rf {file}.inp {file}.log {file}.out {file}.cax'.format(file=self.file))
        
    
    def ConstructAssembly(self,x):
        """
        This function takes input as vector x and convert to assembly strucutre considering 1/2 symmetry 
        This function is used to monitor and plot assembly state, it is optional
        """
        x=np.round(x,2)
        uo2=np.zeros((self.n,self.n), dtype=float)        
        # fill the uo2 matrix 
        index=0
        for i in range (1,self.n+1):
            for j in range(0,i):
                uo2[i-1,j]=x[index]
                index+=1
        uo2 = uo2 + uo2.T - np.diag(np.diag(uo2)) # make a symmetric matrix (repeat the lower diagonal into upper diagonal)
        
        return uo2

    def GetNextLocation(self):
        
        """
        This function decides the next location to be visited. Once the location is assigned, it is removed 
        from the list of locations to ensure it is not visited twice in the same episode.
        Policy: visit one rod at a time
        """
        # change one rod at a time 
        
        if self.seeding:
            random.seed(1)
        # get new location and remove it from future options
        new_loc = random.sample(self.locs,1)[0]
        self.locs.remove(new_loc)  # remove the new location from the list

       
        self.placevec[self.current_loc]=0  # set old location in placement array to 0, set new location to 1
        self.current_loc = new_loc  #update current location with the new one 
        self.placevec[self.current_loc]=1 # update the placevec
        

        self.state[:,1]=self.placevec # initialize the placment
        
    def RunCasmo4(self):
        """
        This function builds a new casmo input, exeucute it, and process the output.
        Input templates are taken from __init__, outputs are kinf, ppf, and enrichment
        """
        f = open(self.file+'.inp', 'w')
        out = self.front_cards # sets the front cards
        for n in range(0,len(self.enrichvec)):
            out = out + "FUE " + str(n+1) + " 10.5 / " + str(self.enrichvec[n]) + "\n"   #this prints the composition for each rod
        
        out = out + self.tail_cards  # merge with tail cards
        
        # write the final input file
        f.write(out)
        f.close()
        
        """
        Run CASMO4 Input 
        """
        #subprocess.call(['casmo4e', self.file+'.inp'])
        os.system('{} {}.inp > tmpout'.format(self.exepath, self.file))
        
        """
        Process CASMO4 Output 
        """
        # strings to extract data between using get_string 
        s1='MWD/KG             TWO-GROUP     PEAK    WT %    WT %    WT %'
        s2="RUN TERMINATED"
        
        self.casmo_log=np.array(get_string(self.file+'.log', s1,s2))
        
        # Define the corresponding parameters for optimisation
        # Vector parameters
        self.bu=self.casmo_log[:,0];      self.kinf=self.casmo_log[:,1]
        self.ppf=self.casmo_log[:,4];     self.enrich=self.casmo_log[:,5]
        self.pu=self.casmo_log[:,7]
        
        #scalar parameters
        self.kinf0=self.kinf[0];        self.maxkinf = self.kinf.max()
        self.maxpppf = self.ppf.max();  self.avgenrich=self.enrich[0]
        
        self.output=[self.kinf0, self.maxpppf, self.avgenrich]
        # This enviroment is built for BOL no burnup, you need to add some interplations to 
        # to get more info about EOL burnup 
        
        return 

    def GetData(self, x):
        """
        This function builds a new casmo input, exeucute it, and process the output.
        Input templates are taken from __init__, outputs are kinf, ppf, and enrichment
        x: is the vector of enrichments to pull the data for 
        """
        
        resp=self.data[tuple(x)]
        
        #scalar parameters
        self.kinf0=resp[0]    
        self.maxpppf = resp[1]  
        self.avgenrich=resp[2]
        
        self.output=[self.kinf0, self.maxpppf, self.avgenrich]
        # This enviroment is built for BOL no burnup, you need to add some interplations to 
        # to get more info about EOL burnup 
        
        return 
        
    def compute_rewards(self):
        """
        Calculate objective function and return rewards as 1/f(x)*100 to show scale differences between patterns
        """
        
        if  self.kinf0 > 1.25:
            wk=-1000
        elif self.kinf0 < 1.24:
            wk=1000
        else:
            wk=1.75
        
        if self.maxpppf >= 1.45:
            wp=-1000
        elif self.maxpppf < 1.45 and self.maxpppf > 1.35:
            wp=-2.5
        else:
            wp=0
        
        if self.avgenrich > 2.3:
            we=-1000
        elif self.avgenrich < 2.2:
            we=1000
        else:
            we=2.0
        
        self.of=wk * (1.25-self.kinf0) + wp*(1.35-self.maxpppf) + we * (2.3-self.avgenrich)        
        self.reward=1/self.of
        
    def compute_rewards2(self):
        """
        Calculate objective function and return rewards as 1/f(x)*100 to show scale differences between patterns
        """

        if self.kinf0 > self.kinf0_thresh:
            wk=-1
        else:
            wk=1
    
        if self.maxpppf > self.ppf_thresh:
            wp=-1
        else:
            wp=0

    
        if self.avgenrich > self.avgenrich_thresh:
            we=-1
        else: 
            we=1
        
        self.of=wk*(1.25-self.kinf0)/1.25 + wp*(1.35-self.maxpppf)/1.35 + we*(2.3-self.avgenrich)/2.3        
        self.reward=1/self.of
        
    def compute_rewards3(self):
        """
        Calculate objective function and return rewards as 1/f(x)*100 to show scale differences between patterns
        """
        
        if self.kinf0 <= self.kinf0_thresh:
            wk=1.25-self.kinf0
        else:
            wk=1000

        if self.avgenrich <= self.avgenrich_thresh:
            we=2.3-self.avgenrich
        else:
            we=1000
        
        if self.maxpppf <= self.ppf_thresh:
            wp=0
        else:
            wp=self.maxpppf
            
        self.of=wp + wk + we
        self.reward=1/self.of

    def check_exes(self):
        
        print ('--debug: checking all exe files')
        if os.path.isdir(self.exepath):
            raise Exception ('--error: the user provided a path for directory not to exefile --> {} --> not complete'.format(self.exepath))
        execheck=os.system('which {}'.format(self.exepath))
        if os.path.exists(self.exepath):
            print('--debug: User provided absolute directory and the binary file reported in {} exists'.format(self.exepath))
            abs_path=self.exepath
        elif (execheck==0):
            exeinfer=subprocess.check_output(['which', str(self.exepath)])
            abs_path=exeinfer.decode('utf-8').strip()
            print('--debug: neorl tried to infer the exepath via which and found {}'.format(self.exepath))
        else:
            raise Exception ('--error: The binary file reported in {} cannot be found'.format(self.exepath))
        
        return abs_path