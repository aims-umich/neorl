# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 11:51:40 2019

@author: Majdi Radaideh
"""

import matplotlib
matplotlib.use('Agg')
import gym
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import random
from gym.spaces import Discrete, Box
import os
import pandas as pd

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
    Implementation of a black-boxed environment for Casmo4 code for a BWR 10x10 assembly with 190 discrete actions
    
    Any RL enviroment consists of four major functions:
        1- initializer: to start the enviroment, define action space, state space 
            * For black-box codes, the initializer would be special as input files or templates 
            should be added. Hard-coded templates are used here without external input files.  
        2- Step: which recieves an action and observe the new state and reward
        3- reset: to reset the enviroment back to first state, when end of episode occurs
        4- render: to visualize the problem (optional)
        
    **************
    Action space 
    **************
    # 4.95 is used instead of 5% to stay within LEU limits < 5%
    UO2 actions 
    2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 4.95 (11 actions)     
    
    UO2 (gad) 
    4.25 (7%), 4.25 (8%), 4.5 (7%), 4.5 (8%), 4.75 (7%), 4.75 (8%), 4.95 (7%), 4.95 (8%)  (8 actions)
    
    ***************
    Game Scenario
    ***************
    1- Intialize the assembly board to a uniform enrichment, and decide the next rod position to be visited (randomly) 
    2- Take an action at that position out of the 19 possibilities above 
    3- Observe next state (next position to be visited), set reward to zero
    4- Take next action, and repeat the process until all 51 positions are visited. 
    5- Provide final reward as the inverse of the objective function, and start a new episode
    
    ***************
    Assembly Board
    ***************
    1
    2 3 
    4 5 6 
    7 8 9 10 
    11 12 13 14 15 
    16 17 18 W W 19
    20 21 22 W W 23 24 
    25 26 27 28 29 30 31 32 
    33 34 35 36 37 38 39 40 41 
    42 43 44 45 46 47 48 49 50 51
    
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
    
    #*************************************************************
    #*************************************************************
    # Required Functions       
    #*************************************************************
    #*************************************************************
    
    def __init__ (self, mode='opt', casename='method', log_dir='./master_log/'):
        
        self.log_dir=log_dir
        os.makedirs(self.log_dir,exist_ok=True)
        self.casename=casename
        print(self.casename)
        # Misc parameters for this assembly case
        self.mode=mode
        self.numlocs=51 #see assembly board
        self.n=10   #6x6 
        self.fueltypes=19  # action space size 
        self.checkpoints=50
        bu=50  #discharge burnup
        self.eocbu=-1
        
        self.fileindex=np.random.randint(1,1e6) # to track casmo input/output names
        self.file=self.casename+'_case'+str(self.fileindex)
        self.flatten=1   # to flatten arrays and return vector state (better to stay as 1 for keras NN)
        
        #*******************
        # Action/State Space
        #*******************
        # !!! These variables are very important, and their names have to be like this to work with Gym
        self.action_space = Discrete(self.fueltypes)  # Action space is discrete with 19 possibilities
        # self.action_space1 = Discrete(self.fueltypes)  # Action space is discrete with 19 possibilities
        # self.action_space2 = Discrete(11)  # Action space is discrete with 19 possibilities
        self.ordered_placement=0
        if self.ordered_placement:
            self.observation_space = Box(low=1, high=self.fueltypes+1, shape=(self.numlocs,), dtype=int)
        else:
            # By default the state space is a flattened version of self.numlocs*2 array 
            # first column has the encirchment value in each rod 
            # second column has vector of zeros except 1 at the position to be visited next
            self.observation_space = Box(low=0, high=self.fueltypes+1, shape=(self.numlocs * 2,), dtype=int)
            
        
        # Clean directory and reset enviroment when intialize the enviroment
        self.reset()
        self.clean()
        self.done = False # true if environement has reached terminal state, false otherwise
        
        
        # Dictionary for action space to convert between integers and enrichments in casmo 
        # first item is Uo2 enrich, second is gad enrichment
        self.Dict = {}
        # for testing 
        self.Dict[0]=[1.5, 0]
        self.Dict[1]=[2.5, 0]; self.Dict[2]=[2.75, 0]; self.Dict[3]=[3.0, 0]; self.Dict[4]=[3.25, 0]; self.Dict[5]=[3.5, 0]
        self.Dict[6]=[3.75, 0]; self.Dict[7]=[4.0, 0]; self.Dict[8]=[4.25, 0]; self.Dict[9]=[4.5, 0]; self.Dict[10]=[4.75, 0]; self.Dict[11]=[4.95, 0]
        
        self.Dict[12]=[4.25, 7]; self.Dict[13]=[4.5, 7]; self.Dict[14]=[4.75, 7]; self.Dict[15]=[4.95, 7]
        
        self.Dict[16]=[4.25, 8]; self.Dict[17]=[4.5, 8]; self.Dict[18]=[4.75, 8]; self.Dict[19]=[4.95, 8]
        
        #--------------------------------------------------------------------
        # These two templates will be used later in construcing Casmo inputs
        #--------------------------------------------------------------------
        
        self.front_cards="""TTL * BWR GE14 10x10 
TFU=900 TMO=560 VOI=40
PDE 65 'KWL'
BWR 10 1.3 13.4 0.19 0.71 0.72 1.33/.3048 3.8928
PIN 1 0.440 0.447 0.510
PIN 2 1.17 1.24 /'MOD' 'BOX' //4
LPI
1
1 1
1 1 1
1 1 1 1
1 1 1 1 1
1 1 1 2 2 1
1 1 1 2 2 1 1 
1 1 1 1 1 1 1 1 
1 1 1 1 1 1 1 1 1 
1 1 1 1 1 1 1 1 1 1 \n"""
 
        self.tail_cards="""LFU
1
2 3 
4 5 6 
7 8 9 10 
11 12 13 14 15 
16 17 18 0 0 19
20 21 22 0 0 23 24 
25 26 27 28 29 30 31 32 
33 34 35 36 37 38 39 40 41 
42 43 44 45 46 47 48 49 50 51
DEP -{bu}
STA
END""".format(bu=bu)
    #--------------------------------------------------------------------------
    
    def step(self, action):
        
        """
        VERY IMPORTANT 
        This function steps in the enviorment by executing RunCasmo4 and computes the reward based on the observed state
        Takes input as an action
        Retruns next state, rewards, termination_flag
        """
        
        action = action + 1 # action goes from 1 to num of fueltypes

        self.file=self.casename+'_case'+str(self.fileindex)
        
        #*********************************************************************************
        #Heuristics
        #*********************************************************************************
        #--- gad hack: this will decieve the agent for the sake of escaping the stuck region. 
        # if self.help and self.gad_rods >= 14 and action > 11:
        #     action=np.random.randint(7,12)
        
        self.rod_index=int(np.where(self.state[:,1]==1)[0]) # the next rod position to be visited
        
        if self.rod_index in [0, 41, 50] and action > 5: # corner rods
            self.invalid_actions+=1
        if self.rod_index in [1, 3, 6, 10, 15, 19 ,24, 32, 42, 43, 44, 45, 46, 47, 48 ,49] and action > 11: # corner rods
            self.invalid_actions+=1
        #if self.gad_rods >= 15 and action > 11:
        #    self.invalid_actions+=1           
                
        #*********************************************************************************      
        # Sanity checks
        #*********************************************************************************
        if (action > self.fueltypes or action <= 0):
            raise("Illegal action (fuel type): {} attempted.".format(action))
            return [self.GetStateView(), -1, self.done, {}]

        #*********************************************************************************
        # Convert the integer data to float enrichment (this is done to fit Gym data structures)
        #*********************************************************************************
        self.enrichvec[self.current_loc]=action
        self.get_stat()

        #-----------------------------------
        #execute casmo4 and compute rewards 
        #----------------------------------- 
        
        if self.counter == self.checkpoints: 
            
            if self.mode=='opt':
                self.RunCasmo4()
                self.compute_rewards()
                
                if self.gad_rods <= 15 and self.gad_rods >= 12:
                    self.reward += 250 
            
                self.monitor()
                                
            if self.mode=='learn':
                
                if self.gad_rods > 15:
                    self.reward += -self.invalid_actions + (15 - self.gad_rods) 
                elif self.gad_rods < 12:
                    self.reward += -self.invalid_actions + (self.gad_rods - 12)
                else:
                    self.reward += -self.invalid_actions + 10
                    
                with open (self.log_dir+self.casename+'_monitor.csv','a') as fin:
                    fin.write(str(self.invalid_actions) + ', '+  str(self.gad_rods) + ', '+str(np.round(self.reward,1)) + ', \n')
    
        else:
             self.reward+=0
            
        
        #self.render()    
        #*********************************************************************************
        #update the assembly state and counter 
        #*********************************************************************************
        self.counter += 1
        if self.ordered_placement:
            self.state = self.enrichvec
        else:
            self.state[:,0] = self.enrichvec
        #*********************************************************************************
        # check if game is over
        #*********************************************************************************
        if self.counter == self.numlocs:
            self.done = True
            
            self.placevec[self.current_loc]=0   #convert the placment vector to all zeros for a new episode
            if not self.ordered_placement:
                self.state[:,1]=self.placevec  #initialize the placment array again for a new episode
            #self.render()      # uncomment to plot the assembly at end of episode     
        else:
            self.GetNextLocation()  #if episode does not end, get the next position to visit
        #*********************************************************************************
        self.fileindex+=1    #update file index for casmo4

        return [self.GetStateView(), self.reward, self.done, {}]    #these must be returned by the step function
            
    def reset(self):
        """
        Important: Reset the enviroment after termination is reached
        """        
        self.locs=list(range(0,self.numlocs))  # generate a list of locations from 0 to numlocs-1
        self.enrichvec=[0]*self.numlocs  # this vector contains the enrichment in each rod, fixed to 4.0wt% based on self.Dict --> identifier is "7"
        self.placevec=[0]*self.numlocs # intialize the pointer vector to zero
        
        if self.ordered_placement:
            self.state=self.enrichvec 
        else:
             self.state = np.zeros((self.numlocs,2))  # initialize the assembly state
        
        
        if self.ordered_placement:
            self.current_loc=0 # !!!depericated
        else:
            self.current_loc=np.random.randint(self.numlocs) # sample random location 
            self.locs.remove(self.current_loc) # remove new location from the list
            self.placevec[self.current_loc]=1   # update the pointer 
            self.state[:,0]=self.enrichvec # update the state array
            self.state[:,1]=self.placevec # update the state array
        
        self.counter = 0  # time step counter
        self.done = 0   # termination flag
        self.of_list=[]  # mir: not sure if this variable is useful for anything 
        self.clean()    # clean the directory 
        self.reward=0
        self.invalid_actions=0

        return self.GetStateView()  # the reset function MUST return the initial state 
    
    def render(self):
        
        """
        Render (optional): plot the assembly and the enrichments for each pin
        """          
        uo2=self.ConstructAssembly(self.uo2vec)
        gad=self.ConstructAssembly(self.gadvec)
        
        # Make the mesh plot
        plt.figure()
        plt.imshow(uo2); plt.set_cmap('summer'); plt.axis('off')
       
        # Loop over data dimensions and create text annotations.
        for i in range(uo2.shape[0]):
            for j in range(uo2.shape[1]):
                if uo2[i,j]==-1 and gad[i,j]==-1:
                    plt.text(j, i, 'W', ha="center", va="center", color="W", fontsize=11)
                elif gad[i,j] > 1:
                    plt.text(j, i, str(uo2[i, j]) + '\n' + str(gad[i, j]), ha="center", va="center", color="b", fontsize=10)
                else:
                    plt.text(j, i, uo2[i, j], ha="center", va="center", color="k", fontsize=10)
                     
        textstr="""Num of UO$_2$ types: {num_fuel}   Avg UO$_2$ enrichment: {avg_en}
Num of Gd$_2$O$_3$ rods: {gad_rods}   Avg Gd$_2$O$_3$ enrichment: {avg_gad}""".format(num_fuel=self.num_fuel, gad_rods=self.gad_rods, avg_en=self.avg_en, avg_gad=self.avg_gad)
        
        if self.mode=='learn':
            plt.title('Step '+str(self.counter)+'/'+str(self.numlocs)+ ', Reward='+ str(np.round(self.reward,1)) +'\n Invalid Actions=' + str(self.invalid_actions))
        else:
            if self.eocbu > 0:  
                plt.title('Step '+str(self.counter)+'/'+str(self.numlocs)+ ', Reward='+ str(np.round(self.reward,2)) +'\n $k_{\infty}^{max}$=' + str(self.maxkinf)+ ', PPF=' + str(self.maxpppf) + ', BU=' + str(np.round(self.eocbu,1)))
            else:
                plt.title('Step '+str(self.counter)+'/'+str(self.numlocs)+ ', Reward= NA' +'\n $k_{\infty}^{max}$= NA, PPF= NA, BU= NA')

        plt.text(0.23, 0, textstr, fontsize=9, transform=plt.gcf().transFigure)
  
        plt.tight_layout()
        plt.savefig(self.log_dir+self.file+'.png',format='png', dpi=300, bbox_inches="tight")
        plt.show()
    
    def fit(self, x):
        """
        This objective function is special for GA, SA, PSO, and other classical optimistion methods
        It recieves x enrichmet as input and returns objective function as output
        """
        self.enrichvec=x
        
        self.fileindex=np.random.randint(1,1e6) # to track casmo input/output names
        self.file=self.casename+'_case'+str(self.fileindex)
        self.reward=0
        #-----------------------------------
        #execute casmo4 and compute rewards 
        #-----------------------------------
        self.RunCasmo4()
        self.compute_rewards()
        self.get_stat()
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
            fin.write(str(np.round(self.fileindex)) + ', ' + str(np.round(self.reward,3)) + ', '+ str(self.maxkinf) + ', '+ str(self.avgenrich) + ', ' + str(self.maxpppf) + 
                     ', ' + str(np.round(self.eocbu,1)) + ', '+str(self.gad_rods) + ', '+str(np.round(self.invalid_actions)) + '\n')
        
        # X-input file logger
        with open (self.log_dir+self.casename+'_inp.csv','a') as fin:
            fin.write(str(self.fileindex) +','+ str(np.round(self.reward,3)) + ',')
            for i in range (len(self.uo2vec)):
                fin.write(str(self.uo2vec[i])+',')
            for j in range (len(self.gadvec)):
                if j==len(self.gadvec)-1:
                  fin.write(str(self.gadvec[j])+'\n')
                else:
                  fin.write(str(self.gadvec[j])+',')
            
                
        # Save good patterns 
        if (self.reward > 0):
            print('GOOD Pattern:', self.uo2vec)
            print('GOOD Pattern:', self.gadvec)
            self.render() 
            
        #Print to screen (debugging)
        print('Method:', self.casename)
        print('CaseID:', self.file)
        print('Reward:', self.reward)
        print('UO2:', self.uo2vec)
        print('GAD:', self.gadvec)
        
        
        
        
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
        if self.ordered_placement:
            return self.state
        else:
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
        For this problem it returns two arrays one for UO2 enrichment and one for Gad enrichments
        """
        x=np.round(x,2)
        uo2=np.zeros((self.n,self.n), dtype=float)     
        # Water rod locations 
        water_xloc=[6,7]
        water_yloc=[3,4]
        # fill the uo2 matrix 
        index=0
        for i in range (1,self.n+1):
            for j in range(0,i):
                if i in water_xloc and j in water_yloc:
                    uo2[i-1,j]=-1
                else:
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
        if self.ordered_placement:
            new_loc = self.locs[self.counter]  # move in order from 0 to numlocs-1
        else:
            # get new location and remove it from future options
            new_loc = random.sample(self.locs,1)[0]
            self.locs.remove(new_loc)  # remove the new location from the list
    
       
        self.placevec[self.current_loc]=0  # set old location in placement array to 0, set new location to 1
        self.current_loc = new_loc  #update current location with the new one 
        self.placevec[self.current_loc]=1 # update the placevec
        
        if not self.ordered_placement:
            self.state[:,1]=self.placevec # update the state array 
        
    def RunCasmo4(self):
        """
        This function builds a new casmo input, exeucute it, and process the output.
        Input templates are taken from __init__, outputs are kinf, ppf, and enrichment
        """
        f = open(self.file+'.inp', 'w')
        out = self.front_cards # sets the front cards
        for n in range(0,len(self.enrichvec)):
            
            #obtain real enrichments from self.Dict 
            ee=self.Dict[self.enrichvec[n]]
            if ee[1] > 0: # Gad rod is identified 
                out = out + "FUE " + str(n+1) + " 10.2 / " + str(ee[0]) \
                + " 64016 = " + str(ee[1]) + "\n"
            else:  # a Uo2 rod is identified 
                out = out + "FUE " + str(n+1) + " 10.5 / " + str(ee[0]) + "\n"   #this prints the composition for each rod
        
        out = out + self.tail_cards  # merge with tail cards
        
        # write the final input file
        f.write(out)
        f.close()
        
        """
        Run CASMO4 Input 
        """
        #subprocess.call(['casmo4e', self.file+'.inp'])
        os.system('casmo4e '+ self.file+'.inp > tmpout')
        
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
        
        
        
        # Estimate EOC burnup.  Get index of BU before kinf dips below 0.95.
        if len (self.kinf) > 1:
            
            inter_list=[0]; nointer_list=[]
            for i in range (1, len(self.kinf)):
                if self.kinf[i-1] > 0.95 and self.kinf[i] < 0.95: 
                    inter_list.append(i)
                elif self.kinf[i] == 0.95 and i>15:   # no need for interpolation 
                    nointer_list.append(i)
                    
            #print(inter_list)
            #print(nointer_list)
                    
            # make the interpolation 
            if len(nointer_list) == 0: # no exact 0.95 is found, so do interpolation
                i=max(inter_list)
                self.eocbu = (0.95-self.kinf[i-1]) * (self.bu[i]-self.bu[i-1])/(self.kinf[i]-self.kinf[i-1]) + self.bu[i-1]
            elif max(inter_list) > max(nointer_list):  
                i=max(inter_list)
                self.eocbu = (0.95-self.kinf[i-1]) * (self.bu[i]-self.bu[i-1])/(self.kinf[i]-self.kinf[i-1]) + self.bu[i-1]
            elif max(nointer_list) > max(inter_list):
                i=max(nointer_list)
                self.eocbu = self.bu[i]
            else:
                print('Warning: no correct burnup was calculated, set to zero')
                print('CaseID:', self.fileindex)
                print('kinf:', self.kinf)
                print('burnup:', self.bu)
                self.eocbu=0
                
        else:   # no depletion here
            self.eocbu = 0
                
        self.output=[self.kinf0, self.maxpppf, self.avgenrich, self.eocbu]
        
        return 
    
    
    def compute_rewards(self):
        """
        Calculate objective function and return rewards as 1/f(x)*100 to show scale differences between patterns
        """
        
        # check all constraints
        
        if self.maxkinf > 1.15:
            self.wk=-5000*(self.maxkinf-1.15)
        elif self.maxkinf < 1.1:
            self.wk=-5000*(1.1-self.maxkinf)
        else:
            self.wk=0
        
        if self.maxpppf > 1.45:
            self.wp=-1000*(self.maxpppf-1.3)
        elif self.maxpppf < 1.45 and self.maxpppf > 1.3:
            self.wp=-500*(self.maxpppf-1.3)
        else:
            self.wp=0
        
        if self.avgenrich > 4.05:
            self.we=-1000*(self.avgenrich-4.05)
        else:
            self.we=0.0
        
        self.reward += self.eocbu + self.wp + self.wk + self.we   
    
    def get_stat(self):
        """
        This function calculates all necessary stats for the pattern, necessary for logging 
        """
        # Convert enrichvec intro real uo2 and gad enrichments 
        self.uo2vec=[]; self.gadvec=[]
        for i in range (len(self.enrichvec)):
            self.uo2vec.append(self.Dict[self.enrichvec[i]][0])
            self.gadvec.append(self.Dict[self.enrichvec[i]][1])
            
            
        # get statistics 
        self.avg_en=np.round(np.mean(self.uo2vec),2)
        gad_check=[i for i in self.gadvec if i > 1]
        
        # count gad rods, don't duplicate if on diagonal 
        self.gad_rods=0
        for i in range (len(self.gadvec)):
            if i in [0,2,5,9,14,18,23,31,40,50] and self.gadvec[i] > 1: # diagonal entries
                self.gad_rods+=1
            if i not in [0,2,5,9,14,18,23,31,40,50] and self.gadvec[i] > 1:
                self.gad_rods+=2
        
        self.avg_gad=np.round(np.mean(gad_check),2)
        self.num_fuel=len(Counter(self.uo2vec).keys())    