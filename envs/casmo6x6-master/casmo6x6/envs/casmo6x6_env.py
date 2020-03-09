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
    
    def __init__ (self, bu=0.0):
        
        # Misc parameters for this assembly case
        self.numlocs=21 #see assembly board
        self.n=6   #6x6 
        self.fueltypes=2  #1.87 and 2.53
        
        self.fileindex=1 # to track casmo input/output names
        self.flatten=1   # to flatten arrays and return vector state (better to stay as 1 for keras NN)
        self.reward_hist=[]   # this list to register reward history for plotting purposes
        
        #*******************
        # Action/State Space
        #*******************
        # !!! These variables are very important, and their names have to be like this to work with Gym
        self.action_space = Discrete(self.fueltypes)  # Action space is discrete with two possibilities
        self.ordered_placement=0 # this option is depreicated and no longer used, random walk is used, so the second state space is used
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
    
    '''

    '''
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
        os.system('find . -type f -name case\* -exec rm {} \;')
        os.system('find . -type f -name tmp\* -exec rm {} \;')
        os.system('find . -type f -name fort\* -exec rm {} \;')
        os.system('find . -type f -name tmpout\* -exec rm {} \;')
        
    
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
        elif self.kinf0 < 1.2:
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
        elif self.avgenrich < 2.0:
            we=1000
        else:
            we=2.0
        
        self.of=wk * (1.25-self.kinf0) + wp*(1.35-self.maxpppf) + we * (2.3-self.avgenrich)
        self.of_list.append(-self.of) #update the objective function list
        
        if self.counter==self.numlocs-1:
            self.reward=100*1/self.of
        else:
            self.reward=0
                
        
        self.creward+=self.reward
        
        # mir: debugging line  
        # print(str(self.counter)+', kinf='+str(self.kinf0), 'enrich='+str(self.avgenrich), 'ppf=' + str(self.maxpppf), 'obj='+str(np.round(self.of,3)), 'reward='+str(np.round(self.creward,3)))
        
    def step(self, action):
        
        """
        VERY IMPORTANT 
        This function steps in the enviorment by executing RunCasmo4 and computes the reward based on the observed state
        Takes input as an action
        Retruns next state, rewards, termination_flag
        """
        action = action + 1 # action goes from 0 to num_colors-1 so we need to add one to get the actual color

        self.file='case'+str(self.fileindex)
        
        #*********************************************************************************
        # mir: Cheating actions, only for debugging (replace what comes from the network)
        #*********************************************************************************
        if (0):
            # These are some high quality patterns taken from brute-force search
            cheat_action=[1,2,2,1,2,1,2,2,2,2,2,2,2,1,2,2,1,1,1,2,1]
            cheat_action=[1,1,2,2,2,1,2,2,2,2,2,1,1,2,1,2,1,2,2,1,2]
            action=cheat_action[self.counter]
        
        #*********************************************************************************      
        # Sanity checks
        #*********************************************************************************
        if (action > self.fueltypes or action <= 0):
            print("Illegal action (fuel type): {} attempted.".format(action))
            raise('error')
            return [self.GetStateView(), -1, self.done, {}]

        if self.done:
            print("Game is over, all trials are done")
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
        #-----------------------------------
        #execute casmo4 and compute rewards 
        #-----------------------------------
        if self.counter==self.numlocs-1:   
            self.RunCasmo4()
            self.compute_rewards()
            # print('casmo reward:', self.reward)
        else:
            # print('No casmo run')
            self.reward=0
        
        # mir: debugging lines
        # if self.counter in [self.numlocs-1]:            
        #     print(str(self.counter)+', kinf='+str(self.kinf0), 'enrich='+str(self.avgenrich), 'ppf=' + str(self.maxpppf), 'obj='+str(np.round(self.of,4)), 'reward='+str(np.round(self.creward,3)))
        #*********************************************************************************
        #update the assembly state and counter 
        #*********************************************************************************
        self.counter += 1
        if self.ordered_placement:
            self.state = self.enrichvec  #!!! depricated
        else:
            self.state[:,0] = self.enrichvec
        #*********************************************************************************
        # check if game is over
        #*********************************************************************************
        if self.counter == self.numlocs:
            self.done = True
            self.reward_hist.append(self.creward)
            
            self.placevec[self.current_loc]=0   #convert the placment vector to all zeros for a new episode
            if not self.ordered_placement:
                self.state[:,1]=self.placevec  #initialize the placment array again for a new episode
            # self.render()      # uncomment to plot the assembly at end of episode     
        else:
            self.GetNextLocation()  #if episode does not end, get the next position to visit
        #*********************************************************************************
        self.fileindex+=1    #update file index for casmo4

        return [self.GetStateView(), self.reward, self.done, {}]    #these must be returned by the step function
    
    def render(self, mode='human'):
        
        """
        Render (optional): plot the assembly and the enrichments for each pin
        """
        uo2=self.ConstructAssembly(self.enrichvec)
        # Make the mesh plot
        plt.figure()
        plt.imshow(uo2); plt.set_cmap('summer'); plt.axis('off')
                
        # Loop over data dimensions and create text annotations.
        for i in range(uo2.shape[0]):
            for j in range(uo2.shape[1]):
                plt.text(j, i, uo2[i, j], ha="center", va="center", color="k", fontsize=13)
                

        self.of=100
        if (self.counter==self.numlocs):
            plt.title('Step '+str(self.counter)+'/'+str(self.numlocs)+ ', Reward='+ str(np.round(self.creward,2)) +'\n $k_{\infty}^0$=' + str(self.kinf0)+ ', PPF=' + str(self.maxpppf) + ', U-235 wt%=' + str(np.round(self.avgenrich,2)))
        else:
            plt.title('Step '+str(self.counter)+'/'+str(self.numlocs) +'\n $k_{\infty}^0$=' + str(self.kinf0)+ ', PPF=' + str(self.maxpppf) + ', U-235 wt%=' + str(np.round(self.avgenrich,2)))
  
        plt.tight_layout()
        #if (self.counter==self.numlocs):
        plt.savefig('graph_'+self.file+'.png',format='png', dpi=300, bbox_inches="tight")
        plt.show()
        
    def reset(self):
        """
        Important: Reset the enviroment after termination is reached
        """        
        self.locs=list(range(0,self.numlocs))  # generate a list of locations from 0 to numlocs-1
        self.enrichvec=[1.5]*self.numlocs  # this vector contains the enrichment in each rod, fixed to 1.5
        self.placevec=[0]*self.numlocs # intialize the pointer vector to zero
        
        if self.ordered_placement:
            self.state=self.enrichvec # !!!depericated
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
        self.creward=0   # this is for cumulative reward sum, used for testing purposes only
        self.clean()    # clean the directory 

        return self.GetStateView()  # the reset function MUST return the initial state 
    
    def eval_of(self, x):
        """
        This objective function is special for GA, SA, PSO, and other classical optimistion methods
        It recieves x enrichmet as input and returns objective function as output
        """
        self.enrichvec=[]
        
        for i in range (len(x)):
            if x[i]==0:
                self.enrichvec.append(1.87)
            elif x[i]==1:
                self.enrichvec.append(2.53)
            else:
                raise('the integer selected is not defined') 
                
        self.file='case'+str(self.fileindex)
        
        #-----------------------------------
        #execute casmo4 and compute rewards 
        #-----------------------------------
        self.RunCasmo4()
        self.compute_rewards()
        #print(self.enrichvec)
        print(str(self.fileindex)+', kinf='+str(self.kinf0), 'enrich='+str(self.avgenrich), 'ppf=' + str(self.maxpppf), 'obj='+str(np.round(self.of,3)))
        
        self.fileindex+=1    #update file index for casmo4
        
        if (0):
          return self.of,
        else:
          return self.of
        
    def plot_rewards(self):
        """
        plot the reward history for logging, this is for mir to check progress
        """
        plt.figure()
        plt.plot(self.reward_hist)
        plt.xlabel('Episode'); plt.ylabel('Reward')
        plt.savefig('dqn_reward.png',format='png', dpi=150)
        plt.close()
