"""
Created on Sun Jan 14 13:06:10 2020

@author: Paul Seurinh
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Discrete, Box, Tuple
import yaml

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import copy
import random
import math
from random import randint
import glob

s1_templates = 's1.inp'
s2_templates = 's2.inp'
s3_templates = 's3.inp'
sEq_templates = 'simenrifbaeq.inp'
SDM_templates = [s1_templates,s2_templates,s3_templates]#sEq_templates]
def split(boron, exposure, radial_peaking,core_exposure):
    """
    Function use to only store eauillibrium cycle born, exposure etc...
    """

    temp_boron = []
    increment = 0
    for index in range(len(boron)):
        increment +=1
        if boron[index] == 0.0 and index != len(boron) - 1:
            # temp_boron = boron[index - increment:index]
            # print(increment)
            increment = 0
    temp_boron = boron[len(boron) - increment:len(boron)]
    temp_exposure = exposure[len(boron) - increment:len(boron)]
    temp_radial_peaking = radial_peaking[len(boron) - increment:len(boron)]
    temp_core_exposure = core_exposure[len(boron) - increment:len(boron)]
    return(temp_boron,temp_exposure,temp_radial_peaking,temp_core_exposure)    

# --- Fixing random seed for reproducibility
np.random.seed(23)

QUARTER_CORE_MAP = [[0,1,2,3],[4,5,6,7,8,9],[10,11,12,13,14,15,16],[17,18,19,20,21,22,23],[24,25,26,27,28,29,30,31],[32,33,34,35,36,37,38,39],[40,41,42,43,44,45,46,47],[48,49,50,51,52,53,54,55]] 
#FULL_CORE_MAP = [[0,1,2,3,4,5,6],[7,8,9,10,11,12,13,14,15,16,17],[17,18,19,20,21,22,23,24,25,26,27,28,29],[30,31,32,33,34,35,36,37,38,39,40,41,42],[43,44,45,46,47,48,49,50,51,52,53,54,55,56,55,56,57],[58,59,60,61,62,63,64,65,66,67,68,69,70,71,72],[73,74,75,76,77,78,79,80,81,82,83,84,85,86,87],[88,89,90,91,92,93,94,95,96,97,98,99,100,101,102],[103,104,105,106,107,108,019,110,111,112,113,114,115,116,117],[118,119,120,121,122,123,124,125,126,127,128,129,130,131,132],[133,134,135,136,137,138,139,140,141,142,143,144,145,146,147],[148,149,150,151,152,153,154,155,156,157,158,159,160],[161,162,163,164,165,166,167,168,169,170,171,172,173],[174,175,176,177,178,179,180,181,182,183,184],[185,186,187,188,189,190,191]]

initial_core = [['K-01', 'K-02', 'H-01', 'J-02', 'B-04', 'D-03', 'F-01'], ['M-04', 'P-07', 'TYPE02', 'TYPE02', 'TYPE02', 'TYPE02', 'TYPE02', 'TYPE02', 'TYPE02', 'B-08', 'D-02'], ['P-04', 'TYPE02', 'TYPE02', 'TYPE02', 'L-02', 'TYPE01', 'N-05', 'TYPE01', 'E-02', 'TYPE02', 'TYPE02', 'TYPE02', 'D-04'], ['H-02', 'TYPE02', 'N-03', 'P-06', 'TYPE01', 'M-06', 'TYPE01', 'C-07', 'TYPE01', 'C-04', 'C-03', 'TYPE02', 'G-02'], ['R-06', 'TYPE02', 'TYPE02', 'M-03', 'TYPE01', 'K-04', 'TYPE01', 'N-07', 'TYPE01', 'G-05', 'TYPE01', 'F-02', 'TYPE02', 'TYPE02', 'A-06'], ['N-04', 'TYPE02', 'P-05', 'TYPE01', 'L-07', 'L-03', 'J-05', 'TYPE01', 'D-08', 'C-05', 'D-06', 'TYPE01', 'B-05', 'TYPE02', 'B-06'], ['M-02', 'TYPE02', 'TYPE01', 'J-03', 'TYPE01', 'H-04', 'H-06', 'L-05', 'F-08', 'E-07', 'TYPE01', 'F-04', 'TYPE01', 'TYPE02', 'A-08'], ['P-09', 'TYPE02', 'L-13', 'TYPE01', 'J-13', 'TYPE01', 'L-11', 'TYPE03', 'E-05', 'TYPE01', 'G-03', 'TYPE01', 'E-03', 'TYPE02', 'B-07'], ['R-08', 'TYPE02', 'TYPE01', 'K-12', 'TYPE01', 'L-09', 'K-08', 'E-11', 'H-10', 'H-12', 'TYPE01', 'G-13', 'TYPE01', 'TYPE02', 'D-14'], ['P-10', 'TYPE02', 'P-11', 'TYPE01', 'M-10', 'N-11', 'M-08', 'TYPE01', 'G-11', 'E-13', 'E-09', 'TYPE01', 'B-11', 'TYPE02', 'C-12'], ['R-10', 'TYPE02', 'TYPE02', 'K-14', 'TYPE01', 'J-11', 'TYPE01', 'C-09', 'TYPE01', 'F-12', 'TYPE01', 'D-13', 'TYPE02', 'TYPE02', 'A-10'], ['J-14', 'TYPE02', 'N-13', 'N-12', 'TYPE01', 'N-09', 'TYPE01', 'D-10', 'TYPE01', 'B-10', 'C-13', 'TYPE02', 'H-14'], ['M-12', 'TYPE02', 'TYPE02', 'TYPE02', 'L-14', 'TYPE01', 'C-11', 'TYPE01', 'E-14', 'TYPE02', 'TYPE02', 'TYPE02', 'B-12'], ['M-14', 'P-08', 'TYPE02', 'TYPE02', 'TYPE02', 'TYPE02', 'TYPE02', 'TYPE02', 'TYPE02', 'B-09', 'D-12'], ['K-15', 'M-13', 'P-12', 'G-14', 'H-15', 'F-14', 'F-15']]

# -----------------
# --- futur heuristic (e.g No fresh fuel assembly on the side)
HEUR_INDICES = [0,1,2,3,4,8,14,20,27,34]
Good_state = [4, 0, 37, 3, 25, 36, 15, 18, 19, 14, 9, 17, 24, 6, 20, 22, 26, 10, 29, 23, 39, 5, 30, 1, 32, 2, 28, 27, 21, 12, 7, 35, 13, 38, 33, 11, 40, 31, 8, 34, 16]
Good_energy = 0.00015534396267663473 # -- minima

vertical_indices = [3,9,16,23,31,39,47]
horizontal_indices   = [48,49,50,51,52,53,54]
# -----------------
# --- Optimization method

optim_method = {1 :'COBYLA', 2 : 'SBPLX', 3 : 'ISRES', 4 : 'ESCH', 5: 'GA', 6:'PSO', 7:'SA', 8 : 'RL'} # --- optim algorithm method

def search_element(target_in, element,flag):
    """
    Search an element in a list
    """
    target = copy.deepcopy(list(target_in))
    if flag == 0: # --- matrix
        for row in range(len(target)):
            try:
                column = target[row].index(element)
            except ValueError:
                continue
            return(row, column)
    elif flag == 1: # --- vector
        try:
            column = target.index(element)
        except ValueError:
            print('Error not found')
            print(element)
            print(target)
            sys.exit()
        return(column)
    return 0

def name_mapping(vector,flag):
    """ Optimization Method list of float not string """
    float_to_core = {}
    core_to_float = {}
    iteration = 0
    if flag == 0: # --- It is a list if horizontal/vertical line is swapped 
        for row in range(len(vector)):
            for col in range(len(vector[row])):
                float_to_core[iteration] = str(vector[row][col])
                core_to_float[str(vector[row][col])] =  iteration
                iteration  += 1
    elif flag == 1:
        for row in range(len(vector)):
            float_to_core[iteration] = str(vector[row])
            core_to_float[str(vector[row])] =  iteration
            iteration  += 1    
    return (float_to_core, core_to_float)

"""
    Implementation of a black-boxed environment for SIMULATE3 code for a PWR 15x15 assemblies with 2 discrete actions
    
    Any RL enviroment consists of four major functions:
        1- initializer: to start the enviroment, define action space, state space 
            * For black-box codes, the initializer would be special as input files or templates 
            should be added. Hard-coded templates are used here without external input files.  
        2- Step: which recieves an action and observe the new state and reward
        3- reset: to reset the enviroment back to first state, when end of episode occurs
        4- render: to visualize the problem (optional)
    
    ***************
    # --- Game Scenario : test
    ***************
    1- Start from an initial configuration : usually equilibirum cycle of a reactor core 
    2- Binary swap two assemblies (randomly choosen)
    3- Observe next state (new core configuration), calculate the reward and update CNN
    4- Take next action, and repeat the process until stopping criterion is reached : n episodes observed or better solution obtained
    5- Repeat m times (batch)
    6- Choose a new initial configuration : with 1 - p probability of being best solution observed and p of randomly generated
    7- Repeat Q times
    
    ***************
    # --- Assembly Board : (initial board)
    ***************
                                 K-01   K-02   H-01   J-02
                   M-04   P-07   TYPE02 TYPE02 TYPE02 TYPE02
            P-04   TYPE02 TYPE02 TYPE02 L-02   TYPE01 N-05
            H-02   TYPE02 N-03   P-06   TYPE01 M-06   TYPE01
     R-06   TYPE02 TYPE02 M-03   TYPE01 K-04   TYPE01 N-07
     N-04   TYPE02 P-05   TYPE01 L-07   L-03   J-05   TYPE01
     M-02   TYPE02 TYPE01 J-03   TYPE01 H-04   H-06   L-05
     P-09   TYPE02 L-13   TYPE01 J-13   TYPE01 L-11   TYPE03
    ***************
    # --- Corresponding integer board :
    ***************

                        0   1   2   3 
                4   5   6   7   8    9   
            10  11  12  13  14  15  16
            17  18  19  20  21  22  23  
        24  25  26  27  28  29  30  31  
        32  33  34  35  36  37  38  39  
        40  41  42  43  44  45  46  47  
        48  49  50  51  52  53  54  55  
        

    
"""
# ----- Decorator : Allow to modify the behavior of function or class
# ----- To call above function of interest 
# ----- Here handle if sanity is obtained before and after calling some specific functions

def check_rep_decorate(func): # --- Isaac github function
    def func_wrapper(self,*args, **kwargs):
        self._check_rep()
        out = func(self,*args, **kwargs)
        self._check_rep()
        return out
    return func_wrapper

def check_swap_decorate(func): # --- Verify that only horizontal to horizontal or vertical to vertical are swapped
    def func_wrapper(self,*args, **kwargs):
        #self._check_swap(*args)
        out = func(self,*args, **kwargs)
        return out
    return func_wrapper

class SIMULcolEnv3(gym.Env):
  """
  class for gym-wise SIMULATE3 environment

  """
  def __init__ (self, path_to_config='./config_col2.yaml', casename='method', log_dir='./master_log/'):#, max_episode = 10, run_number = 12): # --- '12' because of my default initial file
    # -------------------------------------------------------
    # --- Attributs for RL
    self.log_dir=log_dir
    self.casename=casename
    with open(path_to_config,"r") as yamlfile:
        config = yaml.safe_load(yamlfile)
        self.numlocs = config["gym"]["numlocs"] # --- size of the board to make test on : 4 9 16 25 40
        self.max_episode = config["gym"]["max_episode"] # --- tuning parameter
        self.run_number = config["gym"]["run_number"]
        self.swap_type = config["gym"]["swap_type"] # --- number of swap : simple : 1 / binary : 2 / triple : 3 / ... / full size swap treated through color environment
        self.flatten = config["gym"]["flatten"]
        self.Gamma = 0.95
        if self.max_episode % 2 * self.swap_type != 0:
            print("The number of episode must be a multiplier of 2 times the number of swap not {}".format(self.max_episode % 2 * self.swap_type))
        #--------------------------------------------------------------------
        # --- template use for initial input file 3D can be used later on
        #--------------------------------------------------------------------
        self.templates_file = config["gym"]["initial_file"]
        self.ordered_placement = config['gym']['ordered_placement'] # true results in deterministic placement order
        self.disable_checking = config['gym']['disable_checking'] # turns off board color checking at terminal step

    self.flag = 0 # --- 1 for full core (remove parameters in self.free_coords) anything else for swapping
    self.reward_hist = [0]   # this list to register reward history for plotting purposes
    self.current_score = 0.0   # --- reward at time t being in state s taking action a
    self.creward = 0.0 # --- cumulative reward
    self.counter = 0 # --- keeper : number of episode
    self.subcounter = 0 # --- Count the number of assemblies choosen for swap
     
    self.default_state = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55]
    self.VECTOR = []
    self.core = []
    self.librairies = {}
    self.core_correspondance = {}
    self.horizontal_correspondance = {}
    self.vertical_correspondance = {}
    self.float_to_core = {} # --- get the mapping integer --> string
    self.core_to_float = {}


    # --- *******************
    # --- Action/State Space
    # --- *******************
    # --- !!! Pivotal to make it work with gym
    
    self.state = np.zeros((2,len(self.default_state)), dtype = int)
    self.state[0] = self.default_state
    self.free_coords = []
    self._new_free_coords()
    self.current_loc = None
    self._first_location()
    self.state[1][self.current_loc] = 1
    self.buffer = [] # --- record swaps
    self.state_file = "" # --- contain *.out file from a run with self.state
    
    self.action_space = Discrete(self.numlocs)
    if self.flatten:
        self.observation_space = Box(low=0, high=len(self.default_state), shape=(len(self.default_state) * 2,), dtype=int)
    else:
        self.observation_space = Box(low=0, high=len(self.default_state), shape=(len(self.default_state),2), dtype=int)

   

    # -------------------------------------------------------------------
    # --- attributs for render
    # -------------------------------------------------------------------
    self.axial = {} # --- dictionnary containing axial quantities (3D)
    self.radial = {} # --- dictionnary containing axial quantities (2D)

    # ---------------------------------------------
    # ---
    # --- Attributs for reward function

    self.file = "" # --- run_XX_gym.inp for each calculation
    self.kinf0    = 0.0 # --- initial kinf ( > 1.00 )
    self.ppf0    = 0.0 # --- initial power peaking, must satisfy constraint as well ( < 1.35 )
    self.maxkinf  = 0.0 # --- maximum kinf ( < 1.13 )
    self.maxppf  = 0.0 # --- maximum pin power peaking factor ( < 1.35 )
    self.maxeocbu    = 0.0
    self.avgenrich  = 0.0 # --- average enrichment ( minimize )
    self.max_of = 0 # --- Max objective function (keeper for best solution)
    self.best_fit = [] # --- best_state
    self.feasible = 0
    self.max_exposure = 0.0
    self.max_boron = 0.0
    self.max_peaking = 0.0
    self.max_deltah = 0.0

    # --- Clean directory and reset enviroment when intialize the enviroment
    self.reset()
    self.clean()
    #self.Download()
    self.done = False # --- true if environement has reached terminal state, false otherwise
  
  def check_heuristic(self,x,float_to_core):
    """ Control that no fresh fuel are placed on the outer ring """
    def Flipt_bits(particle,bits):
        """ Return possible mutation (at position bits) """
        a = np.random.randint(0,self.numlocs)
        index = search_element(particle,a,1)
        particle[index] = particle[bits]
        particle[bits] = a
        return(particle)
    iter = 0
    while iter  != 10:
        iter = 0 
        for index,elem in enumerate(HEUR_INDICES):
            if 'TYPE' in float_to_core[x[elem]]:
                x = Flipt_bits(x,index)
                # print('Bits were randomly swap because fresh fuels cannot be moved at the outmost regions')
                iter -= 1
            else:
                iter += 1
    return(x)


  def check_heuristic_2(self): # --- further penalty on assembly put at the periphery
        """
        Ask new action if false
        """
        for diff in HEUR_INDICES:
            if 'TYPE' in self.float_to_core[self.state[0][diff]]:
                return False

        return True


  def _get_state_agent_view(self):
    """
    Return the view of the agent state, which can be flatten or not
    """
    if self.flatten:
        return self.state.flatten()
    else:
        return self.state
  def _first_location(self):
        """
        Get the first location to place a piece on the board state before returning the latter for the first time

        """
        if self.ordered_placement: # --- Deterministic placement
            self.current_loc = self.free_coords.pop(0)
        else:
            # set the first location in the placement array
            self.current_loc = random.choice(self.free_coords) # the next location to place a piece at
            self.free_coords.remove(self.current_loc)  
  
  def _new_free_coords(self):
        """
        regenerates the self.ordered_placement collection, which is an ordered list if placement
        is deterministic and a set if placement order is non-deterministic

        """
        if not self.ordered_placement:
          self.free_coords = []
          for i in range(self.numlocs):
                self.free_coords.append((i))
        else:
          self.free_coords = set() # --- a set of all the remaining coordinates to put pieces in
          for i in range(self.numlocs):
            self.free_coords.add((i))  

  def _num_nonzero(self,arr):
        """
        Get the number of non-zero element in th eprovided array

        """
        return len(np.transpose(np.nonzero(arr)))


  @check_rep_decorate
  def _get_new_location(self,action,flag): # --- 
        """ 
        Get a new coordinate from the free_coords set
        Remember to set new location to 1 and old location to 0

        """
        assert len(self.free_coords), "free_coords is empty, there are no more positions available"
        assert len(self.buffer) <= 2 * self.swap_type, "The buffer is larger than it should be"

        if self.ordered_placement: # --- deprecated
            new_loc = self.free_coords.pop(0)
        else:
            new_loc = action
            if flag == 1: # --- When full core no swap activate it
                # get new location and remove it from future options
                #print("free coordinate",self.free_coords)
                #print(new_loc)
                #print(self.buffer)
                self.free_coords.remove(new_loc)

        # set old location in placement array to 0, set new location to 1
        self.state[1][self.current_loc] = 0
        self.current_loc = new_loc
        self.state[1][self.current_loc] = 1

        return 0


  @check_rep_decorate
  def _is_valid_location(self,location):
        """
        Check if the given location x is inside the board and returns boolean accordingly
      
        """

        return location >= 0 and location < self.numlocs 


  def _check_legal_board(self):
        """
        Assume the board is full of piece (end of episode). Check if the board is in a legal configuration according the the rule
        return True or False accordingly

        """
        if self.disable_checking:
            return True

        # --- Check if no fresh fuel assemblies at the outmost frontier
    
        self.state[0] = self.check_heuristic(self.state[0],self.float_to_core)

        board = self.state[0]

        if len(board) != len(list(set(board))): # --- check if not duplicate
            return False

        return True


  def _check_swap(self,a,b):
      """
      Verify that quarter core assemblies are not swapped with in-core assemblies
      """
      if a not in vertical_indices and a not in horizontal_indices:
          assert b not in vertical_indices and b not in horizontal_indices, "a is in quarter core therefore b must be as well"
      elif a in vertical_indices:
          assert b in vertical_indices, "a is in vertical therefore b must be as well"
      elif a in horizontal_indices:
          assert b in horizontal_indices, "a is in horizontal therefore b must be as well"
      


        
  @check_swap_decorate
  def move_quarter(self,a,b): # --- < integer > < integer > < integer > < integer >
      """ Switch two assemblies in the second quadrant 
          Defines one step in the action space
      """
      # -------------------------------------------------------
      # ---
      # --- Store quarter core : slow
      # ---
    #   #print("test",self.state[0])


      #self.VECTOR = []
      #for row in range(8): # --- also store horizontal and vertical line
      #  self.VECTOR.append([])
      #  for col in range(0,int(np.ceil(len(self.core[row]) / 2))):
      #      self.VECTOR[row].append(self.core[row][col])
      #
      #float_to_core, core_to_float = name_mapping(self.VECTOR,0)
      #if a in horizontal_indices and b in vertical_indices:
      #    # --- require to change the dictionnary
      #    temp1 = [float_to_core[a],self.horizontal_correspondance[float_to_core[a]]]
      #    if 'TYPE' not in float_to_core[a]:
      #      del self.horizontal_correspondance[float_to_core[a]]
      #    self.vertical_correspondance[float_to_core[a]] = temp1[1]
      #    temp1 = [float_to_core[b],self.vertical_correspondance[float_to_core[b]]]
      #    if 'TYPE' not in float_to_core[b]:
      #      del self.vertical_correspondance[float_to_core[b]]
      #    self.horizontal_correspondance[float_to_core[b]] = temp1[1]
#
      #elif b in horizontal_indices and a in vertical_indices:
      #    return self.move_quarter(b,a)
      #  
      #elif a not in vertical_indices and a not in horizontal_indices and b in horizontal_indices:
      #    # --- require to change the dictionnary
      #    temp1 = [float_to_core[a],self.core_correspondance[float_to_core[a]][0],self.core_correspondance[float_to_core[a]][1],self.core_correspondance[float_to_core[a]][2]]
      #    if 'TYPE' not in float_to_core[a]:
      #      del self.core_correspondance[float_to_core[a]]
      #    self.horizontal_correspondance[float_to_core[a]] = [temp1[3]]
      #    self.vertical_correspondance[temp1[2]] = [temp1[1]]
#
      #    a_sub,b_sub = search_element(QUARTER_CORE_MAP,b,0)
      #    if 'TYPE' not in self.float_to_core[QUARTER_CORE_MAP[b_sub][len(QUARTER_CORE_MAP[b_sub]) - 1]]:
      #      del self.float_to_core[QUARTER_CORE_MAP[b_sub][len(QUARTER_CORE_MAP[b_sub]) - 1]]
      #    self.float_to_core[QUARTER_CORE_MAP[b_sub][len(QUARTER_CORE_MAP[b_sub]) - 1]] = temp1[2]
      #    
      #    temp1 = [float_to_core[b],self.horizontal_correspondance[float_to_core[b]],[self.VECTOR[b_sub][len(self.VECTOR[b_sub]) - 1]],self.vertical_correspondance[self.VECTOR[b_sub][len(self.VECTOR[b_sub]) - 1]]]
      #    if 'TYPE' not in float_to_core[b]:
      #      del self.horizontal_correspondance[float_to_core[b]]
      #    if 'TYPE' not in self.VECTOR[b_sub][len(self.VECTOR[b_sub]) - 1]:
      #      del self.vertical_correspondance[self.VECTOR[b_sub][len(self.VECTOR[b_sub]) - 1]]
      #    self.core_correspondance[temp1[0]] = []
      #    self.core_correspondance[temp1[0]].append(temp1[3][0])
      #    self.core_correspondance[temp1[0]].append(temp1[2][0])
      #    self.core_correspondance[temp1[0]].append(temp1[1][0])
      #    
#
#
      #elif b not in vertical_indices and b not in horizontal_indices and a in horizontal_indices:
      #    return self.move_quarter(b,a)
#
      #elif a not in vertical_indices and a not in horizontal_indices and b in vertical_indices:
      #    # --- require to change the dictionnary
      #    temp1 = [float_to_core[a],self.core_correspondance[float_to_core[a]][0],self.core_correspondance[float_to_core[a]][1],self.core_correspondance[float_to_core[a]][2]]
      #    if 'TYPE' not in float_to_core[a]:
      #       del self.core_correspondance[float_to_core[a]]
      #    self.vertical_correspondance[float_to_core[a]] = [temp1[3]]
      #    self.horizontal_correspondance[temp1[1]] = [temp1[2]]
      #  
      #    
      #    a_sub,b_sub = search_element(QUARTER_CORE_MAP,b,0)
      #    if 'TYPE' not in self.float_to_core[QUARTER_CORE_MAP[len(QUARTER_CORE_MAP) - 1][a_sub]]:
      #       del self.float_to_core[QUARTER_CORE_MAP[len(QUARTER_CORE_MAP) - 1][a_sub]]
      #    self.float_to_core[QUARTER_CORE_MAP[len(QUARTER_CORE_MAP) - 1][a_sub]] = temp1[1]
#
      #    temp1 = [float_to_core[b],self.vertical_correspondance[float_to_core[b]],[self.VECTOR[len(self.VECTOR) - 1][a_sub]],self.horizontal_correspondance[self.VECTOR[len(self.VECTOR) - 1][a_sub]]]
      #    if 'TYPE' not in float_to_core[b]:
      #      del self.vertical_correspondance[float_to_core[b]]
      #    if 'TYPE' not in self.VECTOR[len(self.VECTOR) - 1][a_sub]:
      #      del self.horizontal_correspondance[self.VECTOR[len(self.VECTOR) - 1][a_sub]]
      #    
      #    self.core_correspondance[temp1[0]] = []
      #    self.core_correspondance[temp1[0]].append(temp1[2][0])
      #    self.core_correspondance[temp1[0]].append(temp1[3][0])
      #    self.core_correspondance[temp1[0]].append(temp1[1][0])
#
      #elif b not in vertical_indices and b not in horizontal_indices and a in vertical_indices:
      #  return self.move_quarter(b,a)

    #   x_code = np.array(list(self.float_to_core.keys()))
    #   a_prim = a
    #   b_prim = b
    #   temp = x_code[b_prim]
    #   x_code[b_prim] = x_code[a_prim]
    #   x_code[a_prim] = temp 
    #   self.state[0] = list(x_code)
      
    #   a_sub,b_sub = search_element(QUARTER_CORE_MAP,a,0)
    #   a_sub1,b_sub1 = search_element(QUARTER_CORE_MAP,b,0)
    #   temp4 = copy.deepcopy(self.VECTOR[a_sub][b_sub])
    #   self.VECTOR[a_sub][b_sub] = copy.deepcopy(self.VECTOR[a_sub1][b_sub1])
    #   self.VECTOR[a_sub1][b_sub1] = copy.deepcopy(temp4)

      a_prim = copy.deepcopy(a)
      b_prim = copy.deepcopy(b)
      temp = copy.deepcopy(self.state[0][b_prim])
      self.state[0][b_prim] = copy.deepcopy(self.state[0][a_prim])
      self.state[0][a_prim] = copy.deepcopy(temp)
      print(self.state[0])
      
      a_sub,aa_sub = search_element(QUARTER_CORE_MAP,a,0)
      b_sub, bb_sub = search_element(QUARTER_CORE_MAP,b,0)

      # --- try to switch
      core_temp = copy.deepcopy(self.core)
      if a not in vertical_indices and b not in vertical_indices and a not in horizontal_indices and b not in horizontal_indices:
        temp = copy.deepcopy(self.core[b_sub][bb_sub])
        core_temp[b_sub][bb_sub] = copy.deepcopy(core_temp[a_sub][aa_sub])
        core_temp[a_sub][aa_sub] = copy.deepcopy(temp)
        print(temp)
        temp = copy.deepcopy(core_temp[-b_sub + len(self.core) - 1][bb_sub])
        core_temp[-b_sub + len(self.core) - 1][bb_sub]  = copy.deepcopy(core_temp[-a_sub + len(self.core) - 1][aa_sub])
        core_temp[-a_sub + len(self.core) - 1][aa_sub] = copy.deepcopy(temp)
        print(temp)
        temp = copy.deepcopy(core_temp[b_sub][-bb_sub + len(self.core[b_sub]) - 1])
        core_temp[b_sub][-bb_sub + len(self.core[b_sub]) - 1]  = copy.deepcopy(core_temp[a_sub][-aa_sub + len(self.core[a_sub]) - 1])
        core_temp[a_sub][-aa_sub + len(self.core[a_sub]) - 1] = copy.deepcopy(temp)
        print(temp)
        temp = copy.deepcopy(core_temp[-b_sub + len(self.core) - 1][-bb_sub + len(self.core[b_sub]) - 1]) 
        core_temp[-b_sub + len(self.core) - 1][-bb_sub + len(self.core[b_sub]) - 1] = copy.deepcopy(core_temp[-a_sub + len(self.core) - 1][-aa_sub + len(self.core[a_sub]) - 1])
        core_temp[-a_sub + len(self.core) - 1][-aa_sub + len(self.core[a_sub]) - 1] = copy.deepcopy(temp)
        print(temp)

      if a in vertical_indices and b in vertical_indices:
            temp = copy.deepcopy(self.core[b_sub][bb_sub])
            core_temp[b_sub][bb_sub] = copy.deepcopy(core_temp[a_sub][aa_sub])
            core_temp[a_sub][aa_sub] = copy.deepcopy(temp)
            print(temp)
            temp = copy.deepcopy(core_temp[-b_sub + len(self.core) - 1][-bb_sub + len(self.core[b_sub]) - 1]) 
            core_temp[-b_sub + len(self.core) - 1][-bb_sub + len(self.core[b_sub]) - 1] = copy.deepcopy(core_temp[-a_sub + len(self.core) - 1][-aa_sub + len(self.core[a_sub]) - 1])
            core_temp[-a_sub + len(self.core) - 1][-aa_sub + len(self.core[a_sub]) - 1] = copy.deepcopy(temp)
            print(temp)


      if a in horizontal_indices and b in horizontal_indices:
            temp = copy.deepcopy(self.core[b_sub][bb_sub])
            core_temp[b_sub][bb_sub] = copy.deepcopy(core_temp[a_sub][aa_sub])
            core_temp[a_sub][aa_sub] = copy.deepcopy(temp)
            print(temp)
            temp = copy.deepcopy(core_temp[-b_sub + len(self.core) - 1][-bb_sub + len(self.core[b_sub]) - 1]) 
            core_temp[-b_sub + len(self.core) - 1][-bb_sub + len(self.core[b_sub]) - 1] = copy.deepcopy(core_temp[-a_sub + len(self.core) - 1][-aa_sub + len(self.core[a_sub]) - 1])
            core_temp[-a_sub + len(self.core) - 1][-aa_sub + len(self.core[a_sub]) - 1] = copy.deepcopy(temp)
            print(temp)

      if a in horizontal_indices and b in vertical_indices:
            temp = copy.deepcopy(self.core[b_sub][bb_sub])
            core_temp[b_sub][bb_sub] = copy.deepcopy(core_temp[a_sub][aa_sub])
            core_temp[a_sub][aa_sub] = copy.deepcopy(temp)
            print(temp)
            temp = copy.deepcopy(core_temp[-b_sub + len(self.core) - 1][-bb_sub + len(self.core[b_sub]) - 1]) 
            core_temp[-b_sub + len(self.core) - 1][-bb_sub + len(self.core[b_sub]) - 1] = copy.deepcopy(core_temp[-a_sub + len(self.core) - 1][-aa_sub + len(self.core[a_sub]) - 1])
            core_temp[-a_sub + len(self.core) - 1][-aa_sub + len(self.core[a_sub]) - 1] = copy.deepcopy(temp)
            print(temp)

      if a in vertical_indices and b in horizontal_indices:
            return(self.move_quarter(b,a))

      if a not in horizontal_indices and a not in vertical_indices and b in vertical_indices:
            temp = copy.deepcopy(self.core[b_sub][bb_sub])
            core_temp[b_sub][bb_sub] = copy.deepcopy(core_temp[a_sub][aa_sub])
            core_temp[a_sub][aa_sub] = copy.deepcopy(temp)
            print(temp)
            temp = copy.deepcopy(core_temp[-b_sub + len(self.core) - 1][-bb_sub + len(self.core[b_sub]) - 1]) 
            core_temp[-b_sub + len(self.core) - 1][-bb_sub + len(self.core[b_sub]) - 1] = copy.deepcopy(core_temp[-a_sub + len(self.core) - 1][-aa_sub + len(self.core[a_sub]) - 1])
            core_temp[-a_sub + len(self.core) - 1][-aa_sub + len(self.core[a_sub]) - 1] = copy.deepcopy(temp)
            print(temp)

      if b not in horizontal_indices and b not in vertical_indices and a in vertical_indices:
            return(self.move_quarter(b,a))
      

      if a not in horizontal_indices and a not in vertical_indices and b in horizontal_indices:
            temp = copy.deepcopy(self.core[b_sub][bb_sub])
            core_temp[b_sub][bb_sub] = copy.deepcopy(core_temp[a_sub][aa_sub])
            core_temp[a_sub][aa_sub] = copy.deepcopy(temp)
            print(temp)
            temp = copy.deepcopy(core_temp[-b_sub + len(self.core) - 1][-bb_sub + len(self.core[b_sub]) - 1]) 
            core_temp[-b_sub + len(self.core) - 1][-bb_sub + len(self.core[b_sub]) - 1] = copy.deepcopy(core_temp[-a_sub + len(self.core) - 1][-aa_sub + len(self.core[a_sub]) - 1])
            core_temp[-a_sub + len(self.core) - 1][-aa_sub + len(self.core[a_sub]) - 1] = copy.deepcopy(temp)
            print(temp)
      
      if b not in horizontal_indices and b not in vertical_indices and a in horizontal_indices:
            return(self.move_quarter(b,a))
          



    #        core_temp[a][b] = self.VECTOR[a][b]
    #        core_temp[-a + len(self.core) - 1][b]  = self.core_correspondance[self.VECTOR[a][b]][0]
    #        core_temp[a][-b + len(self.core[a]) - 1]  = self.core_correspondance[self.VECTOR[a][b]][1]
    #        core_temp[-a + len(self.core) - 1][-b + len(self.core[a]) - 1] = self.core_correspondance[self.VECTOR[a][b]][2]
    #
      self.core = copy.deepcopy(core_temp)
      #print(self.core)
      #sys.exit()
      #self.core_correspondance[self.VECTOR[row][col]] = []
#
      #      # --- Third quadrant
      #      self.core_correspondance[self.VECTOR[row][col]].append(self.core[- row + len(self.core) - 1][col])
#
      #      # --- Second quadrant
      #      self.core_correspondance[self.VECTOR[row][col]].append(self.core[row][- col + len(self.core[row]) - 1])
#
      #      # --- Fourth quadrant
      #      self.core_correspondance[self.VECTOR[row][col]].append(self.core[- row + len(self.core) - 1][-col + len(self.core[row]) - 1])
      #print(self.core)
      #sys.exit()
      #self.float_to_core, self.core_to_float = name_mapping(self.VECTOR,0)
    #   print(self.float_to_core)
      #print(self.core_correspondance)
      #print(self.vertical_correspondance)
      #print(self.horizontal_correspondance)

      return 0


  def _Generate_newBoard(self): # --- depreciated
      """ Generate a random new board """
      self.state[0][0:self.numlocs] = random.sample(self.default_state[0:self.numlocs],self.numlocs)
      return

  def _check_rep(self):
    """ Validate if internal state is in accordance with
    design invariants
    """
    if self.done:
        assert self.counter == self.max_episode, "The counter is not correct"
    else:
        assert self.counter < self.max_episode, "The counter is not correct"

    return
  @check_rep_decorate
  def step(self, action): # --- < int > : position
    """ One move in the game """
    self.file = 'run'+str(self.counter) + '_gym.inp'
    self.state_file = 'run'+str(self.counter) + '_gym.out'

    # --- *********************************************************************************
    # --- check if game is over
    # --- *********************************************************************************

    if self.done:
        if self.render == True:
            print(' ------------------------ ')
            print(' ***** Game is over ***** ')
            print(' ------------------------ ')
            print("\n")
            print("Your cumulative score : {}".format(self.creward))
            print("\n")
            print("Your highest score : {}".format(self.max_of))
            print("\n")
            print("The Configuration :")
            print(self.best_fit)
            print("\n")
        self.state = self.best_fit
        return([self._get_state_agent_view(),0.0, self.done, {}])

    if action < 0 or action > self.numlocs - 1:
        print("Illegal action attempted {} is not a valid choice".format(action))
        print("action must be in %f to 40 included"%(self.numlocs - 1))
        return([self._get_state_agent_view(),0.0, self.done, {}])
    self.counter += 1
    self.subcounter += 1
    self.buffer.append(action)
    #self.state[0][self.current_loc] = copy.deepcopy(action) 
    
    if self.counter == self.max_episode:
        self.done = True
        print("Buffer terminal",self.buffer)
        for idx in range(len(self.buffer) - 1):
            self.move_quarter(self.buffer[idx],self.buffer[idx + 1])
            #self.write_input() # charge new dictionnary for core based on new state
        self.buffer = []
        penalty = 0
        if not self.check_heuristic_2(): # --- penalize bad heuristics
            penalty -= 1000000
        self._check_legal_board() # --- randomly flips bits to avoid TYPE at the boundary. Penalize right before but allow statistic to increase   
        # --- Write input file for next calculation
        self.write_input()
        # --- Run Simulate:
        self.runSIMULATE()
        reward = self._Compute_reward()
        self.current_score = reward + penalty
        self.creward += np.power(self.Gamma,self.counter - 1) * self.current_score # --- discounted reward
        self.state[1][self.current_loc] = 0
        self.subcounter = 0
        self._new_free_coords()
        if self._check_legal_board(): # --- deprecation because does not make sense
            reward = reward
        else:
            reward = -100000000
         # --- cumulative reward record at the end of the game
        self.reward_hist.append(self.creward) # --- in or out of the loop?
    else:
        if self.subcounter == self.swap_type * 2 and self.counter != self.max_episode: # --- control compute reward only after choosing which to swap
            print("Buffer",self.buffer)
            for idx in range(len(self.buffer) - 1):
                self.move_quarter(self.buffer[idx],self.buffer[idx + 1])
                #self.write_input() # charge new dictionnary for core based on new state
            self.buffer = []
            penalty = 0
            if not self.check_heuristic_2(): # --- penalize bad heuristics
                penalty -= 1000000
            self._check_legal_board() # --- randomly flips bits to avoid TYPE at the boundary. Penalize right before but allow statistic to increase
            
            self._new_free_coords()
            # --- Write input file for next calculation
            self.write_input()
            # --- Run Simulate:
            self.runSIMULATE()
            reward = self._Compute_reward()
            if self._check_legal_board(): # --- deprecation because does not make sense
                reward = reward
            else:
                reward = -100000000
            self.current_score = reward + penalty
            self.creward += np.power(self.Gamma,self.counter - 1) * self.current_score
            self.subcounter = 0
        else:
            self._get_new_location(action,self.flag)
    # -------------------------------------------
    # --- return [state, reward, true/false, next_state]

    return ([self._get_state_agent_view(),self.current_score, self.done, {}])

  def reset(self, flag = True):
    """ Reinitialize the game """
    # ----------------------------------------------------------------------------------
    # --- Reward-related functions
    # ---
    #  
    
    
    self.creward = 0.0
    self.current_score = 0.0
    self.counter = 0
    self.subcounter = 0
    self.buffer = []
    # self.num_batch = 0

    # --- counters to zero
    # self.increment = 0 
    self.counter = 0
    self.done = False
    
    if flag: # --- start from previous state or new state
        self.file = self.templates_file
        self.axial = {} 
        self.radial = {} 
        self.core = []

        # ------------------
        # ---
        self.clean()
        self.state[0][0:self.numlocs] = self.default_state[0:self.numlocs]
        # -----------------
        # --- fill templates : dictionnary / temp.core_gym.inp...
        self.Download()
        self.float_to_core, self.core_to_float = name_mapping(self.VECTOR,0)
        self.state = np.zeros((2,len(self.default_state)), dtype = int)
        self.state[0] = self.default_state
        #self._Generate_newBoard()
        
    
    # --------------------------
    # --- Generate new initial state
    self._new_free_coords()
    self._first_location()
    self.state[1][self.current_loc] = 1

    # --- WARNING : !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # --- For now : 
    # --- Do not clean best_fit or max_of since it can be retrieved directly from the environment. 
    # --- The purpose being to follow the track fo them
    # --- self.best_fit, self.max_of = [], 0.0
    return(self._get_state_agent_view())

  def render(self, show_position_number = False, show_heuristic = False, show_peaking_factor = False):
    """ 
    Render function to print :
    """
    
    print(" " * 20, end = "")
    print("Board state :")
    F_in = self.VECTOR
    core_input = ""
    for row in range(0,len(F_in)):
        core_input += "\n"
        if row == 0:
            core_input += "%2d"%(row + 1) + "  1" 
        else:
            core_input += "\n" + "%2d"%(row + 1) + "  1" 
        for index in range(int(np.ceil((self.dim - 2 * len(F_in[row]) - 1) / 2)) + 1):
            core_input += " " + " " + " " + " " + " " + " " + " " 
        for col in range(1,len(F_in[row]) + 1):
            core_input += "|{:6s}".format(F_in[row][col - 1])
    print(core_input)

    position_counting = 0
    if show_position_number:
        print("\n")
        print(" " * 20, end = "")
        print("Corresponding Positions :")
        print("\n")
        core_input = ""
        for row in range(0,len(F_in)):
            if row == 0:
                core_input += "%2d"%(row + 1) + "  1" 
            else:
                core_input += "\n" + "%2d"%(row + 1) + "  1" 
            for index in range(int((int(np.ceil(self.dim)) - 2 * len(F_in[row]) - 1) / 2) + 1):
                core_input += " " + " " + " " + " " + " " + " " + " " 
            for col in range(1,len(F_in[row]) + 1):
                if 'TYPE01' in F_in[row][col - 1]: # --- color fresh fuels : Blue
                    core_input += "\033[94m" + "|{:6d}".format(position_counting) + '\033[0m'
                elif 'TYPE02' in F_in[row][col - 1]: # --- color fresh fuels : Green
                    core_input += "\033[92m" + "|{:6d}".format(position_counting) + '\033[0m'
                elif 'TYPE03' in F_in[row][col - 1]: # --- color fresh fuels : RED
                    core_input += "\033[91m" + "|{:6d}".format(position_counting) + '\033[0m'
                else:
                    core_input += "|{:6d}".format(position_counting)
                position_counting += 1
                # --- remove keys for duplicate fresh fuels
                #F_out.pop(F_in[row][col - 1])
        core_input += "\n"
    print(core_input)

    if show_peaking_factor:
        try:
            RPF = np.zeros((len(self.radial['RPF']) , len(self.radial['RPF'][0])))
        except:
            print("--------------------------------------------------------------------------------------------------------- \n")
            raise RuntimeError(' We cannot plot RPF, Euclidian distance of KIN 2D Map because output_data has not been read')
        for row in range(len(self.radial['RPF'])):
            for col in range(len(self.radial['RPF'][row])):
                RPF[row][col] = float(self.radial['RPF'][row][col])

        RPF = np.rot90(RPF,2)
        quarter_vector = []
        for a in range(len(RPF) - 1):
            quarter_vector.append([])
            for b in range(len(RPF[a]) - 1):
                if RPF[a][b] != 0 :
                    # --- Second quadrant           
                    quarter_vector[a].append(RPF[a][b])
        print(" " * 20, end = "")
        print("Radial Peaking factors :")
        F_in = quarter_vector
        core_input_1 = ""
        for row in range(0,len(F_in)):
            core_input_1 += "\n"
            if row == 0:
                core_input_1 += "%2d"%(row + 1) + "  1" 
            else:
                core_input_1 += "\n" + "%2d"%(row + 1) + "  1" 
            for index in range(int((int(np.floor(self.dim)) - 2 * len(F_in[row]) - 1) / 2)):
                core_input_1 += " " + " " + " " + " " + " " + " " 
            for col in range(1,len(F_in[row]) + 1):
                core_input_1 += "|{:4.3f}".format(F_in[row][col - 1])
        print(core_input_1)


    if show_heuristic :
        outx = " \nFor this state, the Optimization heuristics are :"
        outx += "\n"
        #outx += "+++ iter = " + str(self.counter) + "\n best x = \n" + str(self.best_fit) + "\n best obj = \n" + str(self.max_of) + "\n"
        out = "0F-Exposure [GWd/tHm] = "+ str(self.max_exposure) + ") \n" + \
              "OF-Local-Peaking [-] = " + " (maxdeltaH = " + str(self.max_deltah) + ") \n" + \
              "OF-Axial-Peaking [-] = " + " (max4PIN = " + str(self.max_peaking) + ") \n" + \
              "OF_Boron [ppm] = " + " (max_boron = " + str(self.max_boron) + ") \n" + \
              "Number of feasible solutions found : {} \n".format(self.feasible) + \
              " Current score [-] = " + str(self.current_score) + "\n"

        out = outx + out
        print(out)
        return 0

  def clean(self):
    """ 
    Delete old files to avoid overlap and bug
    """
    # print("Function to clean data must be implemented")
    # raise NotImplementedError('self.clean is not')
    os.system('find . -type f -name "*gym.out" -exec rm {} \;')
    os.system('find . -type f -name "*gym.inp" -exec rm {} \;')
    os.system('find . -type f -name "*gym.res" -exec rm {} \;')
    os.system('find . -type f -name "*gym.txt" -exec rm {} \;')

  def runSIMULATE(self):
    """ Run SIMULATE """
    sys.stdout.write('------------------------------------------ \n')
    sys.stdout.write('--------------- Run Simulate ------------- \n')
    sys.stdout.write('------------------------------------------ \n')
    input_file = self.file
    out_file = copy.deepcopy(input_file)
    out_file = out_file.replace('inp','out')
    os.system('rm -rf %s'%(out_file))

    self.cluster = False
    if self.cluster: # --- job scheduler file for run into the cluster
        os.system("rm -rf SIMULATE3_*")
        simqname = '_SIMUopt.qsub'
        simqfile = open( simqname,'w' )
        simqfile.write( "#!/bin/bash\n" )
        simqfile.write("#PBS -N SIMULATE3_%s"%(input_file))
        simqfile.write( "\n#PBS -q default \n" )
        simqfile.write( "\n#PBS -l nodes=1:ppn=1\n" )
        simqfile.write( "\n#PBS -l walltime=30:00:00\n" )
        simqfile.write("\n#PATH=$PBS_O_PATH\n")
        simqfile.write("\ncd $PBS_O_WORKDIR \n")
        simqfile.write("\nsimulate3 %s > /dev/null 2>&1 \n"%(input_file))
        simqfile.close()
        runcommand  = "qsub _SIMUopt.qsub"
        os.system(runcommand)
    else:
        print('Running case : %s \n'%(input_file))
        os.system('simulate3 %s > /dev/null 2>&1 \n'%(input_file)) # --- silence all command
        print("Done....")
        if os.path.isdir(out_file) is None:
            raise RuntimeError('Simulate did not run correctly')
    return 

  def _Compute_reward(self):
    """
    Objective function for optimization : full core (193 assemblies)

    """
    #print("State is now :",self.state[0])
    board = self.state_file
    self.exposure2,self.eocbu,self.boron,self.average_radial_peaking, self.radial_peaking, self.axial_peaking , self.maxexposure, axial_offset = self.read_output_special(board)
    # -------------------  
    # --- Objective function
    self.max_exposure = np.array(self.maxexposure).max()
    self.max_boron = np.array(self.boron).max()
    self.max_peaking = np.array(self.radial_peaking).max()
    self.max_deltah = np.array(self.axial_peaking).max()

    # --- Penalties :
    if self.max_peaking > 2.00:
      self.max_peaking = self.max_peaking * 1000
    if self.max_deltah > 1.55:
      self.max_deltah = self.max_deltah * 1000

    if self.max_deltah <= 1.55 and self.max_peaking <= 2.00:
        print("Solution is feasible!")
        self.feasible += 1
        #self._Compute_SDM()
        #print(self.buffer_fuel)
        #sys.exit()
    # if self.max_boron > 1300:
    #   self.max_boron = self.max_boron * 1000
    # if self.max_exposure > 62:
    #   self.max_exposure = self.max_exposure * 1000

    score  = self.max_exposure * self.max_boron / (self.max_peaking * self.max_deltah) 
    if score - self.max_of > 0: # --- record best solution
      self.max_of = score
      self.best_fit = self.state[0]

    return score


  def read_output_special(self,out): # --- < *.out > file --> called to subsequently create *.inp file and run simulation < *.out  >
    """ Read SIMULATE3 output file '*.out' 
    new version : parameters for reward calculation
    """
    f = open(out,'r')
    iter1 = 0
    max_kinf = 0.0
    max_ppf = 0.0
    max_kinf0 = 0.0
    max_ppf0 = 0.0

    exposure2 = []
    core_exposure = []
    boron = []
    average_radial_peaking = []

    rad_peak = []
    ax_peak = []
    f_deltaH = []
    max_exposure = []
    axial_offset = []

    iter = 0 # --- keeper to process data only after a certain step
    inner_iteration = 0 # --- kepper to process data only once
    title = self.title
    while True:
        line1 = f.readline()
        if not line1: break
        line2 = line1.split()
        # if 'TIT.CAS' in line1 and '/' in line1:
        #     title = str(line2[1].split('/')[0].split("'")[1])
        if 'DIM.PWR' in line1 and '/' in line1:
            dimension = int(line2[1].split('/')[0])          
        if 'Case  1 Step  0' in line1:
            iter1 += 1
        if iter1 == 30: # --- comes from all unuse occurence of "Case  1 Step  0"
            if 'PRI.STA 2RPF' in line1:
                line1 = f.readline()
                RPF = []
                for idx in range(int(np.ceil(dimension / 2))):
                    line1 = f.readline()
                    line2 = line1.split()
                    RPF.append((list(map(float,line2[1:len(line2) - 1])))) # --- faster method?        
                max_ppf0 = np.array(np.array(RPF).max()).max()
            if 'PRI.STA 2KIN' in line1:
                k_inf = []
                line1  = f.readline()
                for idx in range(int(np.ceil(dimension / 2))):
                    line1 = f.readline()
                    line2 = line1.split()
                    k_inf.append((list(map(float,line2[1:len(line2) - 1])))) # --- faster method?
                max_kinf0 = np.array(np.array(k_inf).max()).max()
            if 'PRI.STA 2EXP' in line1 and 'Assembly' in line1:
                exposure = []
                line1  = f.readline()
                for idx in range(int(np.ceil(dimension / 2))):
                    line1 = f.readline()
                    line2 = line1.split()
                    exposure.append((list(map(float,line2[1:len(line2) - 1])))) # --- faster method?
                max_expo0 = np.array(np.array(exposure).max()).max()

        if 'PRI.STA 2RPF' in line1 and len(line1.split()) != 2:
            line1 = f.readline()
            RPF = []
            for idx in range(int(np.ceil(dimension / 2))):
                line1 = f.readline()
                line2 = line1.split()
                RPF.append((list(map(float,line2[1:len(line2) - 1])))) # --- faster method?
            max_ppf = np.array(np.array(RPF).max()).max()
            inner_iteration += 1
        if 'PRI.STA 2KIN' in line1:
            k_inf = []
            line1  = f.readline()
            for idx in range(int(np.ceil(dimension / 2))):
                line1 = f.readline()
                line2 = line1.split()
                k_inf.append((list(map(float,line2[1:len(line2) - 1])))) # --- faster method?
            max_kinf = np.array(np.array(k_inf).max()).max()
            inner_iteration += 1
        if 'PRI.STA 2EXP' in line1 and '2RPF' not in line1:
            exposure = []
            line1  = f.readline()
            for idx in range(int(np.ceil(dimension / 2))):
                line1 = f.readline()
                line2 = line1.split()
                exposure.append((list(map(float,line2[1:len(line2) - 1])))) # --- faster method?
            max_expo = np.array(np.array(exposure).max()).max()
            inner_iteration += 1
        if "Summary File Name" in line1:
            for i in range(5):
                line1 = f.readline()
            line2 = line1.split()
            while "1S I M U L A T E" not in line1 :
                if len(line2) != 0 :
                    if line2[0] == '1':
                        average_radial_peaking.append(float(line2[9]))
                        exposure2.append(float(line2[2]))
                        core_exposure.append(float(line2[18]))
                        boron.append(float(line2[5]))
                    # else:
                    #     continue
                line1 = f.readline()
                line2 = line1.split()
        # ----------------------------
        # --- Regulatory constraints
        if 'Axial Offset' in line1:
            if len(line2) >= 5:
                axial_offset.append(float(line2[11]))
        if "Max-4PIN" in line1: # --- axial peaking factor
            ax_peak.append(float(line2[8]))
        if "F-delta-H" in line1: # --- local peaking factor
            f_deltaH.append(float(line2[9]))
        if "Max-Fxy" in line1:
            rad_peak.append(float(line2[14]))
        if "2XPO  -  PEAK PIN EXPOSURE" in line1:
            #for i in range(10):
            line2 = line1.split()
            while "CORE" not in line2[0]:
                line1 = f.readline()
                if len(line1.split()) != 0:
                    line2 = line1.split()
                    #print(line2)
            max_exposure.append(float(line2[2]))    
        else : 
                    continue
    self.radial['RPF'] = RPF
    self.radial['KINF'] = k_inf
    self.radial['EXP'] = exposure
    boron,exposure2,average_radial_peaking,core_exposure = split(boron,exposure2,average_radial_peaking,core_exposure)

    return (exposure2,core_exposure,boron,average_radial_peaking,ax_peak,f_deltaH,max_exposure, axial_offset)



  def Getshuffle(self,sample): # --- < list > --> sample of position to randomly swap
    """ Keeper :
    Algorithm for shuffling randomly two assemblies 
    """ 
    # --- Shuffle core randomly options  
    # --- shuffle four elements randomly (Binary swap)
    if len(sample) == 1: # --- Only one difference with the best solution : do not alter it
        return 0
    [a_prim,b_prim] = random.sample(sample,2)
    if a_prim == b_prim:
    # --- re-shuffle
      self.Getshuffle(sample)
    elif self.state[0][a_prim] == self.state[0][b_prim]: # --- should not swap two similar assemblies ==> overhead CPU loss
      self.Getshuffle(sample)
    else:
      temp = self.state[0][b_prim]
      self.state[0][b_prim] = self.state[0][a_prim]
      self.state[0][a_prim] = temp 
    return 0

  def plot_rewards(self):
    """
    plot the reward history for logging, this is for mere to check progress
    """
    #print("Reward History",self.reward_hist)

    plt.figure()
    plt.title("Cumulative Reward History")
    plt.plot(self.reward_hist[1:len(self.reward_hist)])
    plt.xlabel('Episode'); plt.ylabel('Reward')
    plt.savefig('dqn_reward.png',format='png', dpi=150)
    plt.draw()
    plt.close()



  # --------------------------------------------------------------------------
  # -----
  # ----- SIMULATE-specific function 
  # ----- Processing function
  # -----
  # --------------------------------------------------------------------------

  def CoretoString(self,x):
        """ Write Core in string : 
        !!! must have modified core (e.g: shuffle) prior
        """
        # -------------------------------------------------------------       
        # -----------------------------------------------------------------
        # --- write out the new core
        F_IN = copy.deepcopy(x)
        core_input = ""
        for row in range(0,int(np.ceil(len(F_IN) / 2))):
            if row == 0:
                core_input += "%2d"%(row + 1) + "  1" 
            else:
                core_input += "\n" + "%2d"%(row + 1) + "  1" 
            for index in range(int((int(np.floor(self.dim)) - len(F_IN[row])) / 2)):
                core_input += " " + " " + " " + " " + " " + " " + " " 
            for col in range(1,len(F_IN[row]) + 1):
                core_input += " "  + "{:6s}".format(F_IN[row][col - 1])
        for row in range(int(np.ceil(len(F_IN) / 2)),len(F_IN)):
            core_input += "\n" + "%2d"%(row + 1) + "  1" 
            for index in range(int((int(np.ceil(self.dim)) - len(F_IN[row])) / 2)):
                core_input += " " + " " + " " + " " + " " + " " + " "
            for col in range(1,len(F_IN[row]) + 1):
                core_input += " " + "{:6s}".format(F_IN[row][col - 1])
        core_input += "\n" + " 0  0"
        return (core_input)


  def write_input(self, flag = True): # --- write new input file
    """ Write the new *_gym.inp
    useful if call outer libraries for optimization :
    Keep track of the changes. Only work with simple swap
    """
    #x_code = copy.deepcopy(self.state[0])
    ## -------------------------
    ## --- 
    ## --- Build new self.VECTOR
    #x_code_increment = 0
    #for row in range(len(self.VECTOR)):
    #    increment = 0
    #    while increment < len(self.VECTOR[row]):
    #        self.VECTOR[row][increment] = self.float_to_core[x_code[x_code_increment]]
    #        increment += 1 
    #        x_code_increment += 1

    self.VECTOR = []
    for row in range(8): # --- also store horizontal and vertical line
        self.VECTOR.append([])
        for col in range(int(np.ceil(len(self.core[row]) / 2))):
            self.VECTOR[row].append(self.core[row][col])
    #print(self.float_to_core)
    #print(self.VECTOR)
    #print(x_code)
    #core_temp = copy.deepcopy(self.core) # --- futur self.core
    #for diff in x_code:
    #    element1 = diff
    #    a,b  = search_element(QUARTER_CORE_MAP,element1,0)
    #    #if element1 in horizontal_indices:
    #    #    core_temp[a][b] = self.VECTOR[a][b]
    #    #    core_temp[a][-b + len(self.core[a]) - 1] = self.horizontal_correspondance[self.VECTOR[a][b]][0]
    #    #elif element1 in vertical_indices:
    #    #    core_temp[a][b] = self.VECTOR[a][b]
    #    #    core_temp[-a + len(self.core) - 1][b] = self.vertical_correspondance[self.VECTOR[a][b]][0]
    #    #elif element1 == 55: # --- do not touch central element
    #    #    continue
    #    if element1 == 55:
    #        continue
    #    else:
    #        core_temp[a][b] = self.VECTOR[a][b]
    #        core_temp[-a + len(self.core) - 1][b]  = self.core_correspondance[self.VECTOR[a][b]][0]
    #        core_temp[a][-b + len(self.core[a]) - 1]  = self.core_correspondance[self.VECTOR[a][b]][1]
    #        core_temp[-a + len(self.core) - 1][-b + len(self.core[a]) - 1] = self.core_correspondance[self.VECTOR[a][b]][2]
    #
    #self.core = core_temp
    

    if not os.path.isfile('temp_gym.inp') or not os.path.isfile('temp_core_gym.inp'):
        raise RuntimeError('read_input() has not worked') 
    else:
        out = open('temp_gym.inp','r') # --- copy it
        f2 = open(self.file,'w') # --- reproduce new input file to later run it
    
    while True:
        line1 = out.readline()
        if not line1: break          
        if 'FUE.LAB' in line1:
            # --- Write the fuel core assembly
            f2.writelines(line1)
        
            out2 = self.CoretoString(self.core) 

            f2.writelines(out2)
            f2.writelines('\n')
            f2.writelines('\n') # --- More space between the input flags
        # elif 'FUE.NEW' in line1:
        #     simulate_input1 = "\n"
        #     line2 = line1.split(',')
        #     line2[2] = str("'" + line2[2].replace("'","") + "%d"%(self.counter) + "'")
        #     #line2[len(line2) - 1] =  " %d / \n"%(self.counter)
        #     s = ','
        #     simulate_input1 += s.join(line2)
        #     f2.writelines(simulate_input1)     
        #     #print(simulate_input1)
        #     #sys.exit()   
        else:
            f2.writelines(line1)
    self.input = f2
    f2.close()
    out.close()

    if flag: # --- multicycle depletion
        # ----------------------------------------------
        # --- CAVEAT!!!: probably a more efficient way to do it 
        # -----------------------------------
        dirname = os.getcwd()   
        if glob.glob("loadcycle_*"):
            os.system('rm -rf loadcycle_*')

        for cycle_number in range(12,17):#int(self.title.split()[1]) + 1,int(self.title.split()[1]) + 6): # --- write en external file for each depletion cycle
            #print("cycle number : {}".format(cycle_number))
            f3 = open('loadcycle_%d_gym.txt'%(cycle_number),'w')
            if not os.path.isfile('temp_gym.inp'):
                raise RuntimeError('read_input() has not worked') 
            else:
                out = open('temp_gym.inp','r') # --- copy it
            while True:
                line1 = out.readline()
                if not line1: break   
                if 'BAT.LAB' in line1:
                    f3.writelines("'BAT.LAB' " + '%d '%(cycle_number)+ "'CYC-%d' "%(cycle_number)+"/\n")
                    continue
                if 'DEP.CYC' in line1:
                    f3.writelines("'DEP.CYC' " + 'CYCLE%d '%(cycle_number)+ '0.0 ' +'%d'%(cycle_number)+"/\n")
                    continue
                if 'TIT.CAS' in line1 : # --- change title for each cycle
                    f3.writelines("'TIT.CAS'" + "'Cycle %d' "%(cycle_number)+"/\n")
                    continue
                elif 'WRE' in line1 or 'SUM' in line1 or 'RES' in line1:
                    line2 = line1.split()
                    if 'WRE' in line1:
                        if cycle_number == int(self.title.split()[1]) + 1:
                            s = " " 
                            wre_path = line2[1:] # --- store path writing file
                            temp_wre = "'RES'" + " " + s.join(wre_path)  + "\n" # --- previous WRE becomes RES for new cycle
                        else:
                            s = '/'
                            wre_path = line2[1].split('/')
                            wre_path[len(wre_path) - 1] = "run%d_%d_gym.res"%(self.run_number,cycle_number - 1)
                            temp_wre = "'RES'" + " " + s.join(wre_path)  + "' 20000 / \n"
                        f3.writelines(temp_wre)
                        f3.writelines(temp_sum)
                        f3.writelines(temp_res)
                        continue
                    if 'SUM' in line1:
                        sum_path = line2[1].split('/')
                        sum_path[len(sum_path) - 1] = "run%d_%d_gym.sum"%(self.run_number,cycle_number)
                        s = "/"
                        temp_sum =  "'SUM'" + " "  + str(s.join(sum_path))+ "'" + " " +'/ \n'
                        #f3.writelines(temp_sum)
                        continue
                    if 'RES' in line1:
                        res_path = line2[1].split('/')
                        res_path[len(res_path) - 1] =  "run%d_%d_gym.res"%(self.run_number,cycle_number)
                        s = "/"
                        temp_res =  "'WRE'" + " " + str(s.join(res_path)) + "'" + " " + line2[2] + ' / \n'
                        self.res = str(s.join(res_path[1:len(res_path) - 1]))
                        #f3.writelines(temp_res)
                        continue    
                if 'FUE.LAB' in line1:
                    # --- Write the fuel core assembly
                    f3.writelines(line1)
                    f3.writelines(out2)
                    f3.writelines('\n')
                    f3.writelines('\n') # --- More space between the input flags
                # elif 'FUE.NEW' in line1:
                #     simulate_input1 = "\n"
                #     line2 = line1.split(',')
                #     line2[2] = str("'" + line2[2].replace("'","") + "%d"%(self.counter) + "'")
                #     #line2[len(line2) - 1] =  " %d / \n"%(self.counter)
                #     s = ','
                #     simulate_input1 += s.join(line2)
                #     f3.writelines(simulate_input1)     
                #     #print(simulate_input1)
                #     #sys.exit()   

                else:
                    f3.writelines(line1)
            f3.close()
            out.close()
            # --- Write into *.inp file so that it can run it
            f2 = open(self.file,'a')
            f2.writelines("\n")
            s = "'" + dirname + '/loadcycle_%d_gym.txt'%(cycle_number) + "'"
            external_cycle = "'INC.FIL' " + s + ' / \n \n' + \
            "'STA'/ \n" + \
            "'END'/ \n"
            f2.writelines(external_cycle)
            self.input = f2
            f2.close()

    return 0

  def _Compute_SDM(self):
      """
      Write and compute SDM margin
      
      """

      data = sEq_templates
      f = open(data,'r')
      f2 = open('SDM_'+data,'w')
      while True:
            line1 = f.readline()
            if not line1 : break
            line2 = line1.split()
            if len(line2) == 0: continue
            else:
                if 'RES' in line1:
                    simu_sdm = "\n"
                    line2[1] = "'/home/pseurin/simulate/Python--script/restart_files/run%d_%d_gym.res'"%(self.run_number,16)
                    s = ' '
                    simu_sdm += s.join(line2)
                    f2.writelines(simu_sdm)  
                else:
                    f2.writelines(line1)
      f.close()
      f2.close()
      # --- run the file
      os.system("simulate3 SDM_simenrifbaeq.inp > /dev/null 2>&1")
      for data in SDM_templates:
          f = open(data,'r')
          f2 = open('SDM_'+data,'w')
          while True:
              line1 = f.readline()
              if not line1 : break
              line2 = line1.split()
              if len(line2) == 0: continue
              else:
                    f2.writelines(line1)
          f.close()
          f2.close()
          # --- run the calculation
          os.system("simulate3 SDM_%s > /dev/null 2>&1 "%(data))

            
  def Download(self):# --- MUST be called at the beginning to obtain restart file and first simulation (templates are on a text file)
    """ Read input *.inp data file
    Directly store quarter core for self.VECTOR
    use preferably with external optimization libraries
    """
    out = self.templates_file
    f = open(out,'r') 
    simulate_input1 = "" # --- string to reproduce input except full core description ('FUE.SER || FUE.LAB')
    simulate_core = "" # --- string to reproduce full core description
    index_backfile = 0 
    control_index = 0
    while True:
        line1 = f.readline()
        if not line1: break
        line2 = line1.split()  
        simulate_input1 += line1 
        if len(line2) == 0:continue

        # --------------------------------
        # ---
        # --- store libraries
        # ---
        if 'LIB' in line1 and 'REF' not in line1 and 'SEG' not in line1:
            self.librairies['TABLES'] = line2[1]
        if 'REF.LIB' in line1 :
            self.librairies['REF'] = {}
            self.librairies['REF'][line2[2]] = line2[3]
        if 'SEG.LIB' in line1:
            self.librairies['FUEL'] = {}
            self.librairies['FUEL'][line2[2]] = line2[3]
            # print(self.libraries)
        # -----------------------------------
        if 'TIT.CAS' in line1:
            self.title = str(line2[1].split('/')[0].split("'")[1]) + " " + str(line2[2].split('/')[0].split("'")[0])
        if 'DIM.PWR' in line1 :
            self.dim = int(line2[1].split('/')[0])
            # ---------------------
            # ---- Build Euclidien distance !! May be called multiple time if here
            # ---- !! Caveat : this empirical distance from origin. May require real distance
            self.distance = np.zeros((int(np.floor(self.dim / 2)),int(np.floor(self.dim / 2))))
            for row in range(int(np.floor(self.dim / 2))):
                for col in range(int(np.floor(self.dim / 2))):
                    self.distance[row][col] = np.sqrt(row**2 + col**2)
            self.radial['Distance'] = self.distance
            continue          
        if 'FUE.LAB'  in line1:
            length_label = int(line2[0].replace('/',' ').replace(',',' ').replace('FUE.LAB',' ').replace("'"," ")) # --- useful for plotting core later on
            for j in range(self.dim + 2): # --- skip core design for other storage
                simulate_core += line1                     
                line1 = f.readline()
                line2 = line1.split()
                if j <= self.dim - 1 :
                    self.core.append(line2[2:len(line2)])
        if 'FUE.SER'  in line1:
            length_label = int(line2[0].replace('/',' ').replace(',',' ').replace('FUE.SER',' ').replace("'"," ")) # --- useful for plotting core later on
            for j in range(self.dim + 2): # --- skip core design for other storage
                simulate_core += line1                     
                line1 = f.readline()
                line2 = line1.split()
                if j <= self.dim - 1 :
                    self.core.append(line2[2:len(line2)])
    self.VECTOR = []
    for row in range(8): # --- also store horizontal and vertical line
        self.VECTOR.append([])
        for col in range(int(np.ceil(len(self.core[row]) / 2))):
            self.VECTOR[row].append(self.core[row][col])

    # ------------------------------------------
    # --- Build correspondance between vector and core in all four quadrants

    #for row in range(len(self.VECTOR) - 1):
    #    for col in range(len(self.VECTOR[row]) - 1):
    #for row in range(len(self.VECTOR)):
    #    for col in range(len(self.VECTOR[row])):
    #
    #        self.core_correspondance[self.VECTOR[row][col]] = []
#
    #        # --- Third quadrant
    #        self.core_correspondance[self.VECTOR[row][col]].append(self.core[- row + len(self.core) - 1][col])
#
    #        # --- Second quadrant
    #        self.core_correspondance[self.VECTOR[row][col]].append(self.core[row][- col + len(self.core[row]) - 1])
#
    #        # --- Fourth quadrant
    #        self.core_correspondance[self.VECTOR[row][col]].append(self.core[- row + len(self.core) - 1][-col + len(self.core[row]) - 1])
    # ------------------------------------------
    # --- Build correspondance between vector and core in all horizontal and vertical line

    #for col in range(len(self.VECTOR[len(self.VECTOR) - 1])):
    #    self.horizontal_correspondance[self.VECTOR[len(self.VECTOR) - 1][col]] = []
    #    self.horizontal_correspondance[self.VECTOR[len(self.VECTOR) - 1][col]].append(self.core[len(self.VECTOR) - 1][-col + len(self.core[row]) - 1])
#
#
    #for row in range(len(self.VECTOR)):
    #    self.vertical_correspondance[self.VECTOR[row][len(self.VECTOR[row]) - 1]] = []
    #    self.vertical_correspondance[self.VECTOR[row][len(self.VECTOR[row]) - 1]].append(self.core[- row + len(self.core) - 1][len(self.VECTOR[row]) - 1])


    # -------------------------------------------
    # --- Write file without fuel core mapping
    f1 = open('temp_gym.inp','w')
    f1.writelines(simulate_input1)
    # --- Store fuel core mapping somewhere
    f2 = open('temp_core_gym.inp','w')
    f2.writelines(simulate_core)

    return (simulate_input1, simulate_core,length_label)