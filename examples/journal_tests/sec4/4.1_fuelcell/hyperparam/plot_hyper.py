# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 17:24:03 2021

@author: Devin Seyler
"""

#--------------------------------------------------------------------
# Paper: NEORL: A Framework for NeuroEvolution Optimization with RL
# Section: Script for section 4.1
# Contact: Majdi I. Radaideh (radaideh@mit.edu)
# Last update: 9/10/2021
#---------------------------------------------------------------------

#---------------------------------
# Import packages
#---------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

de=pd.read_csv('de.csv', usecols=['score'])
bat=pd.read_csv('bat.csv', usecols=['score'])
pesa2=pd.read_csv('pesa2.csv', usecols=['score'])

de=pd.DataFrame.cummax(de)
bat=pd.DataFrame.cummax(bat)
pesa2=pd.DataFrame.cummax(pesa2)
#---------------------------------
# Plot
#---------------------------------
plt.figure()
plt.plot(de, '-o', label = 'DE', markersize=4)
plt.plot(bat, '--s', label = 'BAT', markersize=4)
plt.plot(pesa2, '-.d', label = 'PESA2', markersize=4)
plt.legend(fontsize =12)
plt.xlabel('Iteration', fontsize = 12)
plt.ylabel('Max Fitness So far', fontsize = 12)
#plt.ylim([4100)

plt.savefig('bayesian.png',format='png', dpi=300, bbox_inches="tight")