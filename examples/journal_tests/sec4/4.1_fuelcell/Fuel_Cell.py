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
import random
import numpy as np
from math import cos, pi, exp, e, sqrt, log
import matplotlib.pyplot as plt
from neorl import  DE, GWO, MFO, BAT, PESA2

# This script calculates the power output and efficiency of FC-TEG hybrid system

def SOFC(X):
            
    # Input parameters
    P = 1 #operating pressure in atm
    j=X[0] # current density 
    T = X[1] #Operating temperature
    A=X[2]     #Polar plate area of SOFC
    #Partial pressures in atm of input fuel and air
    pCO = 1-X[3]; pCO2 = X[3] #Fuel components
    pO2 = X[4]; pN2 = 1-X[4] #Air components
    Eact_a = 100000 #activation energy of anode
    Eact_c = 120000 #activation energy of cathode
    eps=X[5] #Electrode porosity
    EPS=X[6]  #Electrode tortuosity
    Dp=X[7] # average pore diameter
    Ds=X[8] # average grain size
    Xg=X[9]  #Average length of grain contact  --> X in the original model 
    La=X[10] # Anode thickness (This value is 5E-04 in the paper but it does not give the same results, 13E-04 does
                                 #it perfectly on the other hand)
    Lc=X[11] #Cathode thickness
    Le=X[12]  #Electrolyte thickness
    sig_a=X[13] #Anode electric conductivity
    sig_c=X[14]  #Cathode electric conductivity
    sig_e=33400*exp(-10300/T)  #Electrolyte electric conductivity
    sig_N2=X[15]   #Diameters of the N2 molecular collision
    sig_O2=X[16]   #Diameters of theO2 molecular collision


    # Constants
    ne=2   #Number of electrons transferred per reaction
    F=96485  #Faraday's constant
    R=8.314  #Uiversal gas constant in J/mol K
    Ru=0.08205   #Universal gas constant in L atm/mol K
    dg=-282150+86.735*T  #Molar Gibbs free energy
    dh=-248201  #Molar enthalpy change
    k=1.38066e-23  #Boltzmann's constant

    eps_CO=91.7 * k
    eps_CO2= 195.2 * k
    eps_N2= 71.4 * k
    eps_O2= 106.7 * k
    sig_CO=X[17]  #Diameters of the CO molecular collision
    sig_CO2=X[18]  #Diameters of the CO2 molecular collision

    #Molecular weights
    MCO = 28.01
    MCO2 = 44.01
    MO2 = 32
    MN2 = 28.01

    # Molar concentrations 
    C0_CO=pCO*101325/(R*T)
    C0_CO2=pCO2*101325/(R*T)
    C0_O2=pO2*101325/(R*T)
    C0_N2=pN2*101325/(R*T)
    C_Tc=C0_O2+C0_N2

    # Pre-factors 
    lambda_a=X[19]
    lambda_c=X[20]

    # ---------------------------------------------------------------------
    # H2,O2 fuel cell data 
    # ---------------------------------------------------------------------
    # library(simecol)   # for linear interlation in dataframes
    # # import data
    # data=read.csv('./data/h2_o2_data.csv',header = TRUE)
    # # interpolate at temperature T from the dataset
    # interpole=approxTime(data,T)
    # dh=interpole[1,3]  # take the enthaply value 
    # dG=interpole[1,5]  # take the Gibbs free energy value

    # ---------------------------------------------------------------------
    # MCFC fuel cell
    # ---------------------------------------------------------------------
    E0 = -dg / (2 * F)    # rate of energy content in H2 fuel 
    E = E0 + (R * T)/(2*F)*log(pCO*pO2**0.5/pCO2)

    j0a=lambda_a*(72*Xg*(Dp-(Dp+Ds)*eps)*eps)/(Ds**2*Dp**2*(1-sqrt(1-Xg**2)))*pCO/P * pCO2/P*exp(-Eact_a/(R*T))
    j0c=lambda_c*(72*Xg*(Dp-(Dp+Ds)*eps)*eps)/(Ds**2*Dp**2*(1-sqrt(1-Xg**2)))* (pO2/P)**0.25 *exp(-Eact_c/(R*T))

    #-- Activation overpotential

    Vact_a=R*T/F * log(j/(2*j0a)+sqrt((j/(2*j0a))**2+1))
    Vact_c=R*T/F * log(j/(2*j0c)+sqrt((j/(2*j0c))**2+1))
    Vact=Vact_a + Vact_c

    #-- Ohmic overpotential

    Vohm=j*(La/sig_a+Lc/sig_c+Le/sig_e)

    #-- Concentration overpotential

    # Anode 
    sig_O2_N2=(sig_O2+sig_N2)/2
    eps_O2_N2= (eps_O2*eps_N2)**0.5
    tau_D2= k*T/(eps_O2_N2)
    omega_D2=1.06036/(tau_D2**0.1561) + 0.193/(exp(0.47635*tau_D2))+1.03587/(exp(1.52996*tau_D2)) + 1.76474/(3.89411*tau_D2)
    M_CO_CO2= 2/(1/MCO + 1/MCO2)
    sig_CO_CO2=(sig_CO+sig_CO2)/2
    eps_CO_CO2=(eps_CO2*eps_CO)**0.5
    tau_D1=k*T/(eps_CO_CO2)
    omega_D1=1.06036/(tau_D1**0.1561) + 0.193/(exp(0.47635*tau_D1))+1.03587/(exp(1.52996*tau_D1)) + 1.76474/(3.89411*tau_D1)

    D_CO_Kn=1/3*Dp*sqrt(8*R*T/(pi*MCO))
    D_CO2_Kn=1/3*Dp*sqrt(8*R*T/(pi*MCO2))
    D_CO_CO2=0.0025*T**1.5/(P*M_CO_CO2**0.5*sig_CO_CO2**2*omega_D1)
    D_CO_eff=(EPS/eps*(1/D_CO_CO2+1/D_CO_Kn))**-1
    D_CO2_eff=(EPS/eps*(1/D_CO_CO2+1/D_CO2_Kn))**-1
    jl_CO2=2*F*C0_CO2*D_CO2_eff/(A*La)
    jl_CO=2*F*C0_CO*D_CO_eff/(A*La)
    Vconc_a=R*T/(2*F) * log((1+j/jl_CO2)/(1-j/jl_CO))  # Final value 

    # Cathode

    N_O2=-j*A/(4*F)
    M_O2_N2=2/(1/MO2 + 1/MN2)
    D_O2_Kn=1/3*Dp*sqrt(8*R*T/(pi*MO2))
    D_O2_N2=0.0026*T**1.5/(P*M_O2_N2**0.5*sig_O2_N2**2*omega_D2)
    D_O2_eff=(EPS/eps*(1/D_O2_Kn+1/D_O2_N2))**-1
    D_O2_Kn_eff=EPS/eps*D_O2_Kn
    D_O2_N2_eff=EPS/eps*D_O2_N2
    D=D_O2_Kn_eff/(D_O2_Kn_eff+D_O2_N2_eff)
    C_O2_l=C_Tc/D+(C0_O2-C_Tc/D)*exp(-(D*N_O2*Lc)/(C_Tc*D_O2_eff))
    Vconc_c=R*T/(4*F) * log(C0_O2/C_O2_l) # Final value 

    # Total
    Vconc=Vconc_a+Vconc_c

    Vcell=E-Vact-Vohm-Vconc

    P=Vcell*j*A/A
    eta=-ne*F*Vcell/dh

    newlist = {}
    newlist['P'] = P
    newlist['eta'] = eta
    newlist['Vcell'] = Vcell
    newlist['Vconc'] = Vconc
    newlist['Vohm'] = Vohm
    newlist['Vact'] = Vact

    return(newlist)

#---------------------------------
# Fitness function
#---------------------------------
def SOFC_fit(X):
    #Calculates the fitness of the FC-TEG Hybrid system as a linear combinaion of power and efficiency
    
    w1 = 0.5
    w2 = 0.5
    data  = SOFC(X)
    fitness = w1*data['P'] + w2*100*data['eta']
    return fitness

#---------------------------------
# Parameter space
#---------------------------------
bounds = {}
bounds['j'] = ['float', 13200-4*132 , 13200+4*132]
bounds['T'] = ['float', 1073, 1073]
bounds['A'] = ['float', 1.55e-3 , 1.65e-3]
bounds['pCO2'] = ['float', 0.05-4*0.0025 , 0.05+4*0.0025]
bounds['pO2'] = ['float', 0.21-4*0.0105 , 0.21+4*0.0105]
bounds['eps'] = ['float', 0.48-4*0.0216 , 0.48+4*0.0216]
bounds['EPS'] = ['float', 5.4-4*0.27 , 5.4+4*0.27]
bounds['Dp'] = ['float', 2.8e-6, 3.2e-6]
bounds['Ds'] = ['float', 1.4e-6, 1.6e-6]
bounds['Xg'] = ['float', 0.6, 0.8]
bounds['La'] = ['float', 4.7e-4, 5.3e-4]
bounds['Lc'] = ['float', 4.7e-5, 5.3e-5]
bounds['Le'] = ['float', 4.7e-5, 5.3e-5]
bounds['sig_a'] = ['float', 80000-4*8000 , 80000+4*8000]
bounds['sig_c'] = ['float', 8400-4*840 , 8400+4*840]
bounds['sig_N2'] = ['float', 3.798-4*0.228 , 3.798+4*0.228]
bounds['sig_O2'] = ['float', 3.467-4*0.208 , 3.467+4*0.208]
bounds['sig_CO'] = ['float', 2.827-4*0.167 , 2.827+4*0.167]
bounds['sig_CO2'] = ['float', 2.641-4*0.158 , 2.641+4*0.158]
bounds['lambda_a'] = ['float', 1.39e-9, 1.69e-9]
bounds['lambda_c'] = ['float', 5.27e-10, 6.44e-10]

#*************************************************************
# Bayesian hyperparameter tuning for BAT
#*************************************************************

#This process is repeated similarly for the other algorithms

from neorl.tune import BAYESTUNE
from neorl import BAT

#*************************************************************
# Define fitness function for hyperparameter tuning
#*************************************************************
def tune_fit(fmin, fmax, A, r0, alpha, gamma, levy):

    #--setup the algorithm
    bat=BAT(mode='max', bounds=bounds, fit=SOFC_fit, nbats=50, fmin = fmin , fmax = fmax, A=A, r0=r0, alpha = alpha, gamma = gamma, levy = levy, seed = 1, ncores=1)


    #--Evolute the object and obtains y_best
    #--turn off verbose for less algorithm print-out when tuning
    bat_x, bat_y, bat_hist=bat.evolute(ngen=300, verbose=0)

    return bat_y #returns the best score

#*************************************************************
# Tuning
#*************************************************************
#Setup the parameter space
#VERY IMPORTANT: The order of these parameters MUST be similar to their order in tune_fit
#see tune_fit
param_grid={
#def tune_fit(cxpb, mu, alpha, cxmode):
'fmin': ['float', -2, 2],
'fmax': ['float', -2, 2],
'A': ['float', 0, 10],
'r0': ['float', 0, 1],
'alpha': ['float', 0, 1],
'gamma': ['float', 0, 10],
'levy': ['grid', [True, False]]}

#setup a bayesian tune object
btune=BAYESTUNE(mode='max', param_grid=param_grid, fit=tune_fit, ncases=30)
#tune the parameters with method .tune
bayesres=btune.tune(ncores=1, csvname='bayestune.csv', verbose=True)
print(bayesres)
btune.plot_results(pngname='bayes_conv')

#---------------------------------
# DE
#---------------------------------
de=DE(mode='max', bounds=bounds, fit=SOFC_fit, npop=50, F=0.834044, CR=0.720324, ncores=1, seed=1)
de_x, de_y, de_hist=de.evolute(ngen=300, verbose=1)
assert SOFC_fit(de_x) == de_y

#---------------------------------
# BAT
#---------------------------------
bat=BAT(mode='max', bounds=bounds, fit=SOFC_fit, nbats=50, fmin = -0.834362 , fmax = -1.354346, A=4.377198, r0=0.846313, alpha = 0.923407, gamma = 1.493487, levy = True, seed = 1, ncores=1)
bat_x, bat_y, bat_hist=bat.evolute(ngen=300, verbose=1)
assert SOFC_fit(bat_x) == bat_y

#---------------------------------
# GWO
#---------------------------------
gwo=GWO(mode='max', fit=SOFC_fit, bounds=bounds, nwolves=50, ncores=1, seed=1)
gwo_x, gwo_y, gwo_hist=gwo.evolute(ngen=300, verbose=1)
assert SOFC_fit(gwo_x) == gwo_y

#---------------------------------
# MFO
#---------------------------------
mfo=MFO(mode='max', bounds=bounds, fit=SOFC_fit, nmoths=50, b = 4.170220, ncores=1, seed=1)
mfo_x, mfo_y, mfo_hist=mfo.evolute(ngen=300, verbose=1)
assert SOFC_fit(mfo_x) == mfo_y

#---------------------------------
# PESA2
#---------------------------------
pesa2=PESA2(mode='max', bounds=bounds, fit=SOFC_fit, npop=50, R_frac = 0.146756, alpha_init = 0.018468, alpha_end = 0.837252, CR = 0.345561, F = 0.793535, nwolves = 11, nwhales = 9, ncores=1, seed=1)
pesa2_x, pesa2_y, pesa2_hist=pesa2.evolute(ngen=600, replay_every=2, verbose=1)
assert SOFC_fit(pesa2_x) == pesa2_y

#---------------------------------
# Plot
#---------------------------------
plt.figure()
plt.plot(de_hist, label = 'DE')
plt.plot(bat_hist['global_fitness'], '--', label = 'BAT')
plt.plot(gwo_hist['fitness'], '-.', label = 'GWO')
plt.plot(mfo_hist['global_fitness'], ':', label = 'MFO')
plt.plot(pesa2_hist, '-', label = 'PESA2')
plt.legend(fontsize =12)
plt.xlabel('Generation', fontsize = 12)
plt.ylabel('Fitness', fontsize = 12)
plt.ylim(4000,4150)
print('de: ', de_x, de_y)
print('bat: ', bat_x , bat_y)
print('gwo: ', gwo_x , gwo_y)
print('mfo: ', mfo_x , mfo_y)
print('pesa2: ', pesa2_x , pesa2_y)
plt.savefig('SOFC_convergence.png',format='png', dpi=300, bbox_inches="tight")