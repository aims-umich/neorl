#    This file is part of NEORL.

#    Copyright (c) 2021 Exelon Corporation and MIT Nuclear Science and Engineering
#    NEORL is free software: you can redistribute it and/or modify
#    it under the terms of the MIT LICENSE

#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

# -*- coding: utf-8 -*-
#"""
#Created on Thu Dec  3 14:42:29 2020
#
#@author: Majdi
#"""

import random
import numpy as np
from numpy import arange, multiply, zeros, copy
import joblib
from neorl.hybrid.edevcore.helpers import ARCHIVE, generator, sub2ind, gnR1R2, randFCR
from neorl.evolu.discrete import mutate_discrete, encode_grid_to_discrete 
from neorl.evolu.discrete import decode_discrete_to_grid, encode_grid_indv_to_discrete
from neorl.utils.seeding import set_neorl_seed
from neorl.utils.tools import get_population, check_mixed_individual


class EDEV(object):
    """
    Ensemble of differential evolution variants
    
    :param mode: (str) problem type, either ``min`` for minimization problem or ``max`` for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param fit: (function) the fitness function 
    :param npop: (int): total size of the full population, which will be divided into three sub-populations
    :param int_transform: (str): method of handling int/discrete variables, choose from: ``nearest_int``, ``sigmoid``, ``minmax``.
    :param ncores: (int) number of parallel processors (must be ``<= npop``)
    :param seed: (int) random seed for sampling
    """
    #:param lambda_: (float): fraction of ``npop`` to split into 3 sub-populations. ``pop1 = npop - npop * lambda_``, ``pop2 = pop3 =  npop * lambda_`` (see **Notes** below)        
    def __init__(self, mode, bounds, fit, npop=100, 
                 int_transform='nearest_int', ncores=1, seed=None):

        set_neorl_seed(seed)
        
        lambda_=0.1   #for EDEV stability, better to keep this parameter fixed. 
        assert npop >= 70, '--error: EDEV ensemble requires a large population of 70 or more'
        assert ncores <= npop, '--error: ncores ({}) must be less than or equal than npop ({})'.format(ncores, npop)
        assert 0 < lambda_ < 1, '--error: lambda_ must be more than 0 and less than 1'
        #--mir
        self.mode=mode
        if mode == 'min':
            self.fit=fit
        elif mode == 'max':
            def fitness_wrapper(*args, **kwargs):
                return -fit(*args, **kwargs) 
            self.fit=fitness_wrapper
        else:
            raise ValueError('--error: The mode entered by user is invalid, use either `min` or `max`')
          
        self.int_transform=int_transform
        if int_transform not in ["nearest_int", "sigmoid", "minmax"]:
            raise ValueError('--error: int_transform entered by user is invalid, must be `nearest_int`, `sigmoid`, or `minmax`')
            
        self.bounds=bounds
        self.ncores = ncores
        self.npop=npop
        self.lambda_=lambda_
        
        #infer variable types 
        self.var_type = np.array([bounds[item][0] for item in bounds])
        
        #mir-grid
        if "grid" in self.var_type:
            self.grid_flag=True
            self.orig_bounds=bounds  #keep original bounds for decoding
            #print('--debug: grid parameter type is found in the space')
            self.bounds, self.bounds_map=encode_grid_to_discrete(self.bounds) #encoding grid to int
            #define var_types again by converting grid to int
            self.var_type = np.array([self.bounds[item][0] for item in self.bounds])
        else:
            self.grid_flag=False
            self.bounds = bounds
            self.orig_bounds=bounds
        
        self.dim = len(bounds)
        self.lb=np.array([self.bounds[item][1] for item in self.bounds])
        self.ub=np.array([self.bounds[item][2] for item in self.bounds])
        
        slice_index=int(self.lambda_*self.npop)
        if slice_index <= 6:
            raise Exception('--error: the size of pop2 and pop3 ({}) must be more than 6 individuals. Make sure int(lambda_ * npop) >= 7'.format(slice_index))

    def init_sample(self, bounds):
    
        indv=[]
        for key in bounds:
            if bounds[key][0] == 'int':
                indv.append(random.randint(bounds[key][1], bounds[key][2]))
            elif bounds[key][0] == 'float':
                indv.append(random.uniform(bounds[key][1], bounds[key][2]))
            #elif bounds[key][0] == 'grid':
            #    indv.append(random.sample(bounds[key][1],1)[0])
            else:
                raise Exception ('unknown data type is given, either int, float, or grid are allowed for parameter bounds')   
        return np.array(indv)

    def eval_pop(self, pop_to_evaluate):
    
        #---------------------
        # Fitness calcs
        #---------------------
        core_lst=[]
        for case in range (0, pop_to_evaluate.shape[0]):
            core_lst.append(pop_to_evaluate[case, :])
    
        if self.ncores > 1:

            with joblib.Parallel(n_jobs=self.ncores) as parallel:
                fitness_lst=parallel(joblib.delayed(self.fit_worker)(item) for item in core_lst)
                
        else:
            fitness_lst=[]
            for item in core_lst:
                fitness_lst.append(self.fit_worker(item))
        
        return fitness_lst

    def select(self, pos, fit):
        
        best_fit=np.min(fit)
        min_idx=np.argmin(fit)
        best_pos=pos[min_idx,:].copy()
        
        return best_pos, best_fit , min_idx
        
    def ensure_bounds(self, vec):
    
        vec_new = []
        # cycle through each variable in vector 
        for i, (key, val) in enumerate(self.bounds.items()):
    
            # variable exceedes the minimum boundary
            if vec[i] < self.bounds[key][1]:
                vec_new.append(self.bounds[key][1])
    
            # variable exceedes the maximum boundary
            if vec[i] > self.bounds[key][2]:
                vec_new.append(self.bounds[key][2])
    
            # the variable is fine
            if self.bounds[key][1] <= vec[i] <= self.bounds[key][2]:
                vec_new.append(vec[i])
            
        return vec_new
    
    def fit_worker(self, x):
        #This worker is for parallel calculations
        
        # Clip the position outside the lower/upper bounds and return same position
        x=self.ensure_bounds(x)

        
        if self.grid_flag:
            #decode the individual back to the int/float/grid mixed space
            x=decode_discrete_to_grid(x,self.orig_bounds,self.bounds_map)
        
        # Calculate objective function for each search agent
        fitness = self.fit(x)
        self.FES += 1
        
        return fitness
    
    def ensure_discrete(self, vec):
        #"""
        #to mutate a vector if discrete variables exist 
        #handy function to be used within SSA phases

        #Params:
        #vec - position in vector/list form

        #Return:
        #vec - updated position vector with discrete values
        #"""
        
        for dim in range(self.dim):
            if self.var_type[dim] == 'int':
                vec[dim] = mutate_discrete(x_ij=vec[dim], 
                                               x_min=min(vec),
                                               x_max=max(vec),
                                               lb=self.lb[dim], 
                                               ub=self.ub[dim],
                                               alpha=self.a,
                                               method=self.int_transform,
                                               )
        
        return vec
            

    def evolute(self, ngen, ng=20, x0=None, verbose=False):
        """
        This function evolutes the EDEV algorithm for a number of generations.
        
        :param ngen: (int) number of generations to evolute
        :param ng: (int) the period or number of generations to determine the best performing DE variant and reward subpopulation assignment, ``ng < ngen`` (see **Notes** below for more info).
        :param x0: (list of lists) initial position of the individuals (must be of same size as ``npop``)
        :param verbose: (bool) print statistics to screen
        
        :return: (tuple) (best individual, best fitness, and dictionary containing major search results)
        """
        
        assert ng < ngen, '--error: ng must be less than ngen to allow frequent updates'
        self.history = {'global_fitness':[], 'JADE':[], 'CoDE':[], 'EPSDE':[]}
        self.best_fitness=float("inf") 
        self.verbose=verbose
        self.a=1   #for discrete analysis

        self.mixPop = np.zeros((self.npop, self.dim))
        if x0:
            assert len(x0) == self.npop, '--error: the length of x0 ({}) MUST equal the number of npop in the group ({})'.format(len(x0), self.npop)
                
            for i in range(self.npop):
                check_mixed_individual(x=x0[i], bounds=self.orig_bounds) #assert the type provided is consistent
                if self.grid_flag:
                    self.mixPop[i,:] = encode_grid_indv_to_discrete(x0[i], bounds=self.orig_bounds, bounds_map=self.bounds_map)
                else:
                    self.mixPop[i,:] = x0[i]
        else:
            # Initialize the positions
            for i in range(self.npop):
                self.mixPop[i,:]=self.init_sample(self.bounds)
        

        #---------------------------------------------------------------------------
        #-------------------------------EDEV starts here----------------------------
        #---------------------------------------------------------------------------
        # Define the dimension of the problem
        self.FES=0
        arrayGbestChange=np.ones((3), dtype=float)
        arrayGbestChangeRate=np.zeros((3), dtype=float)
        indexBestLN=0
        numViaLN=np.array([0,0,0])
        rateViaLN=zeros((ngen,len(numViaLN)))
        
        #evaluate the first population   
        #for discrete mutation
        for iii in range(self.mixPop.shape[0]):
            self.mixPop[iii, :] = self.ensure_bounds(self.mixPop[iii, :])
            self.mixPop[iii, :] = self.ensure_discrete(self.mixPop[iii, :])
        mixVal=np.array(self.eval_pop(self.mixPop))
        self.best_position, self.overallBestVal, _ = self.select(self.mixPop, mixVal)
        
        permutation=np.random.permutation(self.npop)
                
        slice_index=int(self.lambda_*self.npop)
        arrayThird=permutation[:slice_index] #for EPSDE
        arraySecond=permutation[slice_index : 2*slice_index]  #for CoDE
        arrayFirst=permutation[2*slice_index:]    #for JADE 
        
        if verbose:
            print('--debug: initial size of EDEV population is %i'%self.npop)
            print('--debug: initial size of pop1 is %i'%len(arrayFirst))
            print('--debug: initial size of pop2 is %i'%len(arraySecond))
            print('--debug: initial size of pop3 is %i'%len(arrayThird))
        
        ## Initialize JADE related
        popold=self.mixPop[arrayFirst,:]
        valParents=mixVal[arrayFirst]
        c=1 / 10
        pj=0.05
        CRm=0.5
        Fm=0.5
        
        #initialize archive
        archive=ARCHIVE(NP=len(arrayFirst), dim=self.dim)
        
        #the values and indices of the best solutions
        #valBest=np.sort(valParents)
        indBest=np.argsort(valParents)
        
        #Initialize CoDE related
        popCODE=self.mixPop[arraySecond,:]
        valCODE=mixVal[arraySecond]
        
        #Initialize EPSDE related
        I_D=self.dim
        FM_pop=self.mixPop[arrayThird,:]
        FM_popold=zeros(FM_pop.shape)
        I_NP=len(arrayThird)
        val=zeros((I_NP))
        self.FVr_bestmem=zeros((I_D))
        #FVr_bestmemit=zeros((I_D))
        I_nfeval=0
        RATE=0
        
        #------Evaluate the best member after initialization----------------------
        #for discrete mutation
        for iii in range(FM_pop.shape[0]):
            FM_pop[iii, :] = self.ensure_bounds(FM_pop[iii, :])
            FM_pop[iii, :] = self.ensure_discrete(FM_pop[iii, :])
            
        val=np.array(self.eval_pop(FM_pop))
        self.FVr_bestmem, F_bestval, I_best_index=self.select(pos=FM_pop, fit=val)
        
        #------DE-Minimization---------------------------------------------
        #------FM_popold is the population which has to compete. It is--------
        #------static through one iteration. FM_pop is the newly--------------
        #------emerging population.----------------------------------------
        I_iter=0
        FF=np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        CR=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        spara=np.random.permutation(3)
        PPara=np.empty((0,3), dtype=float)
        inde=np.arange(0,I_NP)
        RR=zeros((self.npop))
        gen=0
        FESj=0
        goodCR=[]
        goodF=[]
        rate=[]
        
        while gen < ngen:
            
            self.a= 1 - gen * ((1) / ngen)  #mir: a decreases linearly between 1 to 0, for discrete mutation
        
            ## update the index for adjust different algorithms
            if gen % ng == 0:
                arrayGbestChangeRate[0]=arrayGbestChange[0] / len(arrayFirst)
                arrayGbestChangeRate[1]=arrayGbestChange[1] / (len(arraySecond)*3)
                arrayGbestChangeRate[2]=arrayGbestChange[2] / len(arrayThird)
                indexBestLN=np.argmax(arrayGbestChangeRate)
        
                arrayGbestChange=np.ones((3), dtype=float)
                arrayGbestChangeRate=np.zeros((3), dtype=float)
        
            permutation=np.random.permutation(self.npop)
            if indexBestLN == 0:
                arrayThird=permutation[:slice_index]
                arraySecond=permutation[slice_index : 2*slice_index] 
                arrayFirst=permutation[2*slice_index:]
                numViaLN[0]=numViaLN[0] + 1
            else:
                if indexBestLN == 1:
                    arrayThird=permutation[:slice_index]
                    arrayFirst=permutation[slice_index : 2*slice_index] 
                    arraySecond=permutation[2*slice_index:]
                    numViaLN[1]=numViaLN[1] + 1
                else:
                    if indexBestLN == 2:
                        arrayFirst=permutation[:slice_index]
                        arraySecond=permutation[slice_index : 2*slice_index] 
                        arrayThird=permutation[2*slice_index:]
                        numViaLN[2]=numViaLN[2] + 1
                        
            rateViaLN[gen,:]=numViaLN / sum(numViaLN)
            
            #----------------------------
            #JADE
            #----------------------------
            pop=self.mixPop[arrayFirst,:]
            valParents=mixVal[arrayFirst]
            popsize=len(arrayFirst)
            if FESj > 1 and len(goodCR) > 0 and sum(goodF) > 0:
                CRm = (1 - c)*CRm + c * np.mean(goodCR)
                Fm = (1 - c)*Fm + c * sum(goodF ** 2) / sum(goodF)
        
            # Generate CR according to a normal distribution with mean CRm, and std 0.1
            # Generate F according to a cauchy distribution with location parameter Fm, and scale parameter 0.1
            Fj,CRj=randFCR(popsize,CRm,0.1,Fm,0.1)
            r0=arange(0,popsize)
            if archive.pop.size == 0:
                popAll=copy(pop)
            else:
                popAll=np.concatenate((pop, archive.pop),axis=0)
            r1,r2=gnR1R2(popsize-1,popAll.shape[0]-1,r0)   #-1 for python indexing from 0
            indBest=np.argsort(valParents)
            pNP=max(round(pj*popsize),2)   #choose at least two best solutions
            randindex=np.ceil(np.random.uniform(size=(popsize))*pNP)  #select from [1, 2, 3, ..., pNP]
            randindex[randindex < 1]=1   #to avoid the problem that rand = 0 and thus ceil(rand) = 0
            randindex=randindex-1 #for python indexing
            assert np.min(randindex) >= 0, '--error: one of the indices in randindex is less than 0, the alg does not function properly'
            pbest=pop[indBest[randindex.astype(int)],:]  #randomly choose one of the top 100p% solutions
            
            # == == == == == == == == == == == == == == == Mutation == == == == == == == == == == == == ==
            Ftr=Fj.reshape(Fj.shape[0],1)   #transform Fj
            Ftr=np.tile(Ftr,(1,self.dim))   #convert Fj from npop dimensions to a npop x dim with fixed value in all dimensions
            vi=pop + multiply(Ftr,(pbest - pop + pop[r1,:] - popAll[r2,:]))
            
            #apply bound constraint check
            for iii in range(len(vi)):
                vi[iii,:]=self.ensure_bounds(vi[iii,:])
                
            # == == == == = Crossover == == == == =
            CRtr=CRj.reshape(CRj.shape[0],1)   #transform Crj
            CRtr=np.tile(CRtr,(1,self.dim))   #convert Fj from npop dimensions to a npop x dim with fixed value in all dimensions
            
            mask=np.random.uniform(size=(popsize,self.dim)) > CRtr  #mask is used to indicate which elements of ui comes from the parent
            rows=arange(popsize)
            cols=np.array(np.floor(np.random.uniform(size=(popsize)) * self.dim) + 1, dtype=int)
            jrand=np.array(sub2ind((popsize,self.dim),rows,cols),dtype=int)
            mask=mask.reshape(-1)
            mask[jrand]=False
            mask=mask.reshape(popsize,self.dim)
            ui=copy(vi)
            ui[mask]=pop[mask]
            
            #evaluate the offspring
            #for discrete mutation
            for iii in range(ui.shape[0]):
                ui[iii, :] = self.ensure_bounds(ui[iii, :])
                ui[iii, :] = self.ensure_discrete(ui[iii, :])
            valOffspring=np.array(self.eval_pop(ui))
            
            #== == == == == == == == == == == == == == == Selection == == == == == == == == == == == == ==
            # I == 1: the parent is better; I == 2: the offspring is better
            valParents=np.minimum(valParents, valOffspring)
            I= np.where(valOffspring <= valParents)[0] #indices where offspring is better than the parents
            popold=copy(pop)
            archive.UpdateArchive(popold[I,:],valParents[I])
            popold[I,:]=ui[I,:]
            goodCR=CRj[I]
            goodCR=Fj[I]
            if np.min(valParents) < self.overallBestVal:
                self.overallBestVal=np.min(valParents)
                min_best_index=np.argmin(valParents)
                self.best_position=popold[min_best_index,:].copy()
                
            self.jade_fitness=np.min(valParents)  #store best JADE fitness
            
            arrayGbestChange[0]=arrayGbestChange[0] + np.sum(mixVal[arrayFirst] - valParents)
            self.mixPop[arrayFirst,:]=popold
            mixVal[arrayFirst]=valParents
            
            #----------------------------
            #CoDE
            #----------------------------
            popCODE=self.mixPop[arraySecond,:]
            valCODE=mixVal[arraySecond]
            popsizeC=len(arraySecond)
            pTemp=copy(popCODE)
            fitTemp=copy(valCODE)
            uSet=zeros((3*popsizeC,self.dim))
            for i in range(popsizeC):
                F=np.array([1.0,1.0,0.8])
                CRC=np.array([0.1,0.9,0.2])
                #Uniformly and randomly select one of the control
                # parameter settings for each trial vector generation strategy
                paraIndex=np.floor(np.random.uniform(size=(3))*len(F))
                u=generator(popCODE, self.bounds, i, F, CRC, popsizeC, self.dim, paraIndex.astype(int))
                uSet[(i+1)*3 - 3 : 3*(i+1),:]=copy(u)
                
            # Evaluate the trial vectors
            for iii in range(uSet.shape[0]):   #discrete mutation
                uSet[iii, :] = self.ensure_bounds(uSet[iii, :])
                uSet[iii, :] = self.ensure_discrete(uSet[iii, :])
            fitSet=np.array(self.eval_pop(uSet))
            
            for i in range(popsizeC):
                #minVal=np.min(fitSet[(i+1)*3 - 3 : 3*(i+1)])
                minID=np.argmin(fitSet[(i+1)*3 - 3 : 3*(i+1)])
                bestInd=uSet[3*i + minID,:]
                bestIndFit=fitSet[3*i + minID]
                # Choose the better one between the trial vector and the target vector
                if valCODE[i] >= bestIndFit:
                    pTemp[i,:]=bestInd.copy()
                    fitTemp[i]=bestIndFit

            popCODE=copy(pTemp)
            valCODE=copy(fitTemp)
                    
            if np.min(fitTemp) < self.overallBestVal:
                self.overallBestVal=np.min(fitTemp)
                min_best_index=np.argmin(fitTemp)
                self.best_position=popCODE[min_best_index,:].copy()
            
            self.code_fitness=np.min(fitTemp)  #store best CoDE fitness

            arrayGbestChange[1]=arrayGbestChange[1] + np.sum(mixVal[arraySecond] - valCODE)
            self.mixPop[arraySecond,:]=popCODE
            mixVal[arraySecond]=valCODE
            
            #----------------------------
            #EPSDE
            #----------------------------
            FM_pop=self.mixPop[arrayThird,:]
            val=mixVal[arrayThird]
            I_NP=len(arrayThird)
            inde=arange(I_NP)
            Para=zeros((I_NP,3))
            I_iter=gen
            
            if (I_iter == 0 or len(Para) < I_NP):
                Para[:,0]=spara[np.random.randint(spara.shape[0],size=inde.shape[0])]
                Para[:,1]=CR[np.random.randint(CR.shape[0],size=inde.shape[0])]
                Para[:,2]=FF[np.random.randint(FF.shape[0],size=inde.shape[0])]
            else:
                for k in range(len(inde)):
                    if (np.random.uniform() <= RATE and len(PPara) > 0) or len(RR) < I_NP:
                        RR[k]=np.random.randint(PPara.shape[0])
                        Para[inde[k],:]=PPara[np.random.randint(PPara.shape[0]),:]
                    else:
                        RR[k]=0
                        Para[inde[k],:]=[
                                        spara[np.random.randint(spara.shape[0])],
                                        CR[np.random.randint(CR.shape[0])],
                                        FF[np.random.randint(FF.shape[0])]
                                        ]
        
            RRR=[]
            count=0
            FM_popold=copy(FM_pop)
            par=np.zeros((I_NP, I_D))
            FM_ui=par.copy()
            
            for i in range(I_NP):
                FM_mui=np.random.uniform(size=(I_D)) < Para[i,1]
                
                dd=np.where(FM_mui == True)[0]
                if len(dd) == 0:
                    ddd=int(np.ceil(np.random.uniform() * I_D) - 1)
                    FM_mui[ddd]=1
                    
                FM_mpo=FM_mui < 0.5
                FM_bm=copy(self.FVr_bestmem)
                
                par[i,:]= np.random.normal(Para[i,2],0.001,size=(I_D))
                
                if (Para[i,0] == 0):
                    #DE/best/2/bin
                    ind=np.random.permutation(I_NP)
                    FM_pm3=FM_popold[ind[0],:]
                    FM_pm4=FM_popold[ind[1],:]
                    FM_pm5=FM_popold[ind[2],:]
                    FM_pm6=FM_popold[ind[3],:]
                    FM_ui[i,:]=FM_bm + (FM_pm3 - FM_pm4 + FM_pm5 - FM_pm6) * par[i,:]
                    FM_ui[i,:]=multiply(FM_popold[i,:],FM_mpo) + multiply(FM_ui[i,:],FM_mui)
                if (Para[i,0] == 1):
                    #DE/rand/1/bin
                    ind=np.random.permutation(I_NP)
                    FM_pm7=FM_popold[ind[0],:]
                    FM_pm8=FM_popold[ind[1],:]
                    FM_pm9=FM_popold[ind[2],:]
                    FM_ui[i,:]=FM_pm7 + par[i,:] * (FM_pm8 - FM_pm9)
                    FM_ui[i,:]=multiply(FM_popold[i,:],FM_mpo) + multiply(FM_ui[i,:],FM_mui)
                if (Para[i,0] == 2):
                    # DE/current-to-rand/1/bin/
                    ind=np.random.permutation(I_NP)
                    FM_pm21=FM_popold[ind[0],:]
                    FM_pm22=FM_popold[ind[1],:]
                    FM_pm23=FM_popold[ind[2],:]
                    FM_ui[i,:]=FM_popold[i,:] + np.random.uniform(size=I_D) * (FM_pm21 - FM_popold[i,:]) + par[i,:] * (FM_pm22 - FM_pm23)
        
                if np.any(FM_ui[i,:] < self.lb) or np.any(FM_ui[i,:] > self.ub):
                    #reset an individual outside the bounds
                    FM_ui[i,:]=self.lb + multiply((self.ub - self.lb),np.random.uniform(size=I_D))
                        
            #evaluate the updated population
            for iii in range(FM_ui.shape[0]):   #discrete mutation
                FM_ui[iii, :] = self.ensure_bounds(FM_ui[iii, :])
                FM_ui[iii, :] = self.ensure_discrete(FM_ui[iii, :])
            tempval=np.array(self.eval_pop(FM_ui))
                    
        
            for i in range(I_NP):
                I_nfeval=I_nfeval + 1
                if (tempval[i] < val[i]):
                    FM_pop[i,:]=FM_ui[i,:]
                    val[i]=tempval[i]
                    PPara=np.append(PPara, Para[i,:].reshape(1,-1), axis=0)
                    if RR[i] != 0:
                        RRR.append(int(RR[i]))
                    if tempval[i] < F_bestval:
                        F_bestval=tempval[i]
                        self.FVr_bestmem=FM_ui[i,:]
                        I_best_index=i
                else:
                    count=count + 1
                    
            if np.min(val) < self.overallBestVal:
                self.overallBestVal=np.min(val)
                min_best_index=np.argmin(val)
                self.best_position=FM_pop[min_best_index,:].copy()
            
            self.epsde_fitness=np.min(val)  #store best EPSDE fitness
                
            arrayGbestChange[2]=arrayGbestChange[2] + np.sum(mixVal[arrayThird] - val)
            self.mixPop[arrayThird,:]=FM_pop
            mixVal[arrayThird]=val
            PPara = np.delete(PPara, RRR, axis=0)
            rate.append(count / I_NP)
            if I_iter > 10:
                RATE=np.mean(rate[(I_iter - 10) : I_iter])
            else:
                RATE=np.mean(rate)
            I_iter=I_iter + 1
            gen=gen + 1

        #---------------------------------------------------------------------------
        #-------------------------------EDEV ends here----------------------------
        #---------------------------------------------------------------------------
                
            #--mir
            if self.mode=='max':
                self.fitness_best_correct=-self.overallBestVal
                self.jade_fitness=-self.jade_fitness
                self.code_fitness=-self.code_fitness
                self.epsde_fitness=-self.epsde_fitness
                
            else:
                self.fitness_best_correct=self.overallBestVal

            self.last_pop=self.mixPop.copy()
            self.last_fit=np.array(mixVal).copy()            
            self.history['global_fitness'].append(self.fitness_best_correct)
            self.history['JADE'].append(self.jade_fitness)
            self.history['CoDE'].append(self.code_fitness)
            self.history['EPSDE'].append(self.epsde_fitness)
                        
            # Print statistics
            if self.verbose:
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                print('EDEV step {}/{}, npop={}, Ncores={}'.format((gen)*self.npop, ngen*self.npop, self.npop, self.ncores))
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                print('Best EDEV Fitness:', np.round(self.fitness_best_correct,6))
                if self.grid_flag:
                    self.ind_decoded = decode_discrete_to_grid(self.best_position, self.orig_bounds, self.bounds_map)
                    print('Best Position:', self.ind_decoded)
                else:
                    print('Best Position:', self.best_position)
                print('JADE Fitness:', self.jade_fitness)
                print('CoDE Fitness:', self.code_fitness)
                print('EPSDE Fitness:', self.epsde_fitness)
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        #mir-grid
        if self.grid_flag:
            self.ind_correct = decode_discrete_to_grid(self.best_position, self.orig_bounds, self.bounds_map)
        else:
            self.ind_correct = self.best_position                

        if self.mode=='max':
            self.last_fit=-self.last_fit

        #--mir return the last population for restart calculations
        if self.grid_flag:
            self.history['last_pop'] = get_population(self.last_pop, fits=self.last_fit, grid_flag=True, 
                                                     bounds=self.orig_bounds, bounds_map=self.bounds_map)
        else:
            self.history['last_pop'] = get_population(self.last_pop, fits=self.last_fit, grid_flag=False)
            
        self.history['F-Evals'] = self.FES   #function evaluations 
        if self.verbose:
            print('------------------------ EDEV Summary --------------------------')
            print('Best fitness (y) found:', self.fitness_best_correct)
            print('Best individual (x) found:', self.ind_correct)
            print('--------------------------------------------------------------')  
            
        return self.ind_correct, self.fitness_best_correct, self.history

