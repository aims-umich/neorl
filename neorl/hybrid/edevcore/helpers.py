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
#Created on Thu March  3 14:42:29 2022
#
#@author: Majdi Radaideh
#"""

import numpy as np
from numpy import arange, multiply

def ensure_bounds(vec, bounds):

    vec_new = []
    # cycle through each variable in vector 
    for i, (key, val) in enumerate(bounds.items()):

        # variable exceedes the minimum boundary
        if vec[i] < bounds[key][1]:
            vec_new.append(bounds[key][1])

        # variable exceedes the maximum boundary
        if vec[i] > bounds[key][2]:
            vec_new.append(bounds[key][2])

        # the variable is fine
        if bounds[key][1] <= vec[i] <= bounds[key][2]:
            vec_new.append(vec[i])
        
    return vec_new

def generator(p,bounds,i,
              F,CR,popsize,
              dim,paraIndex,*args,**kwargs):

    #.... "rand/1/bin" strategy ....#
    
    u=np.zeros((3,dim))
    # Choose the indices for mutation
    indexSet=list(arange(popsize))
    indexSet.remove(indexSet[i])
    index=[]

    # Choose the first Index
    temp=int(np.floor(np.random.uniform() * (popsize - 2)) + 1); index.append(indexSet[temp]); indexSet.remove(indexSet[temp])

    # Choose the second index
    temp=int(np.floor(np.random.uniform() * (popsize - 3)) + 1); index.append(indexSet[temp]); indexSet.remove(indexSet[temp])
    
    # Choose the third index
    temp=int(np.floor(np.random.uniform() * (popsize - 4)) + 1); index.append(indexSet[temp])
    
    index=np.array(index)
    
    # Mutation
    v1=p[index[0],:] + F[paraIndex[0]] * (p[index[1],:] - p[index[2],:])

    # Handle the elements of the mutant vector which violate the boundary
    v1=ensure_bounds(v1, bounds)
        
    # Binomial crossover
    j_rand=int(np.floor(np.random.uniform() * dim))
    t=np.random.uniform(size=(dim)) < CR[paraIndex[0]]
    t[j_rand]=1
    t_=1 - t
    u[0,:]=multiply(t,v1) + multiply(t_,p[i,:])
    
    #... "current to rand/1" strategy ...#
    
    # The mechanism to choose the indices for mutation is slightly different from that of the classic
    # "current to rand/1", we found that using the following mechanism to choose the indices for
    # mutation can improve the performance to certain degree
    index[0:3]=np.floor(np.random.uniform(size=(3)) * popsize)
    
    index=index.astype(int)

    # Mutation
    v2=p[i,:] + np.random.uniform() * (p[index[0],:] - p[i,:]) + F[paraIndex[1]] * (p[index[1],:] - p[index[2],:])
    v2=ensure_bounds(v2, bounds)

    # Binomial crossover is not used to generate the trial vector under this condition
    u[1,:]=v2
    
    #... "rand/2/bin" strategy ...#
    
    # Choose the indices for mutation
    indexSet=list(arange(popsize))
    indexSet.remove(indexSet[i])
    index=[]

    # Choose the first index
    temp=int(np.floor(np.random.uniform() * (popsize - 2)) + 1); index.append(indexSet[temp]); indexSet.remove(indexSet[temp])

    # Choose the second index
    temp=int(np.floor(np.random.uniform() * (popsize - 3)) + 1); index.append(indexSet[temp]); indexSet.remove(indexSet[temp])

    # Choose the third index
    temp=int(np.floor(np.random.uniform() * (popsize - 4)) + 1); index.append(indexSet[temp]); indexSet.remove(indexSet[temp])
    
    # Choose the fourth index
    temp=int(np.floor(np.random.uniform() * (popsize - 5)) + 1); index.append(indexSet[temp]); indexSet.remove(indexSet[temp])

    # Choose the fifth index
    temp=int(np.floor(np.random.uniform() * (popsize - 6)) + 1); index.append(indexSet[temp])

    # Mutation
    # The first scaling factor (F) is randomly chosen from 0 to 1
    v3=p[index[0],:] + np.random.uniform() * (p[index[1],:] - p[index[2],:]) + F[paraIndex[2]] * (p[index[3],:]- p[index[4],:])

    # Handle the elements of the mutant vector which violate the boundary
    v3=ensure_bounds(v3, bounds)
    
    # Binomial crossover
    j_rand=int(np.floor(np.random.uniform() * dim))
    t=np.random.uniform(size=(dim)) < CR[paraIndex[2]]
    t[j_rand]=1
    t_=1 - t
    u[2,:]=multiply(t,v3) + multiply(t_,p[i,:])
    
    return u

def sub2ind(sz, row, col):
    n_rows = sz[0]
    return [n_rows * (c-1) + r for r, c in zip(row, col)]

def gnR1R2(NP1, NP2, r0,*args,**kwargs):
    # gnA1A2 generate two column vectors r1 and r2 of size NP1 & NP2, respectively
    #r1's elements are choosen from {1, 2, ..., NP1} & r1(i) ~= r0(i)
    #r2's elements are choosen from {1, 2, ..., NP2} & r2(i) ~= r1(i) & r2(i) ~= r0(i)
    
    # Call:
    #[r1 r2 ...] = gnA1A2(NP1)   # r0 is set to be (1:NP1)'
    #[r1 r2 ...] = gnA1A2(NP1, r0) # r0 should be of length NP1
    NP0=len(r0)
    r1=np.floor(np.random.uniform(size=(NP0))*NP1) + 1
    for i in range(1,10000):
        pos = (r1 == r0)
        if np.sum(pos) == 0:
            break
        else:
            r1[pos]=np.floor(np.random.uniform(size=(sum(pos))) * NP1) + 1
        if i > 1000:
            raise Exception('--error: gnR1R2 function failed to genrate r1 in 1000 iterations')
    
    r2=np.floor(np.random.uniform(size=(NP0))*NP2) + 1
    for i in range(1,10000):
        pos=np.logical_or((r2 == r1), (r2 == r0))
        if np.sum(pos) == 0:
            break
        else:
            r2[pos]=np.floor(np.random.uniform(size=(sum(pos))) * NP2) + 1
        if i > 1000:
            raise Exception('--error: gnR1R2 function failed to genrate r1 in 1000 iterations')
    
    
    return r1.astype(int), r2.astype(int)

def randFCR(NP,CRm,CRsigma,Fm,Fsigma,*args,**kwargs):

    # this function generate CR according to a normal distribution with mean "CRm" and sigma "CRsigma"
    #If CR > 1, set CR = 1. If CR < 0, set CR = 0.
    #this function generate F  according to a cauchy distribution with location parameter "Fm" and scale parameter "Fsigma"
    #If F > 1, set F = 1. If F <= 0, regenrate F.
    
    ## generate CR
    CR=CRm + CRsigma * np.random.randn(NP)
    CR=np.clip(CR,0,1)
    
    ## generate F
    F=randCauchy(NP,1,Fm,Fsigma)
    F=np.minimum(1,F)
    # we don't want F = 0. So, if F<=0, we regenerate F (instead of trucating it to 0)
    pos=np.argwhere(F <= 0).flatten()
    while pos.size > 0:
        F[pos]=randCauchy(len(pos),1,Fm,Fsigma)
        F=np.minimum(1,F)
        pos=np.argwhere(F <= 0).flatten()
    
    return F, CR
    
    # Cauchy distribution: cauchypdf = @(x, mu, delta) 1/pi*delta./((x-mu).^2+delta^2)
def randCauchy(m,n,mu,delta,*args,**kwargs):

    # http://en.wikipedia.org/wiki/Cauchy_distribution
    result=mu + delta * np.tan(np.pi * (np.random.uniform(size=(m)) - 0.5))
    return result

class ARCHIVE(object):
    """
    Archive class for Ensemble DE algorathim
    
    """
    def __init__(self, NP, dim):
        
        self.NP= NP   #the maximum size of the archive
        self.pop= np.empty((0,dim), dtype=float)
        self.funvalues= np.empty((0), dtype=float)
    
    def UpdateArchive(self, popi, funvalue):
        
        self.pop = np.append(self.pop, popi, axis=0)
        self.funvalues = np.append(self.funvalues, funvalue, axis=0)

        #remove duplicates
        self.pop,self.ind=np.unique(self.pop, axis=0, return_index=True)
        self.funvalues = self.funvalues[self.ind]
        
        #check the archive capacity
        if self.pop.shape[0] > self.NP:
            randpos=np.random.permutation(self.pop.shape[0])[:self.NP]  #keep random NP individuals
            self.pop=self.pop[randpos,:]
            self.funvalues=self.funvalues[randpos]