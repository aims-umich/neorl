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
"""
Created on Fri Jun 19 17:23:10 2020

@author: Majdi
"""

import random 
import numpy as np

#alpha0=0, anneal_alpha=False, anneal_steps=None, alpha_end=1
class ExperienceReplay:
    def __init__(self, size):
        #"""
        #:param size (int): the max size of the memory
        #"""
        random.seed(1)
        self.size=size
        self.storage = []
        self.next_indx=0    
        
    @property
    def memory(self):
        #"""
        #content of the memory buffer
        #[(np.ndarray, float, str)]: 
        #"""
        return self.storage

    @property
    def buffer_size(self):
        #"""Max memory capacity"""
        return self.size
    

    def memory_can_sample(self, n_samples):
        #"""
        #Check if n_samples samples can be sampled from the memory.
        #:param n_samples (int):  number of samples to draw
        #:return (bool): whether we can sample or not
        #"""
        return len(self.storage) >= n_samples

    def memory_is_full(self):
        #"""
        #Check whether the replay buffer is full or not.
        #:return: (bool)
        #"""
        return len(self.storage) == self.buffer_size

    def add(self, xvec, obj, method=None):
        #"""
        #add a new sample to the memory
        #:param xvec (list): a sample of input attributes (x1,x2,...,xn)
        #:param obj (float): objective value of xvec
        #:param method (string): method of which this sample belongs to
        #"""

        
        #check if multiple or single samples is to be added 
        if type(obj) is list: # multiple samples 
            if method:
                data = [(x, o, m) for x,o,m in zip(xvec,obj,method)]
            else:
                data = [(x, o) for x,o in zip(xvec,obj)]
            
            for sample in data:
                # check if sample is in memory
                if sample not in self.storage:
                    if self.next_indx >= len(self.storage):
                        self.storage.append(sample)
                    else:
                        self.storage[self.next_indx] = sample
                    self.next_indx = (self.next_indx + 1) % self.size
            
        else: #single sample
            if method:
                data = (xvec, obj, method)
            else:
                data = (xvec, obj)
            
            # check if sample is in memory
            if data not in self.storage:
                if self.next_indx >= len(self.storage):
                    self.storage.append(data)
                else:
                    self.storage[self.next_indx] = data
                self.next_indx = (self.next_indx + 1) % self.size
                    
    def calc_priorities(self, alpha):
        #"""
        #calculate priorties for each memory sample
        #:param alpha: priortization value 
        #:return
        #  - list of normalized priorities 
        #"""
        #Fixed :)
        
        self.storage.sort(key=lambda e: e[1], reverse=True)
        ranks=np.array(range(1,len(self.storage)+1))
        ranks=1/ranks
        priors=ranks**alpha/np.sum(ranks**alpha)
        assert np.round(np.sum(priors)) == 1.0, 'the calculated priorties are not normalized'
        return priors

    def sample(self, batch_size=1, mode='uniform', alpha=0.5, seed=1):
        #"""
        #Sample a batch of experiences.
        #:param batch_size: number of samples to draw from the memory
        #:param mode (str): the priortization mode
        #  -"uniform": uniform sampling
        #  -"greedy": always use the best samples
        #  -"prior": priortized replay with alpha value
        #:param alpha: the prioritization ceoffcient between 0 (no priority) and 1 (full priority)
        #:return:
        #    - batch_size of samples in a list of tuples [(np.ndarray, float, str),...,(np.ndarray, float, str)]
        #"""
        if mode=='uniform': # uniform sampling
            idxs = [random.randint(0, len(self.storage) - 1) for _ in range(batch_size)]
            return [self.storage[i] for i in idxs]
        elif mode=='greedy': #greedy mode (always take the highest)
            #Fixed :) 
            self.storage.sort(key=lambda e: e[1], reverse=True)
            return self.storage[:batch_size]
        elif mode=='prior':  #priortized replay
            priors=self.calc_priorities(alpha=alpha)
            np.random.seed(seed)
            idxs = np.random.choice(range(len(self.storage)),p=priors, size=batch_size)
            return [self.storage[i] for i in idxs]
        else:
            raise ('unknown mode is entered for experience replay: either uniform, greedy, or prior are allowed')
    
    def remove_duplicates(self):
        
        #TODO: Time consuming step!!!
        if len(self.storage[0])==3: #(x, obj, method) tuple
            seen = []
            # using list comprehension 
            self.storage = [(a, b, c) for a, b, c in self.storage 
                     if not (a in seen or seen.append(a))]
        elif len(self.storage[0])==2:   #(x, obj) tuple
            seen = []
            # using list comprehension 
            self.storage = [(a, b) for a, b in self.storage 
                     if not (a in seen or seen.append(a))]
        else:
            raise ('memory content is corrupted and cannot be filtered')

#if __name__=='__main__':
#    random.seed(1)
#    per=ExperienceReplay(size=100, mode='uniform')
#    
#    for i in range(100):
#        sample=(np.ones((4))*random.randint(1,10), random.randint(1,10), random.choice(['ga','sa','pso']))
#        per.add(xvec=sample[0], obj=sample[1], method=sample[2])
#    
#    x=per.storage
#    samples=per.sample(10,alpha=1)
#    print([item[1] for item in samples])
  
    