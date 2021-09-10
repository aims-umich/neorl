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


from numpy import dot, eye, asarray, array, trace, exp 
from numpy import mean, sum, argsort, arange
from scipy.stats import multivariate_normal, norm
from scipy.linalg import det, expm
import joblib
import random
import numpy as np
import copy

class XNES(object):
    """
    Exponential Natural Evolution Strategies
    
    :param mode: (str) problem type, either "min" for minimization problem or "max" for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param fit: (function) the fitness function 
    :param npop: (int) total number of individuals in the population (default: if None, it will make an approximation, see **Notes** below)
    :param A: (np.array): initial guess of the covariance matrix A (default: identity matrix, see **Notes** below)
    :param eta_mu: (float) learning rate for updating the center of the search distribution ``mu`` (see **Notes** below)
    :param eta_sigma: (float) learning rate for updating the step size ``sigma`` (default: if None, it will make an approximation, see **Notes** below)
    :param eta_Bmat: (float) learning rate for updating the normalized transformation matrix ``B``  (default: if None, it will make an approximation, see **Notes** below)
    :param adapt_sampling: (bool): activate the adaption sampling option
    :param ncores: (int) number of parallel processors
    :param seed: (int) random seed for sampling
    """
    def __init__(self, mode, bounds, fit, A=None, npop=None,
                 eta_mu=1.0, eta_sigma=None, eta_Bmat=None, 
                 adapt_sampling=False, ncores=1, seed=None):
        
        if seed:
            random.seed(seed)
            np.random.seed(seed)
            
        self.seed=seed
        patience=100
        self.fitness_hom=-np.inf
        
        #--mir
        self.mode=mode
        if mode == 'max':
            self.f=fit
        elif mode == 'min':
            def fitness_wrapper(*args, **kwargs):
                return -fit(*args, **kwargs) 
            self.f=fitness_wrapper
        else:
            raise ValueError('--error: The mode entered by user is invalid, use either `min` or `max`')
            
        self.eta_mu = eta_mu
        self.use_adasam = adapt_sampling
        self.ncores = ncores
        self.bounds=bounds

        dim = len(bounds)
        A = np.eye(dim) if A is None else A
        sigma = abs(det(A))**(1.0/dim)
        bmat = A*(1.0/sigma)
        self.dim = dim
        self.sigma = sigma
        self.bmat = bmat

        # default population size and learning rates
        npop = int(4 + 3*np.log(dim)) if npop is None else npop
        eta_sigma = 3*(3+np.log(dim))*(1.0/(5*dim*np.sqrt(dim))) if eta_sigma is None else eta_sigma
        eta_Bmat = 3*(3+np.log(dim))*(1.0/(5*dim*np.sqrt(dim))) if eta_Bmat is None else eta_Bmat
        self.npop = npop
        self.eta_sigma = eta_sigma
        self.eta_bmat = eta_Bmat
        
        use_fshape=True
        # compute utilities if using fitness shaping
        if use_fshape:
            a = np.log(1+0.5*npop)
            utilities = array([max(0, a-np.log(k)) for k in range(1,npop+1)])
            utilities /= sum(utilities)
            utilities -= 1.0/npop           # broadcast
            utilities = utilities[::-1]  # ascending order
        else:
            utilities = None
        self.use_fshape = use_fshape
        self.utilities = utilities

        # stuff for adasam
        self.eta_sigma_init = eta_sigma
        self.sigma_old = None

        # logging
        self.fitness_best = -np.inf
        self.mu_best = None
        self.done = False
        self.counter = 0
        self.patience = patience
        self.history = {'eta_sigma':[], 'sigma':[], 'fitness':[]}
        

        # do not use options below when one individual in population is used
        if npop == 1:
            self.use_fshape = False
            self.use_adasam = False

    def init_sample(self, bounds):
    
        indv=[]
        for key in bounds:
            if bounds[key][0] == 'int':
                indv.append(random.randint(bounds[key][1], bounds[key][2]))
            elif bounds[key][0] == 'float':
                indv.append(random.uniform(bounds[key][1], bounds[key][2]))
            elif bounds[key][0] == 'grid':
                indv.append(random.sample(bounds[key][1],1)[0])
            else:
                raise Exception ('unknown data type is given, either int, float, or grid are allowed for parameter bounds')   
        return indv
    
    def ensure_bounds(self, vec, bounds):
    
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

    def evolute(self, ngen, x0=None, verbose=True):
        """
        This function evolutes the XNES algorithm for number of generations.
        
        :param ngen: (int) number of generations to evolute
        :param x0: (list) initial guess for the search (must be of same size as ``len(bounds)``)
        :param verbose: (bool) print statistics to screen
        
        :return: (dict) dictionary containing major XNES search results
        """
        f = self.f
        self.verbose=verbose
        if x0:
            assert len(x0) == self.dim, 'the length of x0 ({}) MUST equal the number of parameters in bounds ({})'.format(len(x0), self.dim)
            self.mu=x0
        else:
            self.mu=self.init_sample(self.bounds)
        mu, sigma, bmat = self.mu, self.sigma, self.bmat
        eta_mu, eta_sigma, eta_bmat = self.eta_mu, self.eta_sigma, self.eta_bmat
        npop = self.npop
        dim = self.dim
        sigma_old = self.sigma_old

        eyemat = eye(dim)

        with joblib.Parallel(n_jobs=self.ncores) as parallel:

            for i in range(ngen):
                s_try = np.random.randn(npop, dim)
                z_try = mu + sigma * dot(s_try, bmat)     # broadcast
                
                for k in range (len(z_try)):
                    z_try[k] = self.ensure_bounds(vec=z_try[k], bounds=self.bounds)
                
                #print(z_try)
                    
                f_try = parallel(joblib.delayed(f)(z) for z in z_try)
                f_try = asarray(f_try)
                
                # save if best
                fitness = mean(f_try)

                isort = argsort(f_try)                
                f_try = f_try[isort]
                s_try = s_try[isort]
                z_try = z_try[isort]
                
                for m in range (len(f_try)):
                    if f_try[m] > self.fitness_best:
                        self.fitness_best=f_try[m]
                        self.x_best=copy.deepcopy(z_try[m])
                        
                if fitness - 1e-8 > self.fitness_best:
                    self.mu_best = mu.copy()
                    self.counter = 0
                else: 
                    self.counter += 1
                    
                #if self.counter > self.patience:
                #    self.done = True
                #    return
                
                u_try = self.utilities if self.use_fshape else f_try

                if self.use_adasam and sigma_old is not None:  # sigma_old must be available
                    eta_sigma = self.adasam(eta_sigma, mu, sigma, bmat, sigma_old, z_try)

                dj_delta = dot(u_try, s_try)
                dj_mmat = dot(s_try.T, s_try*u_try.reshape(npop,1)) - sum(u_try)*eyemat
                dj_sigma = trace(dj_mmat)*(1.0/dim)
                dj_bmat = dj_mmat - dj_sigma*eyemat

                sigma_old = sigma

                # update
                mu += eta_mu * sigma * dot(bmat, dj_delta)
                sigma *= exp(0.5 * eta_sigma * dj_sigma)
                bmat = dot(bmat, expm(0.5 * eta_bmat * dj_bmat))

                # logging
                self.history['fitness'].append(self.fitness_best)
                self.history['sigma'].append(sigma)
                self.history['eta_sigma'].append(eta_sigma)
                
                #--mir
                if self.mode=='min':
                    self.fitness_best_correct=-self.fitness_best
                else:
                    self.fitness_best_correct=self.fitness_best

                # Print data
                if self.verbose and i % self.npop:
                    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                    print('XNES step {}/{}, NPOP={}, ETA_MU={}, ETA_SIGMA={}, ETA_BMAT={}, Ncores={}'.format((i+1)*self.npop, ngen*self.npop, self.npop, np.round(self.eta_mu,2), np.round(self.eta_sigma,2), np.round(self.eta_bmat,2), self.ncores))
                    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                    print('Best XNES Fitness:', np.round(self.fitness_best_correct,6))
                    print('Best XNES Position:', np.round(self.x_best,6))
                    print('MU:', np.round(mu,3))
                    print('Sigma:', np.round(sigma,3))
                    print('BMAT:', np.round(bmat,3))
                    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
            
        # keep last results
        self.mu, self.sigma, self.bmat = mu, sigma, bmat
        self.eta_sigma = eta_sigma
        self.sigma_old = sigma_old
        
        if self.verbose:
            print('------------------------ NES Summary --------------------------')
            print('Best fitness (y) found:', self.fitness_best_correct)
            print('Best individual (x) found:', self.x_best)
            print('--------------------------------------------------------------')  

        #--mir
        if self.mode=='min':
            self.history['fitness']=[-item for item in self.history['fitness']]
            
        return self.x_best, self.fitness_best_correct, self.history

    def adasam(self, eta_sigma, mu, sigma, bmat, sigma_old, z_try):
        #Adaptation sampling
        eta_sigma_init = self.eta_sigma_init
        dim = self.dim
        c = .1
        rho = 0.5 - 1./(3*(dim+1))  # empirical

        bbmat = dot(bmat.T, bmat)
        cov = sigma**2 * bbmat
        sigma_ = sigma * np.sqrt(sigma*(1./sigma_old))  # increase by 1.5
        cov_ = sigma_**2 * bbmat

        p0 = multivariate_normal.logpdf(z_try, mean=mu, cov=cov)
        p1 = multivariate_normal.logpdf(z_try, mean=mu, cov=cov_)
        w = exp(p1-p0)

        # Mann-Whitney. It is assumed z_try was in ascending order.
        n = self.npop
        n_ = sum(w)
        u_ = sum(w * (arange(n)+0.5))

        u_mu = n*n_*0.5
        u_sigma = np.sqrt(n*n_*(n+n_+1)/12.)
        cum = norm.cdf(u_, loc=u_mu, scale=u_sigma)

        if cum < rho:
            return (1-c)*eta_sigma + c*eta_sigma_init
        else:
            return min(1, (1+c)*eta_sigma)