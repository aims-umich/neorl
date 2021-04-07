# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:42:29 2020

@author: Majdi
"""

"""
xNES from 'Natural Evolution Strategies'
if n_jobs>1, I suggest using "export MKL_NUM_THREADS=1"
See at the bottom (under __main__) for an example of usage
"""
import joblib
import random
import numpy as np

import scipy as sp
from scipy import (dot, eye, randn, asarray, array, trace, log, exp, sqrt, mean, sum, argsort, square, arange)
from scipy.stats import multivariate_normal, norm
from scipy.linalg import (det, expm)
import pandas as pd
import copy

class XNES(object):
    def __init__(self, f, mu, amat, bounds,
                 eta_mu=1.0, eta_sigma=None, eta_bmat=None,
                 npop=None, use_fshape=True, use_adasam=False, patience=100, ncores=1, verbose=True, seed=None):
        
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        self.seed=seed
        
        self.fitness_hom=-np.inf

        self.f = f
        self.mu = mu
        self.eta_mu = eta_mu
        self.use_adasam = use_adasam
        self.ncores = ncores
        self.bounds=bounds

        dim = len(mu)
        sigma = abs(det(amat))**(1.0/dim)
        bmat = amat*(1.0/sigma)
        self.dim = dim
        self.sigma = sigma
        self.bmat = bmat

        # default population size and learning rates
        npop = int(4 + 3*log(dim)) if npop is None else npop
        eta_sigma = 3*(3+log(dim))*(1.0/(5*dim*sqrt(dim))) if eta_sigma is None else eta_sigma
        eta_bmat = 3*(3+log(dim))*(1.0/(5*dim*sqrt(dim))) if eta_bmat is None else eta_bmat
        self.npop = npop
        self.eta_sigma = eta_sigma
        self.eta_bmat = eta_bmat

        # compute utilities if using fitness shaping
        if use_fshape:
            a = log(1+0.5*npop)
            utilities = array([max(0, a-log(k)) for k in range(1,npop+1)])
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
        self.verbose=verbose

        # do not use these when hill-climbing
        if npop == 1:
            self.use_fshape = False
            self.use_adasam = False

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

    def evolute(self, niter):
        """ xNES """
        f = self.f
        mu, sigma, bmat = self.mu, self.sigma, self.bmat
        eta_mu, eta_sigma, eta_bmat = self.eta_mu, self.eta_sigma, self.eta_bmat
        npop = self.npop
        dim = self.dim
        sigma_old = self.sigma_old

        eyemat = eye(dim)

        with joblib.Parallel(n_jobs=self.ncores) as parallel:

            for i in range(niter):
                s_try = randn(npop, dim)
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
                    
                if self.counter > self.patience:
                    self.done = True
                    return
                
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

                # Print data
                if self.verbose and i % self.npop:
                    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                    print('NES step {}/{}, NPOP={}, ETA_MU={}, ETA_SIGMA={}, ETA_BMAT={}, Ncores={}'.format((i+1)*self.npop, niter*self.npop, self.npop, np.round(self.eta_mu,2), np.round(self.eta_sigma,2), np.round(self.eta_bmat,2), self.ncores))
                    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                    print('Best Swarm Fitness:', np.round(self.fitness_best,6))
                    print('Best Swarm Position:', np.round(self.x_best,6))
                    print('MU:', np.round(mu,3))
                    print('Sigma:', np.round(sigma,3))
                    print('BMAT:', np.round(bmat,3))
                    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
            
        # keep last results
        self.mu, self.sigma, self.bmat = mu, sigma, bmat
        self.eta_sigma = eta_sigma
        self.sigma_old = sigma_old
        
        print('------------------------ NES Summary --------------------------')
        print('Best fitness (y) found:', self.fitness_best)
        print('Best individual (x) found:', self.x_best)
        print('--------------------------------------------------------------')  
        
        return self.x_best, self.fitness_best, self.history

    def adasam(self, eta_sigma, mu, sigma, bmat, sigma_old, z_try):
        """ Adaptation sampling """
        eta_sigma_init = self.eta_sigma_init
        dim = self.dim
        c = .1
        rho = 0.5 - 1./(3*(dim+1))  # empirical

        bbmat = dot(bmat.T, bmat)
        cov = sigma**2 * bbmat
        sigma_ = sigma * sqrt(sigma*(1./sigma_old))  # increase by 1.5
        cov_ = sigma_**2 * bbmat

        p0 = multivariate_normal.logpdf(z_try, mean=mu, cov=cov)
        p1 = multivariate_normal.logpdf(z_try, mean=mu, cov=cov_)
        w = exp(p1-p0)

        # Mann-Whitney. It is assumed z_try was in ascending order.
        n = self.npop
        n_ = sum(w)
        u_ = sum(w * (arange(n)+0.5))

        u_mu = n*n_*0.5
        u_sigma = sqrt(n*n_*(n+n_+1)/12.)
        cum = norm.cdf(u_, loc=u_mu, scale=u_sigma)

        if cum < rho:
            return (1-c)*eta_sigma + c*eta_sigma_init
        else:
            return min(1, (1+c)*eta_sigma)