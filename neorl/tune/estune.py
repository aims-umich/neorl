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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neorl.evolu.es import ES

class ESTUNE:
    """
    A module for Evolutionary search for hyperparameter tuning based on ES algorithm
    
    :param param_grid: (dict) the type and range of each hyperparameter in a dictionary form (types are ``int/discrete`` or ``float/continuous`` or ``grid/categorical``).
    :param fit: (function) the self-defined fitness function that includes the hyperparameters as input and algorithm score as output
    :param mode: (str) problem type, either ``min`` for minimization problem 
                 or ``max`` for maximization. Default: Evolutionary tuner 
                 is set to maximize an objective
    :param ngen: (int) number of ES generations to run, total number of hyperparameter tests is ``ngen * 10`` (see **Notes** for an important remark) 
    :param seed: (int) random seed for sampling reproducibility
    """
    def __init__(self, param_grid, fit, mode='max', ngen=10, seed=None):
        #--mir
        self.mode=mode

        self.fit=fit
        assert self.mode in ['min', 'max'], '--error: The mode entered by user is invalid, use either `min` or `max`'
        self.param_grid=param_grid
        self.seed=seed
        self.npop=10
        #if self.ncases < self.npop:
        #    self.ncases = self.npop 
        #    print('--warning: ncases={ncases} < {npop} is given by the user, but ncases must be more than {npop}, reset ncases to {npop}'.format(ncases=self.ncases, npop=self.npop))  
            
        self.ngen = ngen
        self.param_names=[item for item in self.param_grid]

    def plot_results(self, pngname=None):
        if self.mode=='max':
            plt.plot(pd.DataFrame.cummax(self.evolures['score']), '-og')
            plt.ylabel('Max score so far')
        else:
            plt.plot(pd.DataFrame.cummin(self.evolures['score']), '-og')
            plt.ylabel('Min score so far')
            
        plt.xlabel('Generation # (Every generation includes {} cases)'.format(self.npop))
        plt.grid()
        if pngname is not None:
            plt.savefig(str(pngname)+'.png', dpi=200, format='png')
        plt.show()


    def tune(self, ncores=1, csvname=None, verbose=True):
        """
        This function starts the tuning process with specified number of processors
    
        :param ncores: (int) number of parallel processors.
        :param csvname: (str) the name of the csv file name to save the tuning results (useful for expensive cases as the csv file is updated directly after the case is done)
        :param verbose: (bool) whether to print updates to the screen or not
        """
        self.ncores=ncores
        self.csvlogger=csvname
        self.verbose=verbose

        if self.verbose:
            print('***********************************************************************')
            print('****************Evolutionary Search is Running*************************')
            print('***********************************************************************')
            
            if self.ncores > 1:
                print('--- Running in parallel with {} cores'.format(self.ncores))
                
        es=ES(mode=self.mode, bounds=self.param_grid, fit=self.fit, 
              lambda_=self.npop, mu=self.npop, mutpb=0.25,
             cxmode='cx2point', cxpb=0.7, ncores=self.ncores, seed=1)
        x_best, y_best, es_hist=es.evolute(ngen=self.ngen, verbose=1)

        self.evolures = pd.DataFrame(es.best_indvs, columns=self.param_names)
        self.evolures.index += 1
        self.evolures['score'] = es_hist

        if self.csvlogger:
            self.evolures.index.name='id'
            self.evolures.to_csv(self.csvlogger)
                
        return self.evolures