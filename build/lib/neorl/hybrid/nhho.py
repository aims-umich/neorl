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
#@author: Katelin Du
#"""

import random
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import joblib
from neorl.evolu.discrete import mutate_discrete, encode_grid_to_discrete, decode_discrete_to_grid
from neorl.hybrid.nhhocore.nnmodel import NNmodel
from neorl.hybrid.nhhocore.hho import HHO
from tensorflow.keras.models import load_model
import shutil

import multiprocessing
import multiprocessing.pool

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

class NHHO(object):
    """
    Neural Harris Hawks Optimizer

    :param mode: (str) problem type, either "min" for minimization problem or "max" for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param fit: (function) the fitness function
    :param nhawks: (int): number of the hawks in the group
    :param num_warmups: (int) number of warmup samples to train the surrogate which will be evaluated by the real fitness ``fit`` (if ``None``, ``num_warmups=20*len(bounds)``)
    :param int_transform: (str): method of handling int/discrete variables, choose from: ``nearest_int``, ``sigmoid``, ``minmax``.
    :param nn_params: (dict) parameters for building the surrogate models in dictionary form. 
                            Keys are: ``test_split``, ``learning_rate``, 
                            ``activation``, ``num_nodes``, ``batch_size``, ``epochs``, 
                            ``save_models``, ``verbose``, ``plot``. See **Notes** below for descriptions.
    :param ncores: (int) number of parallel processors to train the three surrogate models (only ``ncores=1`` or ``ncores=3`` are allowed)
    :param seed: (int) random seed for sampling
    """
    def __init__(self, mode, bounds, fit, nhawks, num_warmups=None, int_transform='nearest_int', 
                 nn_params = {}, ncores=1, seed=None):
        
        self.seed = seed
        if seed:
            random.seed(seed)
            np.random.seed(seed)

        #-------------------------------------------------------
        #construct a main HHO model based on the core directory
        #--------------------------------------------------------
        self.hho=HHO(mode=mode, bounds=bounds,
                fit=fit, nhawks=nhawks, int_transform=int_transform,
                ncores=ncores, seed=seed)

        assert mode == 'min' or mode == 'max', "Mode must be 'max' or 'min'."
        self.mode = mode
        self.fit = fit
        self.int_transform=int_transform
        self.ncores = ncores
        self.nhawks = nhawks
        self.dim = len(bounds)
        self.bounds = bounds

        #infer variable types
        self.var_type = np.array([bounds[item][0] for item in bounds])

        #mir-grid
        if "grid" in self.var_type:
            self.grid_flag=True
            self.orig_bounds=bounds  #keep original bounds for decoding
            print('--debug: grid parameter type is found in the space')
            self.bounds, self.bounds_map=encode_grid_to_discrete(self.bounds) #encoding grid to int
            #define var_types again by converting grid to int
            self.var_type = np.array([self.bounds[item][0] for item in self.bounds])
        else:
            self.grid_flag=False
            self.bounds = bounds

        self.lb = np.array([self.bounds[item][1] for item in self.bounds])
        self.ub = np.array([self.bounds[item][2] for item in self.bounds])

        self.nn_params = nn_params
        self.num_warmups = num_warmups
        
        #----------------------------------------
        #create logger files as needed
        if nn_params['plot'] or nn_params['save_models']:
            for i in range(100):
                logname='NHHOlog_{0:03d}'.format(i)
                if os.path.exists(logname):
                    continue
                else: 
                    self.logger_dir=logname
                    os.makedirs(logname)
                    break
        
            try:
                self.logger_dir
            except:
                raise('--error: we cannot create a new logger, it seems maximum number of logger folders exist in the current directory, consider removing all NHHOlog_* folders in this directory')
            
            
            self.model_path=os.path.join(self.logger_dir, 'models')
            self.errplot_path=os.path.join(self.logger_dir, 'error_plots')
            self.predplot_path=os.path.join(self.logger_dir, 'prediction_plots')
            
            if nn_params['save_models']:
                os.makedirs(self.model_path)
            if nn_params['plot']:
                os.makedirs(self.errplot_path)
                os.makedirs(self.predplot_path)
                
            self.paths={'models': self.model_path, 'error': self.errplot_path, 'predict': self.predplot_path}
            print('--debug: NHHO logger is created, data will be saved at {}'.format(self.logger_dir))
        else:
            self.paths=None
            
            #----------------------------------------

    def evolute(self, ngen, x0=None, verbose=True):
        """
        This function evolutes the NHHO algorithm for number of generations.

        :param ngen: (int) number of generations to evolute
        :param x0: (list of lists) initial position of the hawks (must be of same size as ``nhawks``)
        :param verbose: (bool) print statistics to screen

        :return: (tuple) (best position, best fitness, and dictionary containing major search results)
        """
        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)
        self.verbose = verbose

        self.history = {'local_fitness':[], 'best_hawk':[]}
        self.rabbit_energy = float("inf")

        self.rabbit_location = np.zeros(self.dim)   #the NHHO variable
        self.hho.rabbit_location=self.rabbit_location  #the internal HHO rabbit_location

        ##############################################
        # Generate warmup samples for first NN model #
        ##############################################
        # nhho: initialize warmup hawks, evaluate fitnesses,
        self.NNhawks = self.hho.init_sample(num_hawks=self.num_warmups)  #note how I accessed the function from HHO
        # evaluate and add to NNfitnesses
            # note: eval_hawks is made to evaluate hawks in self.hawk_positions right now
        self.NNfitnesses = np.array(self.eval_hawks(hawks_to_eval=self.NNhawks))

        # split NNhawks and NNfitnesses into three sets
        self.warmup_hawks = np.array_split(self.NNhawks, 3)
        self.warmup_fitnesses = np.array_split(self.NNfitnesses, 3)


        # train hawks in three models
        #construct a worker for parallel training
        if self.ncores > 1:
            def startup_worker(index):
                NNmodel(self.nn_params, gen=0, model_num=index+1, logger_paths=self.paths).fit(self.warmup_hawks[index], self.warmup_fitnesses[index]) # saved as best_models/model1_0000.h5
            
            with joblib.Parallel(n_jobs=self.ncores) as parallel:
                parallel(joblib.delayed(startup_worker)(i) for i in range(3))    
        else:
            NNmodel(self.nn_params, gen=0, model_num=1, logger_paths=self.paths).fit(self.warmup_hawks[0], self.warmup_fitnesses[0]) # saved as best_models/model1_0000.h5
            NNmodel(self.nn_params, gen=0, model_num=2, logger_paths=self.paths).fit(self.warmup_hawks[1], self.warmup_fitnesses[1])
            NNmodel(self.nn_params, gen=0, model_num=3, logger_paths=self.paths).fit(self.warmup_hawks[2], self.warmup_fitnesses[2])

        ##################################
        # Set initial locations of hawks #
        ##################################
        self.hawk_positions = np.zeros((self.nhawks, self.dim))

        if x0:
            assert len(x0) == self.nhawks, 'Length of x0 array MUST equal the number of hawks (self.nhawks).'
            self.hawk_positions = x0
        else:
            self.hawk_positions = self.hho.init_sample()

        self.hho.hawk_positions = self.hawk_positions #the internal HHO hawk_positions kd: moved to after hawk positions set

        for t in range(ngen):
            # "a" decreases linearly from 1 to 0 for discrete mutation
            self.a= 1 - t * ((1) / ngen)

            #########################################################
            # Update NN model and Evaluate hawk fitnesses 
            #########################################################
            fitness_lst=self.update_model(gen=t+1)

            #######################################################################
            # Update rabbit energy and rabbit location based on best hawk fitness #
            #######################################################################
            for i, fitness in enumerate(fitness_lst):
                fitness = fitness if self.mode == 'min' else -fitness
                if fitness < self.rabbit_energy:
                    self.rabbit_energy = fitness
                    self.rabbit_location = self.hawk_positions[i, :].copy()
                    self.hho.rabbit_location = self.rabbit_location   #update the HHO internal rabbit_location

            #####################################################
            # Update best global and local fitnesses in history #
            #####################################################
            best_fitarg = np.argmin(fitness_lst)
            self.best_local_fitness = fitness_lst[best_fitarg] if self.mode == 'min' else -fitness_lst[best_fitarg]
            self.best_hawk = self.hawk_positions[best_fitarg]

            self.history['local_fitness'].append(self.best_local_fitness)
            self.history['best_hawk'].append(self.best_hawk)

            if self.verbose: # change depending on how often message should be displayed
                print(f'NHHO step {(t+1)*self.nhawks}/{ngen*self.nhawks}, nhawks={self.nhawks}, ncores={self.ncores}')
                print('Best generation fitness:', np.round(self.best_local_fitness, 6))
                #mir-grid
                if self.grid_flag:
                    self.hawk_decoded=decode_discrete_to_grid(self.best_hawk,self.orig_bounds,self.bounds_map)
                    print('Best hawk position:', self.hawk_decoded)
                else:
                    print('Best hawk position:', self.best_hawk)
                print()

            ################################
            # Update the location of hawks #
            ################################
            self.hho.update_hawks(t, ngen, fitness_lst) # now self.hawk_positions is updated
            self.hawk_positions = self.hho.hawk_positions # kd: needed to update self.hawk_positions

            for hawk_i in range(self.nhawks):
                #mir: this bound check  line is needed to ensure that choices.remove option to work
                self.hawk_positions[hawk_i, :] = self.hho.ensure_bounds(self.hawk_positions[hawk_i, :], self.bounds)
                for dim in range(self.dim):
                    if self.var_type[dim] == 'int':
                        self.hawk_positions[hawk_i, dim] = mutate_discrete(x_ij=self.hawk_positions[hawk_i, dim],
                                           x_min=self.hawk_positions[hawk_i, :].min(),
                                           x_max=self.hawk_positions[hawk_i, :].max(),
                                           lb=self.lb[dim],
                                           ub=self.ub[dim],
                                           alpha=self.a,
                                           method=self.int_transform)

        if self.verbose:
            print('------------------------ HHO Summary --------------------------')
            print('Function:', self.fit.__name__)
            print('Final fitness (y) found:', self.best_local_fitness)
            if self.grid_flag:
                print('Final individual (x) found:', self.hawk_decoded)
            else:
                print('Final individual (x) found:', self.best_hawk)
            print('-------------------------------------------------------------- \n \n')
        
        if self.nn_params['save_models']:
            [shutil.move('model{}_0000.h5'.format(i+1), self.paths['models']) for i in range(3)]
        else:
            [os.remove('model{}_0000.h5'.format(i+1)) for i in range(3)]

        return self.history['best_hawk'], self.history['local_fitness']

    def eval_hawks(self, hawks_to_eval=None):
        #"""
        #Evaluate fitness of hawks with parallel processing using fitness function.

        #Return:
        #list - hawk fitnesses
        #"""
        if hawks_to_eval is None:
            hawks_to_eval = self.hawk_positions

        if self.ncores > 1:
            with joblib.Parallel(n_jobs=self.ncores) as parallel:
                fitness_lst = parallel(joblib.delayed(self.fit_worker)(hawks_to_eval[i, :]) for i in range(len(hawks_to_eval)))
        else:
            fitness_lst = []
            for i in range(len(hawks_to_eval)):
                fitness_lst.append(self.fit_worker(hawks_to_eval[i, :]))
        return fitness_lst

    def eval_hawksNN(self, gen):
        # """
        # Evaluate fitness of hawks using generated surrogate model.
        #
        # Return:
        # array - hawk fitnesses
        # """
        # pull previous generation models, make predictions, and average them to get final fitness.
        pred1 = self.models[0].predict(self.hawk_positions).flatten()
        pred2 = self.models[1].predict(self.hawk_positions).flatten()
        pred3 = self.models[2].predict(self.hawk_positions).flatten()

        fitness_lst = []
        for i in range(self.nhawks):
            models_avg = (pred1[i] + pred2[i] + pred3[i])/3
            fitness_lst.append(models_avg)
        return fitness_lst

    def fit_worker(self, hawk_pos):
        #"""
        #Evaluates fitness of a hawk.

        #Params:
        #hawk_pos - hawk position vector

        #Return:
        #float - hawk fitness
        #"""

        #mir---
        hawk_pos=self.hho.ensure_bounds(hawk_pos, self.bounds)

        #mir-grid
        if self.grid_flag:
            #decode the individual back to the int/float/grid mixed space
            hawk_pos=decode_discrete_to_grid(hawk_pos,self.orig_bounds,self.bounds_map)

        fitness = self.fit(hawk_pos)
        return fitness

    def neural_worker(self, inp):
        
        index=inp[0]
        p=inp[1]
        gen=inp[2]
        errors = abs(self.preds[p[0]] - self.preds[p[1]])
        i = np.argmin(errors)
        X = np.row_stack((self.warmup_hawks[index], self.hawk_positions[i]))
        Y = np.append(self.warmup_fitnesses[index], (self.preds[p[0]][i] + self.preds[p[1]][i])/2)
        model = NNmodel(self.nn_params, gen=gen, model_num=index+1, logger_paths=self.paths).fit(X, Y)
        preds=model.predict(self.hawk_positions).flatten()
        
        return preds
            
    def update_model(self, gen):
        self.models = [] # list of three models
        for i in range(3):
            self.models.append(load_model('model{}_0000.h5'.format(i+1)))

        self.preds = [] # list of prediction arrays
        for model in self.models:
            self.preds.append(model.predict(self.hawk_positions))
        self.preds = np.array(self.preds) # array of prediction arrays
        
        del self.models
                
        core_lst=[[0,(1,2),gen], [1,(2,0),gen], [2,(0,1),gen]]
        
        if self.ncores > 1:
            
            with joblib.Parallel(n_jobs=self.ncores) as parallel:
                model_fit=parallel(joblib.delayed(self.neural_worker)(item) for item in core_lst)
            
            pred1, pred2, pred3 = model_fit
        
        else:
            pred1=self.neural_worker(core_lst[0])
            pred2=self.neural_worker(core_lst[1])
            pred3=self.neural_worker(core_lst[2])
            
        fitness=(pred1+pred2+pred3)/3
        
        return fitness
            
            
#        else:       
#            # retrain models with best predicted hawk (smallest difference between two models) included
#            errors = abs(self.preds[1] - self.preds[2])
#            i = np.argmin(errors)
#            X = np.row_stack((self.warmup_hawks[0], self.hawk_positions[i]))
#            Y = np.append(self.warmup_fitnesses[0], (self.preds[1][i] + self.preds[2][i])/2)
#            self.models[0] = NNmodel(self.nn_params, gen=gen, model_num=1, logger_paths=self.paths).fit(X, Y)
#    
#            errors = abs(self.preds[2] - self.preds[0])
#            i = np.argmin(errors)
#            X = np.row_stack((self.warmup_hawks[1], self.hawk_positions[i]))
#            Y = np.append(self.warmup_fitnesses[1], (self.preds[2][i] + self.preds[0][i])/2)
#            self.models[1] = NNmodel(self.nn_params, gen=gen, model_num=2, logger_paths=self.paths).fit(X, Y)
#    
#            errors = abs(self.preds[0] - self.preds[1])
#            i = np.argmin(errors)
#            X = np.row_stack((self.warmup_hawks[2], self.hawk_positions[i]))
#            Y = np.append(self.warmup_fitnesses[2], (self.preds[0][i] + self.preds[1][i])/2)
#            self.models[2] = NNmodel(self.nn_params, gen=gen, model_num=3,logger_paths=self.paths).fit(X, Y)