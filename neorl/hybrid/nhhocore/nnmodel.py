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
Created on Thu Jul 29 09:57:06 2021

@author: Katelin Du
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

class NNmodel(object):
    def __init__(self, nn_params, gen, model_num, logger_paths=None):

        self.test_split = nn_params['test_split'] if 'test_split' in nn_params else 0.2
        self.activation = nn_params['activation'] if 'activation' in nn_params else 'relu'
        self.num_nodes = nn_params['num_nodes'] if 'num_nodes' in nn_params else [100, 50, 25]
        self.batch_size = nn_params['batch_size'] if 'batch_size' in nn_params else 32
        self.learning_rate = nn_params['learning_rate'] if 'learning_rate' in nn_params else 6e-4
        self.epochs = nn_params['epochs'] if 'epochs' in nn_params else 100
        self.plot_flag = nn_params['plot'] if 'plot' in nn_params else True
        self.verbose = nn_params['verbose'] if 'verbose' in nn_params else True
        self.save_models = nn_params['save_models'] if 'save_models' in nn_params else True

        self.gen = gen
        self.model_num = model_num
        self.paths=logger_paths
            
    def fit(self, X, Y):
        # """
        # Main function - generates model using NN parameters and saves the best of model of this generation.
        #
        # Return:
        # tuple - best model, maximum relative error from test set
        # """
        self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = train_test_split(X, Y, test_size=self.test_split)
        # print(self.Xtrain.shape, self.Xtest.shape)
        xscaler = StandardScaler()     #x-scaler object
        yscaler = MinMaxScaler()     #y-scaler object
        self.Xtrain = xscaler.fit_transform(self.Xtrain)
        self.Xtest = xscaler.transform(self.Xtest)

        self.Ytrain = yscaler.fit_transform(self.Ytrain.reshape(-1,1)).flatten()
        self.Ytest = yscaler.transform(self.Ytest.reshape(-1,1)).flatten()

        model = self.model_structure()
        
        if self.save_models:
            cb = [ModelCheckpoint(filepath=os.path.join(self.paths['models'], 'model{0:0}_{1:04}.h5'.format(self.model_num, self.gen)), verbose=0, monitor='val_mean_absolute_error', save_best_only=True, mode='min')]
        else:
            cb = []

        if self.gen == 0:  #force saving the first the three models
            cb = [ModelCheckpoint(filepath='model{0:0}_{1:04}.h5'.format(self.model_num, self.gen), verbose=0, monitor='val_mean_absolute_error', save_best_only=True, mode='min')]

        model.compile(loss='mean_absolute_error', optimizer=Adam(self.learning_rate), metrics = ['mean_absolute_error'])

        #mir-new: consider these hyperparam: epochs, batch_size, validation_split
        self.history = model.fit(self.Xtrain, self.Ytrain, epochs=self.epochs, batch_size=self.batch_size, validation_data=(self.Xtest, self.Ytest), callbacks=cb, verbose=0)
        self.Ynn = model.predict(self.Xtest)
        self.Ynn = yscaler.inverse_transform(self.Ynn).flatten()
        self.Ytest = yscaler.inverse_transform(self.Ytest.reshape(-1,1)).flatten()
        # calculate relative error
        rel_errors = self.rel_errors()
        max_error = rel_errors.max()
        mean_error = rel_errors.mean()
        val_err=self.history.history['val_mean_absolute_error']

        if self.verbose:
            self.arg_max=np.argmax(rel_errors)
            print('-----------------------------------------------------')
            print(f'Generation {self.gen}, Model {self.model_num}')
            print('MAX relative test error:', max_error)
            print('MEAN relative test error:', mean_error)
            print('Val error:', val_err[-1])
            print('Worst target =', self.Ytest[self.arg_max])   #identify worst point
            print('Worst prediction =', self.Ynn[self.arg_max]) #identify worst point
            print()

        # graph prediction vs. target
        if self.plot_flag:
            self.plot()
            #print('Plots generated')
        return model

    def model_structure(self):
        # """
        # Defines the general structure of the neural network according to nn parameters.
        #
        # Return:
        # model structure
        # """
        model = Sequential()

        #mir-new: consider these hyperparam: num_nodes in list form (you infer num_dense_layers)
        num_dense_layers = len(self.num_nodes)
        #mir-new: consider activation in HIDDEN Layers ONLY (relu, sigmoid, etc.)
        model.add(Dense(self.num_nodes[0], kernel_initializer='normal', activation=self.activation, input_dim=self.Xtrain.shape[1]))
        model.add(Dropout(0.5))
        for i in range(1, num_dense_layers):
            model.add(Dense(self.num_nodes[i], activation=self.activation, kernel_initializer='normal'))
        model.add(Dense(1, activation='linear', kernel_initializer='normal'))
        return model

    def rel_errors(self):
        # """
        # Finds relative errors between target test set and predicted set.
        #
        # Return:
        # array - relative errors
        # """
        # abs error / true value
        rel_errors = []
        for i in range(len(self.Ynn)):
            rel_errors.append(100 * abs(self.Ytest[i] - self.Ynn[i]) / self.Ytest[i])
        return np.array(rel_errors)

    def plot(self):
        # """
        # Saves relevant plots for the current generation:
        # 1. Target vs. Predicted
        # 2. Loss (MAE) vs. Epoch
        # """
        plt.figure()
        plt.plot(self.Ytest, self.Ynn, 'o')
        max_val = max([max(self.Ytest), max(self.Ynn)])
        plt.plot(range(0,int(max_val)), range(0,int(max_val)),'-k')
        plt.xlabel('Target')
        plt.ylabel('Predicted')
        plt.savefig(os.path.join(self.paths['predict'], 'prediction{0:0}_{1:04}.png'.format(self.model_num, self.gen))) #mir: make the name as prediction-00001.png, .... based on generation #
        plt.close()

        # obtain the training/validation MAE from fitting history
        train_err=self.history.history['mean_absolute_error']
        val_err=self.history.history['val_mean_absolute_error']
        plt.figure()
        plt.plot(train_err, label='Training')
        plt.plot(val_err, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.paths['error'], 'error{0:0}_{1:04}.png'.format(self.model_num, self.gen)))  #mir: same as the previous figure
        plt.close()
