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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#"""
#Created on Wed Mar  4 11:51:22 2020
#
#@author: majdi
#"""

import numpy as np
import pandas as pd
from neorl.rl.baselines.shared.callbacks import BaseCallback
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os, sys

class SavePlotCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq, avg_step, log_dir, total_timesteps, basecall, plot_mode='subplot'):
        #super(SavePlotCallback, self).__init__(verbose)
        self.base=basecall
        self.plot_mode=plot_mode
        self.n_calls=self.base.n_calls
        self.model=self.base.model
        self.num_timesteps=self.base.num_timesteps
        self.total_timesteps=total_timesteps
        self.verbose=1
        self.check_freq = check_freq
        self.avg_step=avg_step
        self.log_dir = log_dir
        self.save_path = self.log_dir + '_bestmodel.pkl'
        self.best_mean_reward = -np.inf

        #avoid activating 'Agg' in the header so not to affect other classes/algs
        import matplotlib
        matplotlib.use('Agg')

    def runcall(self):
        
        print('num_timesteps={}/{}'.format (self.num_timesteps, self.total_timesteps))
        with open (self.log_dir+'_summary.txt','a') as fin:
            fin.write('*****************************************************\n')
            fin.write('Summary for update step {} \n'.format(int(self.num_timesteps/self.check_freq)))
            fin.write('*****************************************************\n')
            
        # Retrieve training reward
        y= pd.read_csv(self.log_dir+'_out.csv')
        y=y["reward"].values
        # Mean training reward over the last 100 episodes
        mean_reward = np.mean(y[-self.avg_step:])
        if self.verbose > 0:
            with open (self.log_dir+'_summary.txt','a') as fin:
                fin.write("Num  of time steps passed: {}/{} \n".format(self.num_timesteps, self.total_timesteps))
                fin.write("Best mean reward so far: {:.3f} \n".format(self.best_mean_reward))
                fin.write("Mean reward in this update step: {:.3f} \n".format(mean_reward))
               
        # New best model, you could save the agent here
        if mean_reward > self.best_mean_reward:
              self.best_mean_reward = mean_reward
              # Example for saving best model
              if self.verbose > 0:
                with open (self.log_dir+'_summary.txt','a') as fin:
                    fin.write("Saving new best model to {} \n".format(self.save_path))
              self.model.save(self.save_path)
              
        self.out_data=pd.read_csv(self.log_dir+'_out.csv')
        self.inp_data=pd.read_csv(self.log_dir+'_inp.csv')     
        #-------------------
        # Progress Plot
        #-------------------
        self.plot_progress()
                        
        # Print summary for this method

        top=10
        assert isinstance(top,int), 'the user provided a non-integer value for the summary parameter {}'.format(top)
        
        sorted_out=self.out_data.sort_values(by=['reward'],ascending=False)   
        sorted_inp=self.inp_data.sort_values(by=['reward'],ascending=False)   
            
        with open (self.log_dir+'_summary.txt','a') as fin:
            fin.write ('--------------------------------------------------------------------------------------\n')
            fin.write ('Top {} outputs for update step {} \n'.format(top, int(self.num_timesteps/self.check_freq)))
            fin.write(sorted_out.iloc[:top].to_string())
            fin.write('\n')
            fin.write ('-------------------------------------------------------------------------------------- \n')
            fin.write ('Top {} corresponding inputs for update step {} \n'.format(top, int(self.num_timesteps/self.check_freq)))
            fin.write(sorted_inp.iloc[:top].to_string())
            fin.write('\n')
            fin.write ('-------------------------------------------------------------------------------------- \n')
            fin.write('\n\n')
                
    def _on_step(self) -> bool:
        
        try:
            if (self.num_timesteps % self.check_freq == 0) or (self.num_timesteps == self.total_timesteps):
                self.runcall()
        except:
            print('--warning: try to plot empty csv loggers, increase `check_freq` to a value larger than time needed to firstly update csv loggers')
        
        if self.num_timesteps == self.total_timesteps:
            print('system exit')
            os._exit(1)
            
            
        return True
    
    def _on_training_end(self) -> None:
        self.runcall()
        print('Training is finished')
        os._exit(1)
        #pass

    def calc_cumavg(self, data, N):
    
        cum_aves=[np.mean(data[i:i+N]) for i in range(0,len(data),N)]
        cum_std=[np.std(data[i:i+N]) for i in range(0,len(data),N)]
        cum_max=[np.max(data[i:i+N]) for i in range(0,len(data),N)]
        cum_min=[np.min(data[i:i+N]) for i in range(0,len(data),N)]
    
        return cum_aves, cum_std, cum_max, cum_min
    
    
    def plot_progress(self, method_xlabel='Epoch'):

        self.out_data=pd.read_csv(self.log_dir+'_out.csv')
        color_list=['b', 'g', 'r', 'c', 'm', 'y', 'darkorange', 'purple', 'tab:brown', 'lime']
        plot_data=self.out_data.drop(['caseid'], axis=1)  #exclude caseid, which is the first column from plotting (meaningless)
        
        labels=list(plot_data.columns.values)
            
        ny=plot_data.shape[1] 
        
        assert ny == len(labels), 'number of columns ({}) to plot in the csv file {} is not equal to the number of labels provided by the user ({})'.format(ny, self.log_dir+'_out.csv', len(labels))
        
        # classic mode
        if self.plot_mode=='classic' or ny == 1:
            color_index=0
            for i in range (ny): #exclude caseid from plot, which is the first column 
                plt.figure()
                ravg, rstd, rmax, rmin=self.calc_cumavg(plot_data.iloc[:,i],self.avg_step)
                epochs=np.array(range(1,len(ravg)+1),dtype=int)
                plt.plot(epochs, ravg,'-o', c=color_list[color_index], label='Average per {}'.format(method_xlabel))
                
                plt.fill_between(epochs,[a_i - b_i for a_i, b_i in zip(ravg, rstd)], [a_i + b_i for a_i, b_i in zip(ravg, rstd)],
                alpha=0.2, edgecolor=color_list[color_index], facecolor=color_list[color_index], label=r'$1-\sigma$ per {}'.format(method_xlabel))
                
                plt.plot(epochs, rmax,'s', c='k', label='Max per {}'.format(method_xlabel), markersize=4)
                plt.plot(epochs,rmin,'d', c='k', label='Min per {}'.format(method_xlabel), markersize=4)
                #plt.axhline(y=6000, color='k', linestyle='--', label='Desired reward (top 0.1%)')
                plt.legend()
                plt.xlabel(method_xlabel)
                plt.ylabel(labels[i])
                
                if color_index==9:
                    color_index=0
                else:
                    color_index+=1
                    
                plt.tight_layout()
                plt.savefig(self.log_dir+'_'+labels[i]+'.png', format='png', dpi=150)
                plt.close()
        
        # subplot mode           
        elif self.plot_mode=='subplot':
            # determine subplot size
            if ny == 2:
                xx= [(1,2,1),(1,2,2)]
                plt.figure(figsize=(12, 4.0))
            elif ny==3:
                xx= [(1,3,1), (1,3,2), (1,3,3)]
                plt.figure(figsize=(12, 4.0))
            elif ny==4:
                xx= [(2,2,1), (2,2,2), (2,2,3), (2,2,4)]
                plt.figure(figsize=(12, 8))
            elif ny > 4 and ny <= 21:
                nrows=int(np.ceil(ny/3))
                xx= [(nrows,3,item) for item in range(1,ny+1)]
                adj_fac=(nrows - 2.0)*0.25 + 1
                plt.figure(figsize=(12, adj_fac*8))
            elif ny > 21 and ny <= 99:
                nrows=int(np.ceil(ny/4))
                xx= [(nrows,4,item) for item in range(1,ny+1)]
                adj_fac=(nrows - 2.0)*0.25 + 1
                plt.figure(figsize=(15, adj_fac*8))
                
                
            color_index=0
            for i in range (ny): #exclude caseid from plot, which is the first column 
                plt.subplot(xx[i][0], xx[i][1], xx[i][2])
                ravg, rstd, rmax, rmin=self.calc_cumavg(plot_data.iloc[:,i],self.avg_step)
                epochs=np.array(range(1,len(ravg)+1),dtype=int)
                plt.plot(epochs,ravg,'-o', c=color_list[color_index])
                
                plt.fill_between(epochs,[a_i - b_i for a_i, b_i in zip(ravg, rstd)], [a_i + b_i for a_i, b_i in zip(ravg, rstd)],
                alpha=0.2, edgecolor=color_list[color_index], facecolor=color_list[color_index])
                
                plt.plot(epochs,rmax,'s', c='k', markersize=4)
                
                plt.plot(epochs,rmin,'d', c='k', markersize=4)
                plt.xlabel(method_xlabel)
                plt.ylabel(labels[i])
                if color_index==9:
                    color_index=0
                else:
                    color_index+=1
            
            #speical legend is created for all subplots to save space
            legend_elements = [Line2D([0], [0], color='k', marker='o', label='Mean ' + r'$\pm$ ' +r'$1\sigma$' + ' per {} (color changes)'.format(method_xlabel)),
                  Line2D([0], [0], color='k', marker='s', label='Max per {} (color changes)'.format(method_xlabel)),
                  Line2D([0], [0], linestyle='-.', color='k', marker='d', label='Min per {} (color changes)'.format(method_xlabel))]
            plt.figlegend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3)
            plt.tight_layout()
            plt.savefig(self.log_dir+'_res.png', format='png', dpi=200, bbox_inches="tight")
            plt.close()
            
        else:
            raise Exception ('the plot mode defined by the user does not exist')
    
class RLLogger(BaseCallback):
    """
    Callback for logging data of RL algorathims (x,y), compatible with: A2C, ACER, ACKTR, DQN, PPO

    :param check_freq: (int) logging frequency, e.g. 1 will record every time step 
    :param plot_freq: (int) frequency of plotting the fitness progress (if ``None``, plotter is deactivated)
    :param n_avg_steps: (int) if ``plot_freq`` is NOT ``None``, then this is the number of timesteps to group to draw statistics for the plotter (e.g. 10 will group every 10 time steps to estimate min, max, mean, and std).
    :param pngname: (str) name of the plot that will be saved if ``plot_freq`` is NOT ``None``.
    :param save_model: (bool) whether or not to save the RL neural network model (model is saved every ``check_freq``)
    :param model_name: (str) name of the model to be saved  if ``save_model=True``
    :param save_best_only: (bool) if ``save_model = True``, then this flag only saves the model if the fitness value improves. 
    :param verbose: (int) print updates to the screen
    """
    def __init__(self, check_freq=1, plot_freq=None, n_avg_steps=10, pngname='history', 
                 save_model=False, model_name='bestmodel.pkl', save_best_only=True, 
                 verbose=False):
        super(RLLogger, self).__init__(verbose)
        self.check_freq = check_freq
        self.plot_freq=plot_freq
        self.pngname=pngname
        self.n_avg_steps=n_avg_steps
        self.model_name = model_name
        self.save_model=save_model
        self.verbose=verbose
        self.save_best_only=save_best_only
        self.rbest = -np.inf
        self.rbest_maxonly = -np.inf
        self.r_hist=[]
        self.x_hist=[]
        
        if self.plot_freq:
            #avoid activating 'Agg' in the header so not to affect other classes/algs
            import matplotlib
            matplotlib.use('Agg')
            
    def _init_callback(self) -> None:
        # Create folder if needed
        try:
            self.mode=self.training_env.get_attr('mode')[0]   #PPO/ACER/A2C/ACKTR
        except:
            try:
                self.mode=self.training_env.mode       #DQN
            except:
                print('--warning: the logger cannot find mode in the environment, it is set by default to `max`')
                self.mode='max'
        
        if self.mode not in ['min', 'max']:
            self.mode='max'
            print('--warning: The mode entered by user is invalid, use either `min` or `max`')

        #if self.save_model:
        #    if self.log_dir is not None:
        #        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        
        
        if self.n_calls % self.check_freq == 0:
            
            if self.verbose:
                print('----------------------------------------------------------------------------------')
                print('RL callback at step {}/{}'.format(self.n_calls, self.locals['total_timesteps']))
            
            try:
                rwd=self.locals['rew']   #DQN case (special dict naming)
            except:
                rwd=self.locals['rewards'][0] #A2C/PPO/ACER/ACKTR
                
            try:
                x=self.locals['infos'][0]['x'] #A2C/PPO/ACKTR cases
            except:
                if 'mus' in list(self.locals.keys()):
                    x=self.locals['_'][0]['x']     #ACER case (special dict naming)
                else:
                    x=self.locals['info']['x']   #DQN case (special dict naming)
                    
            if self.save_model and not self.save_best_only:
                self.model.save(self.model_name)
                if self.verbose:
                    print('A new model is saved to {}'.format(self.model_name))
                
            if rwd > self.rbest_maxonly:
                self.xbest=x.copy()
                self.rbest_maxonly=rwd
                
                if self.mode=='max':
                    self.rbest=self.rbest_maxonly
                else:
                    self.rbest=-self.rbest_maxonly
                
            
                if self.save_model and self.save_best_only:
                    self.model.save(self.model_name)
                    if self.verbose:
                        print('An improvement is observed, new model is saved to {}'.format(self.model_name))
            
            if self.mode=='max':
                self.r_hist.append(rwd)
            else:
                self.r_hist.append(-rwd)
            
            self.x_hist.append(list(x))
            
            if self.plot_freq:
                if self.n_calls % self.plot_freq == 0:
                    self.plot_progress()
                
            
            if self.verbose:
                print('----------------------------------------------------------------------------------')
        return True
    
    def plot_progress(self): 
    
        plt.figure()
        
        ravg, rstd, rmax, rmin=self.calc_cumavg(self.r_hist,self.n_avg_steps)
        epochs=np.array(range(1,len(ravg)+1),dtype=int)
        plt.plot(epochs, ravg,'-o', c='g', label='Average per epoch')
        
        plt.fill_between(epochs,[a_i - b_i for a_i, b_i in zip(ravg, rstd)], [a_i + b_i for a_i, b_i in zip(ravg, rstd)],
        alpha=0.2, edgecolor='g', facecolor='g', label=r'$1-\sigma$ per epoch')
        
        plt.plot(epochs, rmax,'s', c='k', label='Max per epoch', markersize=4)
        plt.plot(epochs,rmin,'d', c='k', label='Min per epoch', markersize=4)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Fitness')
        plt.savefig(self.pngname+'.png',format='png' ,dpi=300, bbox_inches="tight")
        plt.close()

    def calc_cumavg(self, data, N):
    
        cum_aves=[np.mean(data[i:i+N]) for i in range(0,len(data),N)]
        cum_std=[np.std(data[i:i+N]) for i in range(0,len(data),N)]
        cum_max=[np.max(data[i:i+N]) for i in range(0,len(data),N)]
        cum_min=[np.min(data[i:i+N]) for i in range(0,len(data),N)]
    
        return cum_aves, cum_std, cum_max, cum_min