#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:51:22 2020

@author: majdi
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines.common.callbacks import BaseCallback

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

    def _on_step(self) -> bool:
        
        
        if self.num_timesteps % self.check_freq == 0:
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
       

        return True

    def calc_cumavg(self, data, N):
    
        cum_aves=[np.mean(data[i:i+N]) for i in range(0,len(data),N)]
        cum_std=[np.std(data[i:i+N]) for i in range(0,len(data),N)]
        cum_max=[np.max(data[i:i+N]) for i in range(0,len(data),N)]
    
        return cum_aves, cum_std, cum_max
    
    
    def plot_progress(self):
        
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
                ravg, rstd, rmax=self.calc_cumavg(plot_data.iloc[:,i],self.avg_step)
                epochs=np.array(range(1,len(ravg)+1),dtype=int)
                plt.plot(epochs, ravg,'-o', c=color_list[color_index], label='Average per epoch')
                
                plt.fill_between(epochs,[a_i - b_i for a_i, b_i in zip(ravg, rstd)], [a_i + b_i for a_i, b_i in zip(ravg, rstd)],
                alpha=0.2, edgecolor=color_list[color_index], facecolor=color_list[color_index], label=r'$1-\sigma$ per epoch')
                
                plt.plot(epochs, rmax,'-s', c='k', label='Max per epoch')
                #plt.axhline(y=6000, color='k', linestyle='--', label='Desired reward (top 0.1%)')
                plt.legend()
                plt.xlabel('Epoch')
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
                ravg, rstd, rmax=self.calc_cumavg(plot_data.iloc[:,i],self.avg_step)
                epochs=np.array(range(1,len(ravg)+1),dtype=int)
                plt.plot(epochs,ravg,'-o', c=color_list[color_index], label='Average per epoch')
                
                plt.fill_between(epochs,[a_i - b_i for a_i, b_i in zip(ravg, rstd)], [a_i + b_i for a_i, b_i in zip(ravg, rstd)],
                alpha=0.2, edgecolor=color_list[color_index], facecolor=color_list[color_index], label=r'$1-\sigma$ per epoch')
                
                plt.plot(epochs,rmax,'-s', c='k', label='Max per epoch')
                #plt.axhline(y=6000, color='k', linestyle='--', label='Desired reward (top 0.1%)')
                plt.legend()
                plt.xlabel('Epoch')
                plt.ylabel(labels[i])
                if color_index==9:
                    color_index=0
                else:
                    color_index+=1
                
            plt.tight_layout()
            plt.savefig(self.log_dir+'_res.png', format='png', dpi=200)
            plt.close()
            
        else:
            raise Exception ('the plot mode defined by the user does not exist in neorl')