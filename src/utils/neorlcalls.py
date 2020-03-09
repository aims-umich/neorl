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
    def __init__(self, check_freq, avg_step, log_dir, total_timesteps, basecall):
        #super(SavePlotCallback, self).__init__(verbose)
        self.base=basecall
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

    #def _init_callback(self) -> None:
    #    # Create folder if needed
    #    if self.save_path is not None:
    #        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        
        print('num_timesteps={}/{}'.format (self.num_timesteps, self.total_timesteps))
        if self.num_timesteps % self.check_freq == 0:
            
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
                  
            #-------------------
            # Progress Plot
            #-------------------
            top=10
            color_list=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'darkorange', 'purple', 'tab:brown']
            out_data=pd.read_csv(self.log_dir+'_out.csv')
            inp_data=pd.read_csv(self.log_dir+'_inp.csv')
            labels=list(out_data.columns.values)
                
            ny=out_data.shape[1]
            
            assert ny == len(labels), 'number of columns ({}) to plot in the csv file {} is not equal to the number of labels provided by the user ({})'.format(ny, self.log_dir+'_out.csv', len(labels))
            
            color_index=0
            for i in range (ny):
                plt.figure()
                list1=out_data.iloc[:,i]
                n=self.avg_step
                cum_aves = [sum(list1[i:i+n])/n for i in range(0,len(list1),n)]
                plt.plot(cum_aves,'-o', c=color_list[color_index])
                plt.xlabel('epoch')
                plt.ylabel(labels[i])
                plt.tight_layout()
                plt.savefig(self.log_dir+'_'+labels[i]+'.png', format='png', dpi=150)   
                plt.close()
                
                if color_index==9:
                    color_index=0
                else:
                    color_index+=1
                            
                # Print summary for this method
                assert isinstance(top,int), 'the user provided a non-integer value for the summary parameter {}'.format(top)
                
                sorted_out=out_data.sort_values(by=['reward'],ascending=False)   
                sorted_inp=inp_data.sort_values(by=['reward'],ascending=False)   
                
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