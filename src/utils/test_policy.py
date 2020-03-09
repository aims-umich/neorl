#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 09:33:45 2020

@author: majdi
"""
import numpy as np
import pandas as pd
import os, subprocess

def evaluate_policy(model, env, log_dir, n_eval_episodes=10, deterministic=False, render=False, video_record=False, fps=10):
    """
    Runs policy for `n_eval_episodes` episodes and returns average reward.
    This is made to work only with one env.
    :param model: (BaseRLModel) The RL agent you want to evaluate.
    :param env: (gym.Env or VecEnv) The gym environment. In the case of a `VecEnv`
        this must contain only one environment.
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :param render: (bool) Whether to render the environment or not
    :param callback: (callable) callback function to do additional checks,
        called after each step.
    :param reward_threshold: (float) Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: (bool) If True, a list of reward per episode
        will be returned instead of the mean.
    :return: (float, float) Mean reward per episode, std of reward per episode
        returns ([float], [int]) when `return_episode_rewards` is True
    """
    #if isinstance(env, VecEnv):
    #    assert env.num_envs == 1, "You must pass only one environment when using this function"
    ffmpeg_code=os.system('which ffmpeg')
    if ffmpeg_code == 0: 
        print('--debug: ffmpeg is detected on the machine')
        ffmpeg=subprocess.check_output(['which', 'ffmpeg'])
        ffmpeg.decode('utf-8').strip()
    else:
        raise('The user activated video recording but ffmpeg is not installed on the machine')

    episode_rewards, episode_lengths = [], []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, _info = env.step(action)
            episode_reward += reward
            episode_length += 1
            if render or video_record:
                env.render()
            
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    out_data=pd.read_csv(log_dir+'_out.csv')
    inp_data=pd.read_csv(log_dir+'_inp.csv')
    sorted_out=out_data.sort_values(by=['reward'],ascending=False)   
    sorted_inp=inp_data.sort_values(by=['reward'],ascending=False)   

    with open (log_dir + '_summary.txt', 'a') as fin:
        fin.write('*****************************************************\n')
        fin.write('Model testing is completed for {} episodes \n'.format(n_eval_episodes))
        fin.write('*****************************************************\n')
        fin.write('Mean Reward: {0:.3f} \n'.format(np.mean(episode_rewards)))
        fin.write('Std Reward: {0:.3f} \n'.format(np.std(episode_rewards)))
        fin.write('Max Reward: {0:.3f} \n'.format(np.max(episode_rewards)))
        fin.write('Min Reward: {0:.3f} \n'.format(np.min(episode_rewards)))


        fin.write ('--------------------------------------------------------------------------------------\n')
        fin.write ('Outputs of all episodes ordered from highest reward to lowest \n')
        fin.write ('Original data is saved in {} \n'.format(log_dir+'_out.csv'))
        fin.write(sorted_out.to_string())
        fin.write('\n')
        fin.write ('-------------------------------------------------------------------------------------- \n')
        fin.write ('Corresponding inputs of all episodes ordered from highest reward to lowest \n')
        fin.write ('Original data is saved in {} \n'.format(log_dir+'_inp.csv'))
        fin.write(sorted_inp.to_string())
        fin.write('\n')
        fin.write ('-------------------------------------------------------------------------------------- \n')
        fin.write('\n\n')
        
    if video_record:
        print('--debug: Video recording in progress')
        subprocess.run(['ffmpeg', '-r', '1', '-pattern_type', 'glob', '-i', "*.png", '-vf', 'fps={}'.format(fps), 'format=yuv420p', '{}_video.mp4'.format(log_dir.split('./master_log/')[1])], cwd="./master_log/")