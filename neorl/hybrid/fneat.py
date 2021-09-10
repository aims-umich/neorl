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

# -*- encoding: utf-8 -*-
#'''
#@File    :   FNEAT Class based on RNEAT 
#@Time    :   2021/07/27 10:17:37
#@Author  :   Xubo GU 
#@Email   :   guxubo@alumni.sjtu.edu.cn
#'''

import numpy as np  
import neat        
import pickle
import random
import os
from multiprocessing import Pool
from neorl.rl.make_env import CreateEnvironment

class FNEAT(object):
    """
    Feedforward NeuroEvolution of Augmenting Topologies (FNEAT)
    
    :param mode: (str) problem type, either ``min`` for minimization problem or ``max`` for maximization (RL is default to ``max``)       
    :param fit: (function) the fitness function
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param config: (dict) dictionary of RNEAT hyperparameters, see **Notes** below for available hyperparameters to change
    :param ncores: (int) number of parallel processors
    :param seed: (int) random seed for sampling
    """
    def __init__(self, mode, fit, bounds, config, ncores=1, seed=None):
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        
        self.ncores=ncores
        self.mode=mode
        self.bounds = bounds
        self.fit=fit
        self.nx=len(self.bounds)
        default_config = self.basic_config()  #construct the default config file
        self.config = self.modify_config(default_config, config) #modify the config based on user input
        #force the required NEAT variables (Do not change)
        self.config['NEAT']['fitness_criterion'] = "max"
        self.config['DefaultGenome']['num_inputs'] = self.nx
        self.config['DefaultGenome']['num_outputs'] = self.nx
        self.episode_length=self.config['NEAT']['pop_size']
        
        self.env=CreateEnvironment(method='rneat', fit=self.fit, ncores=1, 
                      bounds=self.bounds, mode=self.mode, episode_length=self.episode_length)
        
    def eval_genomes(self, genomes, config):

        for genome_id, genome in genomes:
            
            if self.x0: # input user's data 
                ob=self.x0.copy()
            else: # no user's data 
                ob = self.env.reset()
                
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            local_fit = float("-inf")
            counter = 0
            xpos = 0
            done = False
            
            while not done:
                
                nnOutput = net.activate(ob)
                ob, rew, done, info = self.env.step(nnOutput)
                xpos = info['x']  
                                
                if rew > local_fit:
                    local_fit = rew
                    counter = 0
                else:
                    counter += 1
                
                if rew > self.best_fit:
                    self.best_fit=rew
                    self.best_x=xpos.copy()

                #--mir
                if self.mode=='max':
                    self.best_fit_correct=self.best_fit
                    local_fit_correct=local_fit
                else:
                    self.best_fit_correct=-self.best_fit
                    local_fit_correct=-local_fit
                
                if done or counter == self.episode_length:
                    done = True
                    self.history['global_fitness'].append(self.best_fit_correct)
                    self.history['local_fitness'].append(local_fit_correct)        
                    #print('best fit:', self.best_fit_correct)
                    
                genome.fitness = rew

    def genome_worker(self, genome, config):
        # parallel worker that passes different eval_genomes to diffrent cores
        worker = NEATWorker(genome, config, self.episode_length, self.x0, env=self.env)
        fitness, local_fit, xpos=worker.work()
        return fitness, local_fit, xpos
             
    def evolute(self, ngen, x0=None, save_best_net=False, checkpoint_itv=None, startpoint=None, verbose=False):
        """
        This function evolutes the FNEAT algorithm for number of generations.
        
        :param ngen: (int) number of generations to evolute
        :param x0: (list) initial position of the NEAT (must have same size as the ``x`` variable)
        :param save_best_net: (bool) save the winner neural network to a pickle file
        :param checkpoint_itv: (int) generation frequency to save checkpoints for restarting purposes (e.g. 1: save every generation, 10: save every 10 generations)
        :param startpoint: (str) name/path to the checkpoint file to use to start the search (the checkpoint file can be saved by invoking the argument ``checkpoint_itv``) 
        :param verbose: (bool) print statistics to screen
        
        :return: (vector, float, dict) best :math:`x`, best fitness, logging dictionary
        """
        
        self.history={'global_fitness': [], 'local_fitness':[]}
        self.best_fit=float("-inf")
        self.verbose=verbose        
        
        self.x0=x0
        if self.x0 is not None:
            self.x0 = list(self.x0)
            assert len(self.x0) == self.nx, '--error: the length of x0 ({}) MUST equal the size of the bounds variable ({})'.format(len(self.x0), self.nx)
        
        # transfer dict-type config to neat type
        path = os.path.dirname(__file__)
        file_data = []
        for section, content in self.config.items():
            file_data.append('\n')
            file_data.append('[' + section + ']'+'\n')
            for key, val in content.items():
                file_data.append(key + '=' + str(val) + '\n')

        tmp_file = os.path.join(path, 'tmp_config')
        if not os.path.exists(tmp_file):
            file = open(tmp_file, 'w')
            file.close()
            print('--debug: Temporary config file created ... ')

        with open(tmp_file, 'w') as ft:
            for line in file_data:
                ft.write(line)
            print('--debug: Modified config has been written ... ') 

        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             tmp_file)
        
        os.remove(tmp_file)
        print('--debug: Temporary config file removed ...')

        p = neat.Population(config)

        # train from start or checkpoint
        if startpoint:
            print('\nTrain model from {}'.format(startpoint))
            p = neat.Checkpointer.restore_checkpoint(startpoint)
            
            if verbose:
                p.add_reporter(neat.StdOutReporter(True))
                stats = neat.StatisticsReporter() 
                p.add_reporter(stats)
            if checkpoint_itv:
                p.add_reporter(neat.Checkpointer(checkpoint_itv))

            #winner = p.run(self.eval_genomes, ngen) # total gen = startpoint+ngen

        else: 
            print('\n--debug: Train model from the start.')
            if verbose:
                p.add_reporter(neat.StdOutReporter(True))
                stats = neat.StatisticsReporter()
                p.add_reporter(stats)
            if checkpoint_itv: # checkpoint saving interval
                print('\n--debug: Save model every {} epochs'.format(checkpoint_itv))
                p.add_reporter(neat.Checkpointer(checkpoint_itv))
            #winner = p.run(self.eval_genomes, ngen)
        
        #parallel runner
        if self.ncores > 1:
            print('--debug: FNEAT is running in parallel with {} cores ...'.format(self.ncores))
            pe = ParallelEvaluator(self.ncores, self.genome_worker, self.mode)
            winner = p.run(pe.evaluate, ngen)
            self.best_fit_correct=pe.best_fit_correct
            self.best_x=pe.best_x
            self.history=pe.history
        else:
            winner = p.run(self.eval_genomes, ngen)
            
        
        if save_best_net:
            with open('winner-net', 'wb') as output:
                pickle.dump(winner, output, 1)
                print('--debug: Winner net saved ...')
            
        if verbose:
            print('------------------------ FNEAT Summary --------------------------')
            print('Best fitness (y) found:', self.best_fit_correct)
            print('Best individual (x) found:', self.best_x)
            print('--------------------------------------------------------------')
            
        return self.best_x, self.best_fit_correct, self.history

    def modify_config(self, config_basic, config_modify_dict):
        #"""
        #config_basic: NEAT's basic configs, type: dict
        #config_modify_dict: storing configs to be changed, type: dict
        #"""
        
        para_change_list = [p for p in config_modify_dict.keys()] # parameters changed 
        
        for section, paras in config_basic.items():
            print('--debug: Dealing with section [{}] ...'.format(str(section)))
            for key, value in paras.items():
                if key in para_change_list:
                    # idx = para_change_list.index(key)
                    config_basic[section][key] = config_modify_dict[key]
                    print('--debug: Change parameter "{}" to "{}" successfully!'\
                        .format(str(key),str(config_modify_dict[key])))
    
        print('--debug: ************NEAT config file is constructed!************')
        return config_basic

    def basic_config(self):
        # This function builds the default NEAT config file.
        a = {
            'NEAT':{
            'fitness_criterion':'max',  #mir: default is max
            'fitness_threshold': 1e5,
            'pop_size': 30,
            'reset_on_extinction': True,
            'no_fitness_termination': False
            },
    
            'DefaultGenome':{
            'activation_default':'identity',
            'activation_mutate_rate':0.05,
            'activation_options': 'sigmoid',
    
            'aggregation_default': 'random',
            'aggregation_mutate_rate':0.05,
            'aggregation_options': 'sum product min max mean median maxabs',
            
            'single_structural_mutation': False,
            'structural_mutation_surer': 'default',
            
            'bias_init_type': 'gaussian',
            'bias_init_mean': 0.05,
            'bias_init_stdev': 1.0,
            'bias_max_value': 30.0,
            'bias_min_value':  -30.0,
            'bias_mutate_power': 0.5,
            'bias_mutate_rate': 0.7,
            'bias_replace_rate': 0.1,
    
            'compatibility_disjoint_coefficient': 1.0,
            'compatibility_weight_coefficient': 0.5,
    
            'conn_add_prob': 0.5,
            'conn_delete_prob': 0.1,
    
            'enabled_default': True,
            'enabled_mutate_rate': 0.2,
            'enabled_rate_to_true_add': 0.0,
            'enabled_rate_to_false_add': 0.0,
    
            'feed_forward': True,
            'initial_connection': 'partial_nodirect 0.5',
    
            'node_add_prob': 0.5,
            'node_delete_prob': 0.5,
    
            'num_hidden': 1,
            'num_inputs': None,  #mir: must be defined by the developer
            'num_outputs': None,  #mir: must be defined by the developer
            
            'response_init_type': 'gaussian',
            'response_init_mean': 1.0,
            'response_init_stdev': 0.05,
            'response_max_value': 30.0,
            'response_min_value': -30.0,
            'response_mutate_power': 0.1,
            'response_mutate_rate': 0.75,
            'response_replace_rate': 0.1,
            
            'weight_init_type': 'gaussian', 
            'weight_init_mean': 0.1,
            'weight_init_stdev': 1.0,
            'weight_max_value': 30,
            'weight_min_value': -30,
            'weight_mutate_power': 0.5,   
            'weight_mutate_rate': 0.8,
            'weight_replace_rate': 0.1
            },
    
            'DefaultSpeciesSet':{
            'compatibility_threshold': 2.5
            },
    
            'DefaultStagnation':{
            'species_fitness_func': 'max',
            'max_stagnation': 50,
            'species_elitism': 0
            },
    
            'DefaultReproduction':{
            'elitism': 1,
            'survival_threshold': 0.3,
            'min_species_size': 2
            }
        }
    
        return a

class NEATWorker(object):
    #This class provides a path to allow passing 
    #different workers with different genomes and configs
    #Inputs:
        #genome: genome structure
        #config: config for the genome
        #episode_length: epsiode length for logging if defined in the original FNEAT class
        #x0: initial guess if defined in the original FNEAT class
        #env: enviroment class
    
    def __init__(self, genome, config, episode_length, x0, env):
        self.genome = genome
        self.config = config
        self.episode_length=episode_length
        self.x0=x0
        self.env=env
        
    def work(self):
        
        if self.x0: # input user's data 
            ob=self.x0.copy()
        else: # no user's data 
            ob = self.env.reset()
            
        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        local_fit = float("-inf")
        counter = 0
        xpos = 0
        done = False
        
        while not done:
            
            nnOutput = net.activate(ob)
            ob, rew, done, info = self.env.step(nnOutput)
            xpos = info['x']  
                            
            if rew > local_fit:
                local_fit = rew
                counter = 0
            else:
                counter += 1

        if done or counter == self.episode_length:
            done = True
                
        return rew, local_fit, xpos
    


class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, mode, timeout=None):
        #"""
        #Runs evaluation functions in parallel subprocesses
        #in order to evaluate multiple genomes at once.
        
        #eval_function should take one argument, a tuple of
        #(genome object, config object), and return
        #a single float (the genome's fitness).
        #"""
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(num_workers)
        self.history={'global_fitness': [], 'local_fitness':[]}
        self.best_fit=float("-inf")
        self.mode=mode
        
    def __del__(self):
        self.pool.close() # should this be terminate?
        self.pool.join()

    def evaluate(self, genomes, config):
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))

        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness, local_fit, xpos = job.get(timeout=self.timeout)
        
            if genome.fitness > self.best_fit:
                self.best_fit=genome.fitness
                self.best_x=xpos.copy()
    
            #--mir
            if self.mode=='max':
                self.best_fit_correct=self.best_fit
                local_fit_correct=local_fit
            else:
                self.best_fit_correct=-self.best_fit
                local_fit_correct=-local_fit

            self.history['global_fitness'].append(self.best_fit_correct)
            self.history['local_fitness'].append(local_fit_correct)