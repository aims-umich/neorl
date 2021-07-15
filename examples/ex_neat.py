# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 20:57:48 2021

@author: majdi
"""

import numpy as np  # pip install numpy
import neat         # pip install neat-python
import pickle       # pip install cloudpickle
import random
random.seed(1)
np.random.seed(1)
from neorl import CreateEnvironment

def Sphere(individual):
        """Sphere test objective function.
                F(x) = sum_{i=1}^d xi^2
                d=1,2,3,...
                Range: [-100,100]
                Minima: 0
        """
        return -sum(x**2 for x in individual)

class NEAT(object):
    def __init__(self, env, config):
        
        self.episode_length=env.episode_length
        self.mode=env.mode
        
    def eval_genomes(self, genomes, config):
    
        for genome_id, genome in genomes:
            ob = env.reset()
            net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
            local_fit = float("-inf")
            counter = 0
            xpos = 0
            done = False
            
            while not done:
                
                #env.render()                
                nnOutput = net.activate(ob)
                #print(nnOutput)
                ob, rew, done, info = env.step(nnOutput)
                xpos = info['x']  
                                
                if rew > local_fit:
                    local_fit = rew
                    counter = 0
                else:
                    counter += 1
                
                if rew > self.best_fit:
                    self.best_fit=rew
                    self.best_x=xpos.copy()
                 
                # for self-check
                #assert env.fit(self.best_x) == self.best_fit

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
                
    def evolute(self, ngen, x0=None, verbose=False):
        """
        This function evolutes the SSA algorithm for number of generations.
        
        :param ngen: (int) number of generations to evolute
        :param x0: (list of lists) initial position of the salps (must be of same size as ``nsalps``)
        :param verbose: (bool) print statistics to screen
        
        :return: (dict) dictionary containing major SSA search results
        """
        self.history={'global_fitness': [], 'local_fitness':[]}
        self.best_fit=float("-inf")
        self.verbose=verbose        
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             'config-feedforward')
        
        p = neat.Population(config)
        
        if verbose:
            p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)
            p.add_reporter(neat.Checkpointer(10))
        
        winner = p.run(self.eval_genomes, ngen)
        
        with open('winner.pkl', 'wb') as output:
            pickle.dump(winner, output, 1)
            
        return self.best_x, self.best_fit_correct, self.history

nx=10
bounds={}
for i in range(1,nx+1):
        bounds['x'+str(i)]=['float', -100, 100]

#create an enviroment class
env=CreateEnvironment(method='neat', 
                      fit=Sphere, 
                      bounds=bounds, 
                      mode='max', 
                      episode_length=50)
neats=NEAT(env=env, config='config-feedforward')
x_best, y_best, neat_hist=neats.evolute(ngen=60)
assert Sphere(x_best) == y_best
print(x_best, y_best)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(neat_hist['global_fitness'])
#plt.plot(neat_hist['local_fitness'])