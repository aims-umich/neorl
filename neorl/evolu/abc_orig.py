# -*- coding: utf-8 -*-
#---------------------------------------------------
#Thanks to Samuel C P Oliveira for sharing the work
# This ABC class is inspired from this github repo 
#https://github.com/renard162/BeeColPy/
#---------------------------------------------------

'''
+----------------------------------------------------------------------+
 
   Samuel C P Oliveira
   samuelcpoliveira@gmail.com
   Artificial Bee Colony Optimization
   This project is licensed under the MIT License.
 
+----------------------------------------------------------------------+
 
 Bibliography
 ------------
 
    [1] Karaboga, D. and Basturk, B., 2007
        A powerful and efficient algorithm for numerical function 
        optimization: artificial bee colony (ABC) algorithm. Journal 
        of global optimization, 39(3), pp.459-471. 
        DOI: https://doi.org/10.1007/s10898-007-9149-x
 
    [2] Liu, T., Zhang, L. and Zhang, J., 2013
        Study of binary artificial bee colony algorithm based on 
        particle swarm optimization. Journal of Computational 
        Information Systems, 9(16), pp.6459-6466. 
        Link: https://api.semanticscholar.org/CorpusID:8789571
 
    [3] Anuar, S., Selamat, A. and Sallehuddin, R., 2016
        A modified scout bee for artificial bee colony algorithm and 
        its performance on optimization problems. Journal of King Saud 
        University-Computer and Information Sciences, 28(4), 
        pp.395-406. 
        DOI: https://doi.org/10.1016/j.jksuci.2016.03.001
 
    [4] Kennedy, J. and Eberhart, R.C., 1997
        A discrete binary version of the particle swarm algorithm. 
        In 1997 IEEE International conference on systems, man, and 
        cybernetics. Computational cybernetics and simulation 
        (Vol. 5, pp. 4104-4108). IEEE. 
        DOI: https://doi.org/10.1109/ICSMC.1997.637339
 
    [5] Pampará, G. and Engelbrecht, A.P., 2011
        Binary artificial bee colony optimization. In 2011 IEEE 
        Symposium on Swarm Intelligence (pp. 1-8). IEEE. 
        DOI: https://doi.org/10.1109/SIS.2011.5952562
 
    [6] Mirjalili, S., Hashim, S., Taherzadeh, G., Mirjalili, S.Z. 
    and Salehi, S., 2011
        A study of different transfer functions for binary version 
        of particle swarm optimization. In International Conference 
        on Genetic and Evolutionary Methods (Vol. 1, No. 1, pp. 2-7). 
        Link: http://hdl.handle.net/10072/48831
 
    [7] Huang, S.C., 2015
        Polygonal approximation using an artificial bee colony 
        algorithm. Mathematical Problems in Engineering, 2015. 
        DOI: https://doi.org/10.1155/2015/375926
 
+----------------------------------------------------------------------+
'''
import numpy as np
import random as rng
import warnings as wrn
from collections import Counter


class ABCO:
    '''
    Class that applies Artificial Bee Colony (ABC) algorithm to find 
    minimum or maximum of a function that's receive a vector of floats 
    as input and returns a float as output.

    https://en.wikipedia.org/wiki/Artificial_bee_colony_algorithm

    Parameters
    ----------
    function : Name
        A name of a function to minimize/maximize.
        Example: if the function is:
            def my_func(x): return x[0]**2 + x[1]**2 + 5*x[1]
            
            Use "my_func" as parameter.


    boundaries : List of Tuples
        A list of tuples containing the lower and upper boundaries of 
        each dimension of function domain.

        Obs.: The number of boundaries determines the dimension of 
        function.

        Example: A function F(x1, x2) = y with:
            (-5 <= x1 <= 5) and (-20 <= x2 <= 20) have the boundaries:
                [(-5,5), (-20,20)]


    [colony_size] : Int --optional-- (default: 40)
        A value that determines the number of bees in algorithm. Half 
        of this amount determines the number of points analyzed (food 
        sources).

        According articles, half of this number determines the amount 
        of Employed bees and other half is Onlooker bees.


    [scouts] : Float --optional-- (default: 0.5)
        Determines the limit of tries for scout bee discard a food 
        source and replace for a new one.
            - If scouts = 0 : 
                Scout_limit = colony_size * dimension

            - If scouts = (0 to 1) : 
                Scout_limit = colony_size * dimension * scouts
                    Obs.: scouts = 0.5 is used in [3] as benchmark.

            - If scouts >= 1 : 
                Scout_limit = scouts

        Obs.: Scout_limit is rounded down in all cases.


    [iterations] : Int --optional-- (default: 50)
        The number of iterations executed by algorithm.


    [min_max] : String --optional-- (default: 'min')
        Determines if algorithm will minimize or maximize the function.
            - If min_max = 'min' : (default)
                Locate the minimum of function.

            - If min_max = 'max' : 
                Locate the maximum of function.


    [nan_protection] : Boolean --optional-- (default: True)
        If true, re-generate food sources that get NaN value as cost 
        during initialization or during scout events. This option 
        usually helps the algorithm stability because, in rare cases, 
        NaN values can lock the algorithm in a infinite loop.
        
        Obs.: NaN protection can drastically increases calculation 
        time if analysed function has too many values of domain 
        returning NaN.


    [log_agents] : Boolean --optional-- (default: False)
        If true, beecolpy will register, before each iteration, the
        position of each food source. Useful to debug but, if there a
        high amount of food sources and/or iterations, this option
        drastically increases memory usage.


    [seed] : Int --optional-- (default: None)
        If defined as an int, set the seed used in all random process.


    Methods
    ----------
    fit()
        Execute the algorithm with defined parameters.

        Obs.: Returns a list with values found as minimum/maximum 
        coordinate.


    get_solution()
        Returns the value obtained after fit() the method.


    get_status()
        Returns a tuple with:
            - Number of complete iterations executed
            - Number of scout events during iterations
            - Number of times that NaN protection was activated


    get_agents()
        Returns a list with the position of each food source during
        each iteration if "log_agents = True".

        Parameters
        ----------
        [reset_agents] : bool --optional-- (default: False)
            If true, the food source position log will be cleaned in
            next fit().


    Bibliography
    ----------
    [1] Karaboga, D. and Basturk, B., 2007
        A powerful and efficient algorithm for numerical function 
        optimization: artificial bee colony (ABC) algorithm. Journal 
        of global optimization, 39(3), pp.459-471. 
        DOI: https://doi.org/10.1007/s10898-007-9149-x

    [2] Liu, T., Zhang, L. and Zhang, J., 2013
        Study of binary artificial bee colony algorithm based on 
        particle swarm optimization. Journal of Computational 
        Information Systems, 9(16), pp.6459-6466. 
        Link: https://api.semanticscholar.org/CorpusID:8789571

    [3] Anuar, S., Selamat, A. and Sallehuddin, R., 2016
        A modified scout bee for artificial bee colony algorithm and 
        its performance on optimization problems. Journal of King Saud 
        University-Computer and Information Sciences, 28(4), 
        pp.395-406. 
        DOI: https://doi.org/10.1016/j.jksuci.2016.03.001

    '''
    def __init__(self, mode, bounds, fit, nbees=40, scouts=0.5, nan_protection=True,
                 log_agents=False, ncores=1, seed=None):

        self.boundaries = bounds
        self.min_max_selector = mode
        self.cost_function = fit
        self.nan_protection = nan_protection
        self.log_agents = log_agents
        self.reset_agents = False
        self.seed = seed
        colony_size=nbees

        #--mir
        self.mode=mode
        if mode == 'min':
            self.fit=fit
        elif mode == 'max':
            def fitness_wrapper(*args, **kwargs):
                return -fit(*args, **kwargs) 
            self.fit=fitness_wrapper
        else:
            raise ValueError('--error: The mode entered by user is invalid, use either `min` or `max`')


        self.employed_onlookers_count = int(max([(colony_size/2), 2]))
        if (colony_size < 4):
            warn_message = 'Using the minimun value of colony_size = 4'
            wrn.warn(warn_message, RuntimeWarning)

        if (scouts <= 0):
            self.scout_limit = int(self.employed_onlookers_count * len(self.boundaries))
            if (scouts < 0):
                warn_message = 'Negative scout count given, using default scout ' \
                    'count: colony_size * dimension = ' + str(self.scout_limit)
                wrn.warn(warn_message, RuntimeWarning)
        elif (scouts < 1):
            self.scout_limit = int(self.employed_onlookers_count * len(self.boundaries) * scouts)
        else:
            self.scout_limit = int(scouts)

        self.scout_status = 0
        self.iteration_status = 0
        self.nan_status = 0

        if (self.seed is not None):
            rng.seed(self.seed)

        self.foods = [None] * self.employed_onlookers_count
        for i in range(len(self.foods)):
            _ABC_engine(self).generate_food_source(i)

        try:
            self.best_food_source = self.foods[np.nanargmax([food.fit for food in self.foods])]
        except:
            self.best_food_source = self.foods[0]
            warn_message = 'All food sources\'s fit resulted in NaN and beecolpy can got stuck ' \
                         'in an infinite loop during fit(). Enable nan_protection to prevent this.'
            wrn.warn(warn_message, RuntimeWarning)

        self.agents = []
        if self.log_agents:
            self.agents.append([food.position for food in self.foods])


    def evolute(self, ngen):
        '''
        Execute the algorithm with defined parameters.

        Obs.: Returns a list with values found as minimum/maximum 
        coordinate.
        '''
        
        self.max_iterations = int(max([ngen, 1]))
        if (ngen < 1):
            warn_message = 'Using the minimun value of iterations = 1'
            wrn.warn(warn_message, RuntimeWarning)
            
        if (self.seed is not None):
            rng.seed(self.seed)

        if self.reset_agents:
            self.agents = []
            if self.log_agents:
                self.agents.append([food.position for food in self.foods])
            self.reset_agents = False

        for _ in range(self.max_iterations):
            #--> Employer bee phase <--
            #Generate and evaluate a neighbor point to every food source
            _ABC_engine(self).employer_bee_phase()

            #--> Onlooker bee phase <--
            #Based in probability, generate a neighbor point and evaluate again some food sources
            #Same food source can be evaluated multiple times
            _ABC_engine(self).onlooker_bee_phase()

            #--> Memorize best solution <--
            _ABC_engine(self).memorize_best_solution()

            #--> Scout bee phase <--
            #Generate up to one new food source that does not improve over scout_limit evaluation tries
            _ABC_engine(self).scout_bee_phase()

            #Update iteration status
            self.iteration_status += 1
            if self.log_agents:
                self.agents.append([food.position for food in self.foods])
            
            self.best_food_fit=self.calculate_fit(self.best_food_source.position)

        return self.best_food_source.position, self.best_food_fit


    def init_food(self):
    
        #When a food source is initialized, randomize a position inside boundaries and calculate the "fit"
        self.trial_counter = 0
        self.position = [rng.uniform(*self.boundaries[i]) for i in range(len(self.boundaries))]
        self.fit = self.calculate_fit(self.position)
        
    def evaluate_neighbor(self, partner_position):
     
        #Randomize one coodinate (one dimension) to generate a neighbor point
        j = rng.randrange(0, len(self.boundaries))

        #eq. (2.2) [1] (new coordinate "x_j" to generate a neighbor point)
        xj_new = self.position[j] + rng.uniform(-1, 1)*(self.position[j] - partner_position[j])

        #Check boundaries
        xj_new = self.boundaries[j][0] if (xj_new < self.boundaries[j][0]) else \
            self.boundaries[j][1] if (xj_new > self.boundaries[j][1]) else xj_new

        #Changes the coordinate "j" from food source to new "x_j" generating the neighbor point
        neighbor_position = [(self.position[i] if (i != j) else xj_new) for i in range(len(self.boundaries))]
        neighbor_fit = self.calculate_fit(neighbor_position)

        #Greedy selection
        if (neighbor_fit > self.fit):
            self.position = neighbor_position
            self.fit = neighbor_fit
            self.trial_counter = 0
        else:
            self.trial_counter += 1

    def check_nan_lock(self):
        if not(self.nan_protection):
            if np.all([np.isnan(food.fit) for food in self.foods]):
                raise Exception('All food sources\'s fit resulted in NaN and beecolpy got ' \
                                'stuck in an infinite loop. Enable nan_protection to prevent this.')


    def execute_nan_protection(self, food_index):
        while (np.isnan(self.foods[food_index].fit) and self.nan_protection):
            self.nan_status += 1
            self.foods[food_index] = self.init_food()


    def generate_food_source(self, index):
        self.foods[index] = self.init_food()
        self.execute_nan_protection(index)


    def prob_i(self, actual_fit, max_fit):
        # Improved probability function [7]
        return 0.9*(actual_fit/max_fit) + 0.1
        # Original probability function [1]
        # return actual_fit/np.sum([food.fit for food in self.foods])


    def calculate_fit(self, evaluated_position):
        #eq. (2) [2] (Convert "cost function" to "fit function")
        cost = self.cost_function(evaluated_position)
        if (self.min_max_selector == 'min'): #Minimize function
            fit_value = (1 + np.abs(cost)) if (cost < 0) else (1/(1 + cost))
        else: #Maximize function
            fit_value = (1 + cost) if (cost > 0) else (1/(1 + np.abs(cost)))
        return cost


    def food_source_dance(self, index):
        #Generate a partner food source to generate a neighbor point to evaluate
        while True: #Criterion from [1] geting another food source at random
            d = int(rng.randrange(0, self.employed_onlookers_count))
            if (d != index):
                break
        self.foods[index].evaluate_neighbor(self.foods[d].position)


    def employer_bee_phase(self):
        #Generate and evaluate a neighbor point to every food source
        for i in range(len(self.foods)):
            self.food_source_dance(i)


    def onlooker_bee_phase(self):
        #Based in probability, generate a neighbor point and evaluate again some food sources
        #Same food source can be evaluated multiple times
        self.check_nan_lock()
        max_fit = np.nanmax([food.fit for food in self.foods])
        onlooker_probability = [self.prob_i(food.fit, max_fit) for food in self.foods]
        p = 0 #Onlooker bee index
        i = 0 #Food source index
        while (p < self.employed_onlookers_count):
            if (rng.uniform(0, 1) <= onlooker_probability[i]):
                p += 1
                self.food_source_dance(i)
                self.check_nan_lock()
                max_fit = np.nanmax([food.fit for food in self.foods])
                if (self.foods[i].fit != max_fit):
                    onlooker_probability[i] = self.prob_i(self.foods[i].fit, max_fit)
                else:
                    onlooker_probability = [self.prob_i(food.fit, max_fit) for food in self.foods]
            i = (i+1) if (i < (len(self.foods)-1)) else 0


    def scout_bee_phase(self):
        #Generate up to one new food source that does not improve over scout_limit evaluation tries
        trial_counters = [food.trial_counter for food in self.foods]
        if (max(trial_counters) > self.scout_limit):
            #Take the index of replaced food source
            trial_counters = np.where(np.array(trial_counters) == max(trial_counters))[0].tolist()
            i = trial_counters[rng.randrange(0, len(trial_counters))]
            self.generate_food_source(i) #Replace food source
            self.scout_status += 1


    def memorize_best_solution(self):
        best_food_index = np.nanargmax([food.fit for food in self.foods])
        if (self.foods[best_food_index].fit >= self.best_food_source.fit):
            self.best_food_source = self.foods[best_food_index]

class _FoodSource:
    def __init__(self, abc, engine):
        #When a food source is initialized, randomize a position inside boundaries and calculate the "fit"
        self.abc = abc
        self.engine = engine
        self.trial_counter = 0
        self.position = [rng.uniform(*self.abc.boundaries[i]) for i in range(len(self.abc.boundaries))]
        self.fit = self.engine.calculate_fit(self.position)


    def evaluate_neighbor(self, partner_position):
        #Randomize one coodinate (one dimension) to generate a neighbor point
        j = rng.randrange(0, len(self.abc.boundaries))

        #eq. (2.2) [1] (new coordinate "x_j" to generate a neighbor point)
        xj_new = self.position[j] + rng.uniform(-1, 1)*(self.position[j] - partner_position[j])

        #Check boundaries
        xj_new = self.abc.boundaries[j][0] if (xj_new < self.abc.boundaries[j][0]) else \
            self.abc.boundaries[j][1] if (xj_new > self.abc.boundaries[j][1]) else xj_new

        #Changes the coordinate "j" from food source to new "x_j" generating the neighbor point
        neighbor_position = [(self.position[i] if (i != j) else xj_new) for i in range(len(self.abc.boundaries))]
        neighbor_fit = self.engine.calculate_fit(neighbor_position)

        #Greedy selection
        if (neighbor_fit > self.fit):
            self.position = neighbor_position
            self.fit = neighbor_fit
            self.trial_counter = 0
        else:
            self.trial_counter += 1




class _ABC_engine:
    def __init__(self, abc):
        self.abc = abc


    def check_nan_lock(self):
        if not(self.abc.nan_protection):
            if np.all([np.isnan(food.fit) for food in self.abc.foods]):
                raise Exception('All food sources\'s fit resulted in NaN and beecolpy got ' \
                                'stuck in an infinite loop. Enable nan_protection to prevent this.')


    def execute_nan_protection(self, food_index):
        while (np.isnan(self.abc.foods[food_index].fit) and self.abc.nan_protection):
            self.abc.nan_status += 1
            self.abc.foods[food_index] = _FoodSource(self.abc, self)


    def generate_food_source(self, index):
        self.abc.foods[index] = _FoodSource(self.abc, self)
        self.execute_nan_protection(index)


    def prob_i(self, actual_fit, max_fit):
        # Improved probability function [7]
        return 0.9*(actual_fit/max_fit) + 0.1
        # Original probability function [1]
        # return actual_fit/np.sum([food.fit for food in self.abc.foods])


    def calculate_fit(self, evaluated_position):
        #eq. (2) [2] (Convert "cost function" to "fit function")
        cost = self.abc.cost_function(evaluated_position)
        if (self.abc.min_max_selector == 'min'): #Minimize function
            fit_value = (1 + np.abs(cost)) if (cost < 0) else (1/(1 + cost))
        else: #Maximize function
            fit_value = (1 + cost) if (cost > 0) else (1/(1 + np.abs(cost)))
        return fit_value


    def food_source_dance(self, index):
        #Generate a partner food source to generate a neighbor point to evaluate
        while True: #Criterion from [1] geting another food source at random
            d = int(rng.randrange(0, self.abc.employed_onlookers_count))
            if (d != index):
                break
        self.abc.foods[index].evaluate_neighbor(self.abc.foods[d].position)


    def employer_bee_phase(self):
        #Generate and evaluate a neighbor point to every food source
        for i in range(len(self.abc.foods)):
            self.food_source_dance(i)


    def onlooker_bee_phase(self):
        #Based in probability, generate a neighbor point and evaluate again some food sources
        #Same food source can be evaluated multiple times
        self.check_nan_lock()
        max_fit = np.nanmax([food.fit for food in self.abc.foods])
        onlooker_probability = [self.prob_i(food.fit, max_fit) for food in self.abc.foods]
        p = 0 #Onlooker bee index
        i = 0 #Food source index
        while (p < self.abc.employed_onlookers_count):
            if (rng.uniform(0, 1) <= onlooker_probability[i]):
                p += 1
                self.food_source_dance(i)
                self.check_nan_lock()
                max_fit = np.nanmax([food.fit for food in self.abc.foods])
                if (self.abc.foods[i].fit != max_fit):
                    onlooker_probability[i] = self.prob_i(self.abc.foods[i].fit, max_fit)
                else:
                    onlooker_probability = [self.prob_i(food.fit, max_fit) for food in self.abc.foods]
            i = (i+1) if (i < (len(self.abc.foods)-1)) else 0


    def scout_bee_phase(self):
        #Generate up to one new food source that does not improve over scout_limit evaluation tries
        trial_counters = [food.trial_counter for food in self.abc.foods]
        if (max(trial_counters) > self.abc.scout_limit):
            #Take the index of replaced food source
            trial_counters = np.where(np.array(trial_counters) == max(trial_counters))[0].tolist()
            i = trial_counters[rng.randrange(0, len(trial_counters))]
            self.generate_food_source(i) #Replace food source
            self.abc.scout_status += 1


    def memorize_best_solution(self):
        best_food_index = np.nanargmax([food.fit for food in self.abc.foods])
        if (self.abc.foods[best_food_index].fit >= self.abc.best_food_source.fit):
            self.abc.best_food_source = self.abc.foods[best_food_index]
