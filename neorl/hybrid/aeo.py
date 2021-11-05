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
import xarray as xr
import itertools
import random
import math
import inspect

from neorl import WOA
from neorl import GWO
from neorl import PSO
from neorl import MFO
from neorl import HHO
from neorl import DE
from neorl import ES

# Note: to incorporate additional algorithms into ensembles, the following needs to be done:
#     1. detect_algo needs to be updated to return appropriate name
#     2. clone_algo_obj needs to be updated to change the correct attribute in the dict
#     3. eval_algo_popnumber needs to be given some criteria for the minimum number of individuals
#        to run the evolution phase

def detect_algo(obj):
    #simple function to detect object type given neorl
    # algorithm object
    if isinstance(obj, WOA):
        return 'WOA'
    elif isinstance(obj, GWO):
        return 'GWO'
    elif isinstance(obj, PSO):
        return 'PSO'
    elif isinstance(obj, MFO):
        return 'MFO'
    elif isinstance(obj, HHO):
        return 'HHO'
    elif isinstance(obj, DE):
        return 'DE'
    elif isinstance(obj, ES):
        return 'ES'
    raise Exception('%s algorithm object not recognized or supported'%obj)

max_algos = ['PSO', 'DE', 'ES']#algos that change fitness function to make a maximum problem
min_algos = ['WOA', 'GWO', 'MFO', 'HHO']#algos that change fitness function to make a minimum problem

def wtd_remove(lst, ei, wts = None):
    #quick helper function to handle removing ei items from lst and returning them with
    #wts probability vector
    if wts is None:
        wts = [1/len(lst) for i in range(len(lst))]
    indxs = np.random.choice(range(len(lst)), size=ei, p = wts, replace = False)
    return [lst.pop(i) for i in reversed(sorted(indxs))]

def clone_algo_obj(obj, nmembers, fit):
    # function to return a copy of an algorithm object with anumber of members given as
    # nmembers. This is to circumvent the error when the x0 passed in the evolute function
    # is a different size than the individuals given originally in the initialization of 
    # the algorithm object.
    # works for now, has potential of breaking

    def filter_kw(dftr, twk):
        sig = inspect.signature(twk)
        fks = [p.name for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD]
        fd = {fk:dftr[fk] for fk in fks if fk in dftr.keys()}
        return fd

    algo = detect_algo(obj)

    attrs = obj.__dict__

    if algo == 'WOA':
        attrs['nwhales'] = nmembers
        attrs['fit'] = fit
        return WOA(**filter_kw(attrs, WOA))
    elif algo == 'GWO':
        attrs['nwolves'] = nmembers
        attrs['fit'] = fit
        return GWO(**filter_kw(attrs, GWO))
    elif algo == 'PSO':
        attrs['npar'] = nmembers
        attrs['fit'] = fit
        return PSO(**filter_kw(attrs, PSO))
    elif algo == 'HHO':
        attrs['nhawks'] = nmembers
        attrs['fit'] = fit
        return HHO(**filter_kw(attrs, HHO))
    elif algo == 'DE':
        attrs['npop'] = nmembers
        attrs['fit'] = fit
        return DE(**filter_kw(attrs, DE))
    elif algo == 'ES':
        attrs['lambda_'] = nmembers
        attrs['fit'] = fit
        return ES(**filter_kw(attrs, ES))

def eval_algo_popnumber(obj, nmembers):
    # check if an algorithm is prepared to participate in evolution phase
    # based on its population information

    algo = detect_algo(obj)
    if algo in ['WOA', 'GWO', 'PSO', 'HHO', 'DE']:
        return nmembers >= 5
    elif algo == 'ES':
        return nmembers >= obj.mu

class Population:
    # Class to store information and functionality related to a single population
    # in the AEO algorithm. Should be characterized by an evolution strategy passed
    # as one of the optimization classes from the other algorithms in NEORL.
    def __init__(self, strategy, algo, init_pop, mode, conv = None):
        # strategy should be one of the optimization objects containing an "evolute" method
        # init_pop needs to match the population given to the strategy object initially
        # algo is string that identifies which class is being used
        # conv is a function which takes ngen and returns number of evaluations
        self.conv = conv
        self.strategy = strategy
        self.algo = algo
        self.members = init_pop
        self.mode = mode

        self.fitlog = []
        self.log = []

    @property
    def fitness(self):
        return self.fitlog[-1]

    def evolute(self, ngen, fit): #fit is included to avoid needing to pull .fit methods
                                  #    from algo objects that may have been flipped
        #check if there are enouh members to evolve
        if not eval_algo_popnumber(self.strategy, len(self.members)):
            self.n = len(self.members)
            return self.fitness

        #update strategy with new population number
        self.strategy = clone_algo_obj(self.strategy, len(self.members), fit)

       #store last generation number
        self.last_ngen = ngen

        #perform evolution and store relevant information
        out = self.strategy.evolute(ngen, x0 = self.members)
        self.members = out[2]['last_pop'].iloc[:, :-1].values.tolist()
        self.member_fitnesses = out[2]['last_pop'].iloc[:, -1].values.tolist()
        print("======")
        print(self.algo)
        print(self.member_fitnesses)
        print([-(a[0]**2 + a[1]**2) for a in self.members])
        print("======")

        if self.mode == 'max':
            self.fitlog.append(max(self.member_fitnesses))
        elif self.mode == 'min':
            self.fitlog.append(min(self.member_fitnesses))

        self.n = len(self.members)

        self.log.append(list(zip(self.members, self.member_fitnesses)))
        return self.fitness

    def strength(self, g, g_burden, fmax, fmin):
        if self.mode == "max":
            fbest, fworst = fmax, fmin
        elif self.mode == "min":
            fworst, fbest = fmax, fmin

        #calculate strength for two different types
        if g == 'improve' and len(self.fitlog) > 1:
            unorm = max([self.fitlog[-1] - self.fitlog[-2], 0]) #in case best indiv got exported
        else: #when g == 'fitness' and also no matter what if on first cycle
            unorm = self.fitness

        #normalize strength measure
        normed = (unorm - fworst)/(fbest - fworst)

        if g_burden:
            normed /= 1 + self.conv(self.last_ngen, self.n)
        return normed + 1e-6*(fmax - fmin) #litle adjustment to avoid divide by zero

    def export(self, ei, wt, order, kf, gfrac):
        #decide what members to export and then remove them from members and return them
        if wt == 'uni': #uniform case very easy
            return wtd_remove(self.members, ei)
        else:
            if order[0] == 'a': #handle the annealed cases
                if gfrac < 0.5:
                    o = order[1:]
                else:
                    o = order[2] + order[1]
            else: #if no annealing
                o = order

            #order the members
            if (o == 'wb' and self.mode == 'max') \
                    or (o == 'bw' and self.mode == 'min'):
                self.members = [a for _, a in sorted(zip(self.member_fitnesses, self.members))]
            if (o == 'bw' and self.mode == 'max') \
                    or (o == 'wb' and self.mode == 'min'):
                self.members = [a for _, a in sorted(zip(self.member_fitnesses, self.members), reverse = True)]

            #calculate the wts
            seq = np.array(range(1, len(self.members) + 1))
            if wt == 'log':
                wts = (np.log(seq)+kf)/(self.n*kf + np.log(math.factorial(self.n)))
            elif wt == 'lin':
                wts = (seq-1+kf)/(self.n*(kf-.5)+.5*self.n**2)
            elif wt == 'exp':
                wts = (np.exp(seq-1) - 1 + kf)/((kf-1)*self.n+(1-np.exp(self.n))/(1-np.exp(1)))

            #draw members and return them
            return wtd_remove(self.members, ei, wts)

    def receive(self, individuals):
        #bring individuals into the populations
        self.members += individuals


class AEO(object):
    """
    Animorphoc Ensemble Optimizer

    :param mode: (str) problem type, either "min" for minimization problem or "max" for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param fit: (function) the fitness function
    :param optimizers: (list) list of optimizer instances to be included in the ensemble
    :param gen_per_cycle: (int) number of generations performed in evolution phase per cycle
    :param alpha: (float or str) option for exponent on g strength measure, if numeric, alpha is taken to be
        that value. If alpha is 'up' alpha is annealed from -1 to 1. If alpha is 'down' it is annealed from
        1 to -1.
    :param g: (str) either 'fitness' or 'improve' for strength measure for exportation number section of migration
    :param g_burden: (bool) True if strength if divided by number of fitness evaluations in evolution phase
    :param wt: (str) 'log', 'lin', 'exp', 'uni' for different weightings in member selection section of migration
    :param beta: (float or str) option for exponent on b strength measure. See alpha for details.
    :param b: (str) either 'fitness' or 'improve' for strength measure for destination selection section of migration
    :param b_burden: (bool) True if strength if divided by number of fitness evaluations in evolution phase
    :param ret: (bool) True if individual can return to original population in destination selection section
    :param order: (str) 'wb' for worst to best, 'bw' for best to worst, prepend 'a' for annealed starting in the given ordering.
    :param kf: (int) 0 or 1 for variant of weighting functions
    :param ngtonevals: (list of callables) list of functions which take number of generations and number of individuals and returns
        number of fitness evaluations ordered according to the algorithms given in optimizers.
    :param ncores: (int) number of parallel processors
    :param seed: (int) random seed for sampling
    """
    def __init__(self, mode, bounds, fit, 
            optimizers, gen_per_cycle,
            alpha, g, g_burden, wt,
            beta, b, b_burden, ret,
            order = None, kf = None, ngtonevals = None,
            ncores = 1, seed = None):

        if not (seed is None):
            random.seed(seed)
            np.random.seed(seed)

        self.mode=mode
        self.fit = fit

        if mode == 'max': #create fit attribute to use for checking consistency of fits
            self.fitcheck=fit
        elif mode == 'min':
            def fitness_wrapper(*args, **kwargs):
                return -fit(*args, **kwargs)
            self.fitcheck=fitness_wrapper
        else:
            raise ValueError('--error: The mode entered by user is invalid, use either `min` or `max`')

        self.optimizers = optimizers
        self.algos = [detect_algo(o) for o in self.optimizers]
        self.gpc = gen_per_cycle

        self.bounds = bounds
        self.ncores = ncores

        #infer variable types
        self.var_type = np.array([bounds[item][0] for item in bounds])

        self.dim = len(bounds)
        self.lb=[self.bounds[item][1] for item in self.bounds]
        self.ub=[self.bounds[item][2] for item in self.bounds]

        #check that all optimizers have options that match AEO
        self.ensure_consistency()

        #process variant options for exportation number
        self.alpha = alpha
        if (not isinstance(self.alpha, float) and
            not self.alpha in ['up', 'down']):
            raise Exception('invalid value for alpha, make sure it is a float!')

        self.g = g
        if not self.g in ['fitness', 'improve']:
            raise Exception('invalid option for g')

        self.g_burden = g_burden
        if not isinstance(g_burden, bool):
            raise Exception('g_burden should be boolean type')

        #process variant options for member selection
        self.wt = wt
        if not self.wt in ['log', 'lin', 'exp', 'uni']:
            raise Exception('invalid option for wt')

        self.order = order
        if not self.order in ['wb', 'bw', 'awb', 'abw']:
            raise Exception('invalid option for order')

        self.kf = kf
        if not self.kf in [0, 1]:
            raise Exception('invalid option for kf')

        if self.wt == 'uni' and ((self.kf is not None)
                or (self.order is not None)):
            print('--warning: kf and order options ignored for uniform weighting')

        #process variant options for destination selection
        self.beta = beta
        if (not isinstance(self.beta, float) and
            not self.beta in ['up', 'down']):
            raise Exception('invalid value for beta, make sure it is a float!')

        self.b = b
        if not self.b in ['fitness', 'improve']:
            raise Exception('invalid option for b')

        self.b_burden = b_burden
        if not isinstance(b_burden, bool):
            raise Exception('b_burden should be boolean type')

        self.ret = ret
        if not isinstance(ret, bool):
            raise Exception('ret should be boolean type')

        #process number of generations to number of evaluations functions
        if g_burden or b_burden:
            self.ngtonevals = ngtonevals


    def ensure_consistency(self):
        #loop through all optimizers and make sure all options are set to be the same
        gen_warning = ', check that options of all optimizers are the same as AEO'
        for o, a in zip(self.optimizers, self.algos):
            if a in max_algos:
                assert self.mode == o.mode,'%s has incorrect optimization mode'%o + gen_warning
                assert self.bounds == o.bounds,'%s has incorrect bounds'%o + gen_warning
                try:
                    assert self.fitcheck(self.lb) == o.fit(self.lb)
                    assert self.fitcheck(self.ub) == o.fit(self.ub)
                    inner_test = [np.random.uniform(self.lb[i], self.ub[i]) for i in range(len(self.ub))]
                    assert self.fitcheck(inner_test) == o.fit(inner_test)
                except:
                    raise Exception('i%s has incorrect fitness function'%o + gen_warning)
            else:
                assert self.mode == o.mode,'%s has incorrect optimization mode'%o + gen_warning
                assert self.bounds == o.bounds,'%s has incorrect bounds'%o + gen_warning
                try:
                    assert self.fitcheck(self.lb) == -o.fit(self.lb)
                    assert self.fitcheck(self.ub) == -o.fit(self.ub)
                    inner_test = [np.random.uniform(self.lb[i], self.ub[i]) for i in range(len(self.ub))]
                    assert self.fitcheck(inner_test) == -o.fit(inner_test)
                except:
                    raise Exception('i%s has incorrect fitness function'%o + gen_warning)

    def init_sample(self, bounds):

        indv=[]
        for key in bounds:
            if bounds[key][0] == 'int':
                indv.append(random.randint(bounds[key][1], bounds[key][2]))
            elif bounds[key][0] == 'float':
                indv.append(random.uniform(bounds[key][1], bounds[key][2]))
            #elif bounds[key][0] == 'grid':
            #    indv.append(random.sample(bounds[key][1],1)[0])
            else:
                raise Exception ('unknown data type is given, either int, float, or grid are allowed for parameter bounds')
        return indv

    def get_alphabeta(self, aorb, ncyc, Ncyc):
        if isinstance(aorb, float):
            return aorb
        elif aorb == 'up':
            return 2*(ncyc-1)/(Ncyc-1) - 1
        elif aorb == 'down':
            return 1 - 2*(ncyc-1)/(Ncyc-1)

    def evolute(self, Ncyc, npop0 = None, x0 = None, pop0 = None, verbose = False):
        """
        This function evolutes the AEO algorithm for a number of cycles. Either
        npop0 or x0 and pop0 are required.

        :param Ncyc: (int) number of cycles to evolute
        :param pop0: (list of ints) number of individuals in starting population for each optimizer
        :param x0: (list of lists) initial positions of individuals in problem space
        :param pop0: (list of ints) population assignments for x0, integer corresponding to assigned population ordered
            according to self.optimize
        """
        #intepret npop0 or x0 and pop0 input
        if x0 is not None:
            if npop0 is not None:
                print('--warning: x0 and npop0 is defined, ignoring npop0')
            if pop0 is None:
                raise Exception('need to assign individuals in x0 to populations with different evolution'\
                        + ' strategies by using the pop0 argument where a list of integers is used of equal'\
                        + ' length to x0 telling where each individual belongs.')
            assert len(x0) == len(pop0), 'x0 and pop0 must be ov equal length'
        else:
            x0 = [self.init_sample(self.bounds) for i in range(sum(npop0))]
            dup = [[i]*npop0[i] for i in range(len(npop0))]
            pop0 = list(itertools.chain.from_iterable(dup))

        #separate starting positions according to optimizer/strategy, initialize Population objs
        self.pops = []
        for i in range(len(self.optimizers)):
            xpop = []
            for x, p in zip(x0, pop0):
                if p == i:
                    xpop.append(x)
            if self.g_burden or self.b_burden:
                self.pops.append(Population(self.optimizers[i], self.algos[i],
                    xpop, self.mode, self.ngtonevals[i]))
            else:
                self.pops.append(Population(self.optimizers[i], self.algos[i], xpop, self.mode))

        #initialize log Dataset
        membercoords = range(len(x0))
        cyclecoords = range(1, Ncyc + 1)
        popcoords = [p.algo for p in self.pops]
        popcoords = []
        for p in self.pops:
            algo = p.algo
            if not (algo in popcoords):
                popcoords.append(algo)
            else:
                i = 2
                while algo + str(i).zfill(3) in popcoords:
                    i += 1
                popcoords.append(algo + str(i).zfill(3))

        nm = len(membercoords)
        nc = len(cyclecoords)
        npp = len(popcoords)
        log = xr.Dataset(
                {
                    "initial_members"    : (["member", "pop"         ], np.zeros((nm, npp    ), dtype = np.float64)),
                    "member_locations"   : (["member", "pop", "cycle"], np.zeros((nm, npp, nc), dtype = np.float64)),
                    "member_fitnesses"   : (["member", "pop", "cycle"], np.zeros((nm, npp, nc), dtype = np.float64)),
                    "nmembers"           : (["pop",           "cycle"], np.zeros((    npp, nc), dtype = np.int32)),
                    "nexport"            : (["pop",           "cycle"], np.zeros((    npp, nc), dtype = np.int32)),
                    "export_pop_wts"     : (["pop",           "cycle"], np.zeros((    npp, nc), dtype = np.float64)),
                    "alpha"              : ([                 "cycle"], np.zeros(          nc , dtype = np.float64)),
                    "wb"                 : ([                 "cycle"], np.zeros(          nc , dtype = np.bool8)),
                    "g"                  : (["pop",           "cycle"], np.zeros((    npp, nc), dtype = np.float64)),
                    'f'                  : (['pop',           'cycle'], np.zeros((    npp, nc), dtype = np.float64)),
                    'unburdened_g'       : (['pop',           'cycle'], np.zeros((    npp, nc), dtype = np.float64)),
                    'Nc'                 : (['pop',           'cycle'], np.zeros((    npp, nc), dtype = np.int32)),
                    'delta_f'            : (['pop',           'cycle'], np.zeros((    npp, nc), dtype = np.float64)),
                    'fmin'               : ([                 'cycle'], np.zeros(          nc , dtype = np.float64)),
                    'fmax'               : ([                 'cycle'], np.zeros(          nc , dtype = np.float64)),
                    'export_wts'         : (['member', 'pop', 'cycle'], np.zeros((nm, npp, nc), dtype = np.float64)),
                    'exported'           : (['member', 'pop', 'cycle'], np.zeros((nm, npp, nc), dtype = np.bool8)),
                    'beta'               : ([                 'cycle'], np.zeros(          nc , dtype = np.float64)),
                    'b'                  : ([          'pop', 'cycle'], np.zeros((    npp, nc), dtype = np.float64)),
                    'A'                  : ([          'pop', 'cycle'], np.zeros((    npp, nc), dtype = np.int32)),
                    'evolute'            : ([          'pop', 'cycle'], np.zeros((    npp, nc), dtype = np.bool8))},
                coords = {
                    'member'  : membercoords,
                    'pop'     : popcoords,
                    'cycle'   : cyclecoords}
                )

        #perform evolution/migration cycle
        for i in range(1, Ncyc + 1):
            #evolution phase
            pop_fits = [p.evolute(self.gpc, self.fit) for p in self.pops]
            print("Cycle", i+1, '/',Ncyc + 1)
            print(pop_fits)
            print("======")

            #exportation number
            #  calc weights
            maxf = max(pop_fits)
            minf = min(pop_fits)
            alpha = self.get_alphabeta(self.alpha, i, Ncyc)
            strengths_exp = [p.strength(self.g, self.g_burden, maxf, minf)**alpha for p in self.pops]
            strengths_exp_scaled = [s/sum(strengths_exp) for s in strengths_exp]
            #  sample binomial to get e_i for each population
            eis = [np.random.binomial(len(p.members), strengths_exp_scaled[j]) for j, p in enumerate(self.pops)]

            #member selection
            #  members removed from population with this export method
            exported = [p.export(eis[j], self.wt, self.order, self.kf, i/Ncyc) for j, p in enumerate(self.pops)]

            #destination selection
            beta = self.get_alphabeta(self.beta, i, Ncyc)
            strengths_exp = [p.strength(self.b, self.b_burden, maxf, minf)**beta for p in self.pops]

            if self.ret:#if population can return to original population
                #manage members that are currently without a home
                exported = list(itertools.chain.from_iterable(exported))
                random.shuffle(exported)#in-place randomize order

                #calculate normalized probabilities and draw samples
                strengths_exp_scaled = [s/sum(strengths_exp) for s in strengths_exp]
                allotments = np.random.multinomial(len(exported), strengths_exp_scaled)

                #distribute individuals according to the sample
                for a, p in zip(allotments, self.pops):
                    p.receive(exported[:a])
                    exported = exported[a:]

            else:
                pop_indxs = list(range(len(self.pops)))
                for j, exported_group in enumerate(exported):
                    strengths_inotj = strengths_exp_scaled[:j] + strengths_exp_scaled[j+1:]
                    pop_indxs_inotj = pop_indxs[:j] + pop_indxs[j+1:]
                    random.shuffle(exported_group)
                    allotment = np.random.multinomial(len(exported_group), strengths_inotj)
                    for a, pi in zip(allotment, pop_indxs_inotj):
                        self.pops[pi].receive(exported_group[:a])
                        exported_group = exported_group[a:]




    #TODO: on second generation, the number of members changes so evolute is unhappy...should make
    # some function to copy and initialize a new versionof the algo object but with n individuals changed
    #TODO: Set up evolute method
        #TODO: write in verbose reporting
    #TODO: Set up migration method with 3 phases and markov matrix calculation

    #TODO: Autodetect initial populations
    #TODO: Incorporate ngtonevals into method autodetection
    #TODO: Ramanujan approx for log(x!)



