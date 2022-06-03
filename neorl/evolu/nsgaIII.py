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
#Created on Tues May 24 19:37:04 2022
#
#@author: Paul
# The following implementation is adapted to NEORL
#  from DEAP implementation: https://github.com/DEAP/deap/blob/master/deap/tools/emo.py#L15 
#"""

import random
import numpy as np
from collections import defaultdict
import copy
import joblib
from neorl.evolu.crossover import cxES2point, cxESBlend
from neorl.evolu.discrete import encode_grid_to_discrete, decode_discrete_to_grid
from neorl.utils.seeding import set_neorl_seed
#from neorl.utils.tools import get_population
from neorl import ES

#from DEAP
from operator import attrgetter, itemgetter
from itertools import chain
import sys
import bisect

import pandas as pd

def isDominated(wvalues1, wvalues2):
    """
    
    Returns whether or not *wvalues2* dominates *wvalues1*.
    :param wvalues1: The weighted fitness values that would be dominated.
    :param wvalues2: The weighted fitness values of the dominant.
    :returns: :obj:`True` if wvalues2 dominates wvalues1, :obj:`False`
              otherwise.
    """
    not_equal = False
    for self_wvalue, other_wvalue in zip(wvalues1, wvalues2):
        if self_wvalue > other_wvalue:
            return False
        elif self_wvalue < other_wvalue:
            not_equal = True
    return not_equal

def sortNondominated(pop, k, first_front_only=False):
    """
    Sort the first *k* *pop* into different nondomination levels
    using the "Fast Nondominated Sorting Approach" proposed by Deb et al.,
    see [Deb2002]_. This algorithm has a time complexity of :math:`O(MN^2)`,
    where :math:`M` is the number of objectives and :math:`N` the number of
    pop.
    :param pop: A list of pop to select from.
    :param k: The number of pop to select.
    :param first_front_only: If :obj:`True` sort only the first front and
                             exit.
    :returns: A list of Pareto fronts (lists), the first list includes
              nondominated pop.
    .. 
    [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
    non-dominated sorting genetic algorithm for multi-objective
    optimization: NSGA-II", 2002.
    """
    if k == 0:
        return []

    pop=list(pop.items())

    map_fit_ind = defaultdict(list)
    for ind in pop:
        map_fit_ind[ind[0]].append(ind)
    fits = list(map_fit_ind.keys())

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)

    # Rank first Pareto front
    for i, fit_i in enumerate(fits):
        for fit_j in fits[i+1:]:
            if isDominated(map_fit_ind[fit_j][0][1][2], map_fit_ind[fit_i][0][1][2]):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif isDominated(map_fit_ind[fit_i][0][1][2], map_fit_ind[fit_j][0][1][2]):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    fronts = [[]]
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    pareto_sorted = len(fronts[-1])

    # Rank the next front until all pop are sorted or
    # the given number of individual are sorted.
    if not first_front_only:
        N = min(len(pop), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d]) # add element to the next solution
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

    return fronts

#######################################
# Generalized Reduced runtime ND sort #
#######################################

def identity(obj):
    """Returns directly the argument *obj*.
    """
    return obj


def median(seq, key=identity):
    """Returns the median of *seq* - the numeric value separating the higher
    half of a sample from the lower half. If there is an even number of
    elements in *seq*, it returns the mean of the two middle values.
    """
    sseq = sorted(seq, key=key)
    length = len(seq)
    if length % 2 == 1:
        return key(sseq[(length - 1) // 2])
    else:
        return (key(sseq[(length - 1) // 2]) + key(sseq[length // 2])) / 2.0

def sortLogNondominated(pop, k, first_front_only=False):
    """Sort *individuals* in pareto non-dominated fronts using the Generalized
    Reduced Run-Time Complexity Non-Dominated Sorting Algorithm presented by
    Fortin et al. (2013).
    :param individuals: A list of individuals to select from.
    :returns: A list of Pareto fronts (lists), with the first list being the
              true Pareto front.
    """
    if k == 0:
        return []

    pop=list(pop.items())
    #Separate individuals according to unique fitnesses
    unique_fits = defaultdict(list)
    for i, ind in enumerate(pop):
        unique_fits[tuple(ind[1][2])].append(ind)
            

    #Launch the sorting algorithm
    obj = len(pop[0][1][2])-1
    fitnesses = list(unique_fits.keys())
    front = dict.fromkeys(fitnesses, 0)

    # Sort the fitnesses lexicographically.
    fitnesses.sort(reverse=True)
    sortNDHelperA(fitnesses, obj, front)

    #Extract pop from front list here
    nbfronts = max(front.values())+1
    pareto_fronts = [[] for i in range(nbfronts)]
    for fit in fitnesses:
        index = front[fit]
        pareto_fronts[index].extend(unique_fits[fit])

    # Keep only the fronts required to have k pop.
    if not first_front_only:
        count = 0
        for i, front in enumerate(pareto_fronts):
            count += len(front)
            if count >= k:
                return pareto_fronts[:i+1]
        return pareto_fronts
    else:
        return pareto_fronts[0]

def sortNDHelperA(fitnesses, obj, front):
    """Create a non-dominated sorting of S on the first M objectives"""
    if len(fitnesses) < 2:
        return
    elif len(fitnesses) == 2:
        # Only two individuals, compare them and adjust front number
        s1, s2 = fitnesses[0], fitnesses[1]
        if isDominated(s2[:obj+1], s1[:obj+1]):
            front[s2] = max(front[s2], front[s1] + 1)
    elif obj == 1:
        sweepA(fitnesses, front)
    elif len(frozenset(map(itemgetter(obj), fitnesses))) == 1:
        #All individuals for objective M are equal: go to objective M-1
        sortNDHelperA(fitnesses, obj-1, front)
    else:
        # More than two individuals, split list and then apply recursion
        best, worst = splitA(fitnesses, obj)
        sortNDHelperA(best, obj, front)
        sortNDHelperB(best, worst, obj-1, front)
        sortNDHelperA(worst, obj, front)

def splitA(fitnesses, obj):
    """Partition the set of fitnesses in two according to the median of
    the objective index *obj*. The values equal to the median are put in
    the set containing the least elements.
    """
    median_ = median(fitnesses, itemgetter(obj))
    best_a, worst_a = [], []
    best_b, worst_b = [], []

    for fit in fitnesses:
        if fit[obj] > median_:
            best_a.append(fit)
            best_b.append(fit)
        elif fit[obj] < median_:
            worst_a.append(fit)
            worst_b.append(fit)
        else:
            best_a.append(fit)
            worst_b.append(fit)

    balance_a = abs(len(best_a) - len(worst_a))
    balance_b = abs(len(best_b) - len(worst_b))

    if balance_a <= balance_b:
        return best_a, worst_a
    else:
        return best_b, worst_b

def sweepA(fitnesses, front):
    """Update rank number associated to the fitnesses according
    to the first two objectives using a geometric sweep procedure.
    """
    stairs = [-fitnesses[0][1]]
    fstairs = [fitnesses[0]]
    for fit in fitnesses[1:]:
        idx = bisect.bisect_right(stairs, -fit[1])
        if 0 < idx <= len(stairs):
            fstair = max(fstairs[:idx], key=front.__getitem__)
            front[fit] = max(front[fit], front[fstair]+1)
        for i, fstair in enumerate(fstairs[idx:], idx):
            if front[fstair] == front[fit]:
                del stairs[i]
                del fstairs[i]
                break
        stairs.insert(idx, -fit[1])
        fstairs.insert(idx, fit)

def sortNDHelperB(best, worst, obj, front):
    """Assign front numbers to the solutions in H according to the solutions
    in L. The solutions in L are assumed to have correct front numbers and the
    solutions in H are not compared with each other, as this is supposed to
    happen after sortNDHelperB is called."""
    key = itemgetter(obj)
    if len(worst) == 0 or len(best) == 0:
        #One of the lists is empty: nothing to do
        return
    elif len(best) == 1 or len(worst) == 1:
        #One of the lists has one individual: compare directly
        for hi in worst:
            for li in best:
                if isDominated(hi[:obj+1], li[:obj+1]) or hi[:obj+1] == li[:obj+1]:
                    front[hi] = max(front[hi], front[li] + 1)
    elif obj == 1:
        sweepB(best, worst, front)
    elif key(min(best, key=key)) >= key(max(worst, key=key)):
        #All individuals from L dominate H for objective M:
        #Also supports the case where every individuals in L and H
        #has the same value for the current objective
        #Skip to objective M-1
        sortNDHelperB(best, worst, obj-1, front)
    elif key(max(best, key=key)) >= key(min(worst, key=key)):
        best1, best2, worst1, worst2 = splitB(best, worst, obj)
        sortNDHelperB(best1, worst1, obj, front)
        sortNDHelperB(best1, worst2, obj-1, front)
        sortNDHelperB(best2, worst2, obj, front)

def splitB(best, worst, obj):
    """Split both best individual and worst sets of fitnesses according
    to the median of objective *obj* computed on the set containing the
    most elements. The values equal to the median are attributed so as
    to balance the four resulting sets as much as possible.
    """
    median_ = median(best if len(best) > len(worst) else worst, itemgetter(obj))
    best1_a, best2_a, best1_b, best2_b = [], [], [], []
    for fit in best:
        if fit[obj] > median_:
            best1_a.append(fit)
            best1_b.append(fit)
        elif fit[obj] < median_:
            best2_a.append(fit)
            best2_b.append(fit)
        else:
            best1_a.append(fit)
            best2_b.append(fit)

    worst1_a, worst2_a, worst1_b, worst2_b = [], [], [], []
    for fit in worst:
        if fit[obj] > median_:
            worst1_a.append(fit)
            worst1_b.append(fit)
        elif fit[obj] < median_:
            worst2_a.append(fit)
            worst2_b.append(fit)
        else:
            worst1_a.append(fit)
            worst2_b.append(fit)

    balance_a = abs(len(best1_a) - len(best2_a) + len(worst1_a) - len(worst2_a))
    balance_b = abs(len(best1_b) - len(best2_b) + len(worst1_b) - len(worst2_b))

    if balance_a <= balance_b:
        return best1_a, best2_a, worst1_a, worst2_a
    else:
        return best1_b, best2_b, worst1_b, worst2_b

def sweepB(best, worst, front):
    """Adjust the rank number of the worst fitnesses according to
    the best fitnesses on the first two objectives using a sweep
    procedure.
    """
    stairs, fstairs = [], []
    iter_best = iter(best)
    next_best = next(iter_best, False)
    for h in worst:
        while next_best and h[:2] <= next_best[:2]:
            insert = True
            for i, fstair in enumerate(fstairs):
                if front[fstair] == front[next_best]:
                    if fstair[1] > next_best[1]:
                        insert = False
                    else:
                        del stairs[i], fstairs[i]
                    break
            if insert:
                idx = bisect.bisect_right(stairs, -next_best[1])
                stairs.insert(idx, -next_best[1])
                fstairs.insert(idx, next_best)
            next_best = next(iter_best, False)

        idx = bisect.bisect_right(stairs, -h[1])
        if 0 < idx <= len(stairs):
            fstair = max(fstairs[:idx], key=front.__getitem__)
            front[h] = max(front[h], front[fstair]+1)

#######################################
# niching - based Selection functions #
#######################################

def find_extreme_points(fitnesses, best_point, extreme_points=None):
    """
    Finds the individuals with extreme values for each objective function.
    """
    # Keep track of last generation extreme points
    if extreme_points is not None:
        fitnesses = np.concatenate((fitnesses, extreme_points), axis=0)

    # Translate objectives
    ft = fitnesses - best_point

    # Find achievement scalarizing function (asf)
    asf = np.eye(best_point.shape[0])
    asf[asf == 0] = 1e6
    asf = np.max(ft * asf[:, np.newaxis, :], axis=2)

    # Extreme point are the fitnesses with minimal asf
    min_asf_idx = np.argmin(asf, axis=1)
    return fitnesses[min_asf_idx, :]


def find_intercepts(extreme_points, best_point, current_worst, front_worst):
    """
    Find intercepts between the hyperplane and each axis with
    the ideal point as origin.
    """
    # Construct hyperplane sum(f_i^n) = 1
    b = np.ones(extreme_points.shape[1])
    A = extreme_points - best_point
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        intercepts = current_worst
    else:
        if np.count_nonzero(x) != len(x):
            intercepts = front_worst
        else:
            intercepts = 1 / x

            if (not np.allclose(np.dot(A, x), b) or
                    np.any(intercepts <= 1e-6) or
                    np.any((intercepts + best_point) > current_worst)):
                intercepts = front_worst

    return intercepts


def associate_to_niche(fitnesses, reference_points, best_point, intercepts):
    """
    Associates individuals to reference points and calculates niche number.
    Corresponds to Algorithm 3 of Deb & Jain (2014).
    """
    # Normalize by ideal point and intercepts
    fn = (fitnesses - best_point) / (intercepts - best_point)

    # Create distance matrix
    fn = np.repeat(np.expand_dims(fn, axis=1), len(reference_points), axis=1)
    norm = np.linalg.norm(reference_points, axis=1)

    distances = np.sum(fn * reference_points, axis=2) / norm.reshape(1, -1)
    distances = distances[:, :, np.newaxis] * reference_points[np.newaxis, :, :] / norm[np.newaxis, :, np.newaxis]
    distances = np.linalg.norm(distances - fn, axis=2)

    # Retrieve min distance niche index
    niches = np.argmin(distances, axis=1)
    distances = distances[range(niches.shape[0]), niches]
    return niches, distances


def niching(pop, k, niches, distances, niche_counts):
    """
    niche preserving operator. Choose elements which are associated to reference points
    with the lowest number of association from the already choosen batch of solution Pt
    """
    selected = []
    available = np.ones(len(pop), dtype=np.bool)
    while len(selected) < k:
        # Maximum number of individuals (niches) to select in that round
        n = k - len(selected)

        # Find the available niches and the minimum niche count in them
        available_niches = np.zeros(len(niche_counts), dtype=np.bool)
        available_niches[np.unique(niches[available])] = True
        min_count = np.min(niche_counts[available_niches])

        # Select at most n niches with the minimum count
        selected_niches = np.flatnonzero(np.logical_and(available_niches, niche_counts == min_count))
        np.random.shuffle(selected_niches)
        selected_niches = selected_niches[:n]

        for niche in selected_niches:
            # Select from available individuals in niche
            niche_individuals = np.flatnonzero(np.logical_and(niches == niche, available))
            np.random.shuffle(niche_individuals)

            # If no individual in that niche, select the closest to reference
            # Else select randomly
            if niche_counts[niche] == 0:
                sel_index = niche_individuals[np.argmin(distances[niche_individuals])]
            else:
                sel_index = niche_individuals[0]

            # Update availability, counts and selection
            available[sel_index] = False
            niche_counts[niche] += 1
            selected.append(pop[sel_index])

    return selected


def uniform_reference_points(nobj, p=4, scaling=None):
    """
    Generate reference points uniformly on the hyperplane intersecting
    each axis at 1. The scaling factor is used to combine multiple layers of
    reference points.
    """
    def gen_refs_recursive(ref, nobj, left, total, depth):
        points = []
        if depth == nobj - 1:
            ref[depth] = left / total
            points.append(ref)
        else:
            for i in range(left + 1):
                ref[depth] = i / total
                points.extend(gen_refs_recursive(ref.copy(), nobj, left - i, total, depth + 1))
        return points

    ref_points = np.array(gen_refs_recursive(np.zeros(nobj), nobj, p, p, 0))
    if scaling is not None:
        ref_points *= scaling
        ref_points += (1 - scaling) / nobj

    return ref_points

class NSGAIII(ES):
    """
    Parallel Fast Non-dominated Sorting Gentic Algorithm - III
    
    Only the seleciton operator differ from the ES implementation
    The statistic also changed.

    :param mode: (str) problem type, either ``min`` for minimization problem or ``max`` for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param lambda\_: (int) total number of individuals in the population. Use Combination(M + p - 1, p) for nicely spaced points on the pareto front.
    :param cxmode: (str): the crossover mode, either 'cx2point' or 'blend'
    :param alpha: (float) Extent of the blending between [0,1], the blend crossover randomly selects a child in the range [x1-alpha(x2-x1), x2+alpha(x2-x1)] (Only used for cxmode='blend')
    :param cxpb: (float) population crossover probability between [0,1]
    :param mutpb: (float) population mutation probability between [0,1] 
    :param smin: (float): minimum bound for the strategy vector
    :param smax: (float): maximum bound for the strategy vector
    :param ncores: (int) number of parallel processors
    :param seed: (int) random seed for sampling
    
    :param sorting: (str) sorting type, ``standard`` or ``log``. The latter should be faster and is used as default.#Paul
    :param: p: (int) number of divisions along each objective for the reference points. The number of reference points is Combination(M + p - 1, p), where M is the number of objective
    :param ref_points: (list) of user inputs reference points. If none the reference points are generated uniformly on the hyperplane intersecting
    each axis at 1.
    """
    def __init__ (self, mode, bounds, fit, lambda_=60, cxmode='cx2point', 
                  alpha=0.5, cxpb=0.6, mutpb=0.3, smin=0.01, smax=0.5, clip=True, ncores=1, seed=None, p = 4,ref_points = None,sorting = 'log',**kwargs):  
        
        set_neorl_seed(seed)
        super().__init__(mode = mode, bounds = bounds, fit = fit, lambda_=lambda_, mu=lambda_, cxmode=cxmode, 
                  alpha=alpha, cxpb=cxpb, mutpb=mutpb, smin=smin, smax=smax, clip=clip, ncores=ncores, seed=seed)
        # new hyper-parameters #Paul
        self.sorting = sorting
        #NSGA-III specific
        self.p = p
        self.ref_points = ref_points

    def fit_worker(self, x):
        #"""
        #Evaluates fitness of an individual.
        #"""
        
        
        #mir-grid
        if self.grid_flag:
            #decode the individual back to the int/float/grid mixed space
            x=decode_discrete_to_grid(x,self.orig_bounds,self.bounds_map) 
                    
        fitness = self.fit(x)
        if isinstance(fitness,np.ndarray):
            return fitness
        else:
            return np.array([fitness])

    def select(self,pop, k, ref_points, nd='standard', best_point=None,
             worst_point=None, extreme_points=None):
        """
        Implementation of NSGA-III selection as presented in [Deb2014]_.
        This implementation is partly based on `lmarti/nsgaiii
        <https://github.com/lmarti/nsgaiii>`_. It departs slightly from the
        original implementation in that it does not use memory to keep track
        of ideal and extreme points. This choice has been made to fit the
        functional api of DEAP. For a version of NSGA-III see
        :class:`~deap.tools.selNSGA3WithMemory`.
        :param pop: A list of pop to select from.
        :param k: The number of pop to select.
        :param ref_points: Reference points to use for niching.
        :param nd: Specify the non-dominated algorithm to use: 'standard' or 'log'.
        :param best_point: Best point found at previous generation. If not provided
            find the best point only from current pop.
        :param worst_point: Worst point found at previous generation. If not provided
            find the worst point only from current pop.
        :param extreme_points: Extreme points found at previous generation. If not provided
            find the extreme points only from current pop.
        :param return_memory: If :data:`True`, return the best, worst and extreme points
            in addition to the chosen pop.
        :returns: A list of selected pop.
        :returns: If `return_memory` is :data:`True`, a namedtuple with the
            `best_point`, `worst_point`, and `extreme_points`.
        You can generate the reference points using the :func:`uniform_reference_points`
        function::
            >>> ref_points = tools.uniform_reference_points(nobj=3, p=12)   # doctest: +SKIP
            >>> selected = selNSGA3(population, k, ref_points)              # doctest: +SKIP
        .. [Deb2014] Deb, K., & Jain, H. (2014). An Evolutionary Many-Objective Optimization
            Algorithm Using Reference-Point-Based Nondominated Sorting Approach,
            Part I: Solving Problems With Box Constraints. IEEE Transactions on
            Evolutionary Computation, 18(4), 577-601. doi:10.1109/TEVC.2013.2281535.
        """
        if nd == "standard":
            pareto_fronts = sortNondominated(pop, k)
        elif nd == 'log':
            pareto_fronts = sortLogNondominated(pop, k)
        else:
            raise Exception("NSGA3: The choice of non-dominated sorting "
                            "method '{0}' is invalid.".format(nd))
        # Extract fitnesses as a np array in the nd-sort order
        # Use * -1 to tackle always as a minimization problem. Necessary here as well
        fitnesses = np.array([ind[1][2] for f in pareto_fronts for ind in f])
        fitnesses *= -1
        # Get best and worst point of population, contrary to pymoo
        # we don't use memory
        if best_point is not None and worst_point is not None:
            best_point = np.min(np.concatenate((fitnesses, best_point), axis=0), axis=0)
            worst_point = np.max(np.concatenate((fitnesses, worst_point), axis=0), axis=0)
        else:
            best_point = np.min(fitnesses, axis=0)
            worst_point = np.max(fitnesses, axis=0)

        extreme_points = find_extreme_points(fitnesses, best_point, extreme_points)
        front_worst = np.max(fitnesses[:sum(len(f) for f in pareto_fronts), :], axis=0)
        intercepts = find_intercepts(extreme_points, best_point, worst_point, front_worst)
        niches, dist = associate_to_niche(fitnesses, ref_points, best_point, intercepts)

        # Get counts per niche for individuals in all front but the last
        niche_counts = np.zeros(len(ref_points), dtype=np.int64)
        index, counts = np.unique(niches[:-len(pareto_fronts[-1])], return_counts=True)
        niche_counts[index] = counts

        # Choose individuals from all fronts but the last
        chosen = list(chain(*pareto_fronts[:-1]))

        # Use niching to select the remaining individuals
        sel_count = len(chosen)
        n = k - sel_count
        selected = niching(pareto_fronts[-1], n, niches[sel_count:], dist[sel_count:], niche_counts)
        chosen.extend(selected)
        # re-cast into a dictionary to comply with NEORL 
        best_dict=defaultdict(list)
        index=0
        for key in chosen:
            best_dict[index] = key[1]
            index+=1
        
        return best_dict
    def GenOffspring(self, pop):
        #"""
        # 
        #This function generates the offspring by applying crossover, mutation **or** reproduction. 
        #The sum of both probabilities self.cxpb and self.mutpb must be in [0,1]
        #The reproduction probability is 1 - cxpb - mutpb
        #The new offspring goes for fitness evaluation
        
        #Inputs:
        #    pop (dict): population in dictionary structure
        #Returns:
        #    offspring (dict): new modified population in dictionary structure    
        #"""
        
        
        pop_indices=list(range(0,len(pop)))
        offspring = defaultdict(list)
        for i in range(self.lambda_):
            rn = random.random()
            #------------------------------
            # Crossover
            #------------------------------
            if rn < self.cxpb:            
                index1, index2 = random.sample(pop_indices,2)
                if self.cxmode.strip() =='cx2point':
                    ind1, ind2, strat1, strat2 = cxES2point(ind1=list(pop[index1][0]),ind2=list(pop[index2][0]), 
                                                            strat1=list(pop[index1][1]),strat2=list(pop[index2][1]))
                elif self.cxmode.strip() == 'blend':
                    ind1, ind2, strat1, strat2=cxESBlend(ind1=list(pop[index1][0]), ind2=list(pop[index2][0]), 
                                                                         strat1=list(pop[index1][1]),strat2=list(pop[index2][1]),
                                                                         alpha=self.alpha)
                else:
                    raise ValueError('--error: the cxmode selected (`{}`) is not available in ES, either choose `cx2point` or `blend`'.format(self.cxmode))
                
                ind1=self.ensure_bounds(ind1)
                ind2=self.ensure_bounds(ind2)
                
                ind1=self.ensure_discrete(ind1)  #check discrete variables after crossover
                ind2=self.ensure_discrete(ind2)  #check discrete variables after crossover
                
                offspring[i + len(pop)].append(ind1)
                offspring[i + len(pop)].append(strat1)
                #print('crossover is done for sample {} between {} and {}'.format(i,index1,index2))
            #------------------------------
            # Mutation
            #------------------------------
            elif rn < self.cxpb + self.mutpb:  # Apply mutation
                index = random.choice(pop_indices)
                ind, strat=self.mutES(ind=list(pop[index][0]), strat=list(pop[index][1]))
                offspring[i + len(pop)].append(ind)
                offspring[i + len(pop)].append(strat)
                #print('mutation is done for sample {} based on {}'.format(i,index))
            #------------------------------
            # Reproduction from population
            #------------------------------
            else:                         
                index=random.choice(pop_indices)
                pop[index][0]=self.ensure_discrete(pop[index][0])
                offspring[i + len(pop)].append(pop[index][0])
                offspring[i + len(pop)].append(pop[index][1])
                #print('reproduction is done for sample {} based on {}'.format(i,index))
                
        if self.clip:
            for item in offspring:
                offspring[item][1]=list(np.clip(offspring[item][1], self.smin, self.smax))
        
        return offspring
               
    def evolute(self, ngen, x0=None, verbose=False):
        """
        This function evolutes the ES algorithm for number of generations.
        
        :param ngen: (int) number of generations to evolute
        :param x0: (list of lists) the initial position of the swarm particles
        :param verbose: (bool) print statistics to screen
        
        :return: (tuple) (best individual, best fitness, and a list of fitness history)
        """
        self.es_hist={}
        self.es_hist['mean_strategy']=[]
        #self.y_opt=-np.inf
        self.best_scores=[]
        self.best_indvs=[]
        if x0:    
            assert len(x0) == self.lambda_, '--error: the length of x0 ({}) (initial population) must equal to the size of lambda ({})'.format(len(x0), self.lambda_)
            self.population=self.init_pop(x0=x0, verbose=verbose)
        else:
            self.population=self.init_pop(verbose=verbose)
        if len(self.population[1][2]) == 1:
            print("--warning: length of output is 1, the sorting method is changed to ``standard``.")
            self.sorting = "standard"
        # generate reference points
        if self.ref_points is None:
            self.ref_points = uniform_reference_points(nobj = len(self.population[1][2]), p = self.p)      
        # Begin the evolution process
        for gen in range(1, ngen + 1):
            
            # Vary the population and generate new offspring
            offspring = self.GenOffspring(pop=self.population)
            # Evaluate the individuals with an invalid fitness with multiprocessign Pool
            # create and run the Pool
            if self.ncores > 1:
                core_list=[]
                for key in offspring:
                    core_list.append(offspring[key][0])

                with joblib.Parallel(n_jobs=self.ncores) as parallel:
                    fitness=parallel(joblib.delayed(self.fit_worker)(item) for item in core_list)
                    
                [offspring[ind + len(self.population)].append(fitness[ind]) for ind in range(len(offspring))]
                
            else: #serial calcs
                
                for ind in offspring.keys():
                    fitness=self.fit_worker(offspring[ind][0])
                    offspring[ind].append(fitness)
        
                
            # Select the next generation population
            offspring.update(self.population) # concatenate offspring and parents dictionnaries
            self.population = copy.deepcopy(self.select(pop=offspring, k=self.mu, ref_points = self.ref_points, nd = self.sorting))
            if self.RLmode:  #perform RL informed ES
                self.population=self.mix_population(self.population)
                
            #inds, rwd=[self.population[i][0] for i in self.population], [self.population[i][2] for i in self.population]
            
            #Paul many changes: simply provide the pareto front (add a layer of calculation)
            if self.sorting == "standard":#Paul
                pareto_front = sortNondominated(self.population, len(self.population))[0]
            elif self.sorting == 'log':
                pareto_front = sortLogNondominated(self.population, len(self.population))[0]  
            inds_par, rwd_par=[i[1][0] for i in pareto_front], [i[1][2] for i in pareto_front]
            self.best_scores.append(rwd_par)
            #arg_max=np.argmax(rwd)
            self.best_indvs.append(inds_par)
            #if rwd[arg_max] > self.y_opt:
            #    self.y_opt=rwd[arg_max]
            #    self.x_opt=copy.deepcopy(inds[arg_max])
            
            #--mir
            #if self.mode=='min':#Paul
            #    self.y_opt_correct=[-x for x in rwd]#self.y_opt
            #else:
            #    self.y_opt_correct=rwd#self.y_opt

            #mir-grid
            #if self.grid_flag:
            #    self.x_opt_correct=decode_discrete_to_grid(inds,self.orig_bounds,self.bounds_map)#self>x_opt
            #else:
            #    self.x_opt_correct=inds#self.x_opt
            if self.mode=='min':#Paul
                self.y_opt_correct=[-x for x in rwd_par]#self.y_opt
            else:
                self.y_opt_correct=rwd_par#self.y_opt

            if self.grid_flag:
                self.x_opt_correct=decode_discrete_to_grid(inds_par,self.orig_bounds,self.bounds_map)#self>x_opt
            else:
                self.x_opt_correct=inds_par#self.x_opt
            
            mean_strategy=[np.mean(self.population[i][1]) for i in self.population]
            self.es_hist['mean_strategy'].append(np.mean(mean_strategy))
            if verbose:
                print('##############################################################################')
                print('ES step {}/{}, CX={}, MUT={}, MU={}, LAMBDA={}, Ncores={}'.format(gen*self.lambda_,ngen*self.lambda_, np.round(self.cxpb,2), np.round(self.mutpb,2), self.mu, self.lambda_, self.ncores))
                print('##############################################################################')
                print('Statistics for generation {}'.format(gen))
                print('Best Fitness:', np.round(rwd_par,6) if self.mode == 'max' else -np.round(rwd_par,6))
                print('Best Individual(s):', inds_par if not self.grid_flag else decode_discrete_to_grid(inds_par,self.orig_bounds,self.bounds_map))
                print('Length of the pareto front / length of the population {} / {}'.format(len(inds_par),len(self.population)))
                print('Max Strategy:', np.round(np.max(mean_strategy),3))
                print('Min Strategy:', np.round(np.min(mean_strategy),3))
                print('Average Strategy:', np.round(np.mean(mean_strategy),3))
                print('##############################################################################')
        if verbose:#Paul
            print('------------------------ ES Summary --------------------------')
            print('Best fitness (y) found:', self.y_opt_correct)
            print('Best individual (x) found:', self.x_opt_correct)
            print('Length of the pareto front / length of the population {} / {}'.format(len(self.y_opt_correct),len(self.population)))
            print('--------------------------------------------------------------') 

        
        #---update final logger
        self.es_hist['last_pop'] = get_population_nsga(self.population,mode = self.mode)
        self.best_scores=[-np.array(item) for item in self.best_scores]
        self.es_hist['global_fitness'] = self.y_opt_correct # pareto of last population
        self.es_hist['local_fitness'] = self.best_scores # full history of pareto front
        
        return self.x_opt_correct, self.y_opt_correct, self.es_hist
    
def get_population_nsga(pop,mode):#Paul
    d=len(pop[0][0])
    p= len(pop[0][2])
    npop=len(pop)
    df_pop=np.zeros((npop, d+p))   #additional column for fitness        
    for i, indv in enumerate(pop):
        df_pop[i,:d]=pop[indv][0]
        if mode == 'min':
            df_pop[i,-p:]= - pop[indv][2]
        else:
            df_pop[i,-p:]=pop[indv][2]
    try:    
        colnames=['var'+str(i) for i in range(1,d+1)] + ['obj'+str(i) for i in range(1,p+1)]
        rownames=['indv'+str(i) for i in range(1,npop+1)]
        df_pop=pd.DataFrame(df_pop, index=rownames, columns=colnames)
    except:
        df_pop=pd.DataFrame(np.zeros((5, 5)))   #return an empty dataframe
    
    return df_pop