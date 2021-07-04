#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:42:24 2020

@author: majdi
"""

import warnings
warnings.filterwarnings("ignore")
import random
import numpy as np
from collections import defaultdict

######################################
# ES Crossovers                      #
######################################

def cxESBlend(ind1, ind2, strat1, strat2, alpha=0.1):
    """Executes a blend crossover on both, the individual and the strategy. The
    individuals shall be a :term:`sequence` and must have a :term:`sequence`
    :attr:`strategy` attribute. Adjustment of the minimal strategy shall be done
    after the call to this function, consider using a decorator.
    :param ind1: The first evolution strategy participating in the crossover.
    :param ind2: The second evolution strategy participating in the crossover.
    :param alpha: Extent of the interval in which the new values can be drawn
                  for each attribute on both side of the parents' attributes.
    :returns: A tuple of two evolution strategies.
    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    
    for i, (x1, s1, x2, s2) in enumerate(zip(ind1, strat1, ind2, strat2)):
        # Blend the individuals
        gamma = (1. + 2. * alpha) * random.random() - alpha
        ind1[i] = (1. - gamma) * x1 + gamma * x2
        ind2[i] = gamma * x1 + (1. - gamma) * x2
        # Blend the strategies
        gamma = (1. + 2. * alpha) * random.random() - alpha
        strat1[i] = (1. - gamma) * s1 + gamma * s2
        strat2[i] = gamma * s1 + (1. - gamma) * s2

    return ind1, ind2, strat1, strat2
    
def cxES2point(ind1, ind2, strat1, strat2):
    """Executes a classical two points crossover on both the individuals and their
    strategy. The individuals /strategies should be a list. The crossover points for the
    individual and the strategy are the same.
    
    Inputs:
        -ind1 (list): The first individual participating in the crossover.
        -ind2 (list): The second individual participating in the crossover.
        -strat1 (list): The first evolution strategy participating in the crossover.
        -strat2 (list): The second evolution strategy participating in the crossover.
    Returns:
        The new ind1, ind2, strat1, strat2 after crossover in list form
    """
    size = min(len(ind1), len(ind2))

    pt1 = random.randint(1, size)
    pt2 = random.randint(1, size - 1)
    if pt2 >= pt1:
        pt2 += 1
    else:  # Swap the two cx points
        pt1, pt2 = pt2, pt1

    ind1[pt1:pt2], ind2[pt1:pt2] = ind2[pt1:pt2], ind1[pt1:pt2]
    strat1[pt1:pt2], strat2[pt1:pt2] = strat2[pt1:pt2], strat1[pt1:pt2]
    
    return ind1, ind2, strat1, strat2

