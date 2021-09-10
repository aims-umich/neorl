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
#Created on Thu Jul  2 16:25:37 2020
#
#@author: Majdi
#"""

import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce

def sphere(individual):
    """Sphere test objective function.
    """
    return sum(x**2 for x in individual)

def cigar(individual):
    """Cigar test objective function.
    """
    return individual[0]**2 + 1e6 * sum(gene * gene for gene in individual)

def rosenbrock(individual):  
    """Rosenbrock test objective function.
    """
    return sum(100 * (x * x - y)**2 + (1. - x)**2 \
                   for x, y in zip(individual[:-1], individual[1:]))

def ackley(individual):
    """Ackley test objective function.
    """
    N = len(individual)
    return 20 - 20 * exp(-0.2*sqrt(1.0/N * sum(x**2 for x in individual))) \
            + e - exp(1.0/N * sum(cos(2*pi*x) for x in individual))
            
def bohachevsky(individual):
    """Bohachevsky test objective function.
    """
    return sum(x**2 + 2*x1**2 - 0.3*cos(3*pi*x) - 0.4*cos(4*pi*x1) + 0.7 
                for x, x1 in zip(individual[:-1], individual[1:]))

def griewank(individual):
    """Griewank test objective function.
    """
    return 1.0/4000.0 * sum(x**2 for x in individual) - \
        reduce(mul, (cos(x/sqrt(i+1.0)) for i, x in enumerate(individual)), 1) + 1
            
def rastrigin(individual):
    """Rastrigin test objective function.
    """     
    return 10 * len(individual) + sum(gene * gene - 10 * \
                        cos(2 * pi * gene) for gene in individual)

def rastrigin_scaled(individual):
    """Scaled Rastrigin test objective function.
    """
    N = len(individual)
    return 10*N + sum((10**(i/(N-1))*x)**2 - 
                      10*cos(2*pi*10**(i/(N-1))*x) for i, x in enumerate(individual))

def rastrigin_skew(individual):
    """Skewed Rastrigin test objective function.
    """
    N = len(individual)
    return 10*N + sum((10*x if x > 0 else x)**2 
                    - 10*cos(2*pi*(10*x if x > 0 else x)) for x in individual)
    

def schaffer(individual):
    """Schaffer test objective function.
    """
    return sum((x**2+x1**2)**0.25 * ((sin(50*(x**2+x1**2)**0.1))**2+1.0) 
                for x, x1 in zip(individual[:-1], individual[1:]))

def schwefel(individual):
    """Schwefel test objective function.
    """    
    N = len(individual)
    return 418.9828872724339*N-sum(x*sin(sqrt(abs(x))) for x in individual)

def alpinen1(individual): 
    return sum(abs(x * sin(x) + 0.1 * x) for x in individual)

def alpinen2(individual):
    return reduce(mul, (sqrt(x)*sin(x) for x in individual), 1)

def brown(individual):
    scores=0
    x=[item**2 for item in individual]
    for i in range(len(individual) - 1):
        scores += x[i] ** (x[i+1] + 1) + x[i+1]**(x[i] + 1)    
    return scores

def expo(individual):
    return -exp(-0.5*sum(x**2 for x in individual))

def yang(individual):
    return sum(random.random()*abs(x)**i for i, x in enumerate(individual,start=1))

def yang2(individual):
    return sum(abs(x) for x in individual) * exp(-sum(sin(x**2) for x in individual))

def yang3(individual):
    beta=15
    m=5
    return exp(-sum((x/beta)**(2*m) for x in individual))  \
            - 2*exp(-sum(x**2 for x in individual)) * reduce(mul, (cos(x)**2 for x in individual), 1)

def yang4(individual):
    return (sum(sin(x)**2 for x in individual) - exp(-sum(x**2 for x in individual))) \
            * exp(-sum(sin(sqrt(abs(x)))**2 for x in individual))
            
def zakharov (individual):
    return sum(x**2 for x in individual) + sum(0.5*i*x for i, x in enumerate(individual,start=1))**2 \
            + sum(0.5*i*x for i, x in enumerate(individual,start=1))**4
            
def salomon (individual):
    return 1-cos(2*pi*sqrt(sum(x**2 for x in individual))) + 0.1*sqrt(sum(x**2 for x in individual))

def st(individual):
    return 0.5*sum(x**4 - 16*x**2 + 5*x for x in individual)

def shubert(individual):  #2d function, global minima at -186.7309
    score=1
    for i in range(len(individual)):
        score*= sum(x*cos((x+1)*individual[i] + x) for x in [1,2,3,4,5])
    return score

def ridge(individual): 
    d=1
    alpha=0.5
    return individual[0] + d*sum(individual[i]**2 for i in range(1,len(individual)))**alpha

def powell(individual):
    return sum(abs(x)**(i+1) for i,x in enumerate (individual,start=1))

#def periodic(individual):
#    return 1+sum(sin(x)**2 for x in individual)-0.1*exp(sum(x**2 for x in individual))

def qing(individual):
    return sum((x**2 - i)**2 for i,x in enumerate (individual,start=1))

def quartic(individual):
    return sum(i*x**4 for i,x in enumerate (individual,start=1)) + random.random()

def happycat(individual):
    l2=sum(x**2 for x in individual)
    n=len(individual)
    alpha=1/8
    return ((l2-n)**2)**alpha + 1/n*(0.5*l2 + sum(individual)) + 0.5

def schwefel2(individual):
    return sum(abs(x) for x in individual) + reduce(mul, (abs(x) for x in individual), 1)

def dixonprice(individual):
    score=0
    for i in range(1,len(individual)):
         score+= (i+1)*(2*individual[i]**2-individual[i-1])**2
    score += (individual[0]-1)**2
    return  score

def levy(individual):

    d = len(individual)
    
    w=[1 + (x - 1)/4 for x in individual]
    
    term1 = sin(pi*w[0])**2
    term3 = (w[-1]-1)**2 * (1+sin(2*pi*w[-1])**2)
    
    term2 = 0;
    for i in range (d-1):
        term2 += (w[i]-1)**2 * (1+10*sin(pi*w[i]+1)**2)
        
    return term1 + term2 + term3

all_functions = [
    sphere,
    cigar,
    rosenbrock,
    bohachevsky,
    griewank,
    rastrigin,
    ackley,
    rastrigin_scaled,
    rastrigin_skew,
    schaffer,
    schwefel,
    schwefel2,
    alpinen1,
    alpinen2,
    brown,
    expo,
    yang,
    yang2,
    yang3,
    yang4,
    zakharov,
    salomon,
    st,
    shubert,
    ridge,
    powell,
    qing,
    quartic,
    happycat,
    dixonprice,
    levy
]