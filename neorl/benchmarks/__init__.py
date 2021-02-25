# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:25:37 2020

@author: Majdi
"""

import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce

def sphere(individual):
    """Sphere test objective function.
        F(x) = x^2
        Range:
        Minima:
    """
    return sum(x**2 for x in individual)

def cigar(individual):
    """Cigar test objective function.
         - :math:`f(\mathbf{x}) = x_0^2 + 10^6\\sum_{i=1}^N\,x_i^2`
    """
    return individual[0]**2 + 1e6 * sum(gene * gene for gene in individual)

def rosenbrock(individual):  
    """Rosenbrock test objective function.
    .. list-table:: 
       :widths: 10 50
       :stub-columns: 1
       * - Type
         - minimization
       * - Range
         - none
       * - Global optima
         - :math:`x_i = 1, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
       * - Function
         - :math:`f(\\mathbf{x}) = \\sum_{i=1}^{N-1} (1-x_i)^2 + 100 (x_{i+1} - x_i^2 )^2`
    
    .. plot:: code/benchmarks/rosenbrock.py
       :width: 67 %
    """
    return sum(100 * (x * x - y)**2 + (1. - x)**2 \
                   for x, y in zip(individual[:-1], individual[1:]))

def ackley(individual):
    """Ackley test objective function.
      f(x) = 20 - 20exp(-0.2*sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2}) + 
                      e - \exp(\frac{1}{N}\sum_{i=1}^N \cos(2\pi x_i))
    """
    N = len(individual)
    return 20 - 20 * exp(-0.2*sqrt(1.0/N * sum(x**2 for x in individual))) \
            + e - exp(1.0/N * sum(cos(2*pi*x) for x in individual))
            
def bohachevsky(individual):
    """Bohachevsky test objective function.
    .. list-table:: 
       :widths: 10 50
       :stub-columns: 1
       * - Type
         - minimization
       * - Range
         - :math:`x_i \in [-100, 100]`
       * - Global optima
         - :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
       * - Function
         -  :math:`f(\mathbf{x}) = \sum_{i=1}^{N-1}(x_i^2 + 2x_{i+1}^2 - \
                   0.3\cos(3\pi x_i) - 0.4\cos(4\pi x_{i+1}) + 0.7)`
    
    .. plot:: code/benchmarks/bohachevsky.py
       :width: 67 %
    """
    return sum(x**2 + 2*x1**2 - 0.3*cos(3*pi*x) - 0.4*cos(4*pi*x1) + 0.7 
                for x, x1 in zip(individual[:-1], individual[1:]))

def griewank(individual):
    """Griewank test objective function.
    
    .. list-table:: 
       :widths: 10 50
       :stub-columns: 1
       * - Type
         - minimization
       * - Range
         - :math:`x_i \in [-600, 600]`
       * - Global optima
         - :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
       * - Function
         - :math:`f(\\mathbf{x}) = \\frac{1}{4000}\\sum_{i=1}^N\,x_i^2 - \
                  \prod_{i=1}^N\\cos\\left(\\frac{x_i}{\sqrt{i}}\\right) + 1`
    .. plot:: code/benchmarks/griewank.py
       :width: 67 %
    """
    return 1.0/4000.0 * sum(x**2 for x in individual) - \
        reduce(mul, (cos(x/sqrt(i+1.0)) for i, x in enumerate(individual)), 1) + 1
            
def rastrigin(individual):
    """Rastrigin test objective function.
    .. list-table:: 
       :widths: 10 50
       :stub-columns: 1
       * - Type
         - minimization
       * - Range
         - :math:`x_i \in [-5.12, 5.12]`
       * - Global optima
         - :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
       * - Function
         - :math:`f(\\mathbf{x}) = 10N + \sum_{i=1}^N x_i^2 - 10 \\cos(2\\pi x_i)`
    .. plot:: code/benchmarks/rastrigin.py
       :width: 67 %
    """     
    return 10 * len(individual) + sum(gene * gene - 10 * \
                        cos(2 * pi * gene) for gene in individual)

def rastrigin_scaled(individual):
    """Scaled Rastrigin test objective function.
    
    :math:`f_{\\text{RastScaled}}(\mathbf{x}) = 10N + \sum_{i=1}^N \
        \left(10^{\left(\\frac{i-1}{N-1}\\right)} x_i \\right)^2 x_i)^2 - \
        10\cos\\left(2\\pi 10^{\left(\\frac{i-1}{N-1}\\right)} x_i \\right)`
    """
    N = len(individual)
    return 10*N + sum((10**(i/(N-1))*x)**2 - 
                      10*cos(2*pi*10**(i/(N-1))*x) for i, x in enumerate(individual))

def rastrigin_skew(individual):
    """Skewed Rastrigin test objective function.
    
     :math:`f_{\\text{RastSkew}}(\mathbf{x}) = 10N \sum_{i=1}^N \left(y_i^2 - 10 \\cos(2\\pi x_i)\\right)`
        
     :math:`\\text{with } y_i = \
                            \\begin{cases} \
                                10\\cdot x_i & \\text{ if } x_i > 0,\\\ \
                                x_i & \\text{ otherwise } \
                            \\end{cases}`
    """
    N = len(individual)
    return 10*N + sum((10*x if x > 0 else x)**2 
                    - 10*cos(2*pi*(10*x if x > 0 else x)) for x in individual)
    

def schaffer(individual):
    """Schaffer test objective function.
    
    .. list-table:: 
       :widths: 10 50
       :stub-columns: 1
       * - Type
         - minimization
       * - Range
         - :math:`x_i \in [-100, 100]`
       * - Global optima
         - :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
       * - Function
         -  :math:`f(\mathbf{x}) = \sum_{i=1}^{N-1} (x_i^2+x_{i+1}^2)^{0.25} \cdot \
                  \\left[ \sin^2(50\cdot(x_i^2+x_{i+1}^2)^{0.10}) + 1.0 \
                  \\right]`
    .. plot:: code/benchmarks/schaffer.py
        :width: 67 %
    """
    return sum((x**2+x1**2)**0.25 * ((sin(50*(x**2+x1**2)**0.1))**2+1.0) 
                for x, x1 in zip(individual[:-1], individual[1:]))

def schwefel(individual):
    """Schwefel test objective function.
    .. list-table:: 
       :widths: 10 50
       :stub-columns: 1
       * - Type
         - minimization
       * - Range
         - :math:`x_i \in [-500, 500]`
       * - Global optima
         - :math:`x_i = 420.96874636, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
       * - Function
         - :math:`f(\mathbf{x}) = 418.9828872724339\cdot N - \
            \sum_{i=1}^N\,x_i\sin\\left(\sqrt{|x_i|}\\right)`
    .. plot:: code/benchmarks/schwefel.py
        :width: 67 %
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