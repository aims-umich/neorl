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
#Created on Sun Jul 25 20:51:03 2021
#
#@author: Majdi
#"""

#--------------------------------------------------------
# This class is inspired from the work by this repo:
# https://github.com/thieunguyen5991  
#--------------------------------------------------------

from pandas import read_csv
from numpy import sum, max, abs, cos, pi, sqrt, exp, e
from numpy.random import seed, choice, uniform
import os 


class Root:
    def __init__(self, f_name=None, f_shift_data_file=None, f_bias=None):
        self.f_name = f_name
        self.f_shift_data_file = f_shift_data_file
        self.f_bias = f_bias
        self.support_path_data=os.path.join(os.path.dirname(__file__), 'tools/data08.csv')
        
    def load_shift_data(self):
        data = read_csv(self.support_path_data, usecols=[self.f_shift_data_file])
        return data.values.reshape(-1)


class F1(Root):
    def __init__(self, f_name="Shifted Sphere Function", f_shift_data_file="F1", f_bias=-450):
        Root.__init__(self, f_name, f_shift_data_file, f_bias)

    def fit(self, solution=None):
        problem_size = len(solution)
        if problem_size > 1000:
            print("CEC 2008 not support for problem size > 1000")
            return 1
        shift_data = self.load_shift_data()[:problem_size]
        return sum((solution - shift_data)**2) + self.f_bias
    
    def return_global(self, nx):
        
        xmin = self.load_shift_data()[:nx]
        ymin = self.fit(xmin)
        
        return xmin, ymin
        
    
    

class F2(Root):
    def __init__(self, f_name="Schwefel’s Problem 2.21", f_shift_data_file="F2", f_bias=-450):
        Root.__init__(self, f_name, f_shift_data_file, f_bias)

    def fit(self, solution=None):
        problem_size = len(solution)
        if problem_size > 1000:
            print("CEC 2008 not support for problem size > 1000")
            return 1
        shift_data = self.load_shift_data()[:problem_size]

        return max(abs(solution - shift_data)) + self.f_bias

    def return_global(self, nx):
        
        xmin = self.load_shift_data()[:nx]
        ymin = self.fit(xmin)
        
        return xmin, ymin
    
class F3(Root):
    def __init__(self, f_name="Shifted Rosenbrock’s Function", f_shift_data_file="F3", f_bias=390, f_matrix=None):
        Root.__init__(self, f_name, f_shift_data_file, f_bias)
        self.f_matrix = f_matrix

    def fit(self, solution=None):
        problem_size = len(solution)
        if problem_size > 1000:
            print("CEC 2008 not support for problem size > 1000")
            return 1
        shift_data = self.load_shift_data()[:problem_size]
        z = solution - shift_data + 1
        result = 0
        for i in range(0, problem_size-1):
            result += 100*(z[i]**2 - z[i+1])**2 + (z[i] - 1)**2
        return result + self.f_bias

    def return_global(self, nx):
        
        xmin = self.load_shift_data()[:nx]
        ymin = self.fit(xmin)
        
        return xmin, ymin
    
class F4(Root):
    def __init__(self, f_name="Shifted Rastrigin’s Function", f_shift_data_file="F4", f_bias=-330):
        Root.__init__(self, f_name, f_shift_data_file, f_bias)

    def fit(self, solution=None):
        problem_size = len(solution)
        if problem_size > 1000:
            print("CEC 2008 not support for problem size > 1000")
            return 1
        shift_data = self.load_shift_data()[:problem_size]
        z = solution - shift_data
        return sum(z**2 - 10*cos(2*pi*z) + 10) + self.f_bias

    def return_global(self, nx):
        
        xmin = self.load_shift_data()[:nx]
        ymin = self.fit(xmin)
        
        return xmin, ymin   

class F5(Root):
    def __init__(self, f_name="Shifted Griewank’s Function", f_shift_data_file="F5", f_bias=-180):
        Root.__init__(self, f_name, f_shift_data_file, f_bias)

    def fit(self, solution=None):
        problem_size = len(solution)
        if problem_size > 1000:
            print("CEC 2008 not support for problem size > 1000")
            return 1
        shift_data = self.load_shift_data()[:problem_size]
        z = solution - shift_data
        result = sum(z**2/4000)
        temp = 1.0
        for i in range(0, problem_size):
            temp *= cos(z[i] / sqrt(i+1))
        return result - temp + 1 + self.f_bias

    def return_global(self, nx):
        
        xmin = self.load_shift_data()[:nx]
        ymin = self.fit(xmin)
        
        return xmin, ymin
    
class F6(Root):
    def __init__(self, f_name="Shifted Ackley’s Function", f_shift_data_file="F6", f_bias=-140):
        Root.__init__(self, f_name, f_shift_data_file, f_bias)

    def fit(self, solution=None):
        problem_size = len(solution)
        if problem_size > 1000:
            print("CEC 2008 not support for problem size > 1000")
            return 1
        shift_data = self.load_shift_data()[:problem_size]
        z = solution - shift_data
        return -20*exp(-0.2*sqrt(sum(z**2)/problem_size)) - exp(sum(cos(2*pi*z))/problem_size) + 20 + e + self.f_bias
    
    def return_global(self, nx):
        
        xmin = self.load_shift_data()[:nx]
        ymin = self.fit(xmin)
        
        return xmin, ymin
    
class F7(Root):
    def __init__(self, f_name="FastFractal “DoubleDip” Function", f_shift_data_file=None, f_bias=None, f_matrix=None):
        Root.__init__(self, f_name, f_shift_data_file, f_bias)
        self.f_matrix = f_matrix

    def fit(self, solution=None):
        seed(0)
        problem_size = len(solution)
        if problem_size > 1000:
            print("CEC 2008 not support for problem size > 1000")
            return 1

        def __doubledip__(x, c, s):
            if -0.5 < x < 0.5:
                return (-6144*(x - c)**6 + 3088*(x - c)**4 - 392*(x - c)**2 + 1)*s
            else:
                return 0

        def __fractal1d__(x):
            result1 = 0.0
            for k in range(1, 4):
                result2 = 0.0
                upper = 2**(k-1)
                for t in range(1, upper):
                    selected = choice([0, 1, 2])
                    result2 += sum([ __doubledip__(x, uniform(), 1.0 / (2**(k-1)*(2-uniform()))) for _ in range(0, selected)])
                result1 += result2
            return result1

        def __twist__(y):
            return 4*(y**4 - 2*y**3 + y**2)

        result = solution[-1] + __twist__(solution[0])
        for i in range(0, problem_size-1):
            x = solution[i] + __twist__(solution[i%problem_size + 1])
            result += __fractal1d__(x)
        return result