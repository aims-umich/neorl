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

# ------------------------------------------------------------------------
# This code is taken from the following paper:
#
# Pengfei Huang,Handing Wang,Yaochu Jin,Offine Data-Driven Evolutionary Optimization Based on Tri-Training, Swarm and Evolutionary Computation, Accepted.
#
# You are free to use it for non-commercial purposes. However, we do not offer any forms of guanrantee or warranty associated with the code. We would appreciate your acknowledgement.
# ------------------------------------------------------------------------

import numpy as np
import random
from neorl.hybrid.ngacore.Latin import latin
import warnings
warnings.filterwarnings("ignore")
class GA():
    def __init__(self, pop_size, dimension, lower_bound, upper_bound):
        self.pop_size = pop_size
        self.max_value = upper_bound
        self.min_value = lower_bound
        self.chrom_length = dimension
        self.pc = 1
        self.pm = 1 / dimension
        self.nc = 15
        self.nm = 15
        self.first = []
        self.pop = np.zeros((self.pop_size, self.chrom_length))

    def init_Population(self):
        self.pop = latin(self.pop_size, self.chrom_length, self.min_value, self.max_value)

    def selection(self,fit_value):
        data = np.column_stack((self.pop, fit_value))
        data = data[np.lexsort(data.T)]
        self.pop = data[:self.pop_size, :self.chrom_length]
        self.first.append(self.pop[0])
        return self.pop

    def crossover(self, pc):
        pop_len = len(self.pop)
        for i in range(pop_len):
            if (random.random() < pc):
                populationList = list(range(pop_len))
                populationList.pop(i)
                p2 = random.choice(populationList)
                par1 = self.pop[i]
                par2 = self.pop[p2]
                a = np.row_stack((par1, par2))
                p1 = a.min(axis=0)
                p2 = a.max(axis=0)
                alfa1 = np.array([0.0] * self.chrom_length)
                for ad in range(self.chrom_length):
                    if p1[ad] - self.min_value[ad] > self.max_value[ad] - p2[ad]:
                        alfa1[ad] = 1 + 2 * (self.max_value[ad] - p2[ad]) / (p2[ad] - p1[ad])
                    else:
                        alfa1[ad] = 1 + 2 * (p1[ad] - self.min_value[ad]) / (p2[ad] - p1[ad])
                alfa1 = 1 / alfa1
                alfa1 = 2 - (alfa1 ** (self.nc + 1))
                alfa2 = alfa1
                alfa1 = np.where(np.isnan(alfa1), 1, alfa1)
                alfa2 = np.where(np.isnan(alfa2), 1, alfa2)
                randList = np.random.random(self.chrom_length)
                orTF = (randList <= 1 / alfa1)
                aq = np.array([0.0] * self.chrom_length)
                for j in range(self.chrom_length):
                    if orTF[j] == True:
                        aq[j] = (randList[j] * alfa1[j]) ** (1.0 / (self.nc + 1))
                    else:
                        expp = 2.0 - (randList[j] * alfa1[j])
                        aq[j] = (1.0 / expp) ** (1.0 / (self.nc + 1))

                randList = np.random.random(self.chrom_length)
                orTF = (randList <= 1 / alfa2)
                bq = np.array([0.0] * self.chrom_length)
                for j in range(self.chrom_length):
                    if orTF[j] == True:
                        bq[j] = (randList[j] * alfa2[j]) ** (1.0 / (self.nc + 1))
                    else:
                        expp = 2.0 - (randList[j] * alfa2[j])
                        bq[j] = (1.0 / expp) ** (1.0 / (self.nc + 1))
                tp1 = 0.5 * ((1 + aq) * p1 + (1 - aq) * p2)
                tp2 = 0.5 * ((1 - bq) * p1 + (1 + bq) * p2)
                tp1 = np.max(np.vstack((tp1, np.array([self.min_value] * self.chrom_length))), 0)
                tp1 = np.min(np.vstack((tp1, np.array([self.max_value] * self.chrom_length))), 0)
                tp2 = np.max(np.vstack((tp2, np.array([self.min_value] * self.chrom_length))), 0)
                tp2 = np.min(np.vstack((tp2, np.array([self.max_value] * self.chrom_length))), 0)
                self.pop = np.row_stack((self.pop, tp1, tp2))

    def mutation(self, pm):
        for i in range(self.pop_size):
            self.pop = np.row_stack((self.pop, self.pop[i]))
            for j in range(self.chrom_length):
                if (random.random() < pm):
                    mpoint = self.pop[i][j]
                    if (mpoint - self.min_value[j]) < (self.max_value[j] - mpoint):
                        temp = 1 - (mpoint - self.min_value[j]) / (self.max_value[j] - self.min_value[j])
                    else:
                        temp = 1 - (self.max_value[j] - mpoint) / (self.max_value[j] - self.min_value[j])
                    u = random.random()
                    if u <= 0.5:
                        delta = ((2 * u + (1 - 2 * u) * (temp ** (self.nm + 1))) ** (1 / (self.nm + 1))) - 1
                    else:
                        delta = 1 - ((2 * (1 - u) + (2 * u - 1) * (temp ** (self.nm + 1))) ** (1 / (self.nm + 1)))
                    self.pop[i][j] = mpoint + delta * (self.max_value[j] - self.min_value[j])
