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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#"""
#Created on Wed Feb 26 09:33:45 2020
#
#@author: majdi
#"""

import numpy as np
import random

def set_neorl_seed(seed=None):
    """
    This function seeds the NEORL algorthim for reproducibility
    :param seed: (int) value of random seed
    """
    
    if not (seed is None):
        random.seed(seed) 
        np.random.seed(seed)