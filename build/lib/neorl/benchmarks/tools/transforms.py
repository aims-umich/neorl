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

#    Thanks to https://github.com/tilleyd/cec2017-py, this script is modified or adapted from this repo 

import numpy as np
import pickle
import os

print(os.path.join(os.path.dirname(__file__), 'data.pkl'))

with open(os.path.join(os.path.dirname(__file__), 'data.pkl'), 'rb') as _pkl_file:
    _pkl = pickle.load(_pkl_file)

# Each has shape (20, N, N) containing an N-dimensional rotation matrix
# for functions f1 to f20
rotations = {
    2: _pkl['M_D2'],
    10: _pkl['M_D10'],
    20: _pkl['M_D20'],
    30: _pkl['M_D30'],
    50: _pkl['M_D50'],
    100: _pkl['M_D100']
}

# Each has shape (10, 10, N, N) containing 10 N-dimensional rotation matrices
# for functions f21 to f30
rotations_cf = {
    2: _pkl['M_cf_d2'],
    10: _pkl['M_cf_D10'],
    20: _pkl['M_cf_D20'],
    30: _pkl['M_cf_D30'],
    50: _pkl['M_cf_D50'],
    100: _pkl['M_cf_D100']
}

# Shape (20, 100)
# Contains 100-dimension shift vectors for functions f1 to f20
shifts = _pkl['shift']

# Shape (10, 10, 100)
# Contains 10 100-dimension shift vectors for functions f21 to f30
shifts_cf = _pkl['shift_cf']

# Each has shape (10, N) containing N-dimensional permutations for functions f11
# to f20 (note: the original were 1-indexed, these are 0-indexed)
shuffles = {
    10: _pkl['shuffle_D10'],
    30: _pkl['shuffle_D30'],
    50: _pkl['shuffle_D50'],
    100: _pkl['shuffle_D100']
}

# Each has shape (2, 10, N) containing 10 N-dimensional permutations for
# functions f29 and f30 (note: the original were 1-indexed, these are 0-indexed)
shuffles_cf = {
    10: _pkl['shuffle_cf_D10'],
    30: _pkl['shuffle_cf_D30'],
    50: _pkl['shuffle_cf_D50'],
    100: _pkl['shuffle_cf_D100']
}
