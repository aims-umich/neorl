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

from neorl.benchmarks.tools import basic
from neorl.benchmarks.tools import transforms

import numpy as np

def f1(x, rotation=None, shift=None):
    """
    Shifted and Rotated Bent Cigar Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][0]
    if shift is None:
        shift = transforms.shifts[0][:nx]
    x_transformed = np.matmul(rotation, x - shift)
    return basic.bent_cigar(x_transformed) + 100.0

def f2(x, rotation=None, shift=None):
    """
    (Deprecated) Shifted and Rotated Sum of Different Power Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    #if 'warned' not in f2.__dict__:
    #    f2.warned = True
    #    print('WARNING: f2 has been deprecated from the CEC 2017 benchmark suite')

    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][1]
    if shift is None:
        shift = transforms.shifts[1][:nx]
    x_transformed = np.matmul(rotation, x - shift)
    return basic.sum_diff_pow(x_transformed) + 200.0

def f3(x, rotation=None, shift=None):
    """
    Shifted and Rotated Zakharov Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][2]
    if shift is None:
        shift = transforms.shifts[2][:nx]
    x_transformed = np.matmul(rotation, x - shift)
    return basic.zakharov(x_transformed) + 300.0

def f4(x, rotation=None, shift=None):
    """
    Shifted and Rotated Rosenbrock’s Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][3]
    if shift is None:
        shift = transforms.shifts[3][:nx]
    x_transformed = np.matmul(rotation, (x - shift))
    return basic.rosenbrock(x_transformed) + 400.0

def f5(x, rotation=None, shift=None):
    """
    Shifted and Rotated Rastrigin's Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][4]
    if shift is None:
        shift = transforms.shifts[4][:nx]
    x_transformed = np.matmul(rotation, (x - shift))
    return basic.rastrigin(x_transformed) + 500.0

def f6(x, rotation=None, shift=None):
    """
    Shifted and Rotated Schaffer’s F7 Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][5]
    if shift is None:
        shift = transforms.shifts[5][:nx]
    x_transformed = np.matmul(rotation, (x - shift))
    return basic.schaffers_f7(x_transformed) + 600.0

def f7(x, rotation=None, shift=None):
    """
    Shifted and Rotated Lunacek Bi-Rastrigin’s Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][6]
    if shift is None:
        shift = transforms.shifts[6][:nx]
    # pass the shift and rotation directly to the function
    return basic.lunacek_bi_rastrigin(x, shift, rotation) + 700.0

def f8(x, rotation=None, shift=None):
    """
    Shifted and Rotated Non-Continuous Rastrigin’s Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][7]
    if shift is None:
        shift = transforms.shifts[7][:nx]
    # pass the shift and rotation directly to the function
    return basic.non_cont_rastrigin(x, shift, rotation) + 800.0

def f9(x, rotation=None, shift=None):
    """
    Shifted and Rotated Levy Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][8]
    if shift is None:
        shift = transforms.shifts[8][:nx]
    x_transformed = np.matmul(rotation, (x - shift))
    return basic.levy(x_transformed) + 900.0

def f10(x, rotation=None, shift=None):
    """
    Shifted and Rotated Schwefel’s Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][9]
    if shift is None:
        shift = transforms.shifts[9][:nx]
    x_transformed = np.matmul(rotation, (x - shift))
    return basic.modified_schwefel(x_transformed) + 1000.0

def _shuffle_and_partition(x, shuffle, partitions):
    """
    First applies the given permutation, then splits x into partitions given
    the percentages.

    Args:
        x (array): Input vector.
        shuffle (array): Shuffle vector.
        partitions (list): List of percentages. Assumed to add up to 1.0.

    Returns:
        (list of arrays): The partitions of x after shuffling.
    """
    nx = len(x)
    # shuffle
    xs = np.zeros(x.shape)
    for i in range(0, nx):
        xs[i] = x[shuffle[i]]
    # and partition
    parts = []
    start, end = 0, 0
    for p in partitions[:-1]:
        end = start + int(np.ceil(p * nx))
        parts.append(xs[start:end])
        start = end
    parts.append(xs[end:])
    return parts

def f11(x, rotation=None, shift=None, shuffle=None):
    """
    Hybrid Function 1 (N=3)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
        shuffle (array): Optionbal shuffle vector. If None (default), the
            official permutation vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][10]
    if shift is None:
        shift = transforms.shifts[10][:nx]
    if shuffle is None:
        shuffle = transforms.shuffles[nx][0]

    x_transformed = np.matmul(rotation, x - shift)
    x_parts = _shuffle_and_partition(x_transformed, shuffle, [0.2, 0.4, 0.4])

    y = basic.zakharov(x_parts[0])
    y += basic.rosenbrock(x_parts[1])
    y += basic.rastrigin(x_parts[2])
    return y + 1100.0

def f12(x, rotation=None, shift=None, shuffle=None):
    """
    Hybrid Function 2 (N=3)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
        shuffle (array): Optionbal shuffle vector. If None (default), the
            official permutation vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][11]
    if shift is None:
        shift = transforms.shifts[11][:nx]
    if shuffle is None:
        shuffle = transforms.shuffles[nx][1]

    x_transformed = np.matmul(rotation, x - shift)
    x_parts = _shuffle_and_partition(x_transformed, shuffle, [0.3, 0.3, 0.4])

    y = basic.high_conditioned_elliptic(x_parts[0])
    y += basic.modified_schwefel(x_parts[1])
    y += basic.bent_cigar(x_parts[2])
    return y + 1200.0

def f13(x, rotation=None, shift=None, shuffle=None):
    """
    Hybrid Function 3 (N=3)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
        shuffle (array): Optionbal shuffle vector. If None (default), the
            official permutation vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][12]
    if shift is None:
        shift = transforms.shifts[12][:nx]
    if shuffle is None:
        shuffle = transforms.shuffles[nx][2]

    x_transformed = np.matmul(rotation, x - shift)
    x_parts = _shuffle_and_partition(x_transformed, shuffle, [0.3, 0.3, 0.4])

    y = basic.bent_cigar(x_parts[0])
    y += basic.rosenbrock(x_parts[1])
    y += basic.lunacek_bi_rastrigin(x_parts[2])
    return y + 1300.0

def f14(x, rotation=None, shift=None, shuffle=None):
    """
    Hybrid Function 4 (N=4)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
        shuffle (array): Optionbal shuffle vector. If None (default), the
            official permutation vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][13]
    if shift is None:
        shift = transforms.shifts[13][:nx]
    if shuffle is None:
        shuffle = transforms.shuffles[nx][3]

    x_transformed = np.matmul(rotation, x - shift)
    x_parts = _shuffle_and_partition(x_transformed, shuffle, [0.2, 0.2, 0.2, 0.4])

    y = basic.high_conditioned_elliptic(x_parts[0])
    y += basic.ackley(x_parts[1])
    y += basic.schaffers_f7(x_parts[2])
    y += basic.rastrigin(x_parts[3])
    return y + 1400.0

def f15(x, rotation=None, shift=None, shuffle=None):
    """
    Hybrid Function 5 (N=4)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
        shuffle (array): Optionbal shuffle vector. If None (default), the
            official permutation vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][14]
    if shift is None:
        shift = transforms.shifts[14][:nx]
    if shuffle is None:
        shuffle = transforms.shuffles[nx][4]

    x_transformed = np.matmul(rotation, x - shift)
    x_parts = _shuffle_and_partition(x_transformed, shuffle, [0.2, 0.2, 0.3, 0.3])

    y = basic.bent_cigar(x_parts[0])
    y += basic.h_g_bat(x_parts[1])
    y += basic.rastrigin(x_parts[2])
    y += basic.rosenbrock(x_parts[3])
    return y + 1500.0

def f16(x, rotation=None, shift=None, shuffle=None):
    """
    Hybrid Function 6 (N=4)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
        shuffle (array): Optionbal shuffle vector. If None (default), the
            official permutation vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][15]
    if shift is None:
        shift = transforms.shifts[15][:nx]
    if shuffle is None:
        shuffle = transforms.shuffles[nx][5]

    x_transformed = np.matmul(rotation, x - shift)
    x_parts = _shuffle_and_partition(x_transformed, shuffle, [0.2, 0.2, 0.3, 0.3])

    y = basic.expanded_schaffers_f6(x_parts[0])
    y += basic.h_g_bat(x_parts[1])
    y += basic.rosenbrock(x_parts[2])
    y += basic.modified_schwefel(x_parts[3])
    return y + 1600.0

def f17(x, rotation=None, shift=None, shuffle=None):
    """
    Hybrid Function 7 (N=5)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
        shuffle (array): Optionbal shuffle vector. If None (default), the
            official permutation vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][16]
    if shift is None:
        shift = transforms.shifts[16][:nx]
    if shuffle is None:
        shuffle = transforms.shuffles[nx][6]

    x_transformed = np.matmul(rotation, x - shift)
    x_parts = _shuffle_and_partition(x_transformed, shuffle, [0.1, 0.2, 0.2, 0.2, 0.3])

    y = basic.katsuura(x_parts[0])
    y += basic.ackley(x_parts[1])
    y += basic.expanded_griewanks_plus_rosenbrock(x_parts[2])
    y += basic.modified_schwefel(x_parts[3])
    y += basic.rastrigin(x_parts[4])
    return y + 1700.0

def f18(x, rotation=None, shift=None, shuffle=None):
    """
    Hybrid Function 8 (N=5)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
        shuffle (array): Optionbal shuffle vector. If None (default), the
            official permutation vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][17]
    if shift is None:
        shift = transforms.shifts[17][:nx]
    if shuffle is None:
        shuffle = transforms.shuffles[nx][7]

    x_transformed = np.matmul(rotation, x - shift)
    x_parts = _shuffle_and_partition(x_transformed, shuffle, [0.2, 0.2, 0.2, 0.2, 0.2])

    y = basic.high_conditioned_elliptic(x_parts[0])
    y += basic.ackley(x_parts[1])
    y += basic.rastrigin(x_parts[2])
    y += basic.h_g_bat(x_parts[3])
    y += basic.discus(x_parts[4])
    return y + 1800.0

def f19(x, rotation=None, shift=None, shuffle=None):
    """
    Hybrid Function 9 (N=5)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
        shuffle (array): Optionbal shuffle vector. If None (default), the
            official permutation vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][18]
    if shift is None:
        shift = transforms.shifts[18][:nx]
    if shuffle is None:
        shuffle = transforms.shuffles[nx][8]

    x_transformed = np.matmul(rotation, x - shift)
    x_parts = _shuffle_and_partition(x_transformed, shuffle, [0.2, 0.2, 0.2, 0.2, 0.2])

    y = basic.bent_cigar(x_parts[0])
    y += basic.rastrigin(x_parts[1])
    y += basic.expanded_griewanks_plus_rosenbrock(x_parts[2])
    y += basic.weierstrass(x_parts[3])
    y += basic.expanded_schaffers_f6(x_parts[4])
    return y + 1900.0

def f20(x, rotation=None, shift=None, shuffle=None):
    """
    Hybrid Function 10 (N=6)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
        shuffle (array): Optionbal shuffle vector. If None (default), the
            official permutation vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][19]
    if shift is None:
        shift = transforms.shifts[19][:nx]
    if shuffle is None:
        shuffle = transforms.shuffles[nx][9]

    x_transformed = np.matmul(rotation, x - shift)
    x_parts = _shuffle_and_partition(x_transformed, shuffle, [0.1, 0.1, 0.2, 0.2, 0.2, 0.2])

    y = basic.happy_cat(x_parts[0])
    y += basic.katsuura(x_parts[1])
    y += basic.ackley(x_parts[2])
    y += basic.rastrigin(x_parts[3])
    y += basic.modified_schwefel(x_parts[4])
    y += basic.schaffers_f7(x_parts[5])
    return y + 2000.0

def _calc_w(x, sigma):
    nx = len(x)
    w = 0
    for i in range(0, nx):
        w += x[i]*x[i]
    if (w != 0):
        w = ((1.0/w)**0.5) * np.exp(-w / (2.0*nx*sigma*sigma))
    else:
        w = float('inf')
    return w

def f21(x, rotations=None, shifts=None):
    """
    Composition Function 1 (N=3)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][0]
    if shifts is None:
        shifts = transforms.shifts_cf[0]

    N = 3
    funcs = [basic.rosenbrock, basic.high_conditioned_elliptic, basic.rastrigin]
    sigmas = np.array([10.0, 20.0, 30.0])
    lambdas = np.array([1.0, 1.0e-6, 1.0])
    biases = np.array([0.0, 100.0, 200.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2100

def f22(x, rotations=None, shifts=None):
    """
    Composition Function 2 (N=3)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][1]
    if shifts is None:
        shifts = transforms.shifts_cf[1]

    N = 3
    funcs = [basic.rastrigin, basic.griewank, basic.modified_schwefel]
    sigmas = np.array([10.0, 20.0, 30.0])
    lambdas = np.array([1.0, 10.0, 1.0])
    biases = np.array([0.0, 100.0, 200.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2200

def f23(x, rotations=None, shifts=None):
    """
    Composition Function 3 (N=4)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][2]
    if shifts is None:
        shifts = transforms.shifts_cf[2]

    N = 4
    funcs = [basic.rosenbrock, basic.ackley, basic.modified_schwefel, basic.rastrigin]
    sigmas = np.array([10.0, 20.0, 30.0, 40.0])
    lambdas = np.array([1.0, 10.0, 1.0, 1.0])
    biases = np.array([0.0, 100.0, 200.0, 300.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2300

def f24(x, rotations=None, shifts=None):
    """
    Composition Function 4 (N=4)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][3]
    if shifts is None:
        shifts = transforms.shifts_cf[3]

    N = 4
    funcs = [basic.ackley, basic.high_conditioned_elliptic, basic.griewank, basic.rastrigin]
    sigmas = np.array([10.0, 20.0, 30.0, 40.0])
    lambdas = np.array([1.0, 1.0e-6, 10.0, 1.0])
    biases = np.array([0.0, 100.0, 200.0, 300.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2400

def f25(x, rotations=None, shifts=None):
    """
    Composition Function 5 (N=5)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][4]
    if shifts is None:
        shifts = transforms.shifts_cf[4]

    N = 5
    funcs = [basic.rastrigin, basic.happy_cat, basic.ackley, basic.discus, basic.rosenbrock]
    sigmas = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    lambdas = np.array([10.0, 1.0, 10.0, 1.0e-6, 1.0])
    biases = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2500

def f26(x, rotations=None, shifts=None):
    """
    Composition Function 6 (N=5)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][5]
    if shifts is None:
        shifts = transforms.shifts_cf[5]

    N = 5
    funcs = [basic.expanded_schaffers_f6, basic.modified_schwefel, basic.griewank, basic.rosenbrock, basic.rastrigin]
    sigmas = np.array([10.0, 20.0, 20.0, 30.0, 40.0])
    # Note: the lambdas specified in the problem definitions (below) differ from what is used in the code
    #lambdas = np.array([1.0e-26, 10.0, 1.0e-6, 10.0, 5.0e-4])
    lambdas = np.array([5.0e-4, 1.0, 10.0, 1.0, 10.0])
    biases = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2600

def f27(x, rotations=None, shifts=None):
    """
    Composition Function 7 (N=6)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][6]
    if shifts is None:
        shifts = transforms.shifts_cf[6]

    N = 6
    funcs = [
        basic.h_g_bat,
        basic.rastrigin,
        basic.modified_schwefel,
        basic.bent_cigar,
        basic.high_conditioned_elliptic,
        basic.expanded_schaffers_f6]
    sigmas = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    lambdas = np.array([10.0, 10.0, 2.5, 1.0e-26, 1.0e-6, 5.0e-4])
    biases = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2700

def f28(x, rotations=None, shifts=None):
    """
    Composition Function 8 (N=6)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][7]
    if shifts is None:
        shifts = transforms.shifts_cf[7]

    N = 6
    funcs = [
        basic.ackley,
        basic.griewank,
        basic.discus,
        basic.rosenbrock,
        basic.happy_cat,
        basic.expanded_schaffers_f6]
    sigmas = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    lambdas = np.array([10.0, 10.0, 1.0e-6, 1.0, 1.0, 5.0e-4])
    biases = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](np.matmul(rotations[i], x_shifted))
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (lambdas*vals + biases)) + 2800

def f29(x, rotations=None, shifts=None, shuffles=None):
    """
    Composition Function 9 (N=3)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
        shuffles (array): Optional shuffle vectors (NxD). If None (default), the
            official permutation vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][8]
    if shifts is None:
        shifts = transforms.shifts_cf[8]
    if shuffles is None:
        shuffles = transforms.shuffles_cf[nx][0]

    N = 3
    funcs = [f15, f16, f17]
    sigmas = np.array([10.0, 30.0, 50.0])
    biases = np.array([0.0, 100.0, 200.0])
    offsets = np.array([1500, 1600, 1700]) # subtract F* added at the end of the functions
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](x, rotation=rotations[i], shift=shifts[i][:nx], shuffle=shuffles[i])
        vals[i] -= offsets[i]
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (vals + biases)) + 2900

def f30(x, rotations=None, shifts=None, shuffles=None):
    """
    Composition Function 10 (N=3)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotations (matrix): Optional rotation matrices (NxDxD). If None
            (default), the official matrices from the benchmark suite will be
            used.
        shifts (array): Optional shift vectors (NxD). If None (default), the
            official vectors from the benchmark suite will be used.
        shuffles (array): Optional shuffle vectors (NxD). If None (default), the
            official permutation vectors from the benchmark suite will be used.
    """
    nx = len(x)
    if rotations is None:
        rotations = transforms.rotations_cf[nx][9]
    if shifts is None:
        shifts = transforms.shifts_cf[9]
    if shuffles is None:
        shuffles = transforms.shuffles_cf[nx][1]

    N = 3
    funcs = [f15, f18, f19]
    sigmas = np.array([10.0, 30.0, 50.0])
    biases = np.array([0.0, 100.0, 200.0])
    offsets = np.array([1500, 1800, 1900]) # subtract F* added at the end of the functions
    vals = np.zeros(N)
    w = np.zeros(N)
    w_sm = 0.0
    for i in range(0, N):
        x_shifted = x-shifts[i][:nx]
        vals[i] = funcs[i](x, rotation=rotations[i], shift=shifts[i][:nx], shuffle=shuffles[i])
        vals[i] -= offsets[i]
        w[i] = _calc_w(x_shifted, sigmas[i])
        w_sm += w[i]

    if (w_sm != 0.0):
        w /= w_sm
    else:
        w = np.full(N, 1/N)

    return np.sum(w * (vals + biases)) + 3000

all_functions = [
    f1,
    f2,
    f3,
    f4,
    f5,
    f6,
    f7,
    f8,
    f9,
    f10,
    f11,
    f12,
    f13,
    f14,
    f15,
    f16,
    f17,
    f18,
    f19,
    f20,
    f21,
    f22,
    f23,
    f24,
    f25,
    f26,
    f27,
    f28,
    f29,
    f30
]
