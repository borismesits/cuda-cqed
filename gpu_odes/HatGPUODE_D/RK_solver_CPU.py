import cupy as cp
import sympy as sp
from numpy import * # we have to do a star import here to use eval() on arbitary equations of motion provided by user
from tqdm import tqdm
import numpy as np

def RK_loop_CPU(M, x0, tlist, numpy_kernel):
    '''
    This is for solving single-variation simulations on the CPU (which is faster in this case).
    M is the number of modes
    '''

    print('Running CPU quick solve...')

    dt = tlist[1]-tlist[0]

    x = zeros([M, len(tlist)])

    x[:,0] = x0

    for i in range(0, len(tlist) - 1):

        k1 = np.array(numpy_kernel(tlist[i], * x[:, i]))
        k2 = np.array(numpy_kernel(tlist[i] + dt/2, * (x[:, i] + k1 * dt / 2)))
        k3 = np.array(numpy_kernel(tlist[i] + dt/2, * (x[:, i] + k2 * dt / 2)))
        k4 = np.array(numpy_kernel(tlist[i] + dt, * (x[:, i] + k3 * dt)))

        x[:,i+1] = x[:,i] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    print('...finished CPU quick solve!')

    return x


