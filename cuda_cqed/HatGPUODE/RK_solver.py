import cupy as cp
import sympy as sp
import numpy as np
from tqdm import tqdm


def RK_loop(x, dt, kernel_op, idxs, S, num_drive_terms):
    '''
    Implements a GPU-accelerated RK4 method for simulating systems of ODEs, with built-in decimation of the
    data to reduce amount of saved information.

    N represents number of modes, M is number of variations (defined outside this function)

    Similar to the RK_loop function of the non-decimated HatGPUODE directory

    '''

    print('Running GPU solve...')

    N, M = np.shape(cp.asnumpy(x))

    x_d = cp.zeros([N, M, S])

    t_d = cp.zeros([M, S])

    for i in tqdm(range(0, S), colour="BLUE"):

        ti = dt*i
        t_d[:, i] = ti
        '''
        These lines implement the RK4 method
        '''

        k1 = f_dxdt(x, ti, dt, kernel_op, idxs)
        k2 = f_dxdt(x + k1 * dt / 2, ti + dt / 2, dt, kernel_op, idxs)
        k3 = f_dxdt(x + k2 * dt / 2, ti + dt / 2, dt, kernel_op, idxs)
        k4 = f_dxdt(x + k3 * dt, ti + dt, dt, kernel_op, idxs)

        x += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        if num_drive_terms > 0:
            x[-num_drive_terms:, :] = (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)[-num_drive_terms:, :]

        x_d[:,:,i] = x


    print(' ')
    print('...finished GPU solve!')

    return x_d, t_d


def f_dxdt(xi, t, dt, kernel_op, idxs):
    '''
    This just formats the array data to be passed into the GPU kernel
    '''

    args = [dt, t]
    args.extend(list(xi))
    args.extend(idxs)
    dxdt = kernel_op(*args)

    return cp.array(dxdt)


def GPUODE(dt, shape, kernel_op, IC_kernel_op, S, num_drive_terms):
    '''
    Wrapper for the RK loop that creates all the necessary arrays, since
    you can't create arrays inside a jit function.

    N is number of modes
    M is number of variations

    only_final option means to only save the final decimation period. Useful for near-steady state sims
    '''

    M = int(np.prod(shape[1:]))
    N = shape[0]

    idxs = []

    for i in range(1, len(shape)):
        idx = np.arange(0, shape[i])
        idxs.append(idx)

    IDXS = np.meshgrid(*idxs, indexing='ij')
    IDXSCP = []

    for i in range(0, len(shape)-1):
        IDXSCP.append(cp.array(cp.array(IDXS[i].flatten(), dtype=np.int32)))

    ICs = IC_kernel_op(*IDXSCP)
    x0 = cp.array(ICs)

    dt = cp.array(dt.flatten(), dtype=cp.float64)

    x, t_d = RK_loop(x0, dt, kernel_op, IDXSCP, S, num_drive_terms)

    x_np = np.reshape(cp.asnumpy(x), (*shape, S))
    t_d_np = np.reshape(cp.asnumpy(t_d), (*shape[1:], S))

    del (x) # free up GPU memory
    del (t_d)
    cp._default_memory_pool.free_all_blocks()

    return x_np, t_d_np