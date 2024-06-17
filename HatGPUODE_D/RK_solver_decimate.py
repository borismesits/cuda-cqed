import cupy as cp
import sympy as sp
import numpy as np


def RK_loop_decimate(x, dt, kernel_op, idxs, d_factor, d_omega, S):
    '''
    Implements a GPU-accelerated RK4 method for simulating systems of ODEs, with built-in decimation of the
    data to reduce amount of saved information.

    N represents number of modes, M is number of variations (defined outside this function)

    Similar to the RK_loop function of the non-decimated HatGPUODE directory

    '''

    N, M = np.shape(cp.asnumpy(x))

    integrated_I = x*0
    integrated_Q = x*0

    I_demod = cp.zeros([N, M, S // d_factor])
    Q_demod = cp.zeros([N, M, S // d_factor])

    t_d = cp.zeros([M, S // d_factor])

    for i in range(0, S - 1):
        '''
        The first five lines implement the RK4 method
        '''

        ti = dt*i

        k1 = f_dxdt(x, ti, dt, kernel_op, idxs)
        k2 = f_dxdt(x + k1 * dt / 2, ti + dt / 2, dt, kernel_op, idxs)
        k3 = f_dxdt(x + k2 * dt / 2, ti + dt / 2, dt, kernel_op, idxs)
        k4 = f_dxdt(x + k3 * dt, ti + dt, dt, kernel_op, idxs)

        x += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        integrated_I += x * cp.cos(d_omega * ti) / d_factor
        integrated_Q += x * cp.sin(d_omega * ti) / d_factor

        if i % d_factor == 0:

            I_demod[:, :, i // d_factor] = integrated_I
            Q_demod[:, :, i // d_factor] = integrated_Q
            t_d[:, i // d_factor] = ti

            integrated_I = x * 0
            integrated_Q = x * 0


    return I_demod, Q_demod, t_d


def f_dxdt(xi, t, dt, kernel_op, idxs):
    '''
    This just formats the array data to be passed into the GPU kernel
    '''

    args = [dt, t]

    args.extend(list(xi))

    args.extend(idxs)

    dxdt = kernel_op(*args)

    return cp.array(dxdt)


def GPUODE_decimate(dt, shape, kernel_op, d_factor, d_omega, S):
    '''
    Wrapper for the RK loop that creates all the necessary arrays, since
    you can't create arrays inside a jit function.

    N is number of modes
    M is number of variations
    '''

    M = np.product(shape[1:])
    N = shape[0]

    x0 = cp.zeros([N, M])

    idxs = []

    for i in range(1, len(shape)):
        idx = np.arange(0, shape[i])
        idxs.append(idx)

    IDXS = np.meshgrid(*idxs, indexing='ij')
    IDXSCP = []

    for i in range(0, len(shape)-1):
        IDXSCP.append(cp.array(cp.array(IDXS[i].flatten(), dtype=np.int32)))

    d_omega = cp.array(d_omega.flatten(), dtype=cp.float64) # convert digitization-related arrays to cupy
    dt = cp.array(dt.flatten(), dtype=cp.float64)

    I_demod, Q_demod, t_d = RK_loop_decimate(x0, dt, kernel_op, IDXSCP, d_factor, d_omega, S)

    I_demod = np.reshape(cp.asnumpy(I_demod), (*shape, S//d_factor))
    Q_demod = np.reshape(cp.asnumpy(Q_demod), (*shape, S//d_factor))
    t_d = np.reshape(cp.asnumpy(t_d), (*shape[1:], S//d_factor))

    return I_demod, Q_demod, t_d