import cupy as cp
import sympy as sp
import numpy as np


def RK_loop(t, x, x_avg, x_var, f_dxdt, dt, kernel_op, idxs, save_i):
    '''
    Implements an RK4 method.

    The indexing for the variable 'x', which contains all the solution data,
    goes like: modes (N), variations/shots (M)

    So for example, 3 modes with 5 variations would have the shape (3, 5).
    '''

    N, M = np.shape(cp.asnumpy(x))

    saved_x = cp.zeros([N, M, len(save_i)])

    for i in range(0, len(t) - 1):
        # x_avg[:, i] = cp.mean(x, axis=1)

        # if np.any(save_i == i):
        #
        #     saved_x[:, :, np.where(save_i == i)[0][0]] = x

        k1 = f_dxdt(x, t[i], dt, kernel_op, idxs)
        k2 = f_dxdt(x + k1 * dt / 2, t[i] + dt / 2, dt, kernel_op, idxs)
        k3 = f_dxdt(x + k2 * dt / 2, t[i] + dt / 2, dt, kernel_op, idxs)
        k4 = f_dxdt(x + k3 * dt, t[i] + dt, dt, kernel_op, idxs)

        x += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    # x_avg[:, -1] = cp.mean(x, axis=1)

    return x, x_avg, saved_x


def f_dxdt(xi, t, dt, kernel_op, idxs):
    '''
    Equation of motion for related rates problem. The off-diagonal elements
    of the rate matrix represent transition rates.
    '''

    args = [dt, t]

    args.extend(list(xi))

    args.extend(idxs)

    dxdt = kernel_op(*args)

    return cp.array(dxdt)


def GPUODE_light(t, shape, kernel_op, save_i=[-1]):
    '''
    Wrapper for the RK loop that creates all the necessary arrays, since
    you can't create arrays inside of a jit function.

    The light means that the data is not saved in full detail. Only snapshots, averages, and final values are saved.

    N is number of modes
    vars is number of variations
    '''

    M = np.product(shape[1:])

    N = shape[0]

    x0 = cp.zeros([N, M])

    dt = cp.array(t[1] - t[0])

    t = cp.array(t, dtype=cp.float64)

    x_avg = cp.zeros([N, len(t)])
    x_var = cp.zeros([N, len(t)])

    idxs = []

    for i in range(1, len(shape)):
        idx = np.arange(0, shape[i])
        idxs.append(idx)

    IDXS = np.meshgrid(*idxs, indexing='ij')
    IDXSCP = []

    for i in range(0, len(shape)-1):
        IDXSCP.append(cp.array(cp.array(IDXS[i].flatten(), dtype=np.int32)))

    x, x_avg, saved_x = RK_loop(t, x0, x_avg, x_var, f_dxdt, dt, kernel_op, IDXSCP, save_i)

    x = np.reshape(cp.asnumpy(x), shape)

    return x, cp.asnumpy(x_avg), cp.asnumpy(saved_x)