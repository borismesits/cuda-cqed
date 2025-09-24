import cupy as cp
import sympy as sp
import numpy as np



def RK_loop_saveall(t, x0, f_dxdt, dt, kernel_op, idxs):

    shape = np.shape(cp.asnumpy(x0))

    x = cp.zeros([N, M, len(t)])

    x[:, :, 0] = x0

    for i in range(0, len(t) - 1):
        k1 = f_dxdt(x[:, :, i], t[i], dt, kernel_op, idxs)
        k2 = f_dxdt(x[:, :, i] + k1 * dt / 2, t[i] + dt / 2, dt, kernel_op, idxs)
        k3 = f_dxdt(x[:, :, i] + k2 * dt / 2, t[i] + dt / 2, dt, kernel_op, idxs)
        k4 = f_dxdt(x[:, :, i] + k3 * dt, t[i] + dt, dt, kernel_op, idxs)

        x[:, :, i + 1] = x[:, :, i] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x


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

    num_variations = np.product(shape[1:])

    x0 = cp.zeros([shape[0], num_variations])

    dt = cp.array(t[1] - t[0])

    t = cp.array(t, dtype=cp.float64)

    x_avg = cp.zeros([shape[0], len(t)])
    x_var = cp.zeros([shape[0], len(t)])

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