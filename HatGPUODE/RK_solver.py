import cupy as cp
import sympy as sp
import numpy as np


def RK_loop_saveall(t, x0, f_dxdt, dt, kernel_op, idxs):
    '''
    Implements an RK4 method.

    The indexing for the variable 'x', which contains all the solution data,
    goes like: modes, variations, time steps

    So for example, 3 modes with 5 variations running for a hundred time steps
    would have the shape (3, 5, 100).
    '''

    N, M = np.shape(cp.asnumpy(x0))

    noise_mask = cp.zeros([N, M])
    noise_mask[0, :] = 1

    x = cp.zeros([N, M, len(t)])

    x[:, :, 0] = x0

    for i in range(0, len(t) - 1):
        k1 = f_dxdt(x[:, :, i], t[i], dt, kernel_op, idxs)
        k2 = f_dxdt(x[:, :, i] + k1 * dt / 2, t[i] + dt / 2, dt, kernel_op, idxs)
        k3 = f_dxdt(x[:, :, i] + k2 * dt / 2, t[i] + dt / 2, dt, kernel_op, idxs)
        k4 = f_dxdt(x[:, :, i] + k3 * dt, t[i] + dt, dt, kernel_op, idxs)

        x[:, :, i + 1] = x[:, :, i] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4) + noise_mask * cp.random.normal(0, 1e-5, size=(N, M))

    return x

def RK_loop(t, x, x_avg, x_var, f_dxdt, dt, kernel_op, idxs, save_i, noise_mask):
    '''
    Implements an RK4 method.

    The indexing for the variable 'x', which contains all the solution data,
    goes like: time steps, modes, variations/shots.

    So for example, 3 modes with 5 variations running for a hundred time steps
    would have the shape (100, 3, 5).
    '''

    N, M = np.shape(cp.asnumpy(x))

    saved_x = cp.zeros([N,M,len(save_i)])

    for i in range(0, len(t) - 1):
        x_avg[:, i] = cp.mean(x, axis=1)

        if np.any(save_i ==i):

            saved_x[:,:,np.where(save_i == i)[0][0]] = x

        k1 = f_dxdt(x, t[i], dt, kernel_op, idxs)
        k2 = f_dxdt(x + k1 * dt / 2, t[i] + dt / 2, dt, kernel_op, idxs)
        k3 = f_dxdt(x + k2 * dt / 2, t[i] + dt / 2, dt, kernel_op, idxs)
        k4 = f_dxdt(x + k3 * dt, t[i] + dt, dt, kernel_op, idxs)

        x += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4) + noise_mask * cp.random.normal(0, 1, size=(N, M))

    return x, x_avg, saved_x


def f_dxdt(xi, t, dt, kernel_op, idxs):
    '''
    Equation of motion for related rates problem. The off-diagonal elements
    of the rate matrix represent transition rates.
    '''

    args = [idxs[0], idxs[1], dt, t]

    args.extend(list(xi))

    dxdt = kernel_op(*args)

    return cp.array(dxdt)


def related_rates_problem(t, N, variations1, variations2, kernel_op, noise_mask, save_i=[-1]):
    '''
    Wrapper for the RK loop that creates all the necessary arrays, since
    you can't create arrays inside of a jit function.

    N is number of modes
    vars is number of variations
    '''

    x0 = cp.zeros([N, variations1 * variations2])

    dt = cp.array(t[1] - t[0])

    t = cp.array(t, dtype=cp.float64)

    x_avg = cp.zeros([N, len(t)])
    x_var = cp.zeros([N, len(t)])

    idxs = []

    idx1 = np.arange(0, variations1)
    idx2 = np.arange(0, variations2)

    IDX2, IDX1 = np.meshgrid(idx2, idx1)

    idxs.append(cp.array(cp.array(IDX1.flatten(), dtype=np.int8)))
    idxs.append(cp.array(cp.array(IDX2.flatten(), dtype=np.int8)))

    x, x_avg, saved_x = RK_loop(t, x0, x_avg, x_var, f_dxdt, dt, kernel_op, idxs, save_i, noise_mask)

    return x, x_avg, saved_x