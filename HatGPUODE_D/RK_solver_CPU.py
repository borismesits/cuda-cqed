import cupy as cp
import sympy as sp
from numpy import * # we have to do a star import here to use eval() on arbitary equations of motion provided by user


def RK_loop_CPU(M, x0, t, var_list, eom_list, param_dict_nosweep):
    '''
    This is for solving single-variation simulations on the CPU (which is faster in this case).
    M is the number of modes
    '''

    dt = t[1]-t[0]

    x = zeros([M, len(t)])

    for i in range(0, len(t) - 1):

        k1 = f_dxdt(x[:,i], t[i], var_list, eom_list, param_dict_nosweep)
        k2 = f_dxdt(x[:,i] + k1 * dt / 2, t[i] + dt / 2, var_list, eom_list, param_dict_nosweep)
        k3 = f_dxdt(x[:,i] + k2 * dt / 2, t[i] + dt / 2, var_list, eom_list, param_dict_nosweep)
        k4 = f_dxdt(x[:,i] + k3 * dt, t[i] + dt, var_list, eom_list, param_dict_nosweep)

        x[:,i+1] = x[:,i] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x


def f_dxdt(xi, t, var_list, eom_list, param_dict_nosweep):

    dxidt = xi * 0

    exec('t = ' + str(t))

    for param in param_dict_nosweep:

        exec(param + ' = ' + str(param_dict_nosweep[param]))

    for i in range(0, len(var_list)):

        exec(var_list[i] + '= ' + str(xi[i]))

    for i in range(0,len(eom_list)):


        dxidt[i] = eval(eom_list[i])

    return dxidt

