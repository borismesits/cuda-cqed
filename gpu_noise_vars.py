import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import numba as nb
import cupy as cp
from numba import jit
import time
from IPython.display import display
from numba import njit
from numba.typed import List

# Make sure you use the GPU runtime, otherwise cupy won't import

variations1 = 300
variations2 = 10000

params = [('omega_0', 1*2*np.pi),
          ('omega_d', [0.5*2*np.pi, 1.5*2*np.pi, variations1]),
          ('A', 0.5),
          ('kappa', 0.1),
          ('B', 10)]

var_strs = ['q_1','p_1']
exp_strs = ['p_1 + A*cos(omega_d*t)','-q_1*omega_0**2 - B*q_1**3 - kappa*p_1']
exp_sps = []

def RK_loop(t, x, x_avg, x_var, f_dxdt, dt, kernel_op, idxs):
    '''
    Implements an RK4 method.

    The indexing for the variable 'x', which contains all the solution data,
    goes like: time steps, modes, variations.

    So for example, 3 modes with 5 variations running for hundred time steps
    would have the shape (100, 3, 5).
    '''

    for i in range(0, len(t)-1):

        k1 = f_dxdt(x, t[i], dt, kernel_op, idxs)
        k2 = f_dxdt(x + k1*dt/2, t[i] + dt/2, dt, kernel_op, idxs)
        k3 = f_dxdt(x + k2*dt/2, t[i] + dt/2, dt, kernel_op, idxs)
        k4 = f_dxdt(x + k3*dt, t[i] + dt, dt, kernel_op, idxs)

        x += (dt/6)*(k1 + 2*k2 + 2*k3 + k4) + cp.random.normal(0,0.005,[N, variations1*variations2])

    return x

def f_dxdt(xi, t, dt, kernel_op, idxs):
    '''
    Equation of motion for related rates problem. The off-diagonal elements
    of the rate matrix represent transition rates.
    '''

    args = [idxs[0], idxs[1], dt, t]

    args.extend(list(xi))

    dxdt = kernel_op(*args)

    return cp.array(dxdt)

def related_rates_problem(t, N, variations1, variations2, kernel_op):
    '''
    Wrapper for the RK loop that creates all the necessary arrays, since
    you can't create arrays inside of a jit function.

    N is number of modes
    vars is number of variations
    '''

    x0 = cp.array(np.random.normal(0,0.1,[N, variations1*variations2]),dtype=cp.float64)

    dt = cp.array(t[1] - t[0])

    t = cp.array(t, dtype=cp.float64)

    x_avg = t*0
    x_var = t*0

    idxs = []

    idx1 = np.arange(0, variations1)
    idx2 = np.arange(0, variations2)

    IDX1, IDX2 = np.meshgrid(idx1, idx2)

    idxs.append(cp.array(cp.array(IDX1.flatten(), dtype=np.int8)))
    idxs.append(cp.array(cp.array(IDX2.flatten(), dtype=np.int8)))

    x = RK_loop(t, x0, x_avg, x_var, f_dxdt, dt, kernel_op, idxs)

    return x


def convert_power_arg_to_float64(inputt):
    try:
        inputt.lower()
    except:
        print('not a string')

    stringy = []

    for i in range(0,len(inputt)):
        stringy.append(inputt[i])

    for i in range(0,len(stringy)-3):
        if (stringy[i] == 'p' and stringy[i+1] == 'o' and stringy[i+2] == 'w'):
            j = i+3
            while(stringy[j] != ')'):
                j += 1
            stringy[j-1]=stringy[j-1]+'.0'

    string = ''

    for i in range(0,len(stringy)):
        string += stringy[i]

    return string

exp_cs = []

for exp_str in exp_strs:
  exp_sp = sp.sympify(exp_str)
  exp_sps.append(exp_sp)
  exp_c = convert_power_arg_to_float64(sp.ccode(exp_sp))
  exp_cs.append(exp_c)


KernelInput = 'int8 index1, int8 index2, float64 dt, float64 t, '

for var_str in var_strs:

  KernelInput += 'float64 ' + var_str + ', '
KernelInput = KernelInput[0:-2] # removes trailing comma

KernelOutput = ''

for var_str in var_strs:

  KernelOutput += 'float64 d' + var_str + 'dt' + ', '
KernelOutput = KernelOutput[0:-2] # removes trailing comma

Kernel = ''

sweep_num = 0

for i in range(0,len(params)):

  try:

    param0 = params[i][1][0]
    param_vars = params[i][1][2]
    param_range = params[i][1][1]-params[i][1][0]

    sweep_num += 1

    Kernel += 'double ' + params[i][0] + ' = ' + str(float(param0)) + ' + index' + str(sweep_num) + '*' + str(float(param_range)) + '/' + str(float(param_vars-1)) + ';\n'

  except:
    Kernel += 'double ' + params[i][0] + ' = ' + str(float(params[i][1])) + 'f;\n'

for i in range(0,2):

  eom_line = 'd' + var_strs[i] + 'dt = ' + exp_cs[i] + '; \n'

  Kernel += eom_line

print(KernelInput)
print(Kernel)
print(KernelOutput)

ODE = cp.ElementwiseKernel(KernelInput, KernelOutput, Kernel, 'demo_ODE')

for i in range(0,len(var_strs)):

  display(sp.Eq(sp.sympify(var_strs[i]),exp_sps[i]))

t = np.linspace(0, 100, 10001)

N = len(var_strs)

time1 = time.time()
x = related_rates_problem(t, N, variations1, variations2, ODE)
print(time.time() - time1)
x = cp.asnumpy(x)