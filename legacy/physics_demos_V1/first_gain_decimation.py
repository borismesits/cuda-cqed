import numpy as np
import sympy as sp
import cupy as cp
import time
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('Qt5Agg')

# Make sure you use a GPU runtime, otherwise cupy won't import

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

    x[:,:,0] = x0

    for i in range(0, len(t)-1):

        k1 = f_dxdt(x[:, :, i], t[i], dt, kernel_op, idxs)
        k2 = f_dxdt(x[:, :, i] + k1*dt/2, t[i] + dt/2, dt, kernel_op, idxs)
        k3 = f_dxdt(x[:, :, i] + k2*dt/2, t[i] + dt/2, dt, kernel_op, idxs)
        k4 = f_dxdt(x[:, :, i] + k3*dt, t[i] + dt, dt, kernel_op, idxs)

        x[:, :, i+1] = x[:, :, i] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4) + noise_mask*cp.random.normal(0, 1e-4, size=(N, M))

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

    x0 = cp.zeros([N, variations1*variations2])

    dt = cp.array(t[1] - t[0])

    t = cp.array(t, dtype=cp.float64)

    idxs = []

    idx1 = np.arange(0, variations1)
    idx2 = np.arange(0, variations2)

    IDX1, IDX2 = np.meshgrid(idx1, idx2)

    idxs.append(cp.array(cp.array(IDX1.flatten(), dtype=np.int8)))
    idxs.append(cp.array(cp.array(IDX2.flatten(), dtype=np.int8)))

    decimation_factor = 100

    xi = cp.zeros([N, variations1 * variations2, len(t) // decimation_factor])
    xq = cp.zeros([N, variations1 * variations2, len(t) // decimation_factor])

    w_demod = 9 * 2 * np.pi

    for i in range(0, len(t)//decimation_factor):

        t_demod = cp.tile(t[i*decimation_factor:(i+1)*decimation_factor], (N, variations1 * variations2, 1))

        x = RK_loop_saveall(t[i*decimation_factor:(i+1)*decimation_factor], x0, f_dxdt, dt, kernel_op, idxs)

        xi[:, :, i] = cp.mean(np.cos(-t_demod*w_demod)*x, axis=2)
        xq[:, :, i] = cp.mean(np.sin(-t_demod*w_demod)*x, axis=2)

        x0 = x[:, :, -1]

    return xi, xq


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
            stringy[j-1] = stringy[j-1]+'.0'

    string = ''

    for i in range(0,len(stringy)):
        string += stringy[i]

    return string


def run_sim(pump_on, signal_on):

    variations1 = 1
    variations2 = 1000
    
    params = [('omega_0', 10*2*np.pi),
              ('omega_p', 20*2*np.pi),
              ('omega_s', 10*2*np.pi),
              ('A_p', 9),
              ('A_s', 0.1),
              ('kappa', 1*2*np.pi),  # mode kappa
              ('kappa_r', 0.5*2*np.pi), # readout kappa
              ('phi', 0),
              ('signal_on', signal_on),
              ('pump_on', pump_on)]

    var_strs = ['q_1','p_1','q_r','p_r'] # define your dependent variables/coordinates
    exp_strs = ['p_1 + signal_on*A_s*cos(omega_s*t) + pump_on*A_p*cos(omega_p*t + phi)',
                '-(q_1+q_1**2)*omega_0**2 - kappa*p_1',
                'p_r',
                '-omega_s**2 * q_r + q_1 - kappa_r*p_r']  # define the time derivatives of each coordinate
    
    exp_sps = []
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
    
        Kernel += 'double ' + params[i][0] + ' = ' + str(float(param0)) + ' + index' + str(sweep_num) + '*' + str(float(param_range)) + '/' + str(float(np.max([param_vars-1,1]))) + ';\n'
    
      except:
        Kernel += 'double ' + params[i][0] + ' = ' + str(float(params[i][1])) + 'f;\n'
    
    for i in range(0,len(var_strs)):
    
      eom_line = 'd' + var_strs[i] + 'dt = ' + exp_cs[i] + '; \n'
    
      Kernel += eom_line
    
    print(KernelInput)
    print(Kernel)
    print(KernelOutput)
    
    ODE = cp.ElementwiseKernel(KernelInput, KernelOutput, Kernel, 'demo_ODE')
    
    # for i in range(0,len(var_strs)):
    #   display(sp.Eq(sp.sympify(var_strs[i]),exp_sps[i]))
    
    t = np.arange(0, 10, 0.001)
    
    N = len(var_strs)
    
    time1 = time.time()
    xi, xq = related_rates_problem(t, N, variations1, variations2, ODE)
    print(time.time() - time1)

    xi_np = cp.asnumpy(xi)
    xq_np = cp.asnumpy(xq)
    
    return xi_np, xq_np
#
# xi, xq = run_sim(0, 1)
#
# plt.plot(xi[2,0:100,:].transpose(),xq[2,0:100,:].transpose(),'k')

xi, xq = run_sim(1, 1)

xiq = xi + 1j*xq

roiq_fft = np.fft.fft(xiq[0,0::100,:], axis=1)
plt.plot(np.log(np.abs(roiq_fft.transpose())))

plt.figure()
plt.plot(np.log(np.abs(roiq_fft)))
plt.figure()
plt.plot(np.real(roiq_sa),np.imag(roiq_sa))