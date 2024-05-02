import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from HatGPUODE.util import generate_kernel
from HatGPUODE.RK_solver import related_rates_problem
import matplotlib
import time

matplotlib.use('Qt5Agg')

variations1 = 1000
variations2 = 1

signal_on = 1
pump_on = 0

var_strs = ['q_1','p_1','q_r','p_r'] # define your dependent variables/coordinates
exp_strs = ['p_1 + signal_on*A_s*cos(omega_s*t) + pump_on*A_p*cos(omega_p*t + phi)',
            '-(q_1+q_1**2)*omega_0**2 - kappa*p_1',
            'p_r',
            '-omega_s**2 * q_r + q_1 - kappa_r*p_r'] # define the time derivatives of each coordinate

params = [('omega_0', 10 * 2 * np.pi),
          ('omega_p', 20 * 2 * np.pi),
          ('omega_s', [1 * 2 * np.pi, 20 * 2 * np.pi, variations1]),
          ('A_p', 7.0),
          ('A_s', 0.1),
          ('kappa', 1 * 2 * np.pi),  # mode kappa
          ('kappa_r', 0.5 * 2 * np.pi),  # readout kappa
          ('phi', 1.5),
          ('signal_on', signal_on),
          ('pump_on', pump_on)]

kernel_input, kernel_output, kernel_body, kernel_op = generate_kernel(var_strs, exp_strs, params)

print(kernel_input)
print(kernel_body)
print(kernel_output)

N = len(var_strs)

dt = 0.1/(10 * 2 * np.pi)
steps = 10000
t = np.linspace(0, dt*steps, steps)

noise_mask = cp.zeros([N, variations1*variations2])
noise_mask[0, :] = 0

save_i = np.round(10**np.linspace(1,3.2,5))

start_time = time.time()
x, x_avg, saved_x = related_rates_problem(t, N, variations1, variations2, kernel_op, noise_mask, save_i=save_i)
print(time.time()-start_time)

#%%

x = cp.asnumpy(x)
x_avg = cp.asnumpy(x_avg)
saved_x = cp.asnumpy(saved_x)

range_I = 0.0002
range_Q = 0.0002

plt.figure()
plt.plot(saved_x[0,:,-1])
