import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from HatGPUODE.util import generate_kernel
from HatGPUODE.RK_solver import related_rates_problem
import matplotlib
import time

matplotlib.use('Qt5Agg')

variations1 = 1000
variations2 = 1000

var_strs = ['a','f','h']
exp_strs = ['1j*(-1)*(a - (a + conjugate(a))**2 )*omega_0 + sin(a/10) + pump_on*A_p*cos(omega_p * t + phi) + signal_on*A_s*cos(omega_s * t) - kappa*a',
            '1j*(-1)*f*omega_0 - 0.001*kappa*f + 1j*a',
            '1j*(-3)*h*omega_0 - 0.001*kappa*h + 1j*a']


omega_s0 = 9 * 2 * np.pi
omega_s1 = 13 * 2 * np.pi

A_s0 = 0
A_s1 = 0.05

params = [('omega_0', 10 * 2 * np.pi),
          ('omega_p', 19.5 * 2 * np.pi),
          ('omega_s', [omega_s0, omega_s1, variations1]),
          ('A_p', 9),
          ('A_s', [A_s0, A_s1, variations2]),
          ('kappa', 1 * 2 * np.pi),  # mode kappa
          ('phi', 0),
          ('signal_on', 1),
          ('pump_on', 1)]

kernel_input, kernel_output, kernel_body, kernel_op = generate_kernel(var_strs, exp_strs, params, use_complex=True)

print(kernel_input)
print(kernel_body)
print(kernel_output)

N = len(var_strs)

dt = 0.1/(10 * 2 * np.pi)
steps = 10000
t = np.linspace(0, dt*steps, steps)

save_i = np.round(np.linspace(0,9990,5))

noise_mask = cp.zeros([2*N, variations1*variations2])
noise_mask[0, :] = 0e-5

start_time = time.time()
x, x_avg, saved_x = related_rates_problem(t, 2*N, variations1, variations2, kernel_op, noise_mask, save_i=save_i)
print(time.time()-start_time)
xx = cp.asnumpy(x)
saved_x = cp.asnumpy(saved_x)
#%%


x = np.reshape(cp.asnumpy(x), (2*N,variations1,variations2))
x_avg = cp.asnumpy(x_avg)
saved_x = np.reshape(cp.asnumpy(saved_x), (2*N,variations1,variations2,len(save_i)))

range_I = 0.0002
range_Q = 0.0002

amplitude = np.sqrt(saved_x[2,:,:,-1]**2 + saved_x[3,:,:,-1]**2)
print(amplitude)

# plt.figure()
# plt.plot(t, x_avg[2,:])
# plt.plot(t, x_avg[3,:])
#
# plt.plot(t, np.sqrt(x_avg[2,:]**2+x_avg[3,:]**2))

omega_s = np.linspace(omega_s0, omega_s1, variations1)
A_s = np.linspace(A_s0, A_s1, variations2)

plt.pcolor(A_s, omega_s/(2*np.pi),amplitude)
plt.colorbar()