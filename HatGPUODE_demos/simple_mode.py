import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from HatGPUODE.util import generate_kernel
from HatGPUODE.RK_solver import related_rates_problem
import matplotlib
import time

matplotlib.use('Qt5Agg')

variations1 = 137
variations2 = 77

var_strs = ['a']
exp_strs = ['1j*(-1)*a*omega_0 + 10**(logA_s)*(1j)*(-1)*exp(-1j*(omega_s * t)) - kappa**2*a']


omega_s0 = 5 * 2 * np.pi
omega_s1 = 15 * 2 * np.pi

A_s0 = 0
A_s1 = 1

params = [('omega_0', 10 * 2 * np.pi),
          ('omega_s', [omega_s0, omega_s1, variations1]),
          ('logA_s', [-5, 1, variations2]),
          ('kappa', 1.0 * 2 * np.pi),  # mode kappa
          ('phi', 0)]

kernel_input, kernel_output, kernel_body, kernel_op, shape = generate_kernel(var_strs, exp_strs, params, use_complex=True)

print('Shape: ' + str(shape))

dt = 0.1/(10 * 2 * np.pi)
steps = 10000
t = np.linspace(0, dt*steps, steps)

save_i = np.round(np.linspace(0,9990,5))

noise_mask = 1

start_time = time.time()
x, x_avg, saved_x = related_rates_problem(t, shape, kernel_op, noise_mask, save_i=save_i)
print(time.time()-start_time)

#%%


amplitude = np.log(np.sqrt(x[0,:,:]**2 + x[1,:,:]**2))
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