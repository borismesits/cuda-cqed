import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from HatGPUODE.util import generate_kernel
from HatGPUODE.RK_solver import related_rates_problem
import matplotlib
import time

matplotlib.use('Qt5Agg')

variations1 = 2
variations2 = 100000

var_strs = ['a','f']
exp_strs = ['1j*(-1)*(a + (a + conjugate(a))**2 )*omega_0 + pump_on*A_p*cos(omega_p * t + phi) + signal_on*A_s*cos(omega_s * t) - kappa*a',
            '1j*(-1)*f*omega_0 - kappa*f + 1j*a']

chi = 1.0 * 2 * np.pi

params = [('omega_0', [10 * 2 * np.pi - chi/2, 10 * 2 * np.pi + chi/2, variations1]),
          ('omega_p', 19.9 * 2 * np.pi),
          ('omega_s', 9.95 * 2 * np.pi),
          ('A_p', 4),
          ('A_s', 0.05),
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
steps = 3000
t = np.linspace(0, dt*steps, steps)

save_i = np.round(np.linspace(0,2990,5))

noise_mask = cp.zeros([2*N, variations1*variations2])
noise_mask[0, :] = 1e-3

start_time = time.time()
x, x_avg, saved_x = related_rates_problem(t, 2*N, variations1, variations2, kernel_op, noise_mask, save_i=save_i)
print(time.time()-start_time)
xx = cp.asnumpy(x)
saved_x = cp.asnumpy(saved_x)
#%%

def gen_hist(I, Q, size=3):

    range_I = size
    range_Q = size

    hist_range = [[-range_I, range_I], [-range_Q, range_Q]]
    # hist_range = None

    hist, x, y = np.histogram2d(I, Q, bins=300, range=hist_range)

    return hist,x,y

rows = len(var_strs)
cols = len(save_i)

fig, axs = plt.subplots(rows,cols)

hist_size = [0.1,0.01]

for i in range(0,rows):
    for j in range(0,cols):

        hist, x, y = gen_hist(saved_x[i*2,:,j],saved_x[i*2+1,:,j], size=hist_size[i])
        axs[i,j].pcolor(x,y,np.log(hist))
        axs[i,j].set_aspect('equal')
