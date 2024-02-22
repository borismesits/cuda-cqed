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

weights = np.zeros(variations1*variations2) - 1
weights[0:variations2] = 1

var_strs = ['a','f']
exp_strs = ['1j*(-1)*(a + (a + conjugate(a))**2 )*omega_0 + pump_on*A_p*cos(omega_p * t + phi) + signal_on*A_s*cos(omega_s * t) - kappa*(a + conjugate(a))',
            '1j*(-1)*f*omega_0 - kappa*(f + conjugate(f)) + 1j*a']

chi = 0.5 * 2 * np.pi

params = [('omega_0', [10 * 2 * np.pi - chi/2, 10 * 2 * np.pi + chi/2, variations1]),
          ('omega_p', 20 * 2 * np.pi),
          ('omega_s', 10 * 2 * np.pi),
          ('A_p', 7),
          ('A_s', 0.5),
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

save_i = np.round(10**np.linspace(1,3.2,5))

noise_mask = cp.zeros([2*N, variations1*variations2])
noise_mask[0, :] = 1e-3

start_time = time.time()
x, x_avg, saved_x = related_rates_problem(t, 2*N, variations1, variations2, kernel_op, noise_mask, save_i=save_i)
print(time.time()-start_time)
xx = cp.asnumpy(x)
saved_x = cp.asnumpy(saved_x)

saved_x = np.reshape(saved_x, [2*N, variations1, variations2, len(save_i)])
#%%

def gen_hist(I, Q, size=3, weights=None):

    range_I = size
    range_Q = size

    hist_range = [[-range_I, range_I], [-range_Q, range_Q]]
    # hist_range = None

    hist, x, y = np.histogram2d(I.flatten(), Q.flatten(), bins=300, range=hist_range, weights=weights)

    return hist, x, y

rows = len(var_strs)
cols = len(save_i)

fig = plt.figure(1)
plt.clf()
fig, axs = plt.subplots(rows, cols, num=1)

hist_size = [0.2, 0.03]

for j in range(0,cols):

    den_hist, x, y = gen_hist(saved_x[0, :, :, j], saved_x[1, :, :, j], size=hist_size[0])
    num_hist, x, y = gen_hist(saved_x[0, :, :, j], saved_x[1, :, :, j], size=hist_size[0], weights=weights)
    axs[0, j].pcolor(x, y, num_hist / den_hist, cmap='seismic', vmin=-1.5, vmax=1.5)
    axs[0, j].set_aspect('equal')
    axs[0,j].set_title('Time = ' + str(np.round(t[int(save_i[j])],4)) + ' s')
    axs[0,j].grid()

    den_hist, x, y = gen_hist(saved_x[2, :, :, j], saved_x[3, :, :, j], size=hist_size[1])
    num_hist, x, y = gen_hist(saved_x[2, :, :, j], saved_x[3, :, :, j], size=hist_size[1], weights=weights)
    axs[1, j].pcolor(x, y, num_hist/den_hist, cmap='seismic',vmin=-1.5,vmax=1.5)
    axs[1, j].set_aspect('equal')
    axs[1, j].grid()
