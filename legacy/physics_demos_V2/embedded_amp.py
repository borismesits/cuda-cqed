import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from HatGPUODE.util import generate_kernel
from HatGPUODE.RK_solver import related_rates_problem
import matplotlib
import time

matplotlib.use('Qt5Agg')

variations1 = 1
variations2 = 100000

var_strs = ['a']
exp_strs = ['1j*(-1)*a*omega + 5*a**5 + A*cos(omega_d * t)']

params = [('omega', 10 * 2 * np.pi),
          ('omega_d', 10 * 2 * np.pi),
          ('A', 1)]

kernel_input, kernel_output, kernel_body, kernel_op = generate_kernel(var_strs, exp_strs, params, use_complex=True)

print(kernel_input)
print(kernel_body)
print(kernel_output)

N = len(var_strs)

dt = 0.1/(10 * 2 * np.pi)
steps = 10000
t = np.linspace(0, dt*steps, steps)

save_i = np.round(np.linspace(0,10000,16))

start_time = time.time()
x, x_avg, saved_x = related_rates_problem(t, 2*N, variations1, variations2, kernel_op, save_i, noise_mask)
print(time.time()-start_time)
xx = cp.asnumpy(x)
saved_x = cp.asnumpy(saved_x)
#%%

def gen_hist(xi):

    range_I = 5
    range_Q = 5

    I1 = xi[0, :]
    Q1 = xi[1, :]

    hist, x, y = np.histogram2d(I1, Q1, bins=300, range=[[-range_I, range_I], [-range_Q, range_Q]])
    return hist,x,y

rows = 4
cols = 4

fig, axs = plt.subplots(rows,cols)

for  i in range(0,rows):
    for j in range(0,cols):

        index = i*cols + j

        hist, x, y = gen_hist(saved_x[:,:,index])
        axs[i,j].pcolor(x,y,np.log(hist))
