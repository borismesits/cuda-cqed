import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from HatGPUODE_D.util import generate_kernel
from HatGPUODE_D.RK_solver_decimate import GPUODE_decimate
import matplotlib
import time

matplotlib.use('Qt5Agg')


var_strs = ['a']
exp_strs = ['1j*(-1)*(a + (a + conjugate(a))**2 )*omega_0 + A_s*(1j)*(-1)*exp(-1j*(omega_s * t)) + A_p*(1j)*(-1)*exp(-1j*(2*omega_s * t + phi)) - kappa*a']


omega_s0 = 5 * 2 * np.pi
omega_s1 = 15 * 2 * np.pi
swp_pts = 500

A_s0 = 0
A_s1 = 1

params = [('omega_0', 10 * 2 * np.pi),
          ('A_s', [0, 0.01, 200]),
          ('A_p', [0, 5, 400]),
          ('kappa', 1 * 2 * np.pi),  # mode kappa
          ('phi', np.pi/2),
          ('omega_s', [omega_s0, omega_s1, swp_pts]),
          ('omega_p', 20 * 2 * np.pi)]

kernel_input, kernel_output, kernel_body, kernel_op, shape = generate_kernel(var_strs, exp_strs, params, use_complex=True)

print('Shape: ' + str(shape))

PTS_PER_CYCLE = 100
NUM_CYCLES = 50

D_FACTOR = PTS_PER_CYCLE # decimation factor

# below is the decimation frequency

d_omega = np.ones(shape[1:])

omega_s = np.linspace(omega_s0, omega_s1, swp_pts) # this will generally need to have shape M (since demod frequency may need to change with drive or mixing products)

d_omega = d_omega*omega_s

dt = (2*np.pi)/(d_omega*PTS_PER_CYCLE)

S = PTS_PER_CYCLE*NUM_CYCLES

start_time = time.time()
I_demod, Q_demod, t_d = GPUODE_decimate(dt, shape, kernel_op, D_FACTOR, d_omega, S)
print(time.time()-start_time)

amp = np.sqrt(I_demod**2 + Q_demod**2)
phase = np.arctan(Q_demod/I_demod)

# plt.figure()
# for i in range(0, shape[2]):
#     plt.plot(t_d[-1, i, :], Q_demod[0, -1, i, :])
#
# plt.figure()
# for i in range(0, shape[1]):
#     plt.plot(t_d[i, 25, :], Q_demod[0, i, 25, :])

plt.figure()
plt.pcolor(np.log(amp[0,1,:,:,-1]/amp[0,1,0,:,-1]))
plt.colorbar()

plt.figure()
plt.pcolor(amp[0,0,:,:,-1])