import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from HatGPUODE_D.util import generate_kernel
from HatGPUODE_D.RK_solver_decimate import GPUODE_decimate
import matplotlib
import time

matplotlib.use('Qt5Agg')


var_strs = ['x1','x2','x3','r1','r2']
exp_strs = ['x2',
            '-kappa*x2 - w_0**2*x1 - w_J**2*( sin(x1 + phi_DC) - sin(phi_DC) - cos(phi_DC)*x1) + 2*kappa*(-amp_in*w_in*sin(w_in*t + phase_in) - 2*phi_max*w_in*cos(2*w_in*t) - 2*phi_block*w_in*cos(1.1*w_in*t))',
            'x2 - amp_in*w_in*sin(w_in*t + phase_in) + 2*phi_max*w_in*cos(2*w_in*t) - 2*phi_block*w_in*cos(1.1*w_in*t)',
            'r2',
            '-w_in**2 * r1 - kappa_r*r2 + x3*(sign(t-wait_t)+1)/2']


w_in_i = 6e9 * 2 * np.pi
w_in_f = 7e9 * 2 * np.pi
swp_pts = 100

params = [('w_0', 6.426939330532695e9 * 2 * np.pi),
          ('amp_in', [0, 0.1, 200]), # signal amplitude
          ('phi_max', [0, 1.0, 2]), # pump amplitude
          ('phi_block', [0, 1.0, 2]), # pump amplitude
          ('kappa', 4.547284088339866e8 * 2 * np.pi),  # mode decay rate
          ('kappa_r', 1e8 * 2 * np.pi), # filter bandwidth
          ('phase_in', -1.3),
          ('phi_DC', 1.5),
          ('wait_t', 1e-8),
          ('w_J', 8.507189549448235e9 * 2 * np.pi),
          ('w_in', 6.426939330532695e9 * 2 * np.pi)] # signal frequency

kernel_input, kernel_output, kernel_body, kernel_op, shape, var_dict = generate_kernel(var_strs, exp_strs, params, use_complex=False)

print('Shape: ' + str(shape))

PTS_PER_CYCLE = 100
NUM_CYCLES = 500

D_FACTOR = PTS_PER_CYCLE # decimation factor

# below is the decimation frequency

d_omega = np.ones(shape[1:])

omega_s = var_dict['w_in'] # this will generally need to have shape M (since demod frequency may need to change with drive or mixing products)

d_omega = d_omega*omega_s

dt = (2*np.pi)/(d_omega*PTS_PER_CYCLE)

S = PTS_PER_CYCLE*NUM_CYCLES

start_time = time.time()
I_demod, Q_demod, t_d = GPUODE_decimate(dt, shape, kernel_op, D_FACTOR, d_omega, S)
print(time.time()-start_time)

amp = np.sqrt(I_demod**2 + Q_demod**2)
phase = np.arctan(Q_demod/I_demod)


plt.figure()
pwrgain = 20*np.log10(amp[4,:,1,:,-1]/amp[4,:,0,:,-1])
plt.pcolor(pwrgain, cmap='seismic')
plt.colorbar()
plt.clim([-40, 40])
plt.contour(pwrgain,'k--', levels=[20])


plt.figure()
plt.plot(t_d[0,-1,41,:], amp[4,0,-4,41,:],label='Signal off, pump on')

plt.plot(t_d[0,-1,41,:], amp[4,1,0,41,:],label='Signal on, pump off')

plt.plot(t_d[0,-1,41,:], amp[4,1,-4,41,:],label='Signal on, pump on')
plt.legend()




plt.figure()
plt.plot(omega_s/(2*np.pi), np.mean(amp[2,0,-5,:,:],axis=1),label='Signal off, pump on')

plt.plot(omega_s/(2*np.pi), np.mean(amp[2,1,0,:,:],axis=1),label='Signal on, pump off')

plt.plot(omega_s/(2*np.pi), np.mean(amp[2,1,-5,:,:],axis=1),label='Signal on, pump on')
plt.legend()



