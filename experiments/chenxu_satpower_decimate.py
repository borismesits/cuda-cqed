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
            '-kappa*x2 - w_0**2*x1 - w_J**2*( sin(x1 + phi_DC) - sin(phi_DC) - cos(phi_DC)*x1) + 2*kappa*(-amp_in*w_in*cos(w_in*t + phase_in) - pf*phi_max*w_in*cos(pf*w_in*t))',
            'x2 + amp_in*w_in*cos(w_in*t + phase_in) + pf*phi_max*w_in*cos(pf*w_in*t)',
            'r2',
            '-w_in**2 * r1 - kappa_r*r2 + x3']

w_in_i = 3e9 * 2 * np.pi
w_in_f = 9e9 * 2 * np.pi
swp_pts = 100

params = [('w_0', 6e9 * 2 * np.pi),
          ('amp_in', [0, 0.1, 50]), # signal amplitude
          ('phi_max', [0, 5, 2]), # pump amplitude
          ('kappa', 0.5e9 * 2 * np.pi),  # mode decay rate
          ('kappa_r', 1e8 * 2 * np.pi), # filter bandwidth
          ('phase_in', [0, 2*np.pi, 21]),
          ('pf', 2), # pump frequency factor
          ('phi_DC', 1.5),
          ('w_J', 3e9 * 2 * np.pi),
          ('w_in', [w_in_i, w_in_f, swp_pts])] # signal frequency

kernel_input, kernel_output, kernel_body, kernel_op, shape, var_dict = generate_kernel(var_strs, exp_strs, params, use_complex=False)

print('Shape: ' + str(shape))

PTS_PER_CYCLE = 100
NUM_CYCLES = 200

D_FACTOR = PTS_PER_CYCLE  # decimation factor

# below is the decimation frequency

omega_s = np.linspace(w_in_i, w_in_f, swp_pts)  # this will generally need to have shape M (since demod frequency may need to change with drive or mixing products)

d_omega = np.ones(shape[1:])*omega_s

d_omega_dt = np.ones(shape[1:])*omega_s  # this is here in case you want to turn demod freq to 0, which would otherwise create infinitely long timesteps

dt = (2*np.pi)/(d_omega_dt*PTS_PER_CYCLE)

S = PTS_PER_CYCLE*NUM_CYCLES

start_time = time.time()
I_demod, Q_demod, t_d = GPUODE_decimate(dt, shape, kernel_op, D_FACTOR, d_omega, S)
print(time.time()-start_time)

amp_max = np.max(np.sqrt(I_demod**2 + Q_demod**2), axis=3)
amp_min = np.min(np.sqrt(I_demod**2 + Q_demod**2), axis=3)
phase = np.arctan(Q_demod/I_demod)

modeN = 2

plt.figure()
pwrgain = 20*np.log10(amp_max[modeN,:,1,:,-1]/amp_max[modeN,:,0,:,-1])
plt.pcolor(var_dict['w_in']/(2*np.pi), var_dict['amp_in'], pwrgain, cmap='seismic')
plt.colorbar()
plt.clim([-20,20])
plt.contour(var_dict['w_in']/(2*np.pi), var_dict['amp_in'], pwrgain,'k--', levels=[20])

plt.figure()
pwrgain = 20*np.log10(amp_min[modeN,:,1,:,-1]/amp_min[modeN,:,0,:,-1])
plt.pcolor(var_dict['w_in']/(2*np.pi), var_dict['amp_in'], pwrgain, cmap='seismic')
plt.colorbar()
plt.clim([-20,20])
plt.contour(var_dict['w_in']/(2*np.pi), var_dict['amp_in'], pwrgain,'k--', levels=[20])

plt.figure()
pwrgain = 20*np.log10(amp_max[modeN,:,1,:,-1]/amp_max[modeN,:,0,:,-1])
plt.plot(var_dict['amp_in'][1:], pwrgain[1:,len(var_dict['w_in'])//2])

phi_max_i = 25
w_in_i = 51
phase_i = 0

plt.figure()
plt.plot(t_d[0,-1,phase_i,w_in_i,:], I_demod[modeN,0,phi_max_i,phase_i,w_in_i,:],label='Signal off, pump on')

plt.plot(t_d[0,-1,phase_i,w_in_i,:], I_demod[modeN,1,0,phase_i,w_in_i,:],label='Signal on, pump off')

plt.plot(t_d[0,-1,phase_i,w_in_i,:], I_demod[modeN,1,phi_max_i,phase_i,w_in_i,:],label='Signal on, pump on')
plt.legend()


plt.figure()
plt.plot(omega_s/(2*np.pi), amp_max[modeN,0,phi_max_i,:,-1],'b',label='Signal off, pump on', linewidth=3)
plt.plot(omega_s/(2*np.pi), amp_min[modeN,0,phi_max_i,:,-1],'b--', linewidth=3)

plt.plot(omega_s/(2*np.pi), amp_max[modeN,1,0,:,-1],'r',label='Signal on, pump off', linewidth=2.5)
plt.plot(omega_s/(2*np.pi), amp_min[modeN,0,phi_max_i,:,-1],'r--', linewidth=2.5)

plt.plot(omega_s/(2*np.pi), amp_max[modeN,1,phi_max_i,:,-1],'g',label='Signal on, pump on', linewidth=2)
plt.plot(omega_s/(2*np.pi), amp_min[modeN,1,phi_max_i,:,-1],'g--', linewidth=2)
plt.legend()



'''
the problem is that calculating the output light (not the intracavity light) has an arbitrary constant of integration,
and if you don't pick the right C (or start integration with offset) depending on RHS phase, then you get a DC component 
in the output that messes up the demodulation.

however, I think this just gets filtered out with demodulation, so you should only see it if demod freq is 0
'''