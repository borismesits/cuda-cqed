from sim_wrapper.sim import Sim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

sim = Sim(use_complex=False)

sim.add_param('w_0', 6e9 * 2 * np.pi)
sim.add_paramsweep('amp_in', 0, 0.1, 50) # signal amplitude
sim.add_paramsweep('phi_max', 0, 5, 2) # pump amplitude
sim.add_param('kappa', 0.5e9 * 2 * np.pi)  # mode decay rate
sim.add_param('kappa_r', 1e8 * 2 * np.pi) # filter bandwidth
sim.add_paramsweep('phase_in', 0, 2*np.pi, 21)
sim.add_param('pf', 2), # pump frequency factor
sim.add_param('phi_DC', 1.5)
sim.add_param('w_J', 3e9 * 2 * np.pi)
sim.add_param('w_in', 6e9*2*np.pi, is_excitation=True) # signal frequency

sim.add_EOM('x1', 'x2')
sim.add_EOM('x2', '-kappa*x2 - w_0**2*x1 - w_J**2*( sin(x1 + phi_DC) - sin(phi_DC) - cos(phi_DC)*x1) + 2*kappa*(-amp_in*w_in*cos(w_in*t + phase_in) - pf*phi_max*w_in*cos(pf*w_in*t))')
sim.add_EOM('x3','x2 + amp_in*w_in*cos(w_in*t + phase_in) + pf*phi_max*w_in*cos(pf*w_in*t)')

sim.set_solve_type('decimate')

sim.specify_time(20, 1000, d_factor=1)

sim.validate()

x, t = sim.quick_trace()


plt.figure(1)
plt.clf()
plt.plot(t*1e9, x[0,:]/np.max(x[0,:]),color=(1,0,0,0.7),label='3WM drive')
plt.plot(t*1e9, x[2,:]/np.max(x[1,:])+2,color=(0,1,0,0.5),label='resonator drive')
plt.xlabel('Time (ns)')
plt.ylabel('Normalized amplitude')
plt.legend()
plt.yticks([])
plt.grid()

plt.figure(2)
fftx = np.fft.fft(x[2, :])
freqs = np.linspace(0, len(t)/t[-1], len(t))
plt.loglog(freqs, np.abs(fftx).transpose())
plt.xlim([5e9, 15e9])

