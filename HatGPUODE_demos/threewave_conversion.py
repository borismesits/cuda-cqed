from sim_wrapper.sim import Sim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

pi = np.pi

SimpleSim = Sim(use_complex=True)

SimpleSim.add_param('omega_a', 4.0e9*2*pi, is_excitation=True)
SimpleSim.add_param('omega_s', 6.0e9*2*pi)
SimpleSim.add_param('omega_p', 2e9*2*pi)
SimpleSim.add_param('ka', 1e6*2*pi)
SimpleSim.add_param('ks', 1e6*2*pi)
SimpleSim.add_param('g_as', 10e6*2*pi)
SimpleSim.add_param('g3', 1e9*2*pi)
SimpleSim.add_param('g4', 0e9*2*pi)
SimpleSim.add_paramsweep('logAp', -1, -0.5, 200)
SimpleSim.add_paramsweep('phi', 0, 2*pi, 9)

Cpulse = SimpleSim.make_pulse('omega_p', '10**logAp', 'phi', 10e-9, 100e-9, 1e-9)

SimpleSim.add_EOM('a', '-1j*omega_a*a - 1j*g_as*(s + conjugate(s))**2 - ka/2*a - 1j*g_as*s', IC=0.01)
SimpleSim.add_EOM('s', '-1j*omega_s*s - 1j*g3*(s + conjugate(s))**2 + 1j*6*g_as*(a + conjugate(a))*(s + conjugate(s)) - ks/2*s- 1j*g_as*a -1j*' + Cpulse)
SimpleSim.set_solve_type('decimate')

SimpleSim.specify_time(100, 500, d_factor=2)

SimpleSim.validate()

x, t = SimpleSim.quick_trace()

plt.figure(1)
plt.plot(x[0,:]/(np.max(x[0,:])))
plt.plot(x[2,:]/(np.max(x[2,:])))
plt.grid()

plt.figure(2)
xs = x[:,len(t)//2:]
ts = t[len(t)//2:] - t[len(t)//2]
fftx = np.fft.fft(xs[0, :])
freqs = np.linspace(0, len(ts)/ts[-1], len(ts))
plt.loglog(freqs, np.abs(fftx).transpose())

I, Q, t = SimpleSim.solve()

phi_i = 5

mag_a = np.sqrt(I[0,:,phi_i,:]**2 + Q[0,:,phi_i, :]**2)

plt.figure(1)
plt.clf()
plt.pcolor(mag_a,vmin=0,vmax=0.02)
plt.colorbar()


