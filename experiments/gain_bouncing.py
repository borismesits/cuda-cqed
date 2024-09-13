from sim_wrapper.sim import Sim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

pi = np.pi

SimpleSim = Sim(use_complex=True)

SimpleSim.add_param('omega_0', 8.1e9*2*pi)
SimpleSim.add_param('omega_d1', 8e9*2*pi, is_excitation=True)
SimpleSim.add_param('omega_p', 16e9*2*pi)
SimpleSim.add_param('g', 1e9*2*pi)

SimpleSim.add_EOM('a', '- 1j*omega_0*a - 1*g*conjugate(a)*a + 2e10*cos(omega_p*t) + 1e7*cos(omega_d1*t)')
SimpleSim.set_solve_type('decimate')

SimpleSim.specify_time(30, 500, d_factor=2)

SimpleSim.validate()


x, t = SimpleSim.quick_trace()

plt.figure(1)
plt.clf()
plt.plot(t,x.transpose())


plt.figure(2)
fftx = np.fft.fft(x[0, :])
freqs = np.linspace(0, len(t)/t[-1], len(t))
plt.loglog(freqs, np.abs(fftx).transpose())

I, Q, t = SimpleSim.solve()

plt.figure(3)
plt.plot(t, I[0,:])
plt.plot(t, Q[0,:])