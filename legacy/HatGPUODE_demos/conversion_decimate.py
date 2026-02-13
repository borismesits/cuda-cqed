from sim_wrapper.sim import Sim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

pi = np.pi

SimpleSim = Sim(use_complex=True)

SimpleSim.add_param('omega_a', 6e9*2*pi)
SimpleSim.add_param('omega_b', 8e9*2*pi, is_excitation=True)
SimpleSim.add_param('omega_d1', 2e9*2*pi)
SimpleSim.add_param('kappa', np.sqrt(1e5*2*pi))
SimpleSim.add_param('As_in', 3)
SimpleSim.add_paramsweep('gab', 1e6*2*pi, 10e6*2*pi, 4)

SimpleSim.add_EOM('s_in', 'omega_d1*As_in*cos(omega_d1*t)')
SimpleSim.add_EOM('a', '-1j*omega_a*a - 1j*gab*b*conjugate(s_in) - kappa**2/2*a', IC=1)
SimpleSim.add_EOM('b', '-1j*omega_b*b - 1j*gab*a*s_in - kappa**2/2*b')
SimpleSim.set_solve_type('decimate')

SimpleSim.specify_time(20, 1200, d_factor=48)

SimpleSim.validate()

x, t = SimpleSim.quick_trace()

plt.figure(1)
plt.plot(t,x[4,:])

plt.figure(2)
fftx = np.fft.fft(x[4, :])
freqs = np.linspace(0, len(t)/t[-1], len(t))
plt.loglog(freqs, np.abs(fftx).transpose())

I, Q, t = SimpleSim.solve()

plt.figure(3)
plt.plot(I[4,:,:].transpose(), Q[4,:,:].transpose())
