from sim_wrapper.sim import Sim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

pi = np.pi

SimpleSim = Sim()

SimpleSim.add_param('omega_0', 5e9*2*pi)
SimpleSim.add_param('omega_d', 5e9*2*pi)
SimpleSim.add_param('omega_adc', 5e9*2*pi, is_excitation=True)
SimpleSim.add_param('kappa', 1e8*2*pi)
SimpleSim.add_param('g', 5e9*2*pi)
SimpleSim.add_paramsweep('logA', 6, 9, 90)

SimpleSim.add_EOM('a', '-1j*omega_0*a -1j*g*(a)**3 + 10**(logA)*cos(omega_d*t) - kappa*a')
SimpleSim.set_solve_type('decimate')

SimpleSim.use_complex = True

SimpleSim.specify_time(30, 150, d_factor=15)

SimpleSim.validate()

x, t = SimpleSim.quick_trace()

plt.figure(1)
plt.plot(t,x[0,:])

plt.figure(2)
fftx = np.fft.fft(x[0, :])
freqs = np.linspace(0, len(t)/t[-1], len(t))
plt.loglog(freqs, np.abs(fftx).transpose())

I, Q, t = SimpleSim.solve()

plt.figure(3)
plt.plot(I[0, :, :].transpose(), Q[0, :, :].transpose())

plt.figure(4)
plt.semilogy(SimpleSim.paramsweep_dict['logA'], I[0, :, -1].transpose())


plt.figure(5)
plt.semilogy(t.transpose(), I[0, :, :].transpose())

SimpleSim.add_param('omega_adc', 15e9*2*pi, is_excitation=True)
I, Q, t = SimpleSim.solve()

plt.figure(3)
plt.plot(I[0, :, :].transpose(), Q[0, :, :].transpose())

plt.figure(4)
plt.semilogy(SimpleSim.paramsweep_dict['logA'], I[0, :, -1].transpose())


plt.figure(5)
plt.semilogy(t.transpose(), I[0, :, :].transpose())
