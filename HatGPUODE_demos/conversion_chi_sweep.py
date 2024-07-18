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
SimpleSim.add_param('sqrtkappa_a', np.sqrt(1e5*2*pi))
SimpleSim.add_param('sqrtkappa_b', np.sqrt(1e5*2*pi))
SimpleSim.add_param('As_in', 2.8)
SimpleSim.add_param('Aa_in', 0.1)
SimpleSim.add_param('gab', 10e6*2*pi)
SimpleSim.add_paramsweep('chi', 0, 1e8*2*pi, 500)
SimpleSim.add_paramsweep('qbstate', -1, 1, 2)

SimpleSim.add_param('startR', 0e-8)
SimpleSim.add_paramsweep('stopR', 0.1e-8, 2e-8, 100)
SimpleSim.add_param('rampR', 1e-9)

SimpleSim.add_param('delayC', 3e-8)
SimpleSim.add_param('lenC', 4e-8)
SimpleSim.add_param('rampC', 1e-9)

SimpleSim.add_EOM('s_in', 'omega_d1*As_in*cos(omega_d1*t)*(tanh((t-(stopR+delayC))/rampC)-tanh((t-(stopR+delayC+lenC))/rampC))')
SimpleSim.add_EOM('a_in', 'omega_a*Aa_in*cos(omega_a*t)*(tanh((t-startR)/rampR)-tanh((t-stopR)/rampR))')
SimpleSim.add_EOM('a', '-1j*(omega_a + chi*qbstate)*a - 1j*gab*b*conjugate(s_in) - (sqrtkappa_a**2/2)*a - sqrtkappa_a*a_in')
SimpleSim.add_EOM('b', '-1j*omega_b*b - 1j*gab*a*s_in - (sqrtkappa_b**2/2)*b')
SimpleSim.set_solve_type('decimate')

SimpleSim.specify_time(20, 1200, d_factor=48)

SimpleSim.validate()

x, t = SimpleSim.quick_trace()

plt.figure(1)
plt.plot(t,x[4,:])
plt.plot(t,x[6,:])

plt.figure(2)
fftx = np.fft.fft(x[4, :])
freqs = np.linspace(0, len(t)/t[-1], len(t))
plt.loglog(freqs, np.abs(fftx).transpose())

I, Q, t = SimpleSim.solve()

a_I = I[4,:,:,:,:]
a_Q = Q[4,:,:,:,:]

ge_distance = np.sqrt((a_I[:, 0, :, :] - a_I[:, 1, :, :])**2 + (a_Q[:, 0, :, :] - a_Q[:, 1, :, :])**2)

plt.figure(3)
plt.pcolor(SimpleSim.paramsweep_dict['stopR']*1e6, SimpleSim.paramsweep_dict['chi']/(1e6*2*np.pi), ge_distance[:,:,-1])

plt.title('GE state distance, simulation')
plt.ylabel('Chi/(2pi) (MHz)')
plt.xlabel('Time (us)')