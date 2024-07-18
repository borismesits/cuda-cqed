from sim_wrapper.sim import Sim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

pi = np.pi

SimpleSim = Sim(use_complex=True)


SimpleSim.add_param('omega_a', 9e9*2*pi)
SimpleSim.add_paramsweep('omega_b', 7e9*2*pi, 11e9*2*pi, 100)
SimpleSim.add_param('g', 1e8*2*pi)
SimpleSim.add_paramsweep('omega_d1', 8e9*2*pi, 10e9*2*pi, 1000, is_excitation=True)
SimpleSim.add_param('sqrtkappa_a', np.sqrt(3e7*2*pi))
SimpleSim.add_param('sqrtkappa_b', np.sqrt(1e6*2*pi))
SimpleSim.add_param('As', 1)

SimpleSim.add_EOM('ain', '-1j*omega_d1*As*exp(-1j*omega_d1*t)')  # intput output theory
SimpleSim.add_EOM('a', '-1j*omega_a*a - 1j*g*(b+conjugate(b)) - (sqrtkappa_a**2/2)*a - sqrtkappa_a*ain')
SimpleSim.add_EOM('b', '-1j*omega_b*b - 1j*g*(a+conjugate(a)) - (sqrtkappa_b**2/2)*b ')
SimpleSim.set_solve_type('decimate')

SimpleSim.specify_time(20, 1200, d_factor=1)

SimpleSim.validate()

x, t = SimpleSim.quick_trace()

plt.figure(1)
plt.plot(t,x[2,:])

plt.figure(2)
fftx = np.fft.fft(x[2, :])
freqs = np.linspace(0, len(t)/t[-1], len(t))
plt.loglog(freqs, np.abs(fftx).transpose())

I, Q, t = SimpleSim.solve()

ain_I = I[0,:,:,:]
ain_Q = Q[0,:,:,:]

a_I = I[2,:,:,:]
a_Q = Q[2,:,:,:]

sqrtkappa_a = SimpleSim.param_dict_nosweep['sqrtkappa_a']

aout_I = ain_I + sqrtkappa_a*a_I
aout_Q = ain_Q + sqrtkappa_a*a_Q

phase = np.angle(1j*aout_Q+aout_I)

plt.figure(3)
plt.pcolor(SimpleSim.paramsweep_dict['omega_b']/(2*pi), SimpleSim.paramsweep_dict['omega_d1']/(2*pi), phase[:,:,-1].transpose())

plt.xlabel('Mode Freq.')
plt.ylabel('Probe Freq.')
