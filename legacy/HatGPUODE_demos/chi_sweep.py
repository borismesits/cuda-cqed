from sim_wrapper.sim import Sim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

pi = np.pi

SimpleSim = Sim(use_complex=True)

SimpleSim.add_param('omega_0', 5.9e9*2*pi)
SimpleSim.add_param('sqrtkappa', np.sqrt(0.05e6*2*pi))
SimpleSim.add_param('As', 1)
SimpleSim.add_paramsweep('chi', 0, 1e9*2*pi, 100)
SimpleSim.add_paramsweep('qbstate', -1, 1, 2)
SimpleSim.add_param('omega_d1', 10e9*2*pi, is_excitation=True)

SimpleSim.add_EOM('bin', 'omega_d1*As*exp(-1j*omega_d1*t)')  # intput output theory
SimpleSim.add_EOM('a', '-1j*(omega_0 + chi*qbstate)*a - (sqrtkappa**2)/2*a - sqrtkappa*bin')
SimpleSim.set_solve_type('decimate')

SimpleSim.specify_time(20, 5000, d_factor=1)

SimpleSim.validate(print_result=True)

# x, t = SimpleSim.quick_trace()
#
# plt.figure(1)
# plt.plot(t,x[2,:])
#
# plt.figure(2)
# fftx = np.fft.fft(x[2, :])
# freqs = np.linspace(0, len(t)/t[-1], len(t))
# plt.loglog(freqs, np.abs(fftx).transpose())

I, Q, t = SimpleSim.solve()

bin_I = I[0,:,:]
bin_Q = Q[0,:,:]

a_I = I[2,:,:]
a_Q = Q[2,:,:]

sqrtkappa = SimpleSim.param_dict_nosweep['sqrtkappa']

bout_I = bin_I + sqrtkappa*a_I
bout_Q = bin_Q + sqrtkappa*a_Q

bout_phase = np.angle(1j*bout_Q+bout_I)

chi_i = 20

plt.figure(3)
plt.plot(a_I[chi_i, 0, :].transpose(), a_Q[chi_i, 0, :].transpose())
plt.plot(a_I[chi_i, 1, :].transpose(), a_Q[chi_i, 1, :].transpose())
plt.grid()
plt.gca().set_aspect('equal')
plt.title('Intracavity fields (IQ)')

plt.figure(4)
plt.plot(bout_I[chi_i, 0, :].transpose(), bout_Q[chi_i, 0, :].transpose())
plt.plot(bout_I[chi_i, 1, :].transpose(), bout_Q[chi_i, 1, :].transpose())
plt.grid()
plt.gca().set_aspect('equal')
plt.title('Reflected fields (IQ)')

ge_distance = np.sqrt((bout_I[:, 0, :] - bout_I[:, 1, :])**2 + (bout_Q[:, 0, :] - bout_Q[:, 1, :])**2)

plt.figure(5)
plt.pcolor(t[0, 0, :]*1e6, SimpleSim.paramsweep_dict['chi']/(2*np.pi*1e6), ge_distance)
plt.colorbar()
plt.title('Output GE state distance, simulation')
plt.ylabel('Chi/(2pi) (MHz)')
plt.xlabel('Time (us)')

ge_distance = np.sqrt((a_I[:, 0, :] - a_I[:, 1, :])**2 + (a_Q[:, 0, :] - a_Q[:, 1, :])**2)

plt.figure(6)
plt.pcolor(t[0, 0, :]*1e6, SimpleSim.paramsweep_dict['chi']/(2*np.pi*1e6), ge_distance)
plt.colorbar()
plt.title('Intracavity GE state distance, simulation')
plt.ylabel('Chi/(2pi) (MHz)')
plt.xlabel('Time (us)')

plt.figure(7)
plt.plot(SimpleSim.paramsweep_dict['chi'], phase[:,0,-1])
plt.plot(SimpleSim.paramsweep_dict['chi'], phase[:,1,-1])



