from sim_wrapper.sim import Sim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

pi = np.pi

SimpleSim = Sim(use_complex=True)

SimpleSim.add_param('omega_0', 10e9*2*pi)
SimpleSim.add_param('sqrtkappa', np.sqrt(1e8*2*pi))
SimpleSim.add_paramsweep('logAs', 4.2, 5, 500)
SimpleSim.add_paramsweep('omega_d1', 9.5e9*2*pi, 10.5e9*2*pi, 500, is_excitation=True)

SimpleSim.add_EOM('bin', '-1j*omega_d1*(10**logAs)*exp(-1j*omega_d1*t)')  # intput output theory
SimpleSim.add_EOM('a', '-1j*omega_0*sin(a) - sqrtkappa*bin - sqrtkappa**2/2*a')
SimpleSim.set_solve_type('decimate')

SimpleSim.specify_time(40, 200, d_factor=1)

SimpleSim.validate()

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

bin_I = I[0,:]
bin_Q = Q[0,:]

a_I = I[2,:]
a_Q = Q[2,:]

kappa = SimpleSim.param_dict_nosweep['sqrtkappa']**2
omega_0 = SimpleSim.param_dict_nosweep['omega_0']
omega_d1 = SimpleSim.paramsweep_dict['omega_d1']
logAs = SimpleSim.paramsweep_dict['logAs']

bout_I = bin_I + np.sqrt(kappa)*a_I
bout_Q = bin_Q + np.sqrt(kappa)*a_Q

mag = np.sqrt(bout_I**2 + bout_Q**2)

phase = np.angle(bout_I+1j*bout_Q)

plt.pcolor(phase[:,:,-1])
plt.colorbar()