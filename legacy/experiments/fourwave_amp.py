from sim_wrapper.sim import Sim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

pi = np.pi

SimpleSim = Sim(use_complex=True)

SimpleSim.add_param('omega_0', 9e9*2*pi)
SimpleSim.add_param('omega_d1', 9e9*2*pi, is_excitation=True)
SimpleSim.add_param('omega_p', 18e9*2*pi)
SimpleSim.add_param('sqrtkappa', np.sqrt(1e9*2*pi))
SimpleSim.add_param('g', 1e9*2*pi)
SimpleSim.add_paramsweep('logAs', 2, 4, 100)
SimpleSim.add_paramsweep('Ap', 0, 500000, 5)

SimpleSim.add_EOM('bin', '10**logAs * omega_d1*sin(omega_d1*t) + Ap*omega_p*sin(omega_p*t)')  # intput output theory
SimpleSim.add_EOM('a', '-1j*omega_0*a -1j*g*conjugate(a)*a*a - (sqrtkappa**2/2)*a - sqrtkappa*bin')
SimpleSim.set_solve_type('decimate')

SimpleSim.specify_time(10, 1200, d_factor=10)

SimpleSim.validate()

x, t = SimpleSim.quick_trace()

plt.figure(1)
plt.plot(t,x[2,:])

plt.figure(2)
fftx = np.fft.fft(x[2, :])
freqs = np.linspace(0, len(t)/t[-1], len(t))
plt.loglog(freqs, np.abs(fftx).transpose())

I, Q, t = SimpleSim.solve()

bin_I = I[0,:,:]
bin_Q = Q[0,:,:]

a_I = I[2,:,:]
a_Q = Q[2,:,:]

sqrtkappa = SimpleSim.param_dict_nosweep['sqrtkappa']

bout_I = bin_I + sqrtkappa*a_I
bout_Q = bin_Q + sqrtkappa*a_Q

plt.figure(3)
plt.plot(SimpleSim.paramsweep_dict['logAs'], np.mean(np.sqrt(bout_I**2+bout_Q**2)[:,4,:],axis=1))
plt.plot(SimpleSim.paramsweep_dict['logAs'], np.mean(np.sqrt(bout_I**2+bout_Q**2)[:,3,:],axis=1))
plt.plot(SimpleSim.paramsweep_dict['logAs'], np.mean(np.sqrt(bout_I**2+bout_Q**2)[:,2,:],axis=1))
plt.plot(SimpleSim.paramsweep_dict['logAs'], np.mean(np.sqrt(bout_I**2+bout_Q**2)[:,1,:],axis=1))
plt.plot(SimpleSim.paramsweep_dict['logAs'], np.mean(np.sqrt(bout_I**2+bout_Q**2)[:,0,:],axis=1))
