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
SimpleSim.add_param('As', 10)
SimpleSim.add_paramsweep('Ap', 0, 100000, 2)


SimpleSim.add_param('start', 1e-8)
SimpleSim.add_paramsweep('stop', 2e-8, 8e-8, 7)
SimpleSim.add_param('ramp', 1e-9)

SimpleSim.add_EOM('bin', 'As*omega_d1*sin(omega_d1*t)*(tanh((t-start)/ramp)-tanh((t-stop)/ramp)) + Ap*omega_p*sin(omega_p*t)')  # intput output theory
SimpleSim.add_EOM('a', '-1j*omega_0*a -1j*g*( a*a + 2*a*conjugate(a) + conjugate(a)*conjugate(a) ) - (sqrtkappa**2/2)*a - sqrtkappa*bin')
SimpleSim.set_solve_type('decimate')

SimpleSim.specify_time(10, 1200, d_factor=10)

SimpleSim.validate()

SimpleSim.param_dict_nosweep['Ap'] = 0
x, t = SimpleSim.quick_trace()

plt.figure(1)
plt.plot(t,x[0,:])

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
plt.plot(t[0,:,:].transpose(), bout_I[0,:,:].transpose())

