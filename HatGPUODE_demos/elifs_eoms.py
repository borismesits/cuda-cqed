from sim_wrapper.sim import Sim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

pi = np.pi

SimpleSim = Sim(use_complex=True)

SimpleSim.add_param('omega_S', 16e6*2*pi)
SimpleSim.add_param('kappa_S', 1e6*2*pi)
SimpleSim.add_param('kappa_a', 1e6*2*pi)
SimpleSim.add_param('kappa_b', 1e6*2*pi)
SimpleSim.add_param('Lambdaa', 0.1e6*2*pi)
SimpleSim.add_paramsweep('A_s', 0, 10, 2)
SimpleSim.add_paramsweep('logA_a', -5, 0, 100)
SimpleSim.add_param('delta', 8e6*2*pi, is_excitation=True)

SimpleSim.add_EOM('S', '1j*omega_S*S - 1j*Lambdaa*a*b - kappa_S/2 * S + 1j*omega_S*A_s*exp(1j*omega_S*t)')  # intput output theory
SimpleSim.add_EOM('a', '(1j*delta - kappa_a/2) * a - 1j* Lambdaa*conjugate(b)*S + 1j*delta*(10**logA_a)*exp(1j*delta*t)')
SimpleSim.add_EOM('b', '(-1j*delta - kappa_b/2) * b - 1j* Lambdaa*conjugate(a)*S ')
SimpleSim.set_solve_type('decimate')

SimpleSim.specify_time(40, 100, d_factor=2)

SimpleSim.validate(print_result=True)

SimpleSim.param_dict_nosweep['A_s'] = 10

x, t = SimpleSim.quick_trace()

plt.figure(1)
plt.plot(t,x[2,:])

plt.figure(2)
xs = x[:,len(t)//2:]
ts = t[len(t)//2:]
fftx = np.fft.fft(xs[2, :])
freqs = np.linspace(0, len(t)/(t[-1]), len(t)//2)
plt.loglog(freqs, np.abs(fftx).transpose())

I, Q, t = SimpleSim.solve()

S_I = I[0,:]
S_Q = Q[0,:]
a_I = I[2,:]
a_Q = Q[2,:]
b_I = I[4,:]
b_Q = Q[4,:]

disp_a_final = np.sqrt(a_I**2+a_Q**2)[:, :, -1]
disp_b_final = np.sqrt(b_I**2+b_Q**2)[:, :, -1]

gain_a = (disp_a_final[1,:]/disp_a_final[0,:])**2

plt.figure(3)
plt.plot(disp_a_final.transpose())

