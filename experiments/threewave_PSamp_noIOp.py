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
SimpleSim.add_param('sqrtkappa', np.sqrt(5e9*2*pi))
SimpleSim.add_param('g', 1e9*2*pi)
SimpleSim.add_param('g4', 0e9*2*pi)
SimpleSim.add_paramsweep('logAs', -7, -4, 20)
SimpleSim.add_paramsweep('Ap', 0, 0.99, 2)
SimpleSim.add_paramsweep('phi', 0, 2*pi, 1001)

SimpleSim.add_EOM('a', '-1j*omega_0*a - 1j*g*(a + conjugate(a))**2 - 1j*g4*(conjugate(a)*a*a) - (sqrtkappa**2/2)*a -1j*10**logAs*omega_d1*exp(-1j*(omega_d1*t + phi)) + -1j*omega_p*Ap*exp(-1j*omega_p*t) ')
SimpleSim.set_solve_type('decimate')

SimpleSim.specify_time(100, 2, d_factor=2)

SimpleSim.validate()

x, t = SimpleSim.quick_trace()

plt.figure(1)
plt.plot(t,x[0,:])

plt.figure(2)
xs = x[:,len(t)//2:]
ts = t[len(t)//2:]
fftx = np.fft.fft(xs[0, :])
freqs = np.linspace(0, len(t)/t[-1]/2, len(t)//2)
plt.loglog(freqs, np.abs(fftx).transpose())

# I, Q, t = SimpleSim.solve()
#
# a_I = I[0,:]
# a_Q = Q[0,:]
#
#
# disp_final = np.sqrt(a_I**2+a_Q**2)[:, :, :, -1]
#
# gain = (disp_final[:,1,:]/disp_final[:,0,:])**2
#
# plt.figure(3)
# plt.pcolor(SimpleSim.paramsweep_dict['phi'], SimpleSim.paramsweep_dict['logAs'], 10*np.log10(gain))
# plt.colorbar()
#
#
# plt.figure(5)
# plt.loglog(10**SimpleSim.paramsweep_dict['logAs'],disp_final[:,0,25],label='pump off')
# plt.loglog(10**SimpleSim.paramsweep_dict['logAs'],disp_final[:,1,25],label='pump on')
# plt.xlabel('Signal in amplitude')
# plt.ylabel('Signal out amplitude')
# plt.grid()
# plt.legend()
#
# plt.figure(6)
# plt.plot(t[50,1,50,:], bout_I[50,1,50,:])
#
# plt.figure(7)
# plt.semilogy(SimpleSim.paramsweep_dict['phi']/(2*pi), gain[50,:],label='gain')
# plt.semilogy(SimpleSim.paramsweep_dict['phi']/(2*pi), gain[50,:]*0+1,label='unity')
# plt.legend()
# plt.xlabel('Signal-pump relative phase (rad/2pi)')