from sim_wrapper.sim import Sim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

pi = np.pi

SimpleSim = Sim(use_complex=True)


SimpleSim.add_param('omega_a', 9e9*2*pi)
SimpleSim.add_param('omega_b', 8e9*2*pi)
SimpleSim.add_param('g', 1e8*2*pi)
SimpleSim.add_param('alpha', -1e8*2*pi)

SimpleSim.add_param('sqrtkappa_a', np.sqrt(1e8*2*pi))
SimpleSim.add_param('sqrtkappa_b', np.sqrt(1e6*2*pi))
SimpleSim.add_paramsweep('logAs', 1, 10, 50)
SimpleSim.add_paramsweep('omega_d1', 8.975e9*2*pi, 9.025e9*2*pi, 300, is_excitation=True)

SimpleSim.add_EOM('ain', 'omega_d1*10**(logAs)*cos(omega_d1*t)')  # intput output theory
SimpleSim.add_EOM('a', '-1j*omega_a*a - 1j*g*(b+conjugate(b))  - (sqrtkappa_a**2/2)*a - sqrtkappa_a*ain')
SimpleSim.add_EOM('b', '-1j*omega_b*b - 1j*g*(a+conjugate(a)) - 1j*alpha*conjugate(b)*b*b - (sqrtkappa_b**2/2)*b ')
SimpleSim.set_solve_type('decimate')

SimpleSim.specify_time(100, 120, d_factor=1)

SimpleSim.validate()

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
plt.pcolor(SimpleSim.paramsweep_dict['omega_d1']/(2*pi), SimpleSim.paramsweep_dict['logAs'], np.unwrap(phase[:,:,-1],axis=1))
plt.xlabel('Drive freq.')
plt.ylabel('Log10 Amplitude')

plt.figure(4)
plt.plot(180/pi*phase[0,:,-1]+90)
plt.xlabel('Drive freq.')
