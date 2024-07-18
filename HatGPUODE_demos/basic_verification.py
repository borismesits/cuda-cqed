from sim_wrapper.sim import Sim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

pi = np.pi

SimpleSim = Sim(use_complex=True)

SimpleSim.add_param('omega_0', 10e9*2*pi)
SimpleSim.add_param('kappa', 1e8*2*pi)
SimpleSim.add_param('As', 1)
SimpleSim.add_param('omega_d1', 10e9*2*pi, is_excitation=True)

SimpleSim.add_EOM('a', '-1j*omega_0*a - kappa*a ', IC=1)
SimpleSim.set_solve_type('decimate')

SimpleSim.specify_time(40, 20, d_factor=1)

SimpleSim.validate(print_result=True)

x, t = SimpleSim.quick_trace()

kappa = SimpleSim.param_dict_nosweep['kappa']
omega_0 = SimpleSim.param_dict_nosweep['omega_0']

plt.figure(1)
plt.plot(t,x[0,:])
plt.plot(t, np.exp(-kappa*t)*np.cos(omega_0*t)) # analytical solution, im very good at math I swear :)
# plt.plot(t,x[1,:])



I, Q, t = SimpleSim.solve()

plt.figure(1)

plt.plot(t, I[0,:])
plt.plot(t, I[1,:])

# plt.plot(t, np.angle(I[0,:]+1j*Q[0,:]))
