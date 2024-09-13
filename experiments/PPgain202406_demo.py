from sim_wrapper.sim import Sim
import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

SimpleSim = Sim()

SimpleSim.add_param('omega_0', 5e9*2*pi)
SimpleSim.add_paramsweep('omega_d', 4e9*2*pi, 6e9*2*pi, 101, isExcitation=True)
SimpleSim.add_param('kappa', 1e8*2*pi)

SimpleSim.add_EOM('a', '-1j*omega_0*a - kappa*a + cos(t*omega_d)')
SimpleSim.set_solve_type('decimate')

SimpleSim.use_complex = True

SimpleSim.specify_time(20,100)

SimpleSim.validate()

x, t = SimpleSim.quick_trace()

plt.plot(t,x.transpose())

I,Q,t = SimpleSim.solve()

plt.plot(I[0,:,:].transpose(),Q[0,:,:].transpose())
