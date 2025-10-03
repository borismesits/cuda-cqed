from cuda_cqed.sim import Sim
# import gpu_odes.HatGPUODE_D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

pi = np.pi

sim = Sim(use_complex=True)


sim.add_param('sqrtka_ext', np.sqrt(2e6*2*pi)) # in MHz
sim.add_param('ka_int', 1e6*2*pi) # in MHz
sim.add_param('g4', 0e6 * 2 * np.pi)
sim.add_param('amplR', 0)
sim.add_param('wa', 5.0e9*2*pi)
sim.add_paramsweep('ICphase', 0, 2*np.pi, 101)
sim.add_param('wR', 5.0e9*2*np.pi, is_excitation=True)
sim.add_param('rampR', 1e-9)
sim.add_param('startR', 5e-9)
sim.add_param('stopR', 200e-9)
sim.add_param('phaseR', 0)

Rpulse = sim.make_pulse('wR', 'amplR', 'phaseR', 'startR', 'stopR', 'rampR')

sim.add_EOM('ain', Rpulse)
sim.add_EOM('a', '-1j*wa*a - ain*sqrtka_ext - (sqrtka_ext**2 + ka_int)/2*a - 1j*a*g4*abs(a)**2', IC_str='cos(ICphase)')

sim.set_solve_type('decimate')

sim.specify_time(20, 50, d_factor=1)

sim.validate(print_result=True)

x, t = sim.quick_trace()

I, Q, t = sim.solve()

Id = I.copy()
Qd = Q.copy()
td = t.copy()

ain = Id[0,:]+1j*Qd[0,:]
a = Id[2,:]+1j*Qd[2,:]

plt.plot(np.real(a).transpose())