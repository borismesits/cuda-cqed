from cuda_cqed.sim import Sim
# import gpu_odes.HatGPUODE_D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

pi = np.pi

sim = Sim(use_complex=True)


sim.add_param('sqrtka_ext', np.sqrt(20e6*2*pi)) # in MHz
sim.add_param('ka_int', 3e6*2*pi) # in MHz
sim.add_param('g4', 10e6 * 2 * np.pi)
sim.add_param('gr', 1e6 * 2 * np.pi)
sim.add_param('amplR', 100)
sim.add_param('wa', 5.0e9*2*pi)
sim.add_paramsweep('wR', 4.8e9 * 2 * np.pi, 5.2e9 * 2 * np.pi, 1001, is_excitation=True)
sim.add_param('rampR', 1e-9)
sim.add_param('startR', 5e-9)
sim.add_param('stopR', 200e-9)
sim.add_param('phaseR', 0)

Rpulse = sim.make_pulse('wR', 'amplR', 'phaseR', 'startR', 'stopR', 'rampR')

sim.add_EOM('ain', Rpulse)

sim.add_EOM('a', '-1j*wa*a - ain*sqrtka_ext - (sqrtka_ext**2 + ka_int)/2*a - 1j*a*g4*abs(a)**2', IC=0)
sim.set_solve_type('decimate')

sim.specify_time(20, 500, d_factor=1)

sim.validate(print_result=True)

# I, Q, t = sim.solve()