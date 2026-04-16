from sim_wrapper.sim import Sim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

sim = Sim(use_complex=False)

sim.add_param('w_0', 6.0e9 * 2 * np.pi)
sim.add_param('amp_in', 0.4) # signal amplitude
sim.add_param('kappa', 0.1e9 * 2 * np.pi)  # mode decay rate
sim.add_param('phase_in', 0)
sim.add_paramsweep('epsilon', 0, 0.1, 100)
sim.add_param('w_phi', 0.5e9 * 2 * np.pi)
sim.add_paramsweep('w_in', 6e9*np.pi*2, 7e9*np.pi*2, 3, is_excitation=True) # signal frequency

sim.add_EOM('x1', 'x2')
sim.add_EOM('x2', '-kappa*x2 - w_0**2*(1 + epsilon*cos(t*w_phi))*sin(x1) + 2*kappa*(amp_in*w_in*cos(w_in*t + phase_in))')
sim.add_EOM('x3','x2 - amp_in*w_in*cos(w_in*t + phase_in)')

sim.set_solve_type('decimate')

sim.specify_time(20, 5000, d_factor=1)

sim.validate()

sim.param_dict_nosweep['amp_in'] = 0.1
sim.param_dict_nosweep['phi_DC'] = 0.28
sim.param_dict_nosweep['epsilon'] = 0.1
sim.param_dict_nosweep['w_phi'] = 0.5e9 * 2 * np.pi
sim.param_dict_nosweep['kappa'] = 1e9 * 2 * np.pi

epsilons = np.linspace(0, 0.1, 10)





