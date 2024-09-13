from sim_wrapper.sim import Sim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

'''
This is as close as possible to the original Clerk description of a nondegenerate parametric amplifier.
The lack of a resonant term makes this simulation almost trivial (and easy pen-and-paper), so just for verification.
'''

pi = np.pi

SimpleSim = Sim(use_complex=True)


SimpleSim.add_param('omega_d', 5e9*2*pi)
SimpleSim.add_param('omega_adc', 5e9*2*pi, is_excitation=True)
SimpleSim.add_param('kappa_S', 2e8*2*pi)
SimpleSim.add_param('kappa_I', 1e8*2*pi)
SimpleSim.add_param('lambdaa', 4e7*2*pi)
SimpleSim.add_paramsweep('logS', 8,10,10)

SimpleSim.add_EOM('bin_S', '0', IC=1)
SimpleSim.add_EOM('bin_I', '0', IC=0)

SimpleSim.add_EOM('a_S', '-(kappa_S/2)*a_S + lambdaa*conjugate(a_I) - sqrt(abs(kappa_S))*bin_S')
SimpleSim.add_EOM('a_I', '-(kappa_I/2)*a_I + lambdaa*conjugate(a_S) - sqrt(abs(kappa_I))*bin_I')
SimpleSim.set_solve_type('decimate')

SimpleSim.specify_time(10, 1000, d_factor=1)

SimpleSim.validate()

lambdaa = SimpleSim.paramsweep_dict['lambdaa']
kappa_S = SimpleSim.paramsweep_dict['kappa_S']
kappa_I = SimpleSim.paramsweep_dict['kappa_I']

x, t = SimpleSim.quick_trace()
bin_S_R = x[0,:]
bin_S_I = x[1,:]
bin_I_R = x[2,:]
bin_I_I = x[3,:]
a_S_R = x[4,:]
a_S_I = x[5,:]
a_I_R = x[6,:]
a_I_I = x[7,:]
bout_S_R = bin_S_R + np.sqrt(kappa_S)*a_S_R
bout_S_I = bin_S_I + np.sqrt(kappa_S)*a_S_I
bout_I_R = bin_I_R + np.sqrt(kappa_I)*a_I_R
bout_I_I = bin_I_I + np.sqrt(kappa_I)*a_I_I

plt.figure(1)
plt.plot(t, bin_S_R)
plt.plot(t, bout_S_I)
plt.plot(t, bin_S_R)
plt.plot(t, bout_S_I)

Q = 2*lambdaa/np.sqrt(kappa_I*kappa_S)

bout_S_I_theory = (Q**2+1)/(Q**2-1)*bin_S_R - 2*Q/(Q**2 - 1) * bin_I_R
bout_S_Q_theory = (Q**2+1)/(Q**2-1)*bin_S_I - 2*Q/(Q**2 - 1) * bin_I_I

G = ((Q**2+1)/(Q**2-1))**2
