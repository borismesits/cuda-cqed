from cuda_cqed.sim import Sim
# import gpu_odes.HatGPUODE_D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

pi = np.pi

sim = Sim(use_complex=True)

sim.add_param('wa', 6.0e9*2*pi)
sim.add_param('wb', 3.8e9*2*pi)
sim.add_param('sqrtka', np.sqrt(10e6*2*pi)) # in MHz
sim.add_param('sqrtkb', np.sqrt(10e6*2*pi)) # in MHz
sim.add_param('gab', 10e6 * 2 * np.pi)
sim.add_param('gbc', 10e6 * 2 * np.pi)
sim.add_param('g3', 50e6 * 2 * np.pi)
sim.add_param('g4', 10e6 * 2 * np.pi)
sim.add_param('gr', 1e6 * 2 * np.pi)
sim.add_param('amplR',  0.1)  # 0 - readout drive
sim.add_param('amplC1', 0.1)  # 6 - c1
sim.add_paramsweep('wR', 5.5e9 * 2 * np.pi, 5.5e9 * 2 * np.pi, 101, is_excitation=True)  # 1
sim.add_param('wC1', -2.2e9*2*np.pi)  # 7
sim.add_param('rampR', 1e-9)  # 2
sim.add_param('rampC1', 1e-9)  # 8
sim.add_param('startR', 5e-9)  # 3
sim.add_param('stopR', 200e-9)  # 4
sim.add_param('startC1', 5e-9)  # 9
sim.add_param('stopC1', 200e-9)  # 10
sim.add_param('phaseR', -np.pi / 4)  # 5
sim.add_param('phaseC1', np.pi / 3)  # 11

Rpulse = sim.make_pulse('wR', 'amplR', 'phaseR', 'startR', 'stopR', 'rampR')
C1pulse = sim.make_pulse('wC1', 'amplC1', 'phaseC1', 'startC1', 'stopC1', 'rampC1')

sim.add_EOM('bin', C1pulse)
sim.add_EOM('ain', Rpulse)

sim.add_EOM('a', '-1j*wa*a - 1j*gab*b*conjugate(bin) - ain*sqrtka - (sqrtka**2/2)*a')
sim.add_EOM('b', '-1j*wb*b - 1j*gab*a*bin - 1j*b*g4*abs(b)**2 - bin*sqrtkb - (sqrtkb**2/2)*b ')
sim.set_solve_type('decimate')

sim.specify_time(20, 1200, d_factor=1)

sim.validate()

x, t = sim.quick_trace()

plt.figure(1)
plt.clf()
plt.plot(t*1e9, x[0,:]/np.max(x[0, :]),color=(1,0,0,0.7),label='3WM drive')
plt.plot(t*1e9, x[2,:]/np.max(x[2, :])+2,color=(0,1,0,0.5),label='resonator drive')
plt.plot(t*1e9, x[4,:]/np.max(x[4, :])+4,color=(0.3,0.3,0.3,0.5),label='a mode')
plt.plot(t*1e9, x[6,:]/np.max(x[6, :])+6,color=(0,0,1,0.5),label='b mode')
plt.xlabel('Time (ns)')
plt.ylabel('Normalized amplitude')
plt.legend()
plt.grid()
plt.show()