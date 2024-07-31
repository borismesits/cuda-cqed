from sim_wrapper.sim import Sim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

pi = np.pi

SimpleSim = Sim(use_complex=True)

SimpleSim.add_param('wa', 5.9e9*2*pi, is_excitation=True)
SimpleSim.add_param('wb', 4.0e9*2*pi)
SimpleSim.add_param('wc', 7.5e9*2*pi)
SimpleSim.add_paramsweep('logchi', 5, 9, 40)
SimpleSim.add_paramsweep('logka', 5, 8, 30)
SimpleSim.add_param('kb', 0.2e6 * 2 * np.pi)
SimpleSim.add_param('kc', 4e6 * 2 * np.pi)
SimpleSim.add_param('gab', 10e6 * 2 * np.pi)
SimpleSim.add_param('gbc', 10e6 * 2 * np.pi)
SimpleSim.add_param('g3', 50e6 * 2 * np.pi)
SimpleSim.add_param('gr', 10e6 * 2 * np.pi)
SimpleSim.add_paramsweep('qbstate', -1, 1, 2)

SimpleSim.add_param('amplR',  0.8)  # 0 - readout drive
SimpleSim.add_paramsweep('amplC1', 0, 2, 51)  # 6 - c1
SimpleSim.add_param('amplG', 0.0)  # 18 - gain
SimpleSim.add_param('amplC2', 0.0)  # 12 - c2
SimpleSim.add_param('wR', 5.90e9 * 2 * np.pi)  # 1
SimpleSim.add_param('wC1', -1.9e9 * 2 * np.pi)  # 7
SimpleSim.add_param('wG', 8.0e9 * 2 * np.pi)  # 19
SimpleSim.add_param('wC2', 3.5e9 * 2 * np.pi)  # 13
SimpleSim.add_param('rampR', 1e-9)  # 2
SimpleSim.add_param('rampC1', 1e-9)  # 8
SimpleSim.add_param('rampG', 1e-9)  # 14
SimpleSim.add_param('rampC2', 1e-9)  # 20
SimpleSim.add_param('startR', 20e-9)  # 3
SimpleSim.add_param('stopR', 40e-9)  # 4
SimpleSim.add_param('startC1', 50e-9)  # 9
SimpleSim.add_param('stopC1', 100e-9)  # 10
SimpleSim.add_param('startG', 110e-9)  # 21
SimpleSim.add_param('stopG', 170e-9)  # 22
SimpleSim.add_param('startC2', 190e-9)  # 15
SimpleSim.add_param('stopC2', 195e-9)  # 16
SimpleSim.add_param('phaseR', -np.pi / 4)  # 5
SimpleSim.add_param('phaseC1', np.pi / 3)  # 11
SimpleSim.add_param('phaseG', np.pi * 0.88)  # 23
SimpleSim.add_param('phaseC2', 0.0)  # 17

Rpulse = SimpleSim.make_pulse('wR', 'amplR', 'phaseR', 'startR', 'stopR', 'rampR')
C1pulse = SimpleSim.make_pulse('wC1', 'amplC1', 'phaseC1', 'startC1', 'stopC1', 'rampC1')
Gpulse = SimpleSim.make_pulse('wG', 'amplG', 'phaseG', 'startG', 'stopG', 'rampG')
C2pulse = SimpleSim.make_pulse('wC2', 'amplC2', 'phaseC2', 'startC2', 'stopC2', 'rampC2')

SimpleSim.add_EOM('s0', C1pulse + ' + ' + Gpulse + ' + ' + C2pulse)
SimpleSim.add_EOM('s1', Rpulse)


SimpleSim.add_EOM('a', '-1j*(wa+10**logchi*qbstate)*a - 1j*gab*b*conjugate(s0) - 1j*s1*gr - 10**(logka)*a')
SimpleSim.add_EOM('b', '-1j*wb*b - 1j*gab*a*s0 - 1j*gbc*c*conjugate(s0) - 1j*g3*conjugate(b)*s0 - kb*b')
SimpleSim.add_EOM('c', '-1j*wc*c - 1j*gbc*b*s0 - kc*c')
SimpleSim.set_solve_type('decimate')

SimpleSim.specify_time(20, 500, d_factor=1)


SimpleSim.validate()

SimpleSim.param_dict_nosweep['logchi'] = 5
SimpleSim.param_dict_nosweep['logka'] = 10

x, t = SimpleSim.quick_trace()

plt.figure(1)
plt.clf()
plt.plot(t, x[0,:]/np.max(x[0,:]),color=(1,0,0,0.7))
plt.plot(t, x[2,:]/np.max(x[2,:])+2,color=(0,1,0,0.5))
plt.plot(t, x[4,:]/np.max(x[4,:])+4,color=(0,0,1,0.5))
plt.plot(t, x[6,:]/np.max(x[6,:])+6,color=(1,0,1,0.5))
plt.plot(t, x[8,:]/np.max(x[8,:])+8,color=(0,0.8,0.8,0.5))

plt.figure(2)
fftx = np.fft.fft(x[4, :])
freqs = np.linspace(0, len(t)/t[-1], len(t))
plt.loglog(freqs, np.abs(fftx).transpose())

# I, Q, t = SimpleSim.solve()
#
# a_nbar = np.sqrt(I[4,:]**2+Q[4,:]**2)
# b_nbar = np.sqrt(I[6,:]**2+Q[6,:]**2)
# c_nbar = np.sqrt(I[8,:]**2+Q[8,:]**2)
#
# a_separation = np.sqrt((I[4,:,:,1,:]-I[4,:,:,0,:])**2 + (Q[4,:,:,1,:]-Q[4,:,:,0,:])**2)
# b_separation = np.sqrt((I[6,:,:,1,:]-I[6,:,:,0,:])**2 + (Q[6,:,:,1,:]-Q[6,:,:,0,:])**2)
# # c_separation = np.sqrt((I[8,:,1,:]-I[8,:,0,:])**2 + (Q[8,:,1,:]-Q[8,:,0,:])**2)
#
# logchi = SimpleSim.paramsweep_dict['logchi'] - np.log10(2*pi)
# logka = SimpleSim.paramsweep_dict['logka'] - np.log10(2*pi)
#
# plt.figure(4)
# plt.pcolor(np.log10(a_nbar[10,10,0,:, :]+a_nbar[10,10,1,:, :]), vmin=-3, vmax=2)
# plt.colorbar()
#
# plt.figure(4)
# plt.plot(np.log10(a_nbar[10,10,0,:, -1]+a_nbar[10,10,1,:, -1]))
#
#
