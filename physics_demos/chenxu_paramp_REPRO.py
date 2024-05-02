import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from HatGPUODE.util import generate_kernel
from HatGPUODE.RK_solver import related_rates_problem
import matplotlib
import time

matplotlib.use('Qt5Agg')

def rum_sim(w_in_i, w_in_f, w_in_pts, signalOn, pumpOn):
    variations1 = w_in_pts
    variations2 = 10


    N = 9
    t = np.linspace(0,5e-8,25001)[0:-1]

    noise_mask = cp.zeros([N, variations1*variations2])
    noise_mask[0, :] = 0

    var_strs = ['phi','deriv_phi','phi_out','filter','deriv_filter','filter2','deriv_filter2','Idemod','Qdemod'] # define your dependent variables/coordinates
    exp_strs = ['deriv_phi',
                '-Gamma*deriv_phi - w_0**2 * phi - w_J**2 * ( sin(phi + phi_DC) - sin(phi_DC) - cos(phi_DC) * phi) + 2*Gamma*(-amp_in * w_in * sin(w_in*t + phase_in)*signalOn - phi_max*w_d * cos(w_d*t)*pumpOn)',
                'deriv_phi - (-amp_in*w_in*sin(w_in*t + phase_in)*signalOn - phi_max*w_d * cos(w_d*t)*pumpOn)',
                'deriv_filter',
                '-w_in**2 * filter + phi_out - kappa*deriv_filter',
                'deriv_filter2',
                '-w_in**2 * filter2 + filter - kappa*deriv_filter2',
                'filter2*cos(w_in*t)*(sign(t-wait_t)+1)/2',
                'filter2*sin(w_in*t)*(sign(t-wait_t)+1)/2'] # define the time derivatives of each coordinate

    params = [('w_0', 6e9 * 2 * np.pi),
              ('w_J', 2.4e9 * 2 * np.pi),
              ('Gamma', 0.6e9 * 2 * np.pi),
              ('phi_DC', 1.5),
              ('w_d', 2 * 6e9 * 2*np.pi),
              ('phi_max', 1.1),  # mode kappa
              ('w_in', [w_in_i, w_in_f, w_in_pts]),  # readout kappa
              ('phase_in', -1.3),
              ('amp_in', [0.0001,1,variations2]),
              ('signalOn', signalOn),
              ('pumpOn', pumpOn),
              ('kappa',3e8 * 2*np.pi),
              ('wait_t',2.5e-8)]

    kernel_input, kernel_output, kernel_body, kernel_op = generate_kernel(var_strs, exp_strs, params)
    print(kernel_input)
    print(kernel_body)
    print(kernel_output)


    start_time = time.time()
    x, x_avg, saved_x = related_rates_problem(t, N, variations1, variations2, kernel_op, noise_mask, save_i=np.array([-1]))
    print(time.time()-start_time)

    return x, x_avg, saved_x, t

w_in_i = 1e9*2*np.pi
w_in_f = 10e9*2*np.pi
w_in_pts = 10000
w_in = np.linspace(w_in_i,w_in_f,w_in_pts)

x1, x_avg1, saved_x1, t = rum_sim(w_in_i, w_in_f, w_in_pts, 1, 0)
x2, x_avg2, saved_x2, t = rum_sim(w_in_i, w_in_f, w_in_pts, 0, 1)
x3, x_avg3, saved_x3, t = rum_sim(w_in_i, w_in_f, w_in_pts, 1, 1)

x1 = np.reshape(cp.asnumpy(x1), (9,w_in_pts, 10))
x2 = np.reshape(cp.asnumpy(x2), (9,w_in_pts, 10))
x3= np.reshape(cp.asnumpy(x3), (9,w_in_pts, 10))

start = 15000

# plt.figure()
#
# freq = np.linspace(0,(len(t[start:])-1)/t[-start-1],len(t[start:])+1)[0:-1]
#
# plt.loglog(freq, np.abs(np.fft.fft(x_avg1[5,start:])))
# plt.loglog(freq, np.abs(np.fft.fft(x_avg2[5,start:])))
# plt.loglog(freq, np.abs(np.fft.fft(x_avg3[5,start:])))



# gain = (x3[7,0:7,:]**2+x3[8,0:7,:]**2)/(x1[7,0:7,:]**2+x1[8,0:7,:]**2)
amp1 = x3[7,:,:]**2+x3[8,:,:]**2
phase1 = np.arctan(x3[8,:,:]/x3[7,:,:])
plt.figure()
plt.plot(w_in/(2*np.pi),amp1)

plt.figure()
plt.plot(w_in/(2*np.pi),phase1)


