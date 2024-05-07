import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from HatGPUODE.util import generate_kernel
from HatGPUODE.RK_solver import related_rates_problem
import matplotlib
import time

matplotlib.use('Qt5Agg')

def rum_sim(w_in_i, w_in_f, w_in_pts, signalOn, pumpOn):
    variations2 = w_in_pts
    variations1 = 10


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

    params = [('w_0', 6.426939330532695e9 * 2 * np.pi),
              ('w_J', 8.507189549448235e9 * 2 * np.pi),
              ('Gamma', 4.547284088339866e9 * 2 * np.pi),
              ('phi_DC', np.pi/2),
              ('w_d', 2 * 6.42e9 * 2*np.pi),
              ('phi_max', [0.5,1.4,variations1]),  # mode kappa
              ('w_in', [w_in_i, w_in_f, w_in_pts]),  # readout kappa
              ('phase_in', -1.3),
              ('amp_in', 0.001),
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

w_in_i = 5e9*2*np.pi
w_in_f = 8e9*2*np.pi
w_in_pts = 1000
w_in = np.linspace(w_in_i,w_in_f,w_in_pts)

x1, x_avg1, saved_x1, t = rum_sim(w_in_i, w_in_f, w_in_pts, 1, 0)
x2, x_avg2, saved_x2, t = rum_sim(w_in_i, w_in_f, w_in_pts, 0, 1)
x3, x_avg3, saved_x3, t = rum_sim(w_in_i, w_in_f, w_in_pts, 1, 1)

x1 = np.reshape(cp.asnumpy(x1), (9,10,w_in_pts))
x2 = np.reshape(cp.asnumpy(x2), (9,10,w_in_pts))
x3= np.reshape(cp.asnumpy(x3), (9,10,w_in_pts))

start = 15000

# plt.figure()
#
# freq = np.linspace(0,(len(t[start:])-1)/t[-start-1],len(t[start:])+1)[0:-1]
#
# plt.loglog(freq, np.abs(np.fft.fft(x_avg1[5,start:])))
# plt.loglog(freq, np.abs(np.fft.fft(x_avg2[5,start:])))
# plt.loglog(freq, np.abs(np.fft.fft(x_avg3[5,start:])))

plt.figure()

gain = (x3[7,0:7,:]**2+x3[8,0:7,:]**2)/(x1[7,0:7,:]**2+x1[8,0:7,:]**2)

plt.semilogy(w_in/(2*np.pi),gain.transpose())


