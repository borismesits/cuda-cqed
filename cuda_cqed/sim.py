import numpy as np
import matplotlib.pyplot as plt
from cuda_cqed.HatGPUODE_D.util import generate_kernel, generate_pycode
from cuda_cqed.HatGPUODE_D.RK_solver_decimate import GPUODE_decimate
from cuda_cqed.HatGPUODE.RK_solver import GPUODE
from cuda_cqed.HatGPUODE_D.RK_solver_CPU import RK_loop_CPU
from cuda_cqed.HatGPUODE_D.RK_solver_CPU_old import RK_loop_CPU_old
import matplotlib
import time
import warnings
from numpy import *

class Sim():
    '''
    This is a bookkeeping class that
        - takes user description of a system
        - generates a GPU-readable kernel
        - sets up the time variables
        - organizes all the variables so that you can sweep over multiple axes
        - calls the solver code
        - presents some plotting functions.
    '''
    def __init__(self, use_complex=False):

        self.param_dict = {}
        self.param_dict_nosweep = {}
        self.paramsweep_dict = {}
        self.solve_type = ''
        self.var_strs = []
        self.eom_strs = []
        self.drive_var_strs = [] # these equations define the variable values, not derivatives, for convenient drive terms
        self.drive_eom_strs = []  # these equations define the variable values, not derivatives, for convenient drive terms
        self.IC_strs = []
        self.use_complex = use_complex  # are the dependent variables complex (like Langevin equation) or real (like classical)?
        self.excitation_freq = '' # the name of a special variable that represents a drive or pump frequency. Determines decimation parameters. Only one variable can be this
        self.PTS_PER_CYCLE = None
        self.NUM_CYCLES = None
        self.is_time_specified = False
        self.num_drive_terms = 0

    def make_pulse(self, omega, ampl, phi, start, stop, ramp, A='0', B='0', C='0', D='0'):

        omega_string = omega + ' + ' + str(A) + '*t + ' + str(B) + '*t**2'

        return '-' + str(ampl) + '/2*exp(-1j*((' + omega_string + ')*t +' + str(phi) + '))*(tanh((t-' + str(
            start) + ')/' + str(
            ramp) + ')-tanh((t-' + str(stop) + ')/' + str(ramp) + '))'

    def make_pulse_derivative(self, omega, A, phi, start, stop, ramp):
        '''
        this function no longer has a use
        '''
        return '-' + str(A) + '/2*' + str(omega) + '*exp(-1j*(' + str(omega) + '*t +' + str(phi) + '))*(tanh((t-' + str(
            start) + ')/' + str(
            ramp) + ')-tanh((t-' + str(stop) + ')/' + str(ramp) + ')) + ' + str(A) + '/(2*' + str(
            ramp) + ')*exp(-1j*(' + str(omega) + \
            '*t +' + str(phi) + '))*(sech((t-' + str(start) + ')/' + str(ramp) + ')**2-sech((t-' + str(
                stop) + ')/' + str(ramp) + ')**2)'

    def make_pulse_debug(self, omega, A, phi, start, stop, ramp):

        return str(A) + '/2*exp(-1j*(' + str(omega) + '*t +' + str(phi) + '))*(tanh((t-' + str(start) + ')/' + str(
            ramp) + ')-tanh((t-' + str(stop) + ')/' + str(ramp) + '))'

    def make_pulse_sequence(self, pulses):

        pulse_sequence = ''
        for pulse in pulses:
            pulse_sequence += pulse + ' + '

        return pulse_sequence[0:-3]

    def add_EOM(self, var_str, eom_str, IC_str='0'):
        '''
        In order for the solver to parse our description of some set of equations, we need to specify the following.
        You can also add parameters in a later step.
        :param var_str: string, the name of dependent var (not time)
        :param eom_str: string, equation of motion
        :param IC_str: string, initial condition
        '''
        self.var_strs.append(var_str)
        self.eom_strs.append(eom_str)
        self.IC_strs.append(IC_str)

    def add_drive_EOM(self, drive_var_str, drive_eom_str):
        '''
        In order for the solver to parse our description of some set of equations, we need to specify the following.
        You can also add parameters in a later step.
        :param var_str: string, the name of dependent var (not time)
        :param drive_str: string, drive equation
        '''
        self.drive_var_strs.append(drive_var_str)
        self.drive_eom_strs.append(drive_eom_str)

        if self.use_complex:
            self.num_drive_terms += 2
        else:
            self.num_drive_terms += 1

    def add_param(self, name, value, is_excitation=False):

        self.param_dict[name] = value
        self.paramsweep_dict[name] = value

        if is_excitation:
            self.excitation_freq = value
            self.excitation_freq_nosweep = self.excitation_freq  # just for quick trace

        self.param_dict_nosweep[name] = value

    def add_paramsweep(self, name, value_i, value_f, pts, is_excitation=False):

        self.param_dict[name] = [value_i, value_f, pts]
        self.paramsweep_dict[name] = np.linspace(value_i, value_f, pts)

        '''
        The reason to store the two param_dicts separately is that we can't pass a list of values efficiently into
        a cupy kernel. We need to generate the linear sweep inside the kernel, and we only build the initial and final
        values and the number of steps into the definition of the kernel. However, it's handy for plotting to have 
        the array of values in an numpy array.
        '''

        self.param_dict_nosweep[name] = np.linspace(value_i, value_f, pts)[pts//2]

        if is_excitation:
            self.excitation_freq = np.linspace(value_i, value_f, pts)
            self.excitation_freq_nosweep = np.linspace(value_i, value_f, pts)[pts // 2] # just for quick trace

    def set_solve_type(self, type_str):
        '''
        decimate does online (fast) decimation on the GPU, and saves on the GPU.
        avg just saves the final values, as well as mean and variance.
        saveall_gpu stores all (undecimated) data on the GPU
        saveall stores all (decimated) data, transferring the data to a specified
        '''
        TYPES = ['decimate', 'avg', 'all']

        if np.any(type_str == np.array(TYPES)):

            self.solve_type = type_str

        else:
            warnings.warn("Specified solve type not found. Options are 'decimate', 'avg', and 'all'.", RuntimeWarning)
            return

    def specify_time(self, pts=None, t_f=None, pts_per_cycle=None, num_cycles=None, d_factor=None):
        '''
        :param pts_per_cycle: Integration points per cycle, where a cycle is the period of the excitation frequency (see add_param())
        :param num_cycles: How many cycles or periods to run the simulation for
        :param d_factor: By default, the data is decimated over every cycle (not half cycle). You can make the decimation window larger by this factor
        '''

        print('Recent change to specify_time(), check implementation')

        if self.solve_type == None:

            warnings.warn("Specify solve type before specifying time.", RuntimeWarning)
            return

        self.PTS_PER_CYCLE = pts_per_cycle
        self.NUM_CYCLES = num_cycles
        self.d_factor_mult = d_factor

        self.S = pts
        self.t_f = t_f

        self.is_time_specified = True

    def initialize_time(self):

        if self.solve_type == 'decimate':

            if self.PTS_PER_CYCLE < 20:
                warnings.warn("Points per cycle recommended to be at least 20 for accuracy.", RuntimeWarning)

            self.D_FACTOR = self.PTS_PER_CYCLE*self.d_factor_mult  # decimation factor

            # below is the decimation frequency
            excitation_freq = self.excitation_freq
            self.d_omega = np.ones(self.shape[1:]) * excitation_freq
            d_omega_dt = np.ones(self.shape[1:]) * excitation_freq  # this is here in case you want to turn demod freq to 0, which would otherwise create infinitely long timesteps
            self.dt = (2 * np.pi) / (d_omega_dt * self.PTS_PER_CYCLE)
            self.S = self.PTS_PER_CYCLE * self.NUM_CYCLES

        elif self.solve_type == 'all':
            self.dt = self.t_f/self.S

        else:
            raise NotImplementedError

    def validate(self, print_result=False):

        if self.eom_strs == []:
            warnings.warn("No equations of motion defined. Use Sim.add_EOM().", RuntimeWarning)
            return

        if np.any(self.excitation_freq) == '':
            warnings.warn("No excitation variable defined. Decimation will not work. Use isExcitation=True on one of the frequency parameters.", RuntimeWarning)
            return

        if self.solve_type == '':
            warnings.warn("No solve type specified. Use Sim.set_solve_type().", RuntimeWarning)
            return

        if self.is_time_specified == False:
            warnings.warn("Time variables not specified. Use Sim.specify_time().", RuntimeWarning)
            return

        try:

            kernel_string, IC_kernel_string, kernel, IC_kernel, shape = generate_kernel(self.var_strs, self.eom_strs, self.drive_var_strs, self.drive_eom_strs, self.IC_strs,
                                                                                                   self.param_dict,
                                                                                                   use_complex=self.use_complex, print_result=print_result)
            self.kernel_string = kernel_string
            self.IC_kernel_string = IC_kernel_string
            self.kernel = kernel
            self.IC_kernel = IC_kernel
            self.shape = shape

            eom_nps, numpy_kernel_string, numpy_IC_kernel_string = generate_pycode(self.var_strs, self.eom_strs, self.drive_var_strs, self.drive_eom_strs, self.IC_strs,  self.param_dict_nosweep, use_complex=self.use_complex, print_result=print_result)

            self.eom_nps = eom_nps
            self.numpy_kernel_string = numpy_kernel_string
            self.numpy_IC_kernel_string = numpy_IC_kernel_string

            ldict = locals()
            exec(self.numpy_kernel_string, globals(), ldict)
            numpy_dxdt = ldict['numpy_dxdt']
            self.numpy_kernel = numpy_dxdt

            exec(self.numpy_IC_kernel_string, globals(), ldict)
            numpy_get_ICs = ldict['numpy_get_ICs']
            self.numpy_get_ICs = numpy_get_ICs

            self.initialize_time()

            print('Simulation validation success!')

        except RuntimeError:
            warnings.warn("Kernel generation failed.", RuntimeWarning)

    def quick_trace(self, print_kernel=False):
        '''
        The tricky thing about GPU accleration is, it's almost never faster for just one or a few parallel sims. The
        advantage comes with massive parallelization. Thus, this function runs a CPU simulation, saving all data, for
        just a single variation.
        '''

        self.validate(print_result=print_kernel)
        if self.solve_type == 'decimate':
            dt = 2*np.pi/(self.excitation_freq_nosweep * self.PTS_PER_CYCLE)
            t = np.linspace(0, dt * self.PTS_PER_CYCLE * self.NUM_CYCLES, self.PTS_PER_CYCLE * self.NUM_CYCLES + 1)[
                0:self.PTS_PER_CYCLE * self.NUM_CYCLES]
        elif self.solve_type == 'all':
            dt = self.dt
            t = np.arange(0, self.t_f, self.dt)

        M = len(self.eom_nps) # number of modes

        self.var_strs_updated = []
        if self.use_complex == False:
            self.var_strs_updated = self.var_strs
        if self.use_complex == True:

            for var_str in self.var_strs:
                self.var_strs_updated.append(var_str + '_R')
                self.var_strs_updated.append(var_str + '_I')

        x = RK_loop_CPU(M, t, self.numpy_kernel, self.numpy_get_ICs, self.num_drive_terms)

        return x, t

    # def make_quicktrace_param_dict(self):
    #
    #     '''Quick solve doesn't solve sweeps, so this function just picks the middle value from a sweep as a default'''
    #
    #     self.param_dict_nosweep = {}
    #
    #     for param in self.param_dict:
    #         try:
    #             L = len(self.param_dict[param])
    #             self.param_dict_nosweep[param] = self.param_dict[param][L//2]
    #         except TypeError:
    #             self.param_dict_nosweep[param] = self.param_dict[param]
    #
    #     if np.shape(self.excitation_freq) != ():
    #         L = len(self.excitation_freq)
    #         self.excitation_freq_nosweep = self.excitation_freq[L // 2]
    #     else:
    #         self.excitation_freq_nosweep = self.excitation_freq

    def solve(self, only_final=False):

        self.validate()

        if self.solve_type == 'decimate':
            I_demod, Q_demod, t_d = GPUODE_decimate(self.dt, self.shape, self.kernel, self.IC_kernel, self.D_FACTOR,
                                                    self.d_omega, self.S, self.num_drive_terms, only_final=only_final)
            return I_demod, Q_demod, t_d

        elif self.solve_type == 'all':
            x_demod, t_d = GPUODE(np.array(self.dt), self.shape, self.kernel, self.IC_kernel, self.S, self.num_drive_terms)
            # todo add only_final option for 'all' solve type

            return x_demod, t_d
        else:
            raise NotImplementedError


