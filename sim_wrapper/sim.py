import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from HatGPUODE_D.util import generate_kernel, generate_pycode
from HatGPUODE_D.RK_solver_decimate import GPUODE_decimate
from HatGPUODE_D.RK_solver_CPU import RK_loop_CPU
import matplotlib
import time
import warnings

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
    def __init__(self):

        self.param_dict = {}
        self.paramsweep_dict = {}
        self.solve_type = ''
        self.var_strs = []
        self.eom_strs = []
        self.use_complex = False  # are the dependent variables complex (like Langevin equation) or real (like classical)?
        self.excitation_freq = '' # the name of a special variable that represents a drive or pump frequency. Determines decimation parameters. Only one variable can be this
        self.PTS_PER_CYCLE = None
        self.NUM_CYCLES = None

    def add_EOM(self, var_str, eom_str):
        '''
        In order for the solver to parse our description of some set of equations, we need to specify the following.
        You can also add parameters in a later step.
        :param var_list: list of strings, the names of dependent vars (not time)
        :param eom_list: the equations of motion
        '''

        self.var_strs.append(var_str)
        self.eom_strs.append(eom_str)

    def add_param(self, name, value, is_excitation=False):

        self.param_dict[name] = value
        self.paramsweep_dict[name] = value

        if is_excitation:
            self.excitation_freq = name

    def add_paramsweep(self, name, value_i, value_f, pts, is_excitation=False):

        self.param_dict[name] = [value_i, value_f, pts]
        self.paramsweep_dict[name] = np.linspace(value_i, value_f, pts)

        '''
        The reason to store the two param_dicts separately is that we can't pass a list of values efficiently into
        a cupy kernel. We need to generate the linear sweep inside the kernel, and we only build the initial and final
        values and the number of steps into the definition of the kernel. However, it's handy for plotting to have 
        the array of values in an numpy array.
        '''

        if is_excitation:
            self.excitation_freq = name

    def set_solve_type(self, type_str):
        '''
        decimate does online (fast) decimation on the GPU, and saves on the GPU.
        avg just saves the final values, as well as mean and variance.
        saveall_gpu stores all (undecimated) data on the GPU
        saveall stores all (decimated) data, transferring the data to a specified
        '''
        TYPES = ['decimate','avg','save_all']

        if np.any(type_str == np.array(TYPES)):

            self.solve_type = type_str

        else:
            warnings.warn("Specified solve type not found.", RuntimeWarning)
            return

    def set_ICs(self):
        '''
        by default initial conditions are all zero. Change that here
        :return:
        '''

        pass

    def specify_time(self, pts_per_cycle, num_cycles, d_factor=1):

        self.PTS_PER_CYCLE = pts_per_cycle
        self.NUM_CYCLES = num_cycles
        self.d_factor_mult = d_factor

    def initialize_time(self):

        if self.solve_type == 'decimate':

            self.D_FACTOR = self.PTS_PER_CYCLE*self.d_factor_mult  # decimation factor

            # below is the decimation frequency

            excitation_freq = self.param_dict[self.excitation_freq]

            self.d_omega = np.ones(self.shape[1:]) * excitation_freq

            d_omega_dt = np.ones(self.shape[1:]) * excitation_freq  # this is here in case you want to turn demod freq to 0, which would otherwise create infinitely long timesteps

            self.dt = (2 * np.pi) / (d_omega_dt * self.PTS_PER_CYCLE)

            self.S = self.PTS_PER_CYCLE * self.NUM_CYCLES

        else:
            raise NotImplementedError

    def validate(self):

        if self.eom_strs == []:
            warnings.warn("No equations of motion defined. Use Sim.add_EOM().", RuntimeWarning)
            return

        if self.excitation_freq == '':
            warnings.warn("No excitation variable defined. Decimation will not work. Use isExcitation=True on one of the frequency parameters.", RuntimeWarning)
            return

        if self.solve_type == '':
            warnings.warn("No solve type specified. Use Sim.set_solve_type().", RuntimeWarning)
            return

        if self.PTS_PER_CYCLE == None:
            warnings.warn("Time variables not specified. Use Sim.specify_time().", RuntimeWarning)
            return

        try:

            kernel_input, kernel_output, kernel_body, kernel_op, shape = generate_kernel(self.var_strs, self.eom_strs,
                                                                                                   self.param_dict,
                                                                                                   use_complex=self.use_complex)
            self.kernel_input = kernel_input
            self.kernel_output = kernel_output
            self.kernel_body = kernel_body
            self.kernel_op = kernel_op
            self.shape = shape

            self.eom_nps = generate_pycode(self.var_strs, self.eom_strs, self.param_dict, use_complex=self.use_complex)

            self.initialize_time()

            self.make_quicktrace_param_dict()

            print('Simulation validation success!')

        except RuntimeError:
            warnings.warn("Kernel generation failed.", RuntimeWarning)

    def quick_trace(self):
        '''
        The tricky thing about GPU accleration is, it's almost never faster for just one or a few parallel sims. The
        advantage comes with massive parallelization. Thus, this function runs a CPU simulation, saving all data, for
        just a single variation.
        '''

        dt = 2*np.pi/(self.param_dict_nosweep[self.excitation_freq]*self.PTS_PER_CYCLE)

        t = np.linspace(0, dt*self.PTS_PER_CYCLE*self.NUM_CYCLES, self.PTS_PER_CYCLE*self.NUM_CYCLES)

        M = len(self.eom_nps) # number of modes

        x0 = 0 # TODO

        self.var_strs_updated = []
        if self.use_complex == False:
            self.var_strs_updated = self.var_strs
        if self.use_complex == True:

            for var_str in self.var_strs:
                self.var_strs_updated.append(var_str + '_R')
                self.var_strs_updated.append(var_str + '_I')

        x = RK_loop_CPU(M, x0, t, self.var_strs_updated, self.eom_nps, self.param_dict_nosweep)

        return x, t

    def make_quicktrace_param_dict(self):

        '''Quick solve doesn't so sweeps, so this function just picks the middle value from a sweep as a default'''

        self.param_dict_nosweep = {}

        for param in self.param_dict:
            try:
                L = len(self.param_dict[param])
                self.param_dict_nosweep[param] = self.param_dict[param][L//2]
            except TypeError:
                self.param_dict_nosweep[param] = self.param_dict[param]

    def solve(self):

        self.validate()

        I_demod, Q_demod, t_d = GPUODE_decimate(self.dt, self.shape, self.kernel_op, self.D_FACTOR, self.d_omega, self.S)

        return I_demod, Q_demod, t_d




