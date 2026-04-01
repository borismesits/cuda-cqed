import cupy as cp
import sympy as sp
import numpy as np
import sympy
from sympy import pycode
from numpy import *

def convert_power_arg_to_float64(inputt):
    try:
        inputt.lower()
    except:
        print('not a string')

    stringy = []

    for i in range(0,len(inputt)):
        stringy.append(inputt[i])

    for i in range(0, len(stringy)-3):
        if stringy[i] == 'p' and stringy[i+1] == 'o' and stringy[i+2] == 'w':
            j = i+3
            while stringy[j] != ')':
                j += 1
            stringy[j-1] = stringy[j-1]+'.0'

    string = ''

    for i in range(0, len(stringy)):
        string += stringy[i]

    return string


def generate_kernel(var_strs, eom_strs, drive_var_strs, drive_eom_strs, IC_strs, param_dict, use_complex=False, print_result=False):
    '''
    :param var_strs: The list of strings containing the names of quadratures (indep. variables in ODEs)
    :param eom_strs: The equations of motion. Should be in terms of var_strs
    :param drive_var_strs: these terms are given explicitly by the EoM (rather than calculating derivatives)
    :param drive_eom_strs:
    :param params: Any variables in the exp_strs not listed in var_strs must be defined here.
        There are two ways:
            e.g., ('omega_0', 10 * 2 * np.pi), OR
            e.g., ('omega_0', [10 * 2 * np.pi, 20 * 2 * np.pi, 100]) to have 100 variations
    :param use_complex: If this is true, then the indep. variables will be treated as complex valued. Twice the number
                        of degrees of freedom.
    :return kernel_input, kernel_output, kernel_body, kernel: Three strings containing sections of the kernel, and the
                                                            cupy kernel itself.
    '''

    eom_sps = []  # sympy translation
    eom_cs = []  # C code translation

    drive_eom_sps = []  # sympy translation
    drive_eom_cs = []  # C code translation

    IC_sps = []  # sympy translation
    IC_cs = []  # C code translation

    for i in range(0, len(eom_strs)):

        eom_str = eom_strs[i]
        IC_str = IC_strs[i]

        param_dict_keys = list(param_dict.keys())

        if use_complex:
            '''
            Sympy has a very convenient function called sympify, but it has tricky requirements. So I need to 
            define all the variables and parameters as sympy objects first, and then tell sympy to assume they are
            real-valued.
            '''

            symbols_and_parameters_dict = {}
            symbols_and_parameters_dict['t'] = sp.Symbol('t', real=True)  # t is a special variable

            for i in range(0, len(param_dict_keys)):
                parameter = sp.Symbol(param_dict_keys[i], real=True)
                symbols_and_parameters_dict[param_dict_keys[i]] = parameter

            for i in range(0, len(var_strs)):
                symbol = sp.Symbol(var_strs[i], real=True)
                symbols_and_parameters_dict[var_strs[i]] = symbol

            for i in range(0, len(drive_var_strs)):
                symbol = sp.Symbol(drive_var_strs[i], real=True)
                symbols_and_parameters_dict[drive_var_strs[i]] = symbol

            eom_sp = sp.sympify(eom_str, locals=symbols_and_parameters_dict)
            IC_sp = sp.sympify(IC_str, locals=symbols_and_parameters_dict)

            for i in range(0, len(var_strs)):
                symbol_R = sp.Symbol(var_strs[i] + '_R', real=True)
                symbol_I = sp.Symbol(var_strs[i] + '_I', real=True)

                eom_sp = eom_sp.subs(symbols_and_parameters_dict[var_strs[i]], symbol_R + 1j * symbol_I)
                IC_sp = IC_sp.subs(symbols_and_parameters_dict[var_strs[i]], symbol_R + 1j * symbol_I)

            for i in range(0, len(drive_var_strs)):
                symbol_R = sp.Symbol(drive_var_strs[i] + '_R', real=True)
                symbol_I = sp.Symbol(drive_var_strs[i] + '_I', real=True)

                eom_sp = eom_sp.subs(symbols_and_parameters_dict[drive_var_strs[i]], symbol_R + 1j * symbol_I)
                IC_sp = IC_sp.subs(symbols_and_parameters_dict[drive_var_strs[i]], symbol_R + 1j * symbol_I)

            eom_sp_R = sp.re(eom_sp)
            eom_sp_I = sp.im(eom_sp)
            IC_sp_R = sp.re(IC_sp)
            IC_sp_I = sp.im(IC_sp)

            eom_sps.append(eom_sp_R)
            eom_sps.append(eom_sp_I)
            IC_sps.append(IC_sp_R)
            IC_sps.append(IC_sp_I)

            eom_c_R = sp.ccode(eom_sp_R)
            eom_c_I = sp.ccode(eom_sp_I)
            eom_cs.append(eom_c_R)
            eom_cs.append(eom_c_I)

            IC_c_R = sp.ccode(IC_sp_R)
            IC_c_I = sp.ccode(IC_sp_I)
            IC_cs.append(IC_c_R)
            IC_cs.append(IC_c_I)

        else:
            eom_sp = sp.sympify(eom_str)
            IC_sp = sp.sympify(IC_str)

            eom_sps.append(eom_sp)
            IC_sps.append(IC_sp)

            eom_c = sp.ccode(eom_sp)
            eom_cs.append(eom_c)
            IC_c = sp.ccode(IC_sp)
            IC_cs.append(IC_c)

    for i in range(0, len(drive_eom_strs)):

        drive_eom_str = drive_eom_strs[i]
        param_dict_keys = list(param_dict.keys())

        if use_complex:
            drive_eom_sp = sp.sympify(drive_eom_str, locals=symbols_and_parameters_dict)

            for i in range(0, len(drive_var_strs)):

                symbol_R = sp.Symbol(drive_var_strs[i] + '_R',real=True)
                symbol_I = sp.Symbol(drive_var_strs[i] + '_I',real=True)

                drive_eom_sp = drive_eom_sp.subs(symbols_and_parameters_dict[drive_var_strs[i]], symbol_R + 1j * symbol_I)

            drive_eom_sp_R = sp.re(drive_eom_sp)
            drive_eom_sp_I = sp.im(drive_eom_sp)

            drive_eom_sps.append(drive_eom_sp_R)
            drive_eom_sps.append(drive_eom_sp_I)

            drive_eom_c_R = sp.ccode(drive_eom_sp_R)
            drive_eom_c_I = sp.ccode(drive_eom_sp_I)
            drive_eom_cs.append(drive_eom_c_R)
            drive_eom_cs.append(drive_eom_c_I)

        else:
            drive_eom_sp = sp.sympify(drive_eom_str)
            drive_eom_sps.append(drive_eom_sp)

            drive_eom_c = sp.ccode(drive_eom_sp)
            drive_eom_cs.append(drive_eom_c)

    kernel_input = 'float64 dt, float64 t,  ' # this "kernel" is a string that will get converted to a cuda GPU kernel. It defines the EoMs
    IC_kernel_input = ''  # this "kernel" handles the initial conditions

    shape = [len(eom_cs)+len(drive_eom_cs)]

    for var_str in var_strs:
        if use_complex:
            kernel_input += 'float64 ' + var_str + '_R, '
            kernel_input += 'float64 ' + var_str + '_I, '
        else:
            kernel_input += 'float64 ' + var_str + ', '

    for drive_var_str in drive_var_strs:
        if use_complex:
            kernel_input += 'float64 ' + drive_var_str + '_R_temp, '
            kernel_input += 'float64 ' + drive_var_str + '_I_temp, '
        else:
            kernel_input += 'float64 ' + drive_var_str + ', '

    kernel_output = ''
    IC_kernel_output = ''

    for var_str in var_strs:
        if use_complex:
            kernel_output += 'float64 d' + var_str + 'dt_R, '
            kernel_output += 'float64 d' + var_str + 'dt_I, '

            IC_kernel_output += 'float64 IC_' + var_str + '_R, '
            IC_kernel_output += 'float64 IC_' + var_str + '_I, '
        else:
            kernel_output += 'float64 d' + var_str + 'dt, '
            IC_kernel_output += 'float64 IC_' + var_str + 'dt, '

    for drive_var_str in drive_var_strs:
        if use_complex:
            kernel_output += 'float64 ' + drive_var_str + '_R, '
            kernel_output += 'float64 ' + drive_var_str + '_I, '

            IC_kernel_output += 'float64 IC_' + drive_var_str + '_R, '
            IC_kernel_output += 'float64 IC_' + drive_var_str + '_I, '
        else:
            kernel_output += 'float64 ' + drive_var_str + ', '

            IC_kernel_output += 'float64 ' + drive_var_str + ', '

    kernel_output = kernel_output[0:-2]
    IC_kernel_output = IC_kernel_output[0:-2]

    kernel_body = ''
    IC_kernel_body = ''

    sweep_num = 0

    for i in range(0, len(param_dict_keys)):
        try: # this is for sweep variables (specifying a range and a number of points)

            param0 = param_dict[param_dict_keys[i]][0]
            param_vars = param_dict[param_dict_keys[i]][2]
            param_range = param_dict[param_dict_keys[i]][1] - param_dict[param_dict_keys[i]][0]

            sweep_num += 1

            kernel_body += 'double ' + param_dict_keys[i] + ' = ' + str(float(param0)) + ' + index' + str(
                sweep_num) + '*' + str(
                float(param_range)) + '/' + str(float(np.max([param_vars - 1, 1]))) + ';\n'

            IC_kernel_body += 'double ' + param_dict_keys[i] + ' = ' + str(float(param0)) + ' + index' + str(
                sweep_num) + '*' + str(
                float(param_range)) + '/' + str(float(np.max([param_vars - 1, 1]))) + ';\n'

            shape.append(param_vars)

        except: # this is for constant variables
            kernel_body += 'double ' + param_dict_keys[i] + ' = ' + str(float(param_dict[param_dict_keys[i]])) + 'f;\n'
            IC_kernel_body += 'double ' + param_dict_keys[i] + ' = ' + str(float(param_dict[param_dict_keys[i]])) + 'f;\n'

    for i in range(0, sweep_num):
        kernel_input += 'int32 index' + str(i + 1) + ', '
        IC_kernel_input += 'int32 index' + str(i + 1) + ', '

    kernel_input = kernel_input[0:-2]  # removes trailing comma
    IC_kernel_input = IC_kernel_input[0:-2]

    for i in range(0, len(drive_var_strs)):
        if use_complex:
            kernel_body += drive_var_strs[i] + '_R=' + drive_eom_cs[2 * i] + '; \n'
            kernel_body += drive_var_strs[i] + '_I=' + drive_eom_cs[2 * i + 1] + '; \n'

            '''
            it may seem weird that we are defining the initial condition of the drive terms, which by definition
            don't have an initial condition (they are defined directly by EoM). This is just a quick fix to make the 
            IC kernel have the same number of outputs as the main kernel. It doesn't actually do anything, so we 
            just set it to zero.
            '''
            IC_kernel_body += 'IC_' + drive_var_strs[i] + '_R = 0; \n'
            IC_kernel_body += 'IC_' + drive_var_strs[i] + '_I = 0 ; \n'
        else:
            kernel_body += drive_var_strs[i] + ' = ' + drive_eom_cs[i] + '; \n'
            IC_kernel_body += 'IC_' + drive_var_strs[i] + ' = 0; \n'

    for i in range(0, len(var_strs)):
        if use_complex:
            kernel_body += 'd' + var_strs[i] + 'dt_R=' + eom_cs[2 * i] + '; \n'
            kernel_body += 'd' + var_strs[i] + 'dt_I= ' + eom_cs[2 * i + 1] + '; \n'

            IC_kernel_body += 'IC_' + var_strs[i] + '_R =' + IC_cs[2 * i] + '; \n'
            IC_kernel_body += 'IC_' + var_strs[i] + '_I=' + IC_cs[2 * i + 1] + '; \n'

        else:
            kernel_body += 'd' + var_strs[i] + 'dt = ' + eom_cs[i] + '; \n'
            IC_kernel_body += 'IC_' + var_strs[i] + ' = ' + IC_cs[i] + '; \n'

    kernel = cp.ElementwiseKernel(kernel_input, kernel_output, kernel_body, 'demo_ODE')
    IC_kernel = cp.ElementwiseKernel(IC_kernel_input, IC_kernel_output, IC_kernel_body, 'IC_kernel')

    kernel_string = '\n INPUT: \n' + kernel_input + '\n BODY: \n' + kernel_body + '\n OUTPUT: \n' + kernel_output
    IC_kernel_string = '\n INPUT: \n' + IC_kernel_input + '\n BODY: \n' + IC_kernel_body + '\n OUTPUT: \n' + IC_kernel_output

    if print_result:
        print('GPU KERNEL: ')
        print(kernel_string)
        print(' ')
        print('GPU IC KERNEL: ')
        print(IC_kernel_string)
        print(' ')

    return kernel_string, IC_kernel_string, kernel, IC_kernel, shape

def generate_pycode(var_strs, eom_strs, drive_var_strs, drive_eom_strs, IC_strs, param_dict, use_complex=False, print_result=False):
    '''
    ~~Similar to above but for generating python code to run simulations, encoded in the same format, on CPU~~
    Sympy's printer is not actually that useful here, so this rewrites M complex-valued EOMs as 2M real-valued EOMs, then
    makes a "kernel" - a string that defines a derivative function, which can be called as an exec() statement in the RK loop
    '''

    eom_sps = []  # sympy translation
    eom_nps = []  # numpy translation
    drive_eom_sps = []  # sympy translation
    drive_eom_nps = []  # numpy translation
    IC_sps = []
    IC_nps = []

    numpy_kernel = '' # this "kernel" is a string that will get converted to a callable python function. It defines the EoMs
    numpy_kernel += 'def numpy_dxdt(t, '

    numpy_IC_kernel = ''  # this will be used to supply the initial conditions, which may depend on a sweepable parameter
    numpy_IC_kernel += 'def numpy_get_ICs(): \n'

    for var_str in var_strs:
        if use_complex:
            numpy_kernel += var_str + '_R, '
            numpy_kernel += var_str + '_I, '
        else:
            numpy_kernel += var_str + ', '

    for drive_var_str in drive_var_strs:
        if use_complex:
            numpy_kernel += drive_var_str + '_R, '
            numpy_kernel += drive_var_str + '_I, '
        else:
            numpy_kernel += drive_var_str + ', '

    numpy_kernel = numpy_kernel[:-2]

    numpy_kernel += '): \n'

    symbols_and_parameters_dict = {}

    symbols_and_parameters_dict['t'] = sp.Symbol('t', real=True)  # t is a special variable

    param_dict_keys = list(param_dict.keys())

    for i in range(0, len(param_dict_keys)):
        parameter = sp.Symbol(param_dict_keys[i], real=True)
        symbols_and_parameters_dict[param_dict_keys[i]] = parameter

        numpy_kernel += '    ' + param_dict_keys[i] + ' = ' + str(param_dict[param_dict_keys[i]]) + '\n'
        numpy_IC_kernel += '    ' + param_dict_keys[i] + ' = ' + str(param_dict[param_dict_keys[i]]) + '\n'

    for i in range(0, len(var_strs)):
        symbol = sp.Symbol(var_strs[i], real=True)
        symbols_and_parameters_dict[var_strs[i]] = symbol

    for i in range(0, len(drive_var_strs)):
        symbol = sp.Symbol(drive_var_strs[i], real=True)
        symbols_and_parameters_dict[drive_var_strs[i]] = symbol

    for j in range(0, len(drive_eom_strs)):

        drive_eom_str = drive_eom_strs[j]

        if use_complex:

            drive_eom_sp = sp.sympify(drive_eom_str, locals=symbols_and_parameters_dict)

            for i in range(0, len(var_strs)):
                symbol_R = sp.Symbol(var_strs[i] + '_R', real=True)
                symbol_I = sp.Symbol(var_strs[i] + '_I', real=True)

                drive_eom_sp = drive_eom_sp.subs(symbols_and_parameters_dict[var_strs[i]], symbol_R + 1j * symbol_I)

            for i in range(0, len(drive_var_strs)):
                symbol_R = sp.Symbol(drive_var_strs[i] + '_R', real=True)
                symbol_I = sp.Symbol(drive_var_strs[i] + '_I', real=True)

                drive_eom_sp = drive_eom_sp.subs(symbols_and_parameters_dict[drive_var_strs[i]], symbol_R + 1j * symbol_I)

            # First, convert expressions for EoMs and initial conditions into sympy format
            drive_eom_sp_R = sp.re(drive_eom_sp)
            drive_eom_sp_I = sp.im(drive_eom_sp)
            eom_sps.append(drive_eom_sp_R)
            eom_sps.append(drive_eom_sp_I)

            # Then, convert into numpy format with pycode(). This catches the most functions
            drive_eom_np_R = pycode(drive_eom_sp_R, fully_qualified_modules=False)
            drive_eom_np_I = pycode(drive_eom_sp_I, fully_qualified_modules=False)
            eom_nps.append(drive_eom_np_R)
            eom_nps.append(drive_eom_np_I)

            numpy_kernel += '    ' + drive_var_strs[j] + '_R = ' + drive_eom_np_R + ' \n'
            numpy_kernel += '    ' + drive_var_strs[j] + '_I = ' + drive_eom_np_I + ' \n'

        else:
            # First, convert expressions for EoMs and initial conditions into sympy format
            eom_sp = sp.sympify(drive_eom_str)
            eom_sps.append(eom_sp)

            numpy_kernel += '    ' + drive_var_str[j] + ' = ' + drive_eom_sp + ' \n'

            numpy_IC_kernel += '    IC_' + drive_var_str[j] + ' = 0 \n'

    for j in range(0, len(eom_strs)):

        eom_str = eom_strs[j]
        IC_str = IC_strs[j]

        if use_complex:

            eom_sp = sp.sympify(eom_str, locals=symbols_and_parameters_dict)
            IC_sp = sp.sympify(IC_str, locals=symbols_and_parameters_dict)

            for i in range(0, len(var_strs)):
                symbol_R = sp.Symbol(var_strs[i] + '_R', real=True)
                symbol_I = sp.Symbol(var_strs[i] + '_I', real=True)

                eom_sp = eom_sp.subs(symbols_and_parameters_dict[var_strs[i]], symbol_R + 1j * symbol_I)

            for i in range(0, len(drive_var_strs)):
                symbol_R = sp.Symbol(drive_var_strs[i] + '_R', real=True)
                symbol_I = sp.Symbol(drive_var_strs[i] + '_I', real=True)

                eom_sp = eom_sp.subs(symbols_and_parameters_dict[drive_var_strs[i]], symbol_R + 1j * symbol_I)

            # First, convert expressions for EoMs and initial conditions into sympy format
            eom_sp_R = sp.re(eom_sp)
            eom_sp_I = sp.im(eom_sp)
            eom_sps.append(eom_sp_R)
            eom_sps.append(eom_sp_I)

            IC_sp_R = sp.re(IC_sp)
            IC_sp_I = sp.im(IC_sp)
            IC_sps.append(IC_sp_R)
            IC_sps.append(IC_sp_I)

            # Then, convert into numpy format with pycode(). This catches the most functions
            eom_np_R = pycode(eom_sp_R, fully_qualified_modules=False)
            eom_np_I = pycode(eom_sp_I, fully_qualified_modules=False)
            eom_nps.append(eom_np_R)
            eom_nps.append(eom_np_I)

            IC_np_R = pycode(IC_sp_R, fully_qualified_modules=False)
            IC_np_I = pycode(IC_sp_I, fully_qualified_modules=False)
            IC_nps.append(IC_np_R)
            IC_nps.append(IC_np_I)

            numpy_kernel += '    d' + var_strs[j] + '_Rdt = ' + eom_np_R + ' \n'
            numpy_kernel += '    d' + var_strs[j] + '_Idt = ' + eom_np_I + ' \n'

            numpy_IC_kernel += '    IC_' + var_strs[j] + '_R = ' + IC_np_R + ' \n'
            numpy_IC_kernel += '    IC_' + var_strs[j] + '_I = ' + IC_np_I + ' \n'

        else:
            # First, convert expressions for EoMs and initial conditions into sympy format
            eom_sp = sp.sympify(eom_str)
            eom_sps.append(eom_sp)

            IC_sp = sp.sympify(IC_str)
            IC_sps.append(IC_sp)

            # Then, convert into numpy format with pycode(). This catches the most functions
            IC_np = pycode(IC_sp, fully_qualified_modules=False)
            IC_nps.append(IC_np)

            numpy_kernel += '    d' + var_strs[j] + 'dt = ' + eom_sp + ' \n'

            numpy_IC_kernel += '    IC_' + var_strs[j] + ' = ' + IC_np + ' \n'

    # Write the return statement for both kernels
    numpy_kernel += '    return '
    numpy_IC_kernel += '    return '

    for i in range(0, len(var_strs)):

        var_str = var_strs[i]

        if use_complex:
            numpy_kernel += 'd' + var_str + '_Rdt, '
            numpy_kernel += 'd' + var_str + '_Idt, '

            numpy_IC_kernel += 'IC_' + var_str + '_R, '
            numpy_IC_kernel += 'IC_' + var_str + '_I, '

        else:
            numpy_kernel += 'd' + var_str + 'dt, '
            numpy_IC_kernel += 'IC_' + var_str + ', '

    for i in range(0, len(drive_var_strs)):

        drive_var_str = drive_var_strs[i]

        if use_complex:
            numpy_kernel += drive_var_str + '_R, '
            numpy_kernel += drive_var_str + '_I, '

            numpy_IC_kernel += '0, '
            numpy_IC_kernel += '0, '

        else:
            numpy_kernel += drive_var_str + 'dt, '
            numpy_IC_kernel += '0, '


    numpy_kernel = numpy_kernel[:-2]
    numpy_IC_kernel = numpy_IC_kernel[:-2]

    if print_result:
        print('NUMPY EOM KERNEL:')
        print(numpy_kernel)
        print(' ')
        print('NUMPY IC KERNEL:')
        print(numpy_IC_kernel)

    return eom_nps, numpy_kernel, numpy_IC_kernel