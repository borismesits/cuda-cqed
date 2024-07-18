import cupy as cp
import sympy as sp
import numpy as np
import sympy
from sympy import pycode


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


def generate_kernel(var_strs, exp_strs, param_dict, use_complex=False, print_result=False):
    '''

    :param var_strs: The list of strings containing the names of quadratures (indep. variables in ODEs)
    :param exp_strs: The equations of motion. Should be in terms of var_strs
    :param params: Any variables in the exp_strs not listed in var_strs must be defined here.
        There are two ways:
            e.g., ('omega_0', 10 * 2 * np.pi), OR
            e.g., ('omega_0', [10 * 2 * np.pi, 20 * 2 * np.pi, 100]) to have 100 variations
    :param use_complex: If this is true, then the indep. variables will be treated as complex valued. Twice the number
                        of degrees of freedom.
    :return kernel_input, kernel_output, kernel_body, kernel: Three strings containing sections of the kernel, and the
                                                            cupy kernel itself.
    '''

    exp_sps = [] # sympy translation
    exp_cs = [] # C code translation

    for exp_str in exp_strs:

        if use_complex:
            '''
            Sympy has a very convenient function called sympify, but it has tricky requirements. So I need to 
            define all the variables and parameters as sympy objects first, and then tell sympy to assume they are
            real-valued.
            '''

            symbols = sp.symbols(var_strs)

            symbols_and_parameters_dict = {}

            symbols_and_parameters_dict['t'] = sp.Symbol('t', real=True) # t is a special variable

            parameters = []

            param_dict_keys = list(param_dict.keys())

            for i in range(0, len(param_dict_keys)):
                parameter = sp.Symbol(param_dict_keys[i], real=True)
                symbols_and_parameters_dict[param_dict_keys[i]] = parameter

            for i in range(0, len(var_strs)):
                symbol = sp.Symbol(var_strs[i], real=True)
                symbols_and_parameters_dict[var_strs[i]] = symbols[i]

            exp_sp = sp.sympify(exp_str, locals=symbols_and_parameters_dict)

            for i in range(0, len(var_strs)):

                symbol_R = sp.Symbol(var_strs[i] + '_R',real=True)
                symbol_I = sp.Symbol(var_strs[i] + '_I',real=True)

                exp_sp = exp_sp.subs(symbols[i], symbol_R + 1j*symbol_I)

            exp_sp_R = sp.re(exp_sp)
            exp_sp_I = sp.im(exp_sp)

            exp_sps.append(exp_sp_R)
            exp_sps.append(exp_sp_I)
            # exp_c_R = convert_power_arg_to_float64(sp.ccode(exp_sp_R))
            # exp_c_I = convert_power_arg_to_float64(sp.ccode(exp_sp_I))
            exp_c_R = sp.ccode(exp_sp_R)
            exp_c_I = sp.ccode(exp_sp_I)
            exp_cs.append(exp_c_R)
            exp_cs.append(exp_c_I)


        else:
            exp_sp = sp.sympify(exp_str)

            exp_sps.append(exp_sp)
            # exp_c = convert_power_arg_to_float64(sp.ccode(exp_sp))
            exp_c = sp.ccode(exp_sp)
            exp_cs.append(exp_c)

    kernel_input = 'float64 dt, float64 t, '

    shape = [len(exp_cs)]

    for var_str in var_strs:

        if use_complex:

            kernel_input += 'float64 ' + var_str + '_R, '
            kernel_input += 'float64 ' + var_str + '_I, '

        else:
            kernel_input += 'float64 ' + var_str + ', '

    kernel_output = ''

    for var_str in var_strs:

        if use_complex:
            kernel_output += 'float64 d' + var_str + 'dt_R' + ', '
            kernel_output += 'float64 d' + var_str + 'dt_I' + ', '

        else:
            kernel_output += 'float64 d' + var_str + 'dt' + ', '


    kernel_output = kernel_output[0:-2]

    kernel_body = ''

    sweep_num = 0

    for i in range(0, len(param_dict_keys)):

        try: # this is for sweep variables (specifying a range and a number of points)

            param0 = param_dict[param_dict_keys[i]][0]
            param_vars = param_dict[param_dict_keys[i]][2]
            param_range = param_dict[param_dict_keys[i]][1] - param_dict[param_dict_keys[i]][0]

            sweep_num += 1

            kernel_body += 'double ' + param_dict_keys[i] + ' = ' + str(float(param0)) + ' + index' + str(sweep_num) + '*' + str(
                float(param_range)) + '/' + str(float(np.max([param_vars - 1, 1]))) + ';\n'

            shape.append(param_vars)

        except: # this is for constant variables
            kernel_body += 'double ' + param_dict_keys[i] + ' = ' + str(float(param_dict[param_dict_keys[i]])) + 'f;\n'

    for i in range(0, sweep_num):

        kernel_input += 'int32 index' + str(i+1) + ', '

    kernel_input = kernel_input[0:-2]  # removes trailing comma

    for i in range(0, len(var_strs)):

        if use_complex:
            eom_line_R = 'd' + var_strs[i] + 'dt_R = ' + exp_cs[2*i] + '; \n'
            kernel_body += eom_line_R

            eom_line_I = 'd' + var_strs[i] + 'dt_I= ' + exp_cs[2*i+1] + '; \n'
            kernel_body += eom_line_I

        else:
            eom_line = 'd' + var_strs[i] + 'dt = ' + exp_cs[i] + '; \n'
            kernel_body += eom_line

    if print_result:
        print(kernel_input)
        print(kernel_body)
        print(kernel_output)

    kernel = cp.ElementwiseKernel(kernel_input, kernel_output, kernel_body, 'demo_ODE')

    return kernel_input, kernel_output, kernel_body, kernel, shape

def generate_pycode(var_strs, exp_strs, param_dict, use_complex=False):
    '''
    ~~Similar to above but for generating python code to run simulations, encoded in the same format, on CPU~~
    Sympy's printer is not actually that useful here, so all this does is rewrite M complex-valued EOMs as 2M real-valued EOMs
    '''

    exp_sps = [] # sympy translation
    exp_nps = [] # numpy translation

    for exp_str in exp_strs:

        if use_complex:

            symbols = sp.symbols(var_strs)

            symbols_and_parameters_dict = {}

            symbols_and_parameters_dict['t'] = sp.Symbol('t', real=True) # t is a special variable

            param_dict_keys = list(param_dict.keys())

            for i in range(0, len(param_dict_keys)):
                parameter = sp.Symbol(param_dict_keys[i], real=True)
                symbols_and_parameters_dict[param_dict_keys[i]] = parameter

            for i in range(0, len(var_strs)):
                symbol = sp.Symbol(var_strs[i], real=True)
                symbols_and_parameters_dict[var_strs[i]] = symbols[i]

            exp_sp = sp.sympify(exp_str, locals=symbols_and_parameters_dict)

            for i in range(0, len(var_strs)):

                symbol_R = sp.Symbol(var_strs[i] + '_R',real=True)
                symbol_I = sp.Symbol(var_strs[i] + '_I',real=True)

                exp_sp = exp_sp.subs(symbols[i], symbol_R + 1j*symbol_I)

            exp_sp_R = sp.re(exp_sp)
            exp_sp_I = sp.im(exp_sp)
            exp_sps.append(exp_sp_R)
            exp_sps.append(exp_sp_I)


            exp_np_R = pycode(exp_sp_R, fully_qualified_modules=False)
            exp_np_I = pycode(exp_sp_I, fully_qualified_modules=False)
            exp_nps.append(exp_np_R)
            exp_nps.append(exp_np_I)


        else:
            exp_sp = sp.sympify(exp_str)
            exp_sps.append(exp_sp)

            exp_np = pycode(exp_sp, fully_qualified_modules=False)
            exp_nps.append(exp_np)

    return exp_nps