import cupy as cp
import sympy as sp
import numpy as np

'''
This file is a util for converting Hamiltonians (symbolic description of Hermitian operators) into Langevin equations of motion
'''

var_strs = ['a']

exp_str = ['a*sdfasdf(a)']

symbols = sp.symbols(var_strs)

symbols_dict = {}

for i in range(0, len(var_strs)):
    symbol = sp.Symbol(var_strs[i], real=True)
    symbols_dict[var_strs[i]] = symbols[i]

exp_sp = sp.sympify(exp_str, locals=symbols_dict)
