def openfermion2numpyold(OF_exp):
    exp_string_full = str(OF_exp) + ' '
    exp_string_arr = exp_string_full.split('\n')

    eom_string = ''

    for exp_string in exp_string_arr:
        eom_term_string = exp_string

        prefactor_str = exp_string.split('[')[0]
        #         op_strs = exp_string.split('[')[1]
        #         op_strs = op_strs.replace(']', '')
        #         op_strs = op_strs.replace('+', '').split(' ')

        eom_term_string = eom_term_string.replace(' [0]', ')*a')
        eom_term_string = eom_term_string.replace(' [0^]', ')*conjugate(a)')
        eom_term_string = eom_term_string.replace(' [0^ 0]', ')*adaga')
        eom_term_string = eom_term_string.replace(' [0 0]', ')*aa')
        eom_term_string = eom_term_string.replace(' [0^ 0^]', ')*conjugate(aa)')
        eom_term_string = eom_term_string.replace(' [0^ 1]', ')*adagb')
        eom_term_string = eom_term_string.replace(' [0 1]', ')*ab')
        eom_term_string = eom_term_string.replace(' [0 1^]', ')*conjugate(adagb)')
        eom_term_string = eom_term_string.replace(' [0^ 1^]', ')*conjugate(ab)')
        eom_term_string = eom_term_string.replace(' [1]', ')*b')
        eom_term_string = eom_term_string.replace(' [1^]', ')*conjugate(b)')
        eom_term_string = eom_term_string.replace(' [1^ 1]', ')*bdagb')
        eom_term_string = eom_term_string.replace(' [1 1]', ')*bb')
        eom_term_string = eom_term_string.replace(' [1^ 1^]', ')*conjugate(bb)')
        eom_term_string = eom_term_string.replace(' [1 2]', ')*bc')
        eom_term_string = eom_term_string.replace(' [1^ 2]', ')*bdagc')
        eom_term_string = eom_term_string.replace(' [1 2^]', ')*conjugate(bdagc)')
        eom_term_string = eom_term_string.replace(' [1^ 2^]', ')*conjugate(bc)')
        eom_term_string = eom_term_string.replace(' [2]', ')*c')
        eom_term_string = eom_term_string.replace(' [2^]', ')*conjugate(c)')
        eom_term_string = eom_term_string.replace(' [2^ 2]', ')*cdagc')
        eom_term_string = eom_term_string.replace(' [2 2]', ')*cc')
        eom_term_string = eom_term_string.replace(' [2^ 2^]', ')*conjugate(cc)')
        eom_term_string = eom_term_string.replace(' [0^ 2]', ')*adagc')
        eom_term_string = eom_term_string.replace(' [0 2]', ')*ac')
        eom_term_string = eom_term_string.replace(' [0 2^]', ')*conjugate(adagc)')
        eom_term_string = eom_term_string.replace(' [0^ 2^]', ')*conjugate(ac)')

        eom_term_string = eom_term_string.replace('[1^ 1 1]', ')*bdagb*b')
        eom_term_string = eom_term_string.replace('[1^ 1 1 1]', ')*bdagb*bb')
        eom_term_string = eom_term_string.replace('[0 1^ 1 1]', ')*conjugate(adagb)*bb')
        eom_term_string = eom_term_string.replace('[0^ 1^ 1 1]', ')*conjugate(ab)*bb')
        eom_term_string = eom_term_string.replace('[1^ 1 1 2]', ')*conjugate(bdagb)*bc')
        eom_term_string = eom_term_string.replace('[1^ 1^ 1 2]', ')*conjugate(bb)*bc')

        eom_string += '(' + eom_term_string

    eom_string = eom_string.replace('*I', '*1j')
    eom_string = eom_string.replace(' [] ', ')')

    eom_string = eom_string.replace('conj_eta_1', 'conjugate(eta_1)')
    eom_string = eom_string.replace('conj_eta_2', 'conjugate(eta_2)')
    eom_string = eom_string.replace('conj_eta_3', 'conjugate(eta_3)')
    eom_string = eom_string.replace('eta_sum', '(abs(eta_1)**2+abs(eta_2)**2+abs(eta_3)**2)')

    return eom_string

def openfermion2numpy(OF_exp):
    exp_string_full = str(OF_exp) + ' '
    exp_string_arr = exp_string_full.split('\n')

    eom_string = ''

    for exp_string in exp_string_arr:
        eom_term_string = exp_string

        eom_term_string = eom_term_string.replace(' [0]', ')*a')
        eom_term_string = eom_term_string.replace(' [0^]', ')*conjugate(a)')
        eom_term_string = eom_term_string.replace(' [0^ 0]', ')*adaga')
        eom_term_string = eom_term_string.replace(' [0 0]', ')*aa')
        eom_term_string = eom_term_string.replace(' [0^ 0^]', ')*conjugate(aa)')
        eom_term_string = eom_term_string.replace(' [0^ 1]', ')*adagb')
        eom_term_string = eom_term_string.replace(' [0 1]', ')*ab')
        eom_term_string = eom_term_string.replace(' [0 1^]', ')*conjugate(adagb)')
        eom_term_string = eom_term_string.replace(' [0^ 1^]', ')*conjugate(ab)')
        eom_term_string = eom_term_string.replace(' [1]', ')*b')
        eom_term_string = eom_term_string.replace(' [1^]', ')*conjugate(b)')
        eom_term_string = eom_term_string.replace(' [1^ 1]', ')*bdagb')
        eom_term_string = eom_term_string.replace(' [1 1]', ')*bb')
        eom_term_string = eom_term_string.replace(' [1^ 1^]', ')*conjugate(bb)')
        eom_term_string = eom_term_string.replace(' [1 2]', ')*bc')
        eom_term_string = eom_term_string.replace(' [1^ 2]', ')*bdagc')
        eom_term_string = eom_term_string.replace(' [1 2^]', ')*conjugate(bdagc)')
        eom_term_string = eom_term_string.replace(' [1^ 2^]', ')*conjugate(bc)')
        eom_term_string = eom_term_string.replace(' [2]', ')*c')
        eom_term_string = eom_term_string.replace(' [2^]', ')*conjugate(c)')
        eom_term_string = eom_term_string.replace(' [2^ 2]', ')*cdagc')
        eom_term_string = eom_term_string.replace(' [2 2]', ')*cc')
        eom_term_string = eom_term_string.replace(' [2^ 2^]', ')*conjugate(cc)')
        eom_term_string = eom_term_string.replace(' [0^ 2]', ')*adagc')
        eom_term_string = eom_term_string.replace(' [0 2]', ')*ac')
        eom_term_string = eom_term_string.replace(' [0 2^]', ')*conjugate(adagc)')
        eom_term_string = eom_term_string.replace(' [0^ 2^]', ')*conjugate(ac)')

        eom_term_string = eom_term_string.replace('[1^ 1 1]', ')*bdagb*b')
        eom_term_string = eom_term_string.replace('[1^ 1 1 1]', ')*bdagb*bb')
        eom_term_string = eom_term_string.replace('[0 1^ 1 1]', ')*conjugate(adagb)*bb')
        eom_term_string = eom_term_string.replace('[0^ 1^ 1 1]', ')*conjugate(ab)*bb')
        eom_term_string = eom_term_string.replace('[1^ 1 1 2]', ')*conjugate(bdagb)*bc')
        eom_term_string = eom_term_string.replace('[1^ 1^ 1 2]', ')*conjugate(bb)*bc')

        eom_string += '(' + eom_term_string

    eom_string = eom_string.replace('*I', '*1j')
    eom_string = eom_string.replace(' [] ', ')')

    eom_string = eom_string.replace('conj_eta_1', 'conjugate(eta_1)')
    eom_string = eom_string.replace('conj_eta_2', 'conjugate(eta_2)')
    eom_string = eom_string.replace('conj_eta_3', 'conjugate(eta_3)')
    eom_string = eom_string.replace('eta_sum', '(abs(eta_1)**2+abs(eta_2)**2+abs(eta_3)**2)')

    return eom_string


from openfermion.ops import BosonOperator
from openfermion.transforms import normal_ordered
from openfermion.utils import commutator

from sympy import symbols
import numpy as np
K4, K6, g3, eta_1, conj_eta_1, eta_2, conj_eta_2, eta_3, conj_eta_3, eta_sum = symbols('K4, K6, g_3, eta_1, conj_eta_1, eta_2, conj_eta_2, eta_3, conj_eta_3, eta_sum', real=True)
chi, lambda_ab, lambda_bc = symbols('chi, lambda_ab, lambda_bc', real=True)

H_lin = BosonOperator('0^ 0', chi)
H_kerr = BosonOperator('1^ 1^ 1 1', K4)
H_stark = BosonOperator('1^ 1', 2*K4*eta_sum+K6*eta_sum**2)
H_parametric = 6*g3*lambda_ab*(conj_eta_1*BosonOperator('0^ 1')+eta_1*BosonOperator('0 1^')) + 3*g3*(eta_2*BosonOperator('1^ 1^')+conj_eta_2*BosonOperator('1 1')) + 6*g3*lambda_bc*(conj_eta_3*BosonOperator('1^ 2')+eta_3*BosonOperator('1 2^'))

H = H_lin + H_kerr

OF_string = normal_ordered(1j*commutator(H, BosonOperator('1')))
print(OF_string)
eom_string = openfermion2numpy(OF_string)
print('Eom str: ' + eom_string)
