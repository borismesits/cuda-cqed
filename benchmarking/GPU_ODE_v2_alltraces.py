import numpy as np
import sympy as sp
import cupy as cp
import time
from numba.typed import List

# Make sure you use a GPU runtime, otherwise cupy won't import

#%%

def run_shots(variations1, variations2):

  def RK_loop(t, x, f_dxdt, dt, args):

      '''
      Implements an RK4 method.
      '''

      for i in range(0, len(t)-1):


          k1 = f_dxdt(x[i,:,:], t[i], args)
          k2 = f_dxdt(x[i,:,:] + k1*dt/2, t[i] + dt/2, args)
          k3 = f_dxdt(x[i,:,:] + k2*dt/2, t[i] + dt/2, args)
          k4 = f_dxdt(x[i,:,:] + k3*dt, t[i] + dt, args)

          x[i+1, :, :] += (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

      return x

  def f_dxdt(xi, t, args):

      omega = args[0]
      A = args[1]

      x1 = xi[0,:] # phi
      x2 = xi[1,:] # derivative of phi

      omega_0 = 1*2*np.pi
      alpha = 1
      kappa = 1

      dx1dt = x2 + A*np.cos(t*omega) + np.random.normal(0, 0.1, np.shape(x1))
      dx2dt = -omega_0**2*np.sin(x1) - kappa*x2

      dxdt = xi*0
      dxdt[0,:] = dx1dt
      dxdt[1,:] = dx2dt

      return dxdt

  def eom_i(args):
      '''
      Wrapper for the RK loop that creates all the necessary arrays, since
      you can't create arrays inside of a jit function.
      '''

      t = np.linspace(0,10,10001)

      N = 2

      x = np.zeros([len(t), N, len(args[0])])

      dt = t[1] - t[0]

      x = RK_loop(t, x, f_dxdt, dt, args)

      return x, t

  def eom(args_list):

      '''
      Wrapper for the solver, to package the arguments in a way that numba can interpret
      '''

      args = List()
      [args.append(arg) for arg in args_list]

      x, t = eom_i(args_list)

      return x, t

  A = np.linspace(5,5, variations1)
  omega = np.linspace(1*2*np.pi, 3*2*np.pi, variations2)

  A, omega = np.meshgrid(A, omega)

  time1 = time.time()

  args_list = [omega.flatten(), A.flatten()]
  x, t = eom(args_list)
  print(time.time()-time1)


variations1 = [10, 30, 50, 100, 100, 300, 500]
variations2 = [10, 10, 20, 30,  100, 100, 200]

for i in range(0,len(variations1)):

  run_shots(variations1[i], variations2[i])
  
#%%
  
  
def run_shots(variations1, variations2):

  params = [('omega_0', 1*2*np.pi),
            ('omega_d', [0.5*2*np.pi, 1.5*2*np.pi, variations1]),
            ('A', [0.3,9,variations2]),
            ('kappa', 1)]

  var_strs = ['q_1','p_1'] # define your dependent variables/coordinates
  exp_strs = ['p_1 + A*cos(omega_d*t)','-sin(q_1)*omega_0**2 - kappa*p_1'] # define the time derivatives of each coordinate

  exp_sps = []

  def RK_loop(t, x, f_dxdt, dt, kernel_op, idxs):
      '''
      Implements an RK4 method.

      The indexing for the variable 'x', which contains all the solution data,
      goes like: time steps, modes, variations.

      So for example, 3 modes with 5 variations running for hundred time steps
      would have the shape (100, 3, 5).
      '''

      for i in range(0, len(t)-1):

          k1 = f_dxdt(x[i,:,:], t[i], dt, kernel_op, idxs)
          k2 = f_dxdt(x[i,:,:] + k1*dt/2, t[i] + dt/2, dt, kernel_op, idxs)
          k3 = f_dxdt(x[i,:,:] + k2*dt/2, t[i] + dt/2, dt, kernel_op, idxs)
          k4 = f_dxdt(x[i,:,:] + k3*dt, t[i] + dt, dt, kernel_op, idxs)

          x[i+1,:,:] = x[i,:,:] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

      return x

  def f_dxdt(xi, t, dt, kernel_op, idxs):
      '''
      Equation of motion for related rates problem. The off-diagonal elements
      of the rate matrix represent transition rates.
      '''

      args = [idxs[0], idxs[1], dt, t]

      args.extend(list(xi))

      dxdt = kernel_op(*args)

      return cp.array(dxdt)

  def related_rates_problem(t, N, variations1, variations2, kernel_op):
      '''
      Wrapper for the RK loop that creates all the necessary arrays, since
      you can't create arrays inside of a jit function.

      N is number of modes
      vars is number of variations
      '''

      x = cp.array(np.zeros([len(t), N, variations1*variations2]),dtype=cp.float64)

      dt = cp.array(t[1] - t[0])

      idxs = []

      idx1 = np.arange(0, variations1)
      idx2 = np.arange(0, variations2)

      IDX1, IDX2 = np.meshgrid(idx1, idx2)

      idxs.append(cp.array(cp.array(IDX1.flatten(), dtype=np.int8)))
      idxs.append(cp.array(cp.array(IDX2.flatten(), dtype=np.int8)))

      x = RK_loop(t, x, f_dxdt, dt, kernel_op, idxs)

      return x


  def convert_power_arg_to_float64(inputt):
      '''
      Helper function for translating python math to C.
      '''
      try:
          inputt.lower()
      except:
          print('not a string')

      stringy = []

      for i in range(0,len(inputt)):
          stringy.append(inputt[i])

      for i in range(0,len(stringy)-3):
          if (stringy[i] == 'p' and stringy[i+1] == 'o' and stringy[i+2] == 'w'):
              j = i+3
              while(stringy[j] != ')'):
                  j += 1
              stringy[j-1]=stringy[j-1]+'.0'

      string = ''

      for i in range(0,len(stringy)):
          string += stringy[i]

      return string

  exp_cs = []

  for exp_str in exp_strs:
    exp_sp = sp.sympify(exp_str)
    exp_sps.append(exp_sp)
    exp_c = convert_power_arg_to_float64(sp.ccode(exp_sp))
    exp_cs.append(exp_c)

  '''
  Here we start building up the three parts of the kernel: input, body, output.
  '''

  KernelInput = 'int8 index1, int8 index2, float64 dt, float64 t, '

  for var_str in var_strs:

    KernelInput += 'float64 ' + var_str + ', '
  KernelInput = KernelInput[0:-2] # removes trailing comma

  KernelOutput = ''

  for var_str in var_strs:

    KernelOutput += 'float64 d' + var_str + 'dt' + ', '
  KernelOutput = KernelOutput[0:-2] # removes trailing comma

  Kernel = ''

  sweep_num = 0

  for i in range(0,len(params)):

    try:

      param0 = params[i][1][0]
      param_vars = params[i][1][2]
      param_range = params[i][1][1]-params[i][1][0]

      sweep_num += 1

      Kernel += 'double ' + params[i][0] + ' = ' + str(float(param0)) + ' + index' + str(sweep_num) + '*' + str(float(param_range)) + '/' + str(float(param_vars-1)) + ';\n'

    except:
      Kernel += 'double ' + params[i][0] + ' = ' + str(float(params[i][1])) + 'f;\n'

  for i in range(0,2):

    eom_line = 'd' + var_strs[i] + 'dt = ' + exp_cs[i] + '; \n'

    Kernel += eom_line

  # print(KernelInput)
  # print(Kernel)
  # print(KernelOutput)

  ODE = cp.ElementwiseKernel(KernelInput, KernelOutput, Kernel, 'demo_ODE')

  # for i in range(0,len(var_strs)):

  #   display(sp.Eq(sp.sympify(var_strs[i]),exp_sps[i]))

  t = np.linspace(0, 100, 10001)

  N = len(var_strs)

  time1 = time.time()
  x = related_rates_problem(t, N, variations1, variations2, ODE)
  print(time.time() - time1)

  del(x)
  cp._default_memory_pool.free_all_blocks()

variations1 = [10, 30, 50, 100, 100, 300, 500]
variations2 = [10, 10, 20, 30,  100, 100, 200]

for i in range(0,len(variations1)):

  run_shots(variations1[i], variations2[i])