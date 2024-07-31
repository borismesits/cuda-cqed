import numpy

expressions = ["a**2 + b", "b**2 - a"]

var_strs = ['a', 'b']

numpy_kernel = ''

numpy_kernel += 'def asdf('

for var_str in var_strs:
    numpy_kernel += var_str + ', '

numpy_kernel = numpy_kernel[:-2]

numpy_kernel += '): \n'

for i in range(0, len(var_strs)):
    numpy_kernel += '    d' + var_strs[i] + 'dt = ' + expressions[i] + ' \n'

numpy_kernel += '    return '

for var_str in var_strs:
    numpy_kernel += 'd' + var_str + 'dt, '

numpy_kernel = numpy_kernel[:-2]

print(numpy_kernel)

a = exec(numpy_kernel)

print(globals()['asdf'])

for i in range(0, 10):
    asdf(1, 2)



def foo():
    exec("ef = 42")
    print(ef)

def foobar():
    ldict = locals()
    exec("ef=42",globals(),ldict)
    ef = ldict['ef']
    print(ef)