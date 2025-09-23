"""
@author: Boris Mesits
"""


from setuptools import setup

setup(name='gpu_odes',
      version='0.1',
      description='A package for parallelizing integration of ODEs on GPUs, where one desires large sweeps of many parameters. Intended for design of superconducting circuits.',
      url='',
      author='Boris Mesits',
      author_email='boris.mesits@yale.edu',
      license='MIT',
      packages=['gpu_odes'],
      requirements = ['tqdm','cupy','numpy','sympy'],
      zip_safe=False)