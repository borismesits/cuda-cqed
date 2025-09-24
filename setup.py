"""
@author: Boris Mesits
"""


from setuptools import setup

setup(name='cuda-cqed',
      version='0.1',
      description='A package for parallelizing integration of ODEs on GPUs, where one desires large sweeps of many parameters. Intended for design of superconducting circuits.',
      url='',
      author='Boris Mesits',
      author_email='boris.mesits@yale.edu',
      license='MIT',
      packages=['cuda_cqed'],
      zip_safe=False)