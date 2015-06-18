from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import sys

sources = ['pyneural.pyx', 'neural.c']
include_dirs = [numpy.get_include()]

if sys.platform == 'darwin':
    # Mac OSX - use Accelerate framework
    extra_compile_args = ['-DACCELERATE']
    extra_link_args = ['-framework', 'accelerate']
else:
    # up to the user for now
    extra_compile_args = []
    extra_link_args = []

ext = Extension("pyneural", sources=sources, include_dirs=include_dirs, 
        extra_compile_args=extra_compile_args, extra_link_args=extra_link_args)

setup(ext_modules=cythonize([ext]))
