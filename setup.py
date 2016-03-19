from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import sys

sources = ['src/pyneural.pyx', 'src/neural.c']
include_dirs = [numpy.get_include()]

if sys.platform == 'darwin':
    # Mac OSX - use Accelerate framework
    extra_compile_args = ['-DACCELERATE']
    extra_link_args = ['-framework', 'accelerate']
    library_dirs = []
    libraries = []
else:
    # up to the user for now
    extra_compile_args = ['-std=c99']
    extra_link_args = []
    library_dirs = ['/opt/OpenBLAS/lib']
    libraries = ['openblas', 'm']
    include_dirs.append('/opt/OpenBLAS/include')

ext = Extension("pyneural", sources=sources, include_dirs=include_dirs,
        library_dirs=library_dirs, libraries=libraries,
        extra_compile_args=extra_compile_args, extra_link_args=extra_link_args)

setup(ext_modules=cythonize([ext]))
