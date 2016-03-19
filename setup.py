from setuptools import setup
from setuptools.command.install import install as _install
from setuptools.extension import Extension
import sys


class install(_install):

    def run(self):
        from Cython.Build import cythonize
        import numpy

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

        ext = Extension(
            name="pyneural",
            sources=sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
        )

        setup(ext_modules=cythonize([ext]))
        _install.run(self)


setup(
    name="pyneural",
    version="0.1.0",
    author="Taylor Burmeister",
    author_email="burmeister.taylor@gmail.com",
    description="A simple but fast Python library for training neural networks",
    url="https://github.com/tburmeister/pyneural",
    setup_requires=['cython>=0.23', 'numpy>=1.9.2'],
    install_requires=['cython>=0.23', 'numpy>=1.9.2'],
    cmdclass={'install': install}
)
