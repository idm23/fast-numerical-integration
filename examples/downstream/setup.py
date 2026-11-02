from Cython.Build import cythonize
from setuptools import setup

setup(ext_modules=cythonize(["downstream_kernels/osc.pyx"], language_level=3))
