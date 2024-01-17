from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Hello world app',
    ext_modules=cythonize("graph.py"),
)
# Command to run to compile: python setup.py build_ext --inplace