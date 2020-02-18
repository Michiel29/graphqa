from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension("construct_graph",  ["construct_graph.pyx"])
]

setup(
    name = 'test',
    ext_modules = cythonize(ext_modules)
)
