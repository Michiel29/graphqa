import numpy
from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "datasets.graph_dataset_util_fast",
        ["datasets/graph_dataset_util_fast.pyx"],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
        extra_link_args = ["-fopenmp"],
        language="c++",
    )
]

setup(
    name = 'develop',
    ext_modules = cythonize(ext_modules, language_level="3"),
    include_dirs=[numpy.get_include()],
)
