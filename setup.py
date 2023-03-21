from setuptools import setup
from setuptools import Extension

import numpy

from Cython.Build import cythonize

ext_modules = [
    Extension(
        name="graphormer_pretrained.data.algos",
        sources=["graphormer_pretrained/data/algos.pyx"],
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)
