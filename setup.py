import numpy
from setuptools import setup
from setuptools import find_packages, Extension
from Cython.Build import cythonize
from Cython.Build import build_ext

install_requires = [
    "numpy",
    "cython",
]

kw = dict(include_dirs=[numpy.get_include()])
ext_modules = [
    Extension(
        "graphormer_pretrained.data.algos",
        sources=["graphormer_pretrained/data/algos.pyx"],
        **kw
    ),
]
setup(
    name="graphormer_pretrained",
    version="0.0.1",
    url="https://github.com/maclandrol/graphormer-pretrained",
    description="Graphormer is a deep learning package by Microsoft that allows researchers and developers to train custom models for molecule modeling tasks.",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/maclandrol/graphormer-pretrained/issues",
        "Source Code": "https://github.com/maclandrol/graphormer-pretrained",
    },
    install_requires=install_requires,
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
    python_requires=">=3.7",
    cmdclass={"build_ext ": build_ext},
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
