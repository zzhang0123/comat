from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

# Automatically detect the compiler
compiler = os.getenv("CC", "clang")  # Default to clang if no CC is set
default_compile_args = ["-O2", "-march=native"]  # Mimic Cython's defaults

if compiler == "clang":
    default_compile_args += ["-ffast-math"]
elif compiler.startswith("gcc"):
    default_compile_args += ["-fopenmp", "-ffast-math"]

ext_modules = [
    Extension(
        "comat",  # Module name
        ["comat.pyx"],  # Source file
        include_dirs=[numpy.get_include()],
        extra_compile_args=default_compile_args,
        extra_link_args=["-O2"] + (["-fopenmp"] if compiler.startswith("gcc") else []),
    )
]

setup(
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            "language_level": "3",  # Python 3 syntax
            "boundscheck": False,   # Disable bounds checking for speed
            "wraparound": False,    # Disable negative index handling
            "cdivision": True,      # Use C-style division
        },
    ),
)