"""Build configuration for the Cython extensions.

Project metadata lives in ``pyproject.toml``. This file only declares the
compiled extension modules.
"""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

COMPILER_DIRECTIVES = {
    "language_level": 3,
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "initializedcheck": False,
}

common = dict(
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    extra_compile_args=["-O3"],
)

extensions = [
    Extension(
        "fast_numerical_integration.integrators",
        ["src/fast_numerical_integration/integrators.pyx"],
        **common,
    ),
    Extension(
        "fast_numerical_integration.nbody",
        ["src/fast_numerical_integration/nbody.pyx"],
        **common,
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives=COMPILER_DIRECTIVES,
        include_path=["src"],
    ),
    package_data={"fast_numerical_integration": ["*.pxd", "*.pyx"]},
)
