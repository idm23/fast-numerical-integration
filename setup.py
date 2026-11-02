"""Build configuration for the Cython extensions.

Project metadata lives in ``pyproject.toml``. This file only declares the
compiled extension modules and their compiler flags.

Floating-point contraction is disabled everywhere (``-ffp-contract=off``):
fusing ``a * b + c`` into one FMA instruction changes rounding. GCC on
x86-64 doesn't do it by default, but Clang on Apple Silicon does. The
engine's results, including the tests that are bit-identical to SciPy,
must be the same on every platform. MSVC keeps contraction off under its
default ``/fp:precise``.
"""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

COMPILER_DIRECTIVES = {
    "language_level": 3,
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "initializedcheck": False,
}

UNIX_FLAGS = ["-O3", "-ffp-contract=off"]
MSVC_FLAGS = ["/O2", "/fp:precise"]


class BuildExt(build_ext):
    """Pick optimisation / floating-point flags for the compiler in use."""

    def build_extensions(self):
        flags = MSVC_FLAGS if self.compiler.compiler_type == "msvc" else UNIX_FLAGS
        for ext in self.extensions:
            ext.extra_compile_args = list(flags)
        super().build_extensions()


common = dict(
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
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
    Extension(
        "fast_numerical_integration.batch",
        ["src/fast_numerical_integration/batch.pyx"],
        **common,
    ),
    Extension(
        "fast_numerical_integration.problems.native",
        ["src/fast_numerical_integration/problems/native.pyx"],
        **common,
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives=COMPILER_DIRECTIVES,
        include_path=["src"],
    ),
    cmdclass={"build_ext": BuildExt},
    # .pxd files ship in the wheel so downstream packages can cimport the C API.
    package_data={
        "fast_numerical_integration": ["*.pxd", "*.pyx"],
        "fast_numerical_integration.problems": ["*.pyx"],
    },
    exclude_package_data={"": ["*.c"]},
)
