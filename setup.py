"""Setup script for the standalone MACE-ICTC package.

MACE-ICTC = MACE in the Irreducible Cartesian Tensor Decomposition basis, extracted
from FSCETP as a self-contained deployment stack: the ICTC-basis MACE model plus
AOTInductor export, make_fx training compilation, the LAMMPS interface, and the
long-range module.
"""

import os
from pathlib import Path

from setuptools import find_packages, setup

try:
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME
except Exception:  # torch may be unavailable during lightweight metadata reads
    BuildExtension = None
    CppExtension = None
    CUDAExtension = None
    CUDA_HOME = None

readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""


def _get_ext_modules():
    # Optional compiled ICTC tensor-product extension. Opt-in: the pure-PyTorch path is
    # the default and the package is fully functional without it.
    if os.environ.get("MFF_BUILD_ICTD_TP_EXT", "0") != "1":
        return []
    if CppExtension is None:
        return []
    use_cuda = (
        os.environ.get("MFF_BUILD_ICTD_TP_CUDA", "1") == "1"
        and CUDAExtension is not None
        and CUDA_HOME is not None
    )
    extension_cls = CUDAExtension if use_cuda else CppExtension
    sources = ["mace_ictc/csrc/ictd_tp.cpp"]
    extra_compile_args = {"cxx": ["-O3"]}
    define_macros = []
    if use_cuda:
        sources.append("mace_ictc/csrc/ictd_tp_cuda.cu")
        extra_compile_args["nvcc"] = ["-O3"]
        define_macros.append(("WITH_CUDA", None))
    return [
        extension_cls(
            name="mace_ictc._C_ictd_tp",
            sources=sources,
            extra_compile_args=extra_compile_args,
            define_macros=define_macros,
        )
    ]


ext_modules = _get_ext_modules()
cmdclass = {"build_ext": BuildExtension} if ext_modules and BuildExtension is not None else {}


setup(
    name="mace-ictc",
    version="0.1.0",
    description="MACE in the Irreducible Cartesian Tensor Decomposition basis, with AOTInductor / make_fx / LAMMPS / long-range deployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(include=["mace_ictc", "mace_ictc.*"]),
    package_data={
        "mace_ictc": [
            "models/_ictd_cache/v1/cg/*.pt",
            "models/_ictd_cache/v1/cg_full/*.pt",
            "models/_ictd_cache/v1/u_so3/*.pt",
            "csrc/*.cpp",
            "csrc/*.cu",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.9",
    install_requires=[
        # The model + AOTI export + make_fx compile need only these.
        "torch>=2.4.0",          # AOTInductor (aoti_compile_and_package) / make_fx flatten; 2.7+ recommended
        "numpy>=1.20.0",
        "e3nn>=0.4.4,<0.6.0",    # kept compatible with mace-torch's e3nn pin
        "ase>=3.22.0",
        "opt-einsum-fx>=0.1.4",
    ],
    extras_require={
        # Faster scatter / radius-graph (pure-PyTorch fallbacks exist for both).
        "pyg": [
            "torch-scatter>=2.0.9",
            "torch-cluster>=1.6.0",
        ],
        # cuEquivariance backend (optional spherical-cue tensor-product path).
        "cue": [
            "cuequivariance-torch>=0.8.1",
            "cuequivariance-ops-torch-cu12>=0.8.1; platform_system=='Linux'",
        ],
        # Parse an optional fitted_E0.csv atomic-energy table.
        "e0": ["pandas>=1.3.0"],
        "full": [
            "torch-scatter>=2.0.9",
            "torch-cluster>=1.6.0",
            "cuequivariance-torch>=0.8.1",
            "pandas>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Deployment CLIs (names kept identical to FSCETP so the LAMMPS docs apply verbatim).
            "mff-export-aoti=mace_ictc.cli.export_aoti_core:main",
            "mff-export-core=mace_ictc.cli.export_libtorch_core:main",
            "mff-lammps=mace_ictc.cli.lammps_interface:main",
            "mff-convert-mace=mace_ictc.cli.convert_mace:main",
            "mff-preprocess=mace_ictc.cli.preprocess:main",
        ],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
