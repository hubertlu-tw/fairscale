#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import os
import re

import setuptools

this_dir = os.path.dirname(os.path.abspath(__file__))


def fetch_requirements():
    with open("requirements.txt") as f:
        reqs = f.read().strip().split("\n")
    return reqs


# https://packaging.python.org/guides/single-sourcing-package-version/
def find_version(version_file_path) -> str:
    with open(version_file_path) as version_file:
        version_match = re.search(r"^__version_tuple__ = (.*)", version_file.read(), re.M)
        if version_match:
            ver_tup = eval(version_match.group(1))
            ver_str = ".".join([str(x) for x in ver_tup])
            return ver_str
        raise RuntimeError("Unable to find version tuple.")


extensions = []
cmdclass = {}
setup_requires = []

if os.getenv("BUILD_CUDA_EXTENSIONS", "0") == "1":
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME, ROCM_HOME
    is_rocm_pytorch = True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
    if CUDA_HOME is None and ROCM_HOME is None:
        raise RuntimeError(
            f"Neither nvcc or hipcc were found.  Are you sure your environment has nvcc or hipcc available?  "
            "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
            "only images whose names contain 'devel' will provide nvcc. "
            "If you are using ROCm, please use rocm/pytorch:latest from "
            "https://hub.docker.com/repository/docker/rocm/pytorch/general."
        )
    setup_requires = ["ninja"]
    nvcc_args_fused_adam = ["-O3", "--use_fast_math"]
    hipcc_args_fused_adam = ["-O3", "-ffast-math"]
    extensions.extend(
        [
            CUDAExtension(
                name="fairscale.fused_adam_cuda",
                sources=[
                    "fairscale/clib/fused_adam/fused_adam_cuda.cpp",
                    "fairscale/clib/fused_adam/fused_adam_cuda_kernel.cu",
                ],
                include_dirs=[os.path.join(this_dir, "fairscale/clib/fused_adam")],
                extra_compile_args={"cxx": ["-O3"],
                    "nvcc": nvcc_args_fused_adam if not is_rocm_pytorch else hipcc_args_fused_adam},
            )
        ]
    )

    cmdclass["build_ext"] = BuildExtension


if __name__ == "__main__":
    setuptools.setup(
        name="fairscale",
        description="FairScale: A PyTorch library for large-scale and high-performance training.",
        version=find_version("fairscale/version.py"),
        setup_requires=setup_requires,
        install_requires=fetch_requirements(),
        include_package_data=False,
        packages=setuptools.find_packages(include=["fairscale*"]),  # Only include code within fairscale.
        ext_modules=extensions,
        cmdclass=cmdclass,
        python_requires=">=3.8",
        author="Foundational AI Research @ Meta AI",
        author_email="todo@meta.com",
        long_description=(
            "FairScale is a PyTorch extension library for high performance and "
            "large scale training on one or multiple machines/nodes. This library "
            "extends basic PyTorch capabilities while adding new experimental ones."
        ),
        long_description_content_type="text/markdown",
        entry_points={"console_scripts": ["wgit = fairscale.experimental.wgit.__main__:main"]},
        classifiers=[
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Operating System :: OS Independent",
        ],
    )
