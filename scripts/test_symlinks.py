#!/usr/bin/env python
"""Create the missing unversioned CUDA .so symlinks and verify ld resolves them."""
import os

from benchmax.backends import _vllm_cuda_home

home = _vllm_cuda_home(os.path.expanduser("~/benchmax-env/venv-vllm/bin"))
print("CUDA_HOME:", home)
libdir = os.path.join(home, "lib")
print("libcudart.so exists:", os.path.exists(os.path.join(libdir, "libcudart.so")))
print("libcublas.so exists:", os.path.exists(os.path.join(libdir, "libcublas.so")))