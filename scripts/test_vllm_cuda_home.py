#!/usr/bin/env python
"""Confirm the vLLM CUDA_HOME resolver finds a usable nvcc."""
import os
import subprocess

from benchmax.backends import _vllm_cuda_home

venv_bin = os.path.expanduser("~/benchmax-env/venv-vllm/bin")
home = _vllm_cuda_home(venv_bin)
print("CUDA_HOME ->", home)
if not home:
    raise SystemExit("FAIL: resolver returned None")

nvcc = os.path.join(home, "bin", "nvcc")
print("nvcc exists:", os.path.exists(nvcc))
out = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
print("nvcc version:", (out.stdout or out.stderr).strip().splitlines()[-1][:70])
print("lib64 ->", os.path.realpath(os.path.join(home, "lib64")))

import torch  # noqa: E402
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("RESULT: PASS")