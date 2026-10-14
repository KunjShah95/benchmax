#!/usr/bin/env python
"""Is GGUF still a supported quantization backend in this vLLM build?"""
import importlib
import pkgutil

try:
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    print("registered quantization methods:")
    for m in sorted(QUANTIZATION_METHODS):
        print("   ", m)
except Exception as e:
    print("could not read QUANTIZATION_METHODS:", type(e).__name__, e)

print()
try:
    import vllm.model_executor.layers.quantization.gguf as g
    print("gguf module present:", g.__file__)
except Exception as e:
    print("gguf module import failed:", type(e).__name__, str(e)[:120])

print()
try:
    from vllm.transformers_utils.config import GGUFConfig
    print("GGUFConfig available:", GGUFConfig)
except Exception as e:
    print("GGUFConfig unavailable:", type(e).__name__, str(e)[:120])