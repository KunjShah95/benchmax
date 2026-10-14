#!/usr/bin/env bash
# Fetch NVIDIA's official redist manifest for CUDA 12.8.1 and list the
# component tarballs we need. No sudo, no conda, just tarballs.
set -euo pipefail
cd "$HOME/benchmax-env"
curl -fsSL -o redistrib_12.8.1.json \
  https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json
echo "manifest bytes: $(wc -c < redistrib_12.8.1.json)"
python3 - <<'PY'
import json
d = json.load(open("redistrib_12.8.1.json"))
print("release_label:", d.get("release_label"))
want = ["cuda_nvcc","cuda_cudart","libcublas","cuda_cccl","cuda_nvrtc",
        "cuda_cupti","cuda_profiler_api","libcurand","libcusparse","cuda_nvml_dev","libcufile"]
for k in want:
    e = d.get(k, {}).get("linux-x86_64")
    if e:
        size = float(e.get("size", 0))/1e6
        print(f"{k:22} {size:8.1f} MB  {e['relative_path']}")
    else:
        print(f"{k:22}   (absent)")
PY