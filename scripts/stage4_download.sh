#!/usr/bin/env bash
# Download model artifacts for a subset of the matrix.
# Usage: stage4_download.sh <model_key> [<model_key> ...]
set -uo pipefail

ENV_ROOT="$HOME/benchmax-env"
source "$ENV_ROOT/venv-bench/bin/activate"
cd /mnt/c/benchmaxing

MODELS_DIR="$ENV_ROOT/models"
mkdir -p "$MODELS_DIR"

MODELS=("$@")
echo "downloading for models: ${MODELS[*]}"

python -m benchmax.cli \
  --results results \
  --models-dir "$MODELS_DIR" \
  download \
  --backends llamacpp vllm vllm_gguf \
  --models "${MODELS[@]}"

# The retry driver polls for this marker. Written only when every requested
# artifact already resolves to a local file with no transfer pending, so a
# WSL restart mid-transfer cannot be mistaken for completion.
missing=0
python - <<'PY'
import os
from benchmax import config as C
from benchmax.models import _hf, _pick_gguf, GGUF_BACKENDS
api, hf_hub_download, snapshot_download = _hf()
need = 0
for spec in C.MODELS:
    for backend in ("llamacpp", "vllm", "vllm_gguf"):
        for cell in C.build_matrix([spec], [backend], "cuda"):
            if cell.status == "skipped":
                continue
            repo = cell.source_repo
            if not repo:
                continue
            local = os.path.expanduser(
                f"~/benchmax-env/models/{repo.replace('/', '__')}")
            if backend in GGUF_BACKENDS:
                try:
                    files = api.list_repo_files(repo)
                    want = _pick_gguf(files, cell.quant_level)
                except Exception:
                    want = None
                ok = bool(want) and os.path.exists(os.path.join(local, want))
            else:
                ok = os.path.isdir(local) and any(
                    f.endswith('.safetensors') for f in os.listdir(local))
            if not ok:
                need += 1
                print(f"  MISSING {backend} {spec.key} {cell.quant_level} <- {repo}")
print(f"missing_artifacts={need}")
with open(os.path.expanduser('~/benchmax-logs/.remaining'), 'w') as f:
    f.write(str(need))
PY

echo
echo "=== disk used by models ==="
du -sh "$MODELS_DIR" 2>/dev/null || true
echo "STAGE4 DONE"