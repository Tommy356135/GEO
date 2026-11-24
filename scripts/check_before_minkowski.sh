#!/usr/bin/env bash
# Read-only prerequisites report. This script never installs or changes anything.

set -u

echo "[python]"
if command -v python >/dev/null 2>&1; then
  python --version
  python - <<'PY'
try:
    import torch
except ImportError as error:
    print(f"torch: not importable ({error})")
else:
    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
PY
else
  echo "python: not found"
fi

echo "[CUDA_HOME]"
echo "${CUDA_HOME:-not set}"

echo "[nvcc]"
if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
  "${CUDA_HOME}/bin/nvcc" --version
elif command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "nvcc: not found"
fi

echo "[gcc]"
if command -v gcc >/dev/null 2>&1; then
  gcc --version
else
  echo "gcc: not found"
fi

echo "[g++]"
if command -v g++ >/dev/null 2>&1; then
  g++ --version
else
  echo "g++: not found"
fi

