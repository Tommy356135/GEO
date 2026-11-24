#!/usr/bin/env bash
# Create the isolated environment when the user explicitly runs this script.
# This repository change only prepares the script; it does not execute it.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVIRONMENT_FILE="${PROJECT_ROOT}/environment-minkowski-cu121.yml"
ENVIRONMENT_NAME="geomorph-mink"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found on PATH." >&2
  exit 1
fi

if [[ ! -f "${ENVIRONMENT_FILE}" ]]; then
  echo "ERROR: environment file not found: ${ENVIRONMENT_FILE}" >&2
  exit 1
fi

conda env create --file "${ENVIRONMENT_FILE}"

conda run --no-capture-output --name "${ENVIRONMENT_NAME}" \
  python -m pip install \
    torch==2.3.1 \
    torchvision==0.18.1 \
    torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

conda run --no-capture-output --name "${ENVIRONMENT_NAME}" \
  python -m pip install numpy==1.26.4

conda run --no-capture-output --name "${ENVIRONMENT_NAME}" \
  python -c 'import torch; assert torch.__version__.split("+")[0] == "2.3.1", torch.__version__; assert torch.version.cuda == "12.1", torch.version.cuda; print(f"Verified PyTorch {torch.__version__}, CUDA runtime {torch.version.cuda}")'

if ! conda list --name "${ENVIRONMENT_NAME}" --json | \
  conda run --no-capture-output --name "${ENVIRONMENT_NAME}" \
    python -c 'import json, sys; packages = json.load(sys.stdin); bad = ["{}={}".format(p["name"], p["version"]) for p in packages if p["name"] in {"mkl", "intel-openmp"} and p["version"].startswith("2025.")]; print("Unexpected Conda packages: " + ", ".join(bad), file=sys.stderr) if bad else print("Verified: no Conda MKL/Intel OpenMP 2025 packages"); raise SystemExit(bool(bad))'
then
  echo "ERROR: Conda MKL/Intel OpenMP 2025 dependency detected." >&2
  exit 1
fi

echo "Environment creation and validation completed. Activate it with:"
echo "  conda activate geomorph-mink"
