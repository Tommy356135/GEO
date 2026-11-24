#!/usr/bin/env bash
# Build an already-downloaded official MinkowskiEngine 0.5.x source tree.
# This script is intentionally not invoked by Phase 1A--1C preparation.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/minkowski_build_cu121.log"
mkdir -p "${PROJECT_ROOT}/logs"
exec > >(tee "${LOG_FILE}") 2>&1

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ $# -eq 1 ]] || fail "usage: $0 /absolute/path/to/official-MinkowskiEngine-0.5.x"
SOURCE_DIR="$(realpath "$1")"
[[ -d "${SOURCE_DIR}" ]] || fail "source directory does not exist: ${SOURCE_DIR}"
[[ -f "${SOURCE_DIR}/setup.py" ]] || fail "setup.py not found in source directory"
[[ -f "${SOURCE_DIR}/MinkowskiEngine/__init__.py" ]] || fail "not a MinkowskiEngine source tree"
python - "${SOURCE_DIR}" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
text = (root / "MinkowskiEngine" / "__init__.py").read_text(encoding="utf-8")
match = re.search(r"__version__\s*=\s*['\"](0\.5\.[0-9]+)['\"]", text)
if match is None:
    raise SystemExit("official MinkowskiEngine version 0.5.x was not detected")
print(f"MinkowskiEngine source version={match.group(1)}")
PY

[[ "${CONDA_DEFAULT_ENV:-}" == "geomorph-mink" ]] || \
  fail "activate the geomorph-mink environment first"

python - <<'PY'
import sys
import torch

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Python 3.10 required, found {sys.version.split()[0]}")
if torch.__version__.split("+")[0] != "2.3.1":
    raise SystemExit(f"torch 2.3.1 required, found {torch.__version__}")
if torch.version.cuda != "12.1":
    raise SystemExit(f"torch.version.cuda 12.1 required, found {torch.version.cuda}")
PY

[[ "${CUDA_HOME:-}" == "/usr/local/cuda-12.1" ]] || \
  fail "CUDA_HOME must be explicitly set to /usr/local/cuda-12.1"
export CUDA_HOME
[[ -x "${CUDA_HOME}/bin/nvcc" ]] || fail "missing ${CUDA_HOME}/bin/nvcc"
"${CUDA_HOME}/bin/nvcc" --version | grep -Eq 'release 12\.1([,.]|$)' || \
  fail "nvcc is not CUDA 12.1"

CC_BIN="${CC:-gcc}"
CXX_BIN="${CXX:-g++}"
command -v "${CC_BIN}" >/dev/null || fail "C compiler not found: ${CC_BIN}"
command -v "${CXX_BIN}" >/dev/null || fail "C++ compiler not found: ${CXX_BIN}"
[[ "$("${CC_BIN}" -dumpversion | cut -d. -f1)" == "10" ]] || \
  fail "GCC 10 required; ${CC_BIN} reports $("${CC_BIN}" -dumpfullversion)"
[[ "$("${CXX_BIN}" -dumpversion | cut -d. -f1)" == "10" ]] || \
  fail "G++ 10 required; ${CXX_BIN} reports $("${CXX_BIN}" -dumpfullversion)"

command -v pkg-config >/dev/null || fail "pkg-config is required to check OpenBLAS"
pkg-config --exists openblas || \
  fail "OpenBLAS development metadata not found (pkg-config openblas failed)"

export CC="${CC_BIN}"
export CXX="${CXX_BIN}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS="${MAX_JOBS:-8}"

echo "source_dir=${SOURCE_DIR}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "CC=$(${CC} --version | head -n 1)"
echo "CXX=$(${CXX} --version | head -n 1)"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "MAX_JOBS=${MAX_JOBS}"
echo "OpenBLAS=$(pkg-config --modversion openblas)"

cd "${SOURCE_DIR}"
python setup.py install --force_cuda --blas=openblas
