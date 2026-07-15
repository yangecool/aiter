#!/usr/bin/env bash
# Run gfx1201 unified-attention tests inside a ROCm container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AITER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEST_FILE="op_tests/triton_tests/attention/test_unified_attention.py"
MODE="${1:-focused}"

if [[ $# -gt 0 ]]; then
    shift
fi

case "${MODE}" in
    focused)
        TEST_TARGET="${TEST_FILE} -k gfx1201"
        ;;
    full)
        TEST_TARGET="${TEST_FILE}"
        ;;
    collect)
        TEST_TARGET="${TEST_FILE} --collect-only"
        ;;
    *)
        echo "Usage: $0 [focused|full|collect] [pytest arguments...]" >&2
        exit 2
        ;;
esac

if [[ ! -e /dev/kfd ]]; then
    echo "ERROR: /dev/kfd is missing. Start the container with ROCm devices:" >&2
    echo "  --device=/dev/kfd --device=/dev/dri --group-add video" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN=python
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: python3 or python is required" >&2
    exit 1
fi

cd "${AITER_ROOT}"
export PYTHONPATH="${AITER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache-gfx1201-ua}"
mkdir -p "${TRITON_CACHE_DIR}"

read -r GPU_NAME GPU_ARCH < <(
    "${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot access a ROCm GPU")
props = torch.cuda.get_device_properties(0)
print(torch.cuda.get_device_name(0).replace(" ", "_"), props.gcnArchName)
PY
)

if [[ "${GPU_ARCH}" != gfx1201* ]]; then
    echo "ERROR: expected gfx1201, detected ${GPU_ARCH} (${GPU_NAME//_/ })" >&2
    exit 1
fi

echo "=== gfx1201 unified-attention test ==="
echo "mode:         ${MODE}"
echo "repo:         ${AITER_ROOT}"
echo "test:         ${TEST_TARGET}"
echo "gpu:          ${GPU_NAME//_/ }"
echo "arch:         ${GPU_ARCH}"
echo "triton cache: ${TRITON_CACHE_DIR}"

"${PYTHON_BIN}" - <<'PY'
import aiter
import pytest
import torch
import triton

print(f"python aiter:  {aiter.__file__}")
print(f"torch:         {torch.__version__}")
print(f"triton:        {triton.__version__}")
print(f"pytest:        {pytest.__version__}")
PY

echo
# TEST_TARGET intentionally contains the file and optional pytest selector.
# shellcheck disable=SC2086
"${PYTHON_BIN}" -m pytest -vv ${TEST_TARGET} "$@"
