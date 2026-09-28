#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 BUILD_DIR GRPC_PREFIX [JOBS]" >&2
  exit 2
fi

BUILD_DIR=$1
GRPC_PREFIX=$2
JOBS=${3:-2}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

cmake -S "${ROOT}/cadical" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGRPC_ROOT="${GRPC_PREFIX}"
cmake --build "${BUILD_DIR}" --target cadical-satact -j "${JOBS}"
