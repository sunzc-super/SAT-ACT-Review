#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 PROTOC_PREFIX PYTHON" >&2
  exit 2
fi

PROTOC_PREFIX=$1
PYTHON=$2
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROTO=${ROOT}/cadical/satact_trace.proto

"${PROTOC_PREFIX}/bin/protoc" \
  -I "${ROOT}/cadical" \
  --cpp_out="${ROOT}/cadical" \
  --grpc_out="${ROOT}/cadical" \
  --plugin=protoc-gen-grpc="${PROTOC_PREFIX}/bin/grpc_cpp_plugin" \
  "${PROTO}"

"${PYTHON}" -m grpc_tools.protoc \
  -I "${ROOT}/cadical" \
  --python_out="${ROOT}/inference/generated" \
  --grpc_python_out="${ROOT}/inference/generated" \
  "${PROTO}"
