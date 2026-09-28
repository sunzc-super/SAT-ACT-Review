# Environment and build requirements

## Tested environment

The archive was validated with:

- Linux x86-64;
- CMake 3.22.1;
- GCC/G++ 11.4;
- Python 3.12;
- gRPC C++ 1.76;
- Protobuf compiler 31.1.

These are tested versions, not strict minimum versions.

## Common C/C++ tools

Both C++ targets require:

- CMake 3.10 or newer;
- a C99 compiler;
- a C++11 compiler;
- a standard build tool such as GNU Make or Ninja.

LLVM `ld.lld` is optional. The build uses the default system linker when it is
not available.

## DecisionTrace

The `decisiontrace` target under `gen_data/cadical/src` uses only the system
C/C++ runtime. It does not require gRPC, Protobuf, CUDA, or PyTorch.

## Neural-assisted CaDiCaL

The solver under `solver/cadical` additionally requires gRPC C++ and Protobuf
C++. Both CMake packages must be available under the prefix passed to
`solver/build.sh`. This prefix is the installed library root, not the gRPC
source directory. For example, an installation under `/opt/grpc-1.76.0` uses
`GRPC_PREFIX=/opt/grpc-1.76.0` and has the following layout:

```text
/opt/grpc-1.76.0/lib/cmake/grpc/gRPCConfig.cmake
/opt/grpc-1.76.0/lib/cmake/protobuf/protobuf-config.cmake
```

The generated protobuf C++ and Python sources are included. A normal solver
build does not invoke `protoc`. The following tools are required only after
editing `solver/cadical/satact_trace.proto`:

```text
<grpc-prefix>/bin/protoc
<grpc-prefix>/bin/grpc_cpp_plugin
```

Compiling either C++ target does not require CUDA. The neural model runs in the
separate Python inference service.

## Python environment

Install the packages listed in `requirements.txt` in a virtual or Conda
environment. One complete virtual-environment setup is:

```bash
# Create a Python environment and install the runtime dependencies.
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Training and GPU inference require a PyTorch build compatible with the local
CUDA driver. CPU inference is supported but is slower. The `grpcio` and
`protobuf` packages in `requirements.txt` are Python packages; they do not
replace the gRPC C++ and Protobuf C++ libraries needed to compile the solver.
