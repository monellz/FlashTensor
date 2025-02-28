#!/bin/bash
set -x
PROJECT_FOLDER=$(dirname $(readlink -f "$0"))
BUILD=${PROJECT_FOLDER}/build

MLIR_DIR=/home/zhongrx/llm/tune/llvm-project/build/lib/cmake/mlir
LLVM_BUILD=/home/zhongrx/llm/tune/llvm-project/build/

# triton: https://github.com/triton-lang/triton/pull/3325

# cmake .. -GNinja \
#   -DCMAKE_BUILD_TYPE=Release \
#   -DLLVM_ENABLE_ASSERTIONS=ON \
#   -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \

#   -DTRITON_BUILD_PYTHON_MODULE=ON \
#   -DPython3_EXECUTABLE:FILEPATH=/home/zhongrx/miniconda3/envs/asuka/bin/python \
#   -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
#   -DPYTHON_INCLUDE_DIRS=/home/zhongrx/miniconda3/envs/asuka/include/python3.10 \
#   -DPYBIND11_INCLUDE_DIR=/home/zhongrx/miniconda3/envs/asuka/lib/python3.10/site-packages/pybind11/include \
#   -DTRITON_CODEGEN_BACKENDS="nvidia;amd" \
#   -DCUPTI_INCLUDE_DIR=/home/zhongrx/llm/tune/asuka/3rd/triton/third_party/nvidia/backend/include \
#   -DTRITON_BUILD_PROTON=OFF \
#   -DMLIR_DIR=${MLIR_DIR}

BUILD_TYPE=Debug

# without python
# ref: https://github.com/triton-lang/triton/pull/3325
if [ -n "$1" ]; then
  rm -rf ${BUILD}
  mkdir ${BUILD} && cd ${BUILD}
  cmake .. -GNinja \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_LINKER=lld \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \
    -DTRITON_BUILD_PYTHON_MODULE=OFF \
    -DTRITON_CODEGEN_BACKENDS="nvidia;amd" \
    -DMLIR_DIR=${MLIR_DIR}
else
  cd ${BUILD}
fi

cmake --build . -j64