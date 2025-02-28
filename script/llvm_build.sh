#!/bin/bash
set -x


PROJECT_FOLDER=$(dirname  $(dirname $(readlink -f "$0")))

# LLVM_COMMIT_HASH="$(cat ${PROJECT_FOLDER}/3rd/triton/cmake/llvm-hash.txt)"
LLVM_COMMIT_HASH=765206e050453018e861637a08a4520f29238074


git clone https://github.com/llvm/llvm-project.git --recursive || exit -1
cd llvm-project
git checkout $LLVM_COMMIT_HASH



mkdir build
cd build
cmake -G Ninja ../llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX" 
ninja