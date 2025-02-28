#!/bin/bash
spack install -j64 gcc@12.2.0%gcc@11.4.0
spack load gcc@12.2.0
spack compiler add

# DONT install clang (pycuda will confuse)
spack install -j64 llvm@18.1.7%gcc@12.2.0 ~clang +mlir targets=nvptx,x86 ^python@3.10

spack install -j64 cuda@12.1.1%gcc@12.2.0
spack install -j64 py-pip%gcc@12.2.0 ^python@3.10
spack install -j64 git%gcc@12.2.0
spack install -j64 openmpi@4.1.5 %gcc@12.2.0 # for tensorrt_llm
