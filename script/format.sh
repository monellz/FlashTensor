#!/bin/bash
set -x

PROJECT_FOLDER=$(dirname  $(dirname $(readlink -f "$0")))


find ${PROJECT_FOLDER}/include -type f -name "*.cpp" -o -name "*.h" | xargs clang-format -i 
find ${PROJECT_FOLDER}/lib -type f -name "*.cpp" -o -name "*.h" | xargs clang-format -i 
find ${PROJECT_FOLDER}/bin -type f -name "*.cpp" -o -name "*.h" | xargs clang-format -i 
find ${PROJECT_FOLDER}/ffi -type f -name "*.cpp" -o -name "*.h" | xargs clang-format -i 

yapf --recursive --in-place --style ${PROJECT_FOLDER}/.style.yapf ${PROJECT_FOLDER}/python