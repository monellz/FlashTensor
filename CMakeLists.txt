cmake_minimum_required(VERSION 3.18.0)
if(POLICY CMP0116)
# Introduced in cmake 3.20
# https://cmake.org/cmake/help/latest/policy/CMP0116.html
  cmake_policy(SET CMP0116 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)

project(asuka LANGUAGES CXX C)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "MLIR_DIR=${MLIR_DIR}")
message(STATUS "LLVM_DIR=${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")


include(TableGen) # required by AddMLIR
include(AddLLVM)
include(AddMLIR)

set(LLVM_MAIN_SRC_DIR "${LLVM_BUILD_MAIN_SRC_DIR}")

# set(NO_RTTI "-fno-rtti")
# add_definitions(${NO_RTTI})
add_definitions("-fPIC")

# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rd/triton)


include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})


# Utilities
function(add_asuka_object name)
  cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})
  add_library(${name} OBJECT)
  target_sources(${name}
    PRIVATE ${ARG_UNPARSED_ARGUMENTS}
    INTERFACE $<TARGET_OBJECTS:${name}>
  )
  if(ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()
  if(ARG_LINK_LIBS)
    target_link_libraries(${name} PUBLIC ${ARG_LINK_LIBS})
  endif()
endfunction(add_asuka_object)

set_property(GLOBAL PROPERTY ASUKA_LIBS "")
function(add_asuka_library name)
  set_property(GLOBAL APPEND PROPERTY ASUKA_LIBS ${name})
  add_asuka_object(${name} ${ARGN})
  llvm_update_compile_flags(${name})
endfunction()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/3rd/dbg)
# include_directories(${PROJECT_SOURCE_DIR}/3rd/triton/include)
# include_directories(${PROJECT_SOURCE_DIR}/3rd/triton/third_party)

# tablegen
include_directories(${PROJECT_BINARY_DIR}/include)
# include_directories(${PROJECT_BINARY_DIR}/3rd/triton/include)
# include_directories(${PROJECT_BINARY_DIR}/3rd/triton/third_party)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/bin)



# python
include(${PROJECT_SOURCE_DIR}/cmake/QueryPythonForPybind11.cmake)
find_pybind11_python_first()
file(GLOB_RECURSE FFI_SRC ${PROJECT_SOURCE_DIR}/ffi/*.cpp)
pybind11_add_module(asuka_ffi SHARED ${FFI_SRC})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
# get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)
get_property(asuka_libs GLOBAL PROPERTY ASUKA_LIBS)

target_link_libraries(asuka_ffi PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
#  ${triton_libs}
  ${asuka_libs}

  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRTransforms

  AsukaIR
  AsukaTritonIR
)
install(TARGETS asuka_ffi LIBRARY DESTINATION ${PY_BUILD_CMAKE_MODULE_NAME} COMPONENT python_module)
