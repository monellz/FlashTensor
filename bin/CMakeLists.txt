get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
# get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)
get_property(asuka_libs GLOBAL PROPERTY ASUKA_LIBS)

message(STATUS "triton_libs=${triton_libs}")
message(STATUS "asuka_libs=${asuka_libs}")

add_llvm_executable(asuka-opt asuka-opt.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(asuka-opt)
target_link_libraries(asuka-opt PRIVATE
  # TritonLLVMIR
  # TritonAnalysis
  # TritonTransforms
  # TritonGPUTransforms
  # TritonNvidiaGPUTransforms
  # MLIRGPUToROCDLTransforms

  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  # ${triton_libs}
  ${asuka_libs}

  # # tests
  # TritonTestAnalysis
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRTransforms

  AsukaIR
  AsukaTritonIR
)
mlir_check_all_link_libraries(asuka-opt)


# translate
add_llvm_executable(asuka-translate asuka-translate.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(asuka-translate)
target_link_libraries(asuka-translate PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  ${triton_libs}
  ${asuka_libs}

  # MLIR core
  MLIRIR
  MLIRPass
  MLIROptLib
  MLIRSupport
  MLIRTransforms
  MLIRTranslateLib
)
mlir_check_all_link_libraries(asuka-translate)