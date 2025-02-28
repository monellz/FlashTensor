set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS AsukaTritonOps.td)
mlir_tablegen(AsukaTritonOps.h.inc -gen-op-decls)
mlir_tablegen(AsukaTritonOps.cpp.inc -gen-op-defs)
add_mlir_doc(AsukaTritonOps AsukaTritonOps AsukaTriton/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS AsukaTritonDialect.td)
mlir_tablegen(AsukaTritonDialect.h.inc -gen-dialect-decls)
mlir_tablegen(AsukaTritonDialect.cpp.inc -gen-dialect-defs)
add_mlir_doc(AsukaTritonDialect AsukaTritonDialect AsukaTriton/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS AsukaTritonTypes.td)
mlir_tablegen(AsukaTritonTypes.h.inc -gen-typedef-decls)
mlir_tablegen(AsukaTritonTypes.cpp.inc -gen-typedef-defs)


add_public_tablegen_target(AsukaTritonTableGen)