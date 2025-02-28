# set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# set(LLVM_TARGET_DEFINITIONS AsukaDialect.td)
# mlir_tablegen(AsukaDialect.h.inc -gen-dialect-decls)
# mlir_tablegen(AsukaDialect.cpp.inc -gen-dialect-defs)
# add_mlir_doc(AsukaDialect AsukaDialect Asuka/ -gen-dialect-doc)


# add_mlir_dialect(AsukaOps asuka)
# add_mlir_doc(AsukaDialect AsukaDialect Asuka/ -gen-dialect-doc)
# # add_mlir_doc(AsukaOps AsukaOps Asuka/ -gen-op-doc)


set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS AsukaOps.td)
mlir_tablegen(AsukaOps.h.inc -gen-op-decls)
mlir_tablegen(AsukaOps.cpp.inc -gen-op-defs)
add_mlir_doc(AsukaOps AsukaOps Asuka/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS AsukaDialect.td)
mlir_tablegen(AsukaDialect.h.inc -gen-dialect-decls)
mlir_tablegen(AsukaDialect.cpp.inc -gen-dialect-defs)
add_mlir_doc(AsukaDialect AsukaDialect Asuka/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS AsukaTypes.td)
mlir_tablegen(AsukaTypes.h.inc -gen-typedef-decls)
mlir_tablegen(AsukaTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS AsukaEnums.td)
mlir_tablegen(AsukaEnums.h.inc -gen-enum-decls)
mlir_tablegen(AsukaEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(AsukaAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(AsukaAttrs.cpp.inc -gen-attrdef-defs)


set(LLVM_TARGET_DEFINITIONS AsukaInterfaces.td)
mlir_tablegen(AsukaInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(AsukaInterfaces.cpp.inc -gen-op-interface-defs)


add_public_tablegen_target(AsukaTableGen)