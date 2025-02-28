add_asuka_library(AsukaTritonIR
  AsukaTritonOps.cpp
  AsukaTritonTypes.cpp

  DEPENDS
  AsukaTritonTableGen

  LINK_LIBS PUBLIC
  AsukaIR

  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)


# add_mlir_dialect_library(AsukaIR
#   AsukaDialect.cpp
#   AsukaOps.cpp

#   DEPENDS
#   AsukaTableGen

#   LINK_LIBS PUBLIC
#   MLIRIR
#   MLIRArithDialect
#   MLIRMathDialect
#   MLIRSCFDialect
# )