add_asuka_library(AsukaIR
  AsukaOps.cpp
  AsukaTypes.cpp

  DEPENDS
  AsukaTableGen

  LINK_LIBS PUBLIC
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