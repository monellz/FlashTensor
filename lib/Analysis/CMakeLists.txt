add_asuka_library(AsukaAnalysis
  Parallelism.cpp

  DEPENDS
  AsukaTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRIR
  AsukaIR 
)