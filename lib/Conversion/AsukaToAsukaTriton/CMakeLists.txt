add_asuka_library(AsukaToAsukaTriton
  AsukaToAsukaTritonPass.cpp

  DEPENDS
  AsukaConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  AsukaIR 
  AsukaTritonIR
)