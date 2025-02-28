add_asuka_library(AsukaTranslate
  to_triton.cpp

  DEPENDS
  AsukaTritonTableGen

  LINK_LIBS PUBLIC
  AsukaIR
  MLIRIR
)