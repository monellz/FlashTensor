add_asuka_library(AsukaTritonTransforms
  SqueezeBlock.cpp
  # Blocking.cpp
  UserReplicate.cpp

  DEPENDS
  AsukaTritonTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  AsukaIR
  AsukaTritonIR
)