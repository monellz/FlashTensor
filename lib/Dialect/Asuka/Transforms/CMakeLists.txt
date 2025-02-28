add_asuka_library(AsukaTransforms
  Simplify.cpp
  ReplaceExpAndLog.cpp

  LowerComplexReduce.cpp

  EraseTypeInKernel.cpp

  MulScalarHoisting.cpp
  BroadcastTransform.cpp

  EquivalentTransform.cpp

  PermuteHoisting.cpp

  AnnotateParallelism.cpp

  ToMask.cpp
  
  Parallelize.cpp

  Tiling.cpp

  DynamicFor.cpp

  RecoverTypeInKernel.cpp



  UserReplicate.cpp

  DEPENDS
  AsukaTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  AsukaIR
)