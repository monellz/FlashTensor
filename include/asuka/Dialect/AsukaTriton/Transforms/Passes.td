#ifndef ASUKATRITON_PASSES
#define ASUKATRITON_PASSES

include "mlir/Pass/PassBase.td"

def AsukaTritonSqueezeBlock: Pass<"asukatriton-squeeze-block", "::mlir::asuka::KernelOp"> {
  let summary = ""; 
  let description = [{}];
  let constructor = "mlir::asuka::triton::createSqueezeBlockPass()";
  let dependentDialects = [
    "mlir::asuka::AsukaDialect",
    "mlir::asuka::triton::AsukaTritonDialect",
    "mlir::scf::SCFDialect"
  ];
}

def AsukaTritonUserReplicate: Pass<"asukatriton-user-replicate", "mlir::ModuleOp"> {
  let summary = "asuka triton replicate for only one use in device_kernel";
  let description = [{}];

  let constructor = "mlir::asuka::triton::createUserReplicatePass()";

  let dependentDialects = [
    "mlir::asuka::AsukaDialect",
    "mlir::asuka::triton::AsukaTritonDialect"
  ];
}


#endif // ASUKATRITON_PASSES
