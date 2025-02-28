#ifndef ASUKA_PASSES
#define ASUKA_PASSES

include "mlir/Pass/PassBase.td"

def AsukaSimplify: InterfacePass<"asuka-simplify", "FunctionOpInterface"> {
  let summary = "Asuka simplify";
  let description = [{
    remove redundent op
  }];

  let constructor = "mlir::asuka::createSimplifyPass()";

  let dependentDialects = [
    "mlir::asuka::AsukaDialect",
  ];
}

def AsukaReplaceExpAndLog: Pass<"asuka-replace-exp-and-log", "::mlir::asuka::KernelOp"> {
  let summary = "";
  let description = [{
    replace exp and log by base 2
  }];
  let constructor = "mlir::asuka::createReplaceExpAndLogPass()";
  let dependentDialects = [
    "mlir::asuka::AsukaDialect"
  ];
}


// FIXME: this pass should be applied on module_op or function op interface?
def AsukaLowerComplexReduce: InterfacePass<"asuka-lower-complex-reduce", "FunctionOpInterface"> {
  let summary = "";
  let description = [{}];
  let constructor = "mlir::asuka::createLowerComplexReducePass()";

  let dependentDialects = [
    "mlir::asuka::AsukaDialect"
  ];
}


def AsukaEraseTypeInKernel: Pass<"asuka-erase-type-in-kernel", "::mlir::asuka::KernelOp"> {
  let summary = "";
  let description = [{}];  
  let constructor = "mlir::asuka::createEraseTypeInKernelPass()";
  let dependentDialects = [
    "mlir::asuka::AsukaDialect"
  ];
}

def AsukaBroadcastTransform: Pass<"asuka-broadcast-transform", "mlir::ModuleOp"> {
  let summary = "";  
  let description = [{}];
  let constructor = "mlir::asuka::createBroadcastTransformPass()";
  let dependentDialects = [
    "mlir::asuka::AsukaDialect"
  ];
}

def AsukaEquivalentTransform: Pass<"asuka-equivalent-transform", "::mlir::asuka::KernelOp"> {
  let summary = "";  
  let description = [{}];
  let constructor = "mlir::asuka::createEquivalentTransformPass()";
  let dependentDialects = [
    "mlir::asuka::AsukaDialect"
  ];
}

def AsukaMulScalarHoisting: Pass<"asuka-mul-scalar-hoisting", "mlir::ModuleOp"> {
  let summary = "asuka mul scalar hoisting";
  let description = [{
  }];

  let constructor = "mlir::asuka::createMulScalarHoistingPass()";

  let dependentDialects = [
    "mlir::asuka::AsukaDialect",
  ];
}

def AsukaPermuteHoisting: Pass<"asuka-permute-hoisting", "mlir::ModuleOp"> {
  let summary = "asuka permute hoisting";
  let description = [{
  }];

  let constructor = "mlir::asuka::createPermuteHoistingPass()";

  let dependentDialects = [
    "mlir::asuka::AsukaDialect"
  ];

}

def AsukaAnnotateParallelism: Pass<"asuka-annotate-parallelism", "::mlir::asuka::KernelOp"> {
  let summary = "asuka annotate parallelism";
  let description = [{}];

  let constructor = "mlir::asuka::createAnnotateParallelismPass()";

  let options = [
    Option<"verbose", "verbose", "bool", /*default=*/"false", "verbose output">,
  ];

  let dependentDialects = [
    "mlir::asuka::AsukaDialect"
  ];
}

// FIXME: or function op interface?
def AsukaToMask: Pass<"asuka-to-mask", "::mlir::asuka::KernelOp"> {
  let summary = "asuka other op to mask op";
  let description = [{}];

  let constructor = "mlir::asuka::createToMaskPass()";

  let dependentDialects = [
    "mlir::asuka::AsukaDialect"
  ];
}

def AsukaParallelize: Pass<"asuka-parallelize", "::mlir::asuka::KernelOp"> {
  let summary = "asuka parallelize";
  let description = [{}];

  let constructor = "mlir::asuka::createParallelizePass()";

  let options = [
    Option<"partition", "partition", "bool", /*default=*/"true", "whether to partition">,
  ];

  let dependentDialects = [
    "mlir::asuka::AsukaDialect"
  ];
}

def AsukaTiling: Pass<"asuka-tiling", "::mlir::asuka::KernelOp"> {
  let summary = "asuka tiling";
  let description = [{}];
  let constructor = "mlir::asuka::createTilingPass()";
  let dependentDialects = [
    "mlir::asuka::AsukaDialect"
  ];
}

def AsukaDynamicFor: Pass<"asuka-dynamic-for", "::mlir::asuka::KernelOp"> {
  let summary = "asuka dynamic for";
  let description = [{}];
  let constructor = "mlir::asuka::createDynamicForPass()";
  let dependentDialects = [
    "mlir::asuka::AsukaDialect"
  ];
}

def AsukaRecoverTypeInKernel: Pass<"asuka-recover-type-in-kernel", "::mlir::asuka::KernelOp"> {
  let summary = "";
  let description = [{}];
  let constructor = "mlir::asuka::createRecoverTypeInKernelPass()";
  let dependentDialects = [
    "mlir::asuka::AsukaDialect"
  ];
}

def AsukaUserReplicate: Pass<"asuka-user-replicate", "mlir::ModuleOp"> {
  let summary = "asuka replicate for only one use in device_kernel";
  let description = [{}];

  let constructor = "mlir::asuka::createUserReplicatePass()";

  let dependentDialects = [
    "mlir::asuka::AsukaDialect"
  ];
}


#endif // ASUKA_PASSES