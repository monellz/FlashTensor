#ifndef ASUKATRITON_DIALECT
#define ASUKATRITON_DIALECT

include "mlir/IR/OpBase.td"


def AsukaTriton_Dialect : Dialect {
  let name = "asuka_triton";

  let cppNamespace = "::mlir::asuka::triton";

  let summary = "AsukaTriton Dialact in MLIR";

  let description = [{
    AsukaTriton Dialect.
  }];

  let dependentDialects = [
    "::mlir::asuka::AsukaDialect",

    // "::mlir::triton::TritonDialect",
    // "::mlir::triton::gpu::TritonGPUDialect",
    // "::mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
    // "::mlir::triton::nvgpu::NVGPUDialect",

    "::mlir::func::FuncDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::cf::ControlFlowDialect"
  ];

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  // need explicitly set
  let useDefaultTypePrinterParser = 1;
}

#endif // ASUKA_DIALECT