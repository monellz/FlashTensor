#ifndef ASUKA_DIALECT
#define ASUKA_DIALECT

include "mlir/IR/OpBase.td"


def Asuka_Dialect : Dialect {
  let name = "asuka";

  let cppNamespace = "::mlir::asuka";

  let summary = "Asuka Dialact in MLIR";

  let description = [{
    Asuka Dialect.
  }];

  let dependentDialects = [
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

  let useDefaultAttributePrinterParser = 1;
}

#endif // ASUKA_DIALECT