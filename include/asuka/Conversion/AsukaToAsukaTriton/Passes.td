#ifndef ASUKATOASUKATRITON_CONVERSION_PASSES
#define ASUKATOASUKATRITON_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertAsukaToAsukaTriton: Pass<"convert-asuka-to-asukatriton", "::mlir::asuka::KernelOp"> {
  let summary = "convert asuka to asuka triton";
  let description = [{}];

  let constructor = "mlir::asuka::createConvertAsukaToAsukaTritonPass()";

  let dependentDialects = [
    "mlir::asuka::AsukaDialect",
    "mlir::asuka::triton::AsukaTritonDialect"
  ];
}



#endif // ASUKATOASUKATRITON_CONVERSION_PASSES