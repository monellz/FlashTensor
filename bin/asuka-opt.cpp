#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"
#include "asuka/Dialect/AsukaTriton/IR/AsukaTritonDialect.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

// pass
#include "asuka/Dialect/Asuka/Transforms/Passes.h"
#include "asuka/Dialect/AsukaTriton/Transforms/Passes.h"
// #include "triton/Dialect/Triton/Transforms/Passes.h"
// #include "triton/Dialect/TritonGPU/Transforms/Passes.h"
// #include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
// #include "triton/Target/LLVMIR/Passes.h"

// conversion pass
#include "asuka/Conversion/AsukaToAsukaTriton/Passes.h"
// #include "triton/Conversion/TritonToTritonGPU/Passes.h"
// #include "triton/Conversion/TritonGPUToLLVM/Passes.h"
// #include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
// #include "nvidia/include/NVGPUToLLVM/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  // // triton passes
  // mlir::registerTritonPasses();
  // mlir::triton::gpu::registerTritonGPUPasses();
  // mlir::registerTritonNvidiaGPUPasses();

  // // conversion passes
  // mlir::triton::registerTritonToTritonGPUPasses();
  // mlir::triton::registerTritonNVIDIAGPUToLLVMPasses();
  // mlir::triton::registerNVGPUToLLVMPasses();
  // mlir::triton::registerTritonGPUToLLVMPasses();
  // mlir::registerLLVMIRPasses();

  // asuka
  mlir::asuka::registerAsukaPasses();
  mlir::asuka::triton::registerAsukaTritonPasses();
  mlir::asuka::registerAsukaToAsukaTritonPasses();

  mlir::DialectRegistry registry;

  // there is another NVGPU in mlir which will cause conflict
  // mlir::registerAllDialects(registry);

  mlir::registerAllExtensions(registry);

  // clang-format off
  registry.insert<mlir::asuka::AsukaDialect,
                  mlir::asuka::triton::AsukaTritonDialect,

                  // mlir::triton::TritonDialect,
                  // mlir::triton::gpu::TritonGPUDialect,
                  // mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
                  // mlir::triton::nvgpu::NVGPUDialect,

                  mlir::func::FuncDialect,
                  mlir::arith::ArithDialect,
                  mlir::math::MathDialect,
                  mlir::scf::SCFDialect,
                  mlir::cf::ControlFlowDialect>();
  // clang-format on

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "Asuka optimizer\n", registry));
}