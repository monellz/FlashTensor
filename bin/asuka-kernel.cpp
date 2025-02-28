#include "mlir/Parser/Parser.h"

#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/SMLoc.h"

#include "dbg.h"

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

namespace mlir {} // namespace mlir

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

  static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional, llvm::cl::desc("<input file>"),
                                                  llvm::cl::init("-"));

  static llvm::cl::opt<std::string> outputFilename("o", llvm::cl::desc("Output filename"),
                                                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

  llvm::cl::ParseCommandLineOptions(argc, argv, "asuka kernel");

  std::string errorMessage;
  auto input = mlir::openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  std::unique_ptr<llvm::MemoryBuffer> ownedBuffer = std::move(input);
  llvm::raw_ostream &os = output->os();

  mlir::MLIRContext context;
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), llvm::SMLoc());

  mlir::OwningOpRef<mlir::ModuleOp> module(parseSourceFile<mlir::ModuleOp>(sourceMgr, &context));

  module.get()->dump();

  mlir::PassManager pm(&context);
  pm.addPass(mlir::asuka::createLowerComplexReducePass());
  if (mlir::failed(pm.run(module.get()))) {
    dbg("failed");
  } else {
    dbg("succ");
  }

  mlir::func::FuncOp func_op;
  module.get()->walk([&](mlir::func::FuncOp op) {
    func_op = op;
    return mlir::WalkResult::interrupt();
  });

  func_op->dump();

  return 0;
}