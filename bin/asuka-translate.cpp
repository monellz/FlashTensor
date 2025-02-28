#include "mlir/InitAllDialects.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"
#include "asuka/Dialect/AsukaTriton/IR/AsukaTritonDialect.h"
#include "asuka/Translate/translate.h"

namespace mlir::asuka {

void registerAllDialects(DialectRegistry &registry) {
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
                  mlir::cf::ControlFlowDialect,
                  mlir::asuka::AsukaDialect>();
  // clang-format on
}

} // namespace mlir::asuka

namespace mlir {

TranslateFromMLIRRegistration serializeRegistration(
    "mlir-to-py", "Asuka mlir to py",
    [](ModuleOp module, raw_ostream &os) -> LogicalResult { return ::mlir::asuka::module_to_py_impl(module, os); },
    [](DialectRegistry &registry) { ::mlir::asuka::registerAllDialects(registry); });

} // namespace mlir

int main(int argc, char **argv) { return failed(mlir::mlirTranslateMain(argc, argv, "Asuka translator\n")); }