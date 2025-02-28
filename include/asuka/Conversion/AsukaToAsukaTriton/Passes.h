#ifndef ASUKA_CONVERSION_ASUKATOASUKATRITON_ASUKATOASUKATRITON_PASS_H
#define ASUKA_CONVERSION_ASUKATOASUKATRITON_ASUKATOASUKATRITON_PASS_H

#include "mlir/Pass/Pass.h"

namespace mlir::asuka {

std::unique_ptr<mlir::Pass> createConvertAsukaToAsukaTritonPass();

#define GEN_PASS_REGISTRATION
#include "asuka/Conversion/AsukaToAsukaTriton/Passes.h.inc"

} // namespace mlir::asuka

#endif