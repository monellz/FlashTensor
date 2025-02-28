#ifndef ASUKATRITON_TRANSFORMS_PASSES_H_
#define ASUKATRITON_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir::asuka::triton {

std::unique_ptr<Pass> createSqueezeBlockPass();

// std::unique_ptr<Pass> createBlockingPass();
std::unique_ptr<Pass> createUserReplicatePass();

#define GEN_PASS_DECL
#include "asuka/Dialect/AsukaTriton/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "asuka/Dialect/AsukaTriton/Transforms/Passes.h.inc"

} // namespace mlir::asuka::triton

#endif