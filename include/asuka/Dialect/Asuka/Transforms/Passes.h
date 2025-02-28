#ifndef ASUKA_TRANSFORMS_PASSES_H_
#define ASUKA_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir::asuka {

// FIXME: remove all manual constructors in tablegen and use the default one?
#define GEN_PASS_DECL
#include "asuka/Dialect/Asuka/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createSimplifyPass();
std::unique_ptr<Pass> createReplaceExpAndLogPass();

std::unique_ptr<Pass> createLowerComplexReducePass();

std::unique_ptr<Pass> createEraseTypeInKernelPass();

std::unique_ptr<Pass> createMulScalarHoistingPass();
std::unique_ptr<Pass> createBroadcastTransformPass();

std::unique_ptr<Pass> createEquivalentTransformPass();

std::unique_ptr<Pass> createPermuteHoistingPass();

std::unique_ptr<Pass> createAnnotateParallelismPass();
std::unique_ptr<Pass> createAnnotateParallelismPass(const AsukaAnnotateParallelismOptions &);

std::unique_ptr<Pass> createToMaskPass();

std::unique_ptr<Pass> createParallelizePass();
std::unique_ptr<Pass> createParallelizePass(const AsukaParallelizeOptions &);

std::unique_ptr<Pass> createTilingPass();

std::unique_ptr<Pass> createDynamicForPass();

std::unique_ptr<Pass> createRecoverTypeInKernelPass();

std::unique_ptr<Pass> createUserReplicatePass();

#define GEN_PASS_REGISTRATION
#include "asuka/Dialect/Asuka/Transforms/Passes.h.inc"

} // namespace mlir::asuka

#endif