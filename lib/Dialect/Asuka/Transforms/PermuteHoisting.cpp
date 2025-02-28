#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"
#include "asuka/Dialect/Asuka/Transforms/Passes.h"

#include "dbg.h"

namespace mlir::asuka {

#define GEN_PASS_DEF_ASUKAPERMUTEHOISTING
#include "asuka/Dialect/Asuka/Transforms/Passes.h.inc"

} // namespace mlir::asuka

namespace mlir::asuka {

namespace {

struct PermuteHoistingPattern : public OpRewritePattern<PermuteOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteOp op, PatternRewriter &rewriter) const override {
    auto src = op.getOperand();
    auto dims = op.getDims();
    if (!src.getDefiningOp())
      return failure();
    if (auto def_op = dyn_cast<SoftmaxOp>(src.getDefiningOp())) {
      int64_t reduce_dim = def_op.getReduceDimensionAttr().getValue().getSExtValue();
      if (reduce_dim < 0) {
        reduce_dim += cast<RankedTensorType>(def_op.getOperand().getType()).getRank();
      }
      int64_t new_reduce_dim = -1;
      for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] == reduce_dim) {
          new_reduce_dim = (int64_t)i;
          break;
        }
      }
      auto new_permute_op = rewriter.create<PermuteOp>(op.getLoc(), def_op.getOperand(), dims);
      auto new_softmax_op = rewriter.create<SoftmaxOp>(op.getLoc(), new_permute_op.getResult(),
                                                       rewriter.getI64IntegerAttr(new_reduce_dim));
      rewriter.replaceAllUsesWith(op.getResult(), new_softmax_op.getResult());
      rewriter.eraseOp(op);
      rewriter.eraseOp(def_op);
      return success();
    } else if (auto def_op = dyn_cast<NormalizeOp>(src.getDefiningOp())) {
      int64_t reduce_dim = def_op.getReduceDimensionAttr().getValue().getSExtValue();
      if (reduce_dim < 0) {
        reduce_dim += cast<RankedTensorType>(def_op.getOperand().getType()).getRank();
      }
      auto lp_attr = def_op.getLpAttr();
      int64_t new_reduce_dim = -1;
      for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] == reduce_dim) {
          new_reduce_dim = (int64_t)i;
          break;
        }
      }
      auto new_permute_op = rewriter.create<PermuteOp>(op.getLoc(), def_op.getOperand(), dims);
      auto new_normalize_op = rewriter.create<NormalizeOp>(op.getLoc(), new_permute_op.getResult(),
                                                           rewriter.getI64IntegerAttr(new_reduce_dim), lp_attr);
      rewriter.replaceAllUsesWith(op.getResult(), new_normalize_op.getResult());
      rewriter.eraseOp(op);
      rewriter.eraseOp(def_op);
      return success();

    } else if (auto def_op = dyn_cast<PermuteOp>(src.getDefiningOp())) {
      auto def_dims = def_op.getDims();
      llvm::SmallVector<int64_t, 4> new_dims(def_dims.size());
      for (size_t i = 0; i < new_dims.size(); ++i) {
        new_dims[dims[i]] = def_dims[i];
      }
      auto new_permute_op = rewriter.create<PermuteOp>(def_op.getLoc(), def_op.getOperand(), new_dims);
      rewriter.replaceAllUsesWith(op.getResult(), new_permute_op.getResult());
      rewriter.eraseOp(op);
      rewriter.eraseOp(def_op);
      return success();
    }
    return failure();
  }
};

class PermuteHoistingPass : public ::mlir::asuka::impl::AsukaPermuteHoistingBase<PermuteHoistingPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp m = getOperation();

    patterns.add<PermuteHoistingPattern>(context);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createPermuteHoistingPass() { return std::make_unique<PermuteHoistingPass>(); }

} // namespace mlir::asuka