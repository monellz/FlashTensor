#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"
#include "asuka/Dialect/Asuka/Transforms/Passes.h"

#include "dbg.h"

namespace mlir::asuka {

#define GEN_PASS_DEF_ASUKAMULSCALARHOISTING
#include "asuka/Dialect/Asuka/Transforms/Passes.h.inc"

} // namespace mlir::asuka

namespace mlir::asuka {

namespace {

struct MulScalarHoistingPattern : public OpRewritePattern<MulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MulOp mul_op, PatternRewriter &rewriter) const override {
    auto mul_rhs = mul_op.getRhs();
    auto mul_rhs_type = dyn_cast<RankedTensorType>(mul_rhs.getType());
    if (mul_rhs_type.getRank() > 1) {
      return failure();
    }
    auto mul_lhs = mul_op.getLhs();
    if (mul_lhs.getDefiningOp() == nullptr) {
      return failure();
    }

    if (auto add_op = dyn_cast<AddOp>(mul_lhs.getDefiningOp())) {
      auto add_lhs = add_op.getLhs();
      auto add_rhs = add_op.getRhs();

      auto new_mul_op_0 = rewriter.create<MulOp>(mul_op.getLoc(), add_lhs, mul_rhs);
      auto new_mul_op_1 = rewriter.create<MulOp>(mul_op.getLoc(), add_rhs, mul_rhs);
      auto new_add_op = rewriter.create<AddOp>(add_op.getLoc(), new_mul_op_0.getResult(), new_mul_op_1.getResult());

      rewriter.replaceAllUsesWith(mul_op.getResult(), new_add_op.getResult());
      rewriter.eraseOp(mul_op);
      // DONT erase add_op or it will crash (mul_op cannot find its defining op)
      return success();
    }
    return failure();
  }
};

class MulScalarHoistingPass : public ::mlir::asuka::impl::AsukaMulScalarHoistingBase<MulScalarHoistingPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp m = getOperation();

    patterns.add<MulScalarHoistingPattern>(context);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createMulScalarHoistingPass() { return std::make_unique<MulScalarHoistingPass>(); }

} // namespace mlir::asuka