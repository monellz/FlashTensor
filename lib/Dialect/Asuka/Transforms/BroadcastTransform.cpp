#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"
#include "asuka/Dialect/Asuka/Transforms/Passes.h"

#include "dbg.h"

namespace mlir::asuka {

#define GEN_PASS_DEF_ASUKABROADCASTTRANSFORM
#include "asuka/Dialect/Asuka/Transforms/Passes.h.inc"

} // namespace mlir::asuka

namespace mlir::asuka {

namespace {

bool hasLessAccess(ArrayRef<Operation *> previous_ops, ArrayRef<Operation *> later_ops) {
  int64_t previous_access = 0;
  int64_t later_access = 0;
  for (auto op : previous_ops) {
    assert(isa<TotalAccessedElementsInterface>(op));
    auto _op = cast<TotalAccessedElementsInterface>(op);
    previous_access += _op.totalAccessedElements();
  }
  for (auto op : later_ops) {
    assert(isa<TotalAccessedElementsInterface>(op));
    auto _op = cast<TotalAccessedElementsInterface>(op);
    later_access += _op.totalAccessedElements();
  }
  return later_access < previous_access;
}

struct DotDivPattern : public OpRewritePattern<DivOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DivOp div_op, PatternRewriter &rewriter) const override {
    auto div_lhs = div_op.getLhs();
    auto div_rhs = div_op.getRhs();

    if (div_lhs.getDefiningOp() == nullptr) {
      return failure();
    }

    if (auto dot_op = dyn_cast<DotOp>(div_lhs.getDefiningOp())) {
      if (!div_lhs.hasOneUse())
        return failure();
      // check if reorder is legal
      auto div_rhs_broadcasted_shape = cast<BroadcastableBinaryOpInterface>(&*div_op).getRhsBroadcastedShape();
      if (div_rhs_broadcasted_shape[div_rhs_broadcasted_shape.size() - 1] == 1) {
        // valid
        auto dot_lhs = dot_op.getLhs();
        auto dot_rhs = dot_op.getRhs();

        auto new_div_op = rewriter.create<DivOp>(div_op.getLoc(), dot_lhs, div_rhs);
        auto new_dot_op = rewriter.create<DotOp>(dot_op.getLoc(), new_div_op.getResult(), dot_rhs);
        if (hasLessAccess({&*dot_op, &*div_op}, {&*new_div_op, &*new_dot_op})) {
          dbg("dot, div -> div, dot");
          rewriter.replaceAllUsesWith(div_op.getResult(), new_dot_op.getResult());
          rewriter.eraseOp(div_op);
          rewriter.eraseOp(dot_op);
          return success();
        } else {
          rewriter.eraseOp(new_dot_op);
          rewriter.eraseOp(new_div_op);
        }
      }
    }
    return failure();
  }
};

struct DivDotPattern : public OpRewritePattern<DivOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DivOp div_op, PatternRewriter &rewriter) const override {
    auto div_lhs = div_op.getLhs();
    auto div_rhs = div_op.getRhs();
    auto div_res = div_op.getResult();
    // TODO: we can replicate uses before doing it
    if (div_res.hasOneUse()) {
      if (auto dot_op = dyn_cast<DotOp>(*div_res.getUsers().begin())) {
        // check if reorder is legal
        auto div_rhs_broadcasted_shape = cast<BroadcastableBinaryOpInterface>(&*div_op).getRhsBroadcastedShape();
        if (dot_op.getLhs() == div_res && div_rhs_broadcasted_shape[div_rhs_broadcasted_shape.size() - 1] == 1) {
          // valid
          auto dot_rhs = dot_op.getRhs();

          auto new_dot_op = rewriter.create<DotOp>(dot_op.getLoc(), div_lhs, dot_rhs);
          auto new_div_op = rewriter.create<DivOp>(div_op.getLoc(), new_dot_op.getResult(), div_rhs);
          if (hasLessAccess({&*div_op, &*dot_op}, {&*new_dot_op, &*new_div_op})) {
            dbg("div, dot -> dot, div");
            rewriter.replaceAllUsesWith(dot_op.getResult(), new_div_op.getResult());
            rewriter.eraseOp(dot_op);
            rewriter.eraseOp(div_op);
            return success();
          } else {
            rewriter.eraseOp(new_div_op);
            rewriter.eraseOp(new_dot_op);
          }
        }
      }
    }
    return failure();
  }
};

struct DotMulPattern : public OpRewritePattern<MulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MulOp mul_op, PatternRewriter &rewriter) const override {
    auto mul_lhs = mul_op.getLhs();
    auto mul_rhs = mul_op.getRhs();

    if (mul_lhs.getDefiningOp() == nullptr) {
      return failure();
    }

    if (auto dot_op = dyn_cast<DotOp>(mul_lhs.getDefiningOp())) {
      if (!mul_lhs.hasOneUse())
        return failure();
      // check if reorder is legal
      auto mul_rhs_broadcasted_shape = cast<BroadcastableBinaryOpInterface>(&*mul_op).getRhsBroadcastedShape();
      if (mul_rhs_broadcasted_shape[mul_rhs_broadcasted_shape.size() - 1] == 1) {
        // valid
        auto dot_lhs = dot_op.getLhs();
        auto dot_rhs = dot_op.getRhs();

        auto new_mul_op = rewriter.create<MulOp>(mul_op.getLoc(), dot_lhs, mul_rhs);
        auto new_dot_op = rewriter.create<DotOp>(dot_op.getLoc(), new_mul_op.getResult(), dot_rhs);
        if (hasLessAccess({&*dot_op, &*mul_op}, {&*new_mul_op, &*new_dot_op})) {
          dbg("dot, mul -> mul, dot");
          rewriter.replaceAllUsesWith(mul_op.getResult(), new_dot_op.getResult());
          rewriter.eraseOp(mul_op);
          rewriter.eraseOp(dot_op);
          return success();
        } else {
          rewriter.eraseOp(new_dot_op);
          rewriter.eraseOp(new_mul_op);
        }
      }
    }
    return failure();
  }
};

// xy + a + zy -> (x + z)y + a
// TODO

class BroadcastTransformPass : public ::mlir::asuka::impl::AsukaBroadcastTransformBase<BroadcastTransformPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp m = getOperation();

    patterns.add<DotDivPattern>(context);
    patterns.add<DotMulPattern>(context);
    patterns.add<DivDotPattern>(context);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createBroadcastTransformPass() { return std::make_unique<BroadcastTransformPass>(); }

} // namespace mlir::asuka