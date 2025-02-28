#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"
#include "asuka/Dialect/Asuka/Transforms/Passes.h"

#include "dbg.h"

namespace mlir::asuka {

#define GEN_PASS_DEF_ASUKAREPLACEEXPANDLOG
#include "asuka/Dialect/Asuka/Transforms/Passes.h.inc"

} // namespace mlir::asuka

namespace mlir::asuka {

namespace {

constexpr double log2e = 1.44269504088896340736f;

arith::ConstantOp create_log2e_constant_op(PatternRewriter &rewriter, Type elem_type, Location loc) {
  assert(isa<FloatType>(elem_type));
  SmallVector<int64_t> shape(1, 1);
  auto tensor_type = RankedTensorType::get(shape, elem_type);
  SmallVector<Attribute> vals = {rewriter.getFloatAttr(elem_type, log2e)};
  auto data = DenseElementsAttr::get(tensor_type, vals);
  return rewriter.create<arith::ConstantOp>(loc, data);
}

struct ReplaceExpWithExp2Pattern : public OpRewritePattern<ExpOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExpOp op, PatternRewriter &rewriter) const override {
    auto operand_elem_type = cast<RankedTensorType>(op.getOperand().getType()).getElementType();
    auto log2e_const_op = create_log2e_constant_op(rewriter, operand_elem_type, op.getLoc());
    auto mul_op = rewriter.create<MulOp>(op.getLoc(), op.getOperand(), log2e_const_op.getResult());
    rewriter.replaceOpWithNewOp<Exp2Op>(op, mul_op.getResult());
    return success();
  }
};

class ReplaceExpAndLogPass : public ::mlir::asuka::impl::AsukaReplaceExpAndLogBase<ReplaceExpAndLogPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    KernelOp k = getOperation();

    patterns.add<ReplaceExpWithExp2Pattern>(context);
    if (applyPatternsAndFoldGreedily(k, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createReplaceExpAndLogPass() { return std::make_unique<ReplaceExpAndLogPass>(); }

} // namespace mlir::asuka