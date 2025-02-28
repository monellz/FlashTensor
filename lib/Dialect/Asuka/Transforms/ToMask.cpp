#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"
#include "asuka/Dialect/Asuka/Transforms/Passes.h"

#include "dbg.h"

namespace mlir::asuka {

#define GEN_PASS_DEF_ASUKATOMASK
#include "asuka/Dialect/Asuka/Transforms/Passes.h.inc"

} // namespace mlir::asuka

namespace mlir::asuka {

namespace {

struct TriluToMaskPattern : public OpRewritePattern<TriluOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(TriluOp op, mlir::PatternRewriter &rewriter) const override {
    int64_t diagonal = op.getDiagonalAttr().getInt();
    bool is_upper = op.getIsUpper();
    assert(is_upper);
    auto shape = op.getShape();
    auto val = op.getVal();
    assert(shape.size() == 2);

    auto const_zero_op = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(0));
    SmallVector<Value> starts;
    starts.push_back(const_zero_op.getResult());
    starts.push_back(const_zero_op.getResult());

    auto mask_op = rewriter.create<MaskOp>(op.getLoc(), starts, shape, val.getType());
    Block *entry = rewriter.createBlock(&mask_op.getRegion());
    for (int i = 0; i < 2; ++i) {
      entry->addArgument(rewriter.getIndexType(), op.getLoc());
    }
    rewriter.setInsertionPointToStart(entry);

    Value cmp_lhs = nullptr;
    if (diagonal > 0) {
      auto i_plus_diagonal_op = rewriter.create<::mlir::arith::AddIOp>(
          op.getLoc(), entry->getArgument(0),
          rewriter.create<::mlir::arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(diagonal)));
      cmp_lhs = i_plus_diagonal_op.getResult();
    } else {
      auto i_sub_diagonal_op = rewriter.create<::mlir::arith::SubIOp>(
          op.getLoc(), entry->getArgument(0),
          rewriter.create<::mlir::arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(-diagonal)));
      cmp_lhs = i_sub_diagonal_op.getResult();
    }

    auto cmp_op = rewriter.create<::mlir::arith::CmpIOp>(op.getLoc(), ::mlir::arith::CmpIPredicate::ule, cmp_lhs,
                                                         entry->getArgument(1));
    auto if_op = rewriter.create<::mlir::scf::IfOp>(op.getLoc(), val.getType(), cmp_op.getResult(), true);
    Block *then_block = &if_op.getThenRegion().front();
    Block *else_block = &if_op.getElseRegion().front();
    rewriter.setInsertionPointToStart(then_block);
    auto val_constant_op = rewriter.create<::mlir::arith::ConstantOp>(op.getLoc(), val);
    rewriter.create<::mlir::scf::YieldOp>(op.getLoc(), val_constant_op.getResult());
    rewriter.setInsertionPointToStart(else_block);
    auto zero_constant_op =
        rewriter.create<::mlir::arith::ConstantOp>(op.getLoc(), rewriter.getZeroAttr(val.getType()));
    rewriter.create<::mlir::scf::YieldOp>(op.getLoc(), zero_constant_op.getResult());

    rewriter.setInsertionPointAfter(if_op);
    rewriter.create<::mlir::asuka::MaskYieldOp>(op.getLoc(), if_op->getResults());

    rewriter.replaceOp(op, mask_op.getResult());
    return mlir::success();
  }
};

class ToMaskPass : public ::mlir::asuka::impl::AsukaToMaskBase<ToMaskPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    KernelOp k = getOperation();

    patterns.add<TriluToMaskPattern>(context);
    if (applyPatternsAndFoldGreedily(k, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createToMaskPass() { return std::make_unique<ToMaskPass>(); }

} // namespace mlir::asuka