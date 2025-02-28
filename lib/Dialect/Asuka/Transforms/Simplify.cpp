#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"
#include "asuka/Dialect/Asuka/Transforms/Passes.h"

#include "dbg.h"

namespace mlir::asuka {

#define GEN_PASS_DEF_ASUKASIMPLIFY
#include "asuka/Dialect/Asuka/Transforms/Passes.h.inc"

} // namespace mlir::asuka

namespace mlir::asuka {

namespace {

// convert(from i8 to i64) -> reduce(add) -> cmp > 0
struct ToReduceAnyPattern : public OpRewritePattern<ReduceOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(ReduceOp op, mlir::PatternRewriter &rewriter) const override {
    auto arg = op.getOperand();
    auto res = op.getResult();
    int64_t dim = op.getReduceDimensionAttr().getInt();
    auto reduce_type = op.getReduceType();
    bool keep_dim = op.getKeepDim();

    if (reduce_type != ReduceType::ADD) {
      return failure();
    }
    if (!cast<RankedTensorType>(arg.getType()).getElementType().isInteger(64)) {
      return failure();
    }
    auto def_op = arg.getDefiningOp();
    if (def_op == nullptr || !isa<ConvertOp>(def_op)) {
      return failure();
    }
    auto convert_op = cast<ConvertOp>(def_op);
    auto convert_arg_type = convert_op.getOperand().getType();
    if (!isa<RankedTensorType>(convert_arg_type)) {
      return failure();
    }
    if (!cast<RankedTensorType>(convert_arg_type).getElementType().isInteger(8)) {
      return failure();
    }
    if (!convert_op.getResult().hasOneUse()) {
      return failure();
    }
    if (!res.hasOneUse()) {
      return failure();
    }
    auto use_op = *res.getUsers().begin();
    if (!isa<CmpOp>(use_op)) {
      return failure();
    }
    auto cmp_op = cast<CmpOp>(use_op);
    auto cmp_rhs = cmp_op.getRhs();
    auto cmp_rhs_def_op = cmp_rhs.getDefiningOp();
    if (cmp_rhs_def_op == nullptr || !isa<arith::ConstantOp>(cmp_rhs_def_op)) {
      return failure();
    }
    auto cmp_rhs_const_op = cast<arith::ConstantOp>(cmp_rhs_def_op);
    auto cmp_rhs_const_val = cmp_rhs_const_op.getValue();
    if (!isa<DenseElementsAttr>(cmp_rhs_const_val)) {
      return failure();
    }
    auto dense_attr = cast<DenseElementsAttr>(cmp_rhs_const_val);
    auto elem_type = dense_attr.getType().getElementType();
    for (auto e : dense_attr.getValues<Attribute>()) {
      if (!isa<IntegerAttr>(e)) {
        return failure();
      }
      auto int_attr = cast<IntegerAttr>(e);
      if (int_attr.getInt() != 0) {
        return failure();
      }
    }

    // matched
    auto new_reduce_op =
        rewriter.create<ReduceOp>(op->getLoc(), convert_op.getOperand(), dim, ReduceType::ANY, keep_dim);
    rewriter.replaceAllUsesWith(cmp_op, new_reduce_op.getResult());
    rewriter.eraseOp(cmp_op);
    rewriter.eraseOp(op);
    rewriter.eraseOp(convert_op);
    return success();
  }
};

struct RemoveNoEffectPermutePattern : public OpRewritePattern<PermuteOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(PermuteOp op, mlir::PatternRewriter &rewriter) const override {
    auto dims = op.getDims();
    bool order = true;
    for (int i = 0; i < dims.size(); i++) {
      if (dims[i] != i) {
        order = false;
        break;
      }
    }
    if (order) {
      rewriter.replaceOp(op, op.getOperand());
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct RemoveNoEffectReshapePattern : public OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(ReshapeOp op, mlir::PatternRewriter &rewriter) const override {
    auto input = op.getOperand();
    auto result = op.getResult();
    if (input.getType() == result.getType()) {
      rewriter.replaceOp(op, input);
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct RemoveNoEffectConvertPattern : public OpRewritePattern<ConvertOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(ConvertOp op, mlir::PatternRewriter &rewriter) const override {
    auto input = op.getOperand();
    auto result = op.getResult();
    if (input.getType() == result.getType()) {
      rewriter.replaceOp(op, input);
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct MulAfterTriluPattern : public OpRewritePattern<TriluOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  bool isAllNonZeroConstant(arith::ConstantOp op) const {
    auto val_attr = op.getValue();
    if (auto float_attr = dyn_cast<FloatAttr>(val_attr)) {
      return float_attr.getValueAsDouble() != 0.0;
    }
    if (auto int_attr = dyn_cast<IntegerAttr>(val_attr)) {
      return int_attr.getInt() != 0;
    }

    // dense
    if (auto dense_attr = dyn_cast<DenseElementsAttr>(val_attr)) {
      bool all_non_zero = true;
      for (auto e : dense_attr.getValues<Attribute>()) {
        assert(isa<FloatAttr>(e));
        auto _e = cast<FloatAttr>(e).getValueAsDouble();
        // FIXME: how to compare safely
        if (_e == 0.0) {
          all_non_zero = false;
        }
      }
      return all_non_zero;
    } else {
      op->dump();
      llvm_unreachable("not supported");
    }
  }

  mlir::LogicalResult matchAndRewrite(TriluOp trilu_op, PatternRewriter &rewriter) const override {
    auto res = trilu_op.getResult();
    for (auto &use : res.getUses()) {
      auto user = use.getOwner();
      if (auto mul_op = dyn_cast<MulOp>(user)) {
        auto use_op_idx = use.getOperandNumber();
        assert(use_op_idx == 0 || use_op_idx == 1);
        unsigned other_idx = use_op_idx == 0 ? 1 : 0;

        auto other_operand = mul_op->getOperand(other_idx);
        if (auto constant_op = dyn_cast<arith::ConstantOp>(other_operand.getDefiningOp())) {
          if (isAllNonZeroConstant(constant_op)) {
            // matched
            rewriter.replaceAllUsesWith(mul_op.getResult(), mul_op->getOperand(use_op_idx));
            rewriter.eraseOp(mul_op);
            return success();
          }
        }
      }
    }
    return failure();
  }
};

// TODO: this pattern should be applyed on all elementwise op, but now we dont have such interface
struct ConsecutiveElementWiseComputationWithConstant : public OpRewritePattern<DivOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  bool hasSameShape(DenseElementsAttr &a, DenseElementsAttr &b) const {
    return a.getType().getShape() == b.getType().getShape();
  }

  mlir::LogicalResult matchAndRewrite(DivOp op, mlir::PatternRewriter &rewriter) const override {
    auto rhs = op.getRhs();
    auto res = op.getResult();
    if (rhs.getDefiningOp() == nullptr)
      return failure();

    if (auto rhs_def_const_op = dyn_cast<arith::ConstantOp>(rhs.getDefiningOp())) {
      assert(isa<DenseElementsAttr>(rhs_def_const_op.getValue()));
      auto rhs_def_const_op_dense_attr = cast<DenseElementsAttr>(rhs_def_const_op.getValue());
      if (res.hasOneUse()) {
        if (auto mul_op = dyn_cast<MulOp>(*res.getUsers().begin())) {
          auto mul_rhs = mul_op.getRhs();
          if (mul_rhs.getDefiningOp() == nullptr)
            return failure();

          if (auto mul_rhs_def_const_op = dyn_cast<arith::ConstantOp>(mul_rhs.getDefiningOp())) {
            // matched
            assert(isa<DenseElementsAttr>(mul_rhs_def_const_op.getValue()));
            auto mul_rhs_def_const_op_dense_attr = cast<DenseElementsAttr>(mul_rhs_def_const_op.getValue());
            // TODO: we can stil optimize it even they dont have same shape
            if (hasSameShape(rhs_def_const_op_dense_attr, mul_rhs_def_const_op_dense_attr)) {
              auto elem_type = rhs_def_const_op_dense_attr.getElementType();
              assert(elem_type == mul_rhs_def_const_op_dense_attr.getElementType());
              auto shape = rhs_def_const_op_dense_attr.getType().getShape();
              SmallVector<Attribute> new_vals;
              for (auto [rhs_v, mul_rhs_v] : llvm::zip(rhs_def_const_op_dense_attr.getValues<FloatAttr>(),
                                                       mul_rhs_def_const_op_dense_attr.getValues<FloatAttr>())) {
                double _rhs_v = rhs_v.getValueAsDouble();
                double _mul_rhs_v = mul_rhs_v.getValueAsDouble();
                new_vals.push_back(rewriter.getFloatAttr(rewriter.getF64Type(), _mul_rhs_v / _rhs_v));
              }
              auto new_dense_attr = DenseElementsAttr::get(RankedTensorType::get(shape, elem_type), new_vals);
              auto new_op = rewriter.create<arith::ConstantOp>(op->getLoc(), new_dense_attr);

              auto fused_op = rewriter.create<MulOp>(op->getLoc(), op.getLhs(), new_op.getResult());
              rewriter.replaceAllUsesWith(mul_op.getResult(), fused_op.getResult());
              rewriter.eraseOp(mul_op);
              rewriter.eraseOp(op);
              return success();
            } else {
              llvm_unreachable("not supported");
            }
          }
        }
      }
    }
    return mlir::failure();
  }
};

class SimplifyPass : public ::mlir::asuka::impl::AsukaSimplifyBase<SimplifyPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    FunctionOpInterface op = getOperation();

    patterns.add<ToReduceAnyPattern>(context);
    patterns.add<RemoveNoEffectPermutePattern>(context);
    patterns.add<RemoveNoEffectReshapePattern>(context);
    patterns.add<RemoveNoEffectConvertPattern>(context);
    patterns.add<MulAfterTriluPattern>(context);
    patterns.add<ConsecutiveElementWiseComputationWithConstant>(context);
    if (applyPatternsAndFoldGreedily(op, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSimplifyPass() { return std::make_unique<SimplifyPass>(); }

} // namespace mlir::asuka