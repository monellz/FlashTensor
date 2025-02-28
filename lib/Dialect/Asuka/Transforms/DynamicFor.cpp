#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"
#include "asuka/Dialect/Asuka/Transforms/Passes.h"

#include "dbg.h"

namespace mlir::asuka {

#define GEN_PASS_DEF_ASUKADYNAMICFOR
#include "asuka/Dialect/Asuka/Transforms/Passes.h.inc"

} // namespace mlir::asuka

namespace mlir::asuka {

namespace {

struct ToDynamicBlockForPattern : public OpRewritePattern<BlockForOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(BlockForOp op, mlir::PatternRewriter &rewriter) const override {
    int64_t lb = op.getLowerBoundAttr().getInt();
    int64_t ub = op.getUpperBoundAttr().getInt();
    int64_t step = op.getStepAttr().getInt();

    auto lb_const_op = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(lb));
    auto ub_const_op = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(ub));

    SmallVector<Type> res_types(op.getResultTypes());
    SmallVector<Value> block_args(op.getBlockArgs());
    SmallVector<int64_t> block_dims(op.getBlockDims());
    SmallVector<Value> init_args(op.getInitArgs());
    auto dynamic_for_op =
        rewriter.create<DynamicBlockForOp>(op.getLoc(), res_types, lb_const_op.getResult(), ub_const_op.getResult(),
                                           step, block_args, block_dims, init_args);
    dynamic_for_op.getRegion().takeBody(op.getRegion());
    rewriter.replaceOp(op, dynamic_for_op.getResults());
    return mlir::success();
  }
};

// FIXME: move it to an interface
// maybe use a real float/int for inference?
struct ValueType {
  enum Kind { kVar = 1, kZero = 2, kPosConst = 3, kNegConst = 4, kPosInf = 5, kNegInf = 6, kNan = 7 };
  Kind kind;
  ValueType() : kind(Kind::kVar) {}
  ValueType(Kind kind) : kind(kind) {}

  std::string str() {
    std::string ret;
    switch (kind) {
    case kVar:
      ret = "Var";
      break;
    case kZero:
      ret = "Zero";
      break;
    case kPosConst:
      ret = "PosConst";
      break;
    case kNegConst:
      ret = "NegConst";
      break;
    case kPosInf:
      ret = "PosInf";
      break;
    case kNegInf:
      ret = "NegInf";
      break;
    case kNan:
      ret = "Nan";
      break;
    default:
      ret = "???";
      break;
    }
    return ret;
  }

  static ValueType join_add(ValueType lhs, ValueType rhs) {
    // nan + ? = nan
    if (lhs.kind == Kind::kNan || rhs.kind == Kind::kNan)
      return ValueType(Kind::kNan);
    // inf + -inf = nan
    if (lhs.kind == Kind::kPosInf && rhs.kind == Kind::kNegInf)
      return ValueType(Kind::kNan);
    if (lhs.kind == Kind::kNegInf && rhs.kind == Kind::kPosInf)
      return ValueType(Kind::kNan);

    // -inf + ? = -inf
    if (lhs.kind == Kind::kNegInf || rhs.kind == Kind::kNegInf)
      return ValueType(Kind::kNegInf);
    // inf + ? = inf
    if (lhs.kind == Kind::kPosInf || rhs.kind == Kind::kPosInf)
      return ValueType(Kind::kPosInf);

    // 0 + 0 = 0
    if (lhs.kind == Kind::kZero && rhs.kind == Kind::kZero)
      return ValueType(Kind::kZero);

    // 0 + -c = -c
    if (lhs.kind == Kind::kZero && rhs.kind == Kind::kNegConst)
      return ValueType(Kind::kNegConst);
    if (lhs.kind == Kind::kNegConst && rhs.kind == Kind::kZero)
      return ValueType(Kind::kNegConst);

    // 0 + +c = +c
    if (lhs.kind == Kind::kZero && rhs.kind == Kind::kPosConst)
      return ValueType(Kind::kPosConst);
    if (lhs.kind == Kind::kPosConst && rhs.kind == Kind::kZero)
      return ValueType(Kind::kPosConst);

    return ValueType(Kind::kVar);
  }

  static ValueType join_mul(ValueType lhs, ValueType rhs) {
    // nan * ? = nan
    if (lhs.kind == Kind::kNan || rhs.kind == Kind::kNan)
      return ValueType(Kind::kNan);
    // -inf * 0 = nan
    if (lhs.kind == Kind::kNegInf && rhs.kind == Kind::kZero)
      return ValueType(Kind::kNan);
    if (lhs.kind == Kind::kZero && rhs.kind == Kind::kNegInf)
      return ValueType(Kind::kNan);
    // inf * 0 = nan
    if (lhs.kind == Kind::kPosInf && rhs.kind == Kind::kZero)
      return ValueType(Kind::kNan);
    if (lhs.kind == Kind::kZero && rhs.kind == Kind::kPosInf)
      return ValueType(Kind::kNan);

    // inf * ? = nan -inf * ? = nan
    if (lhs.kind == Kind::kPosInf && rhs.kind == Kind::kVar)
      return ValueType(Kind::kNan);
    if (lhs.kind == Kind::kNegInf && rhs.kind == Kind::kVar)
      return ValueType(Kind::kNan);
    if (lhs.kind == Kind::kVar && rhs.kind == Kind::kPosInf)
      return ValueType(Kind::kNan);
    if (lhs.kind == Kind::kVar && rhs.kind == Kind::kNegInf)
      return ValueType(Kind::kNan);

    // -inf * -c = inf
    if (lhs.kind == Kind::kNegInf && rhs.kind == Kind::kNegConst)
      return ValueType(Kind::kPosInf);
    if (lhs.kind == Kind::kNegConst && rhs.kind == Kind::kNegInf)
      return ValueType(Kind::kPosInf);
    // -inf * c = -inf
    if (lhs.kind == Kind::kNegInf && rhs.kind == Kind::kPosConst)
      return ValueType(Kind::kNegInf);
    if (lhs.kind == Kind::kPosConst && rhs.kind == Kind::kNegInf)
      return ValueType(Kind::kNegInf);
    // inf * -c = -inf
    if (lhs.kind == Kind::kPosInf && rhs.kind == Kind::kNegConst)
      return ValueType(Kind::kNegInf);
    if (lhs.kind == Kind::kNegConst && rhs.kind == Kind::kPosInf)
      return ValueType(Kind::kNegInf);
    // inf * c = inf
    if (lhs.kind == Kind::kPosInf && rhs.kind == Kind::kPosConst)
      return ValueType(Kind::kPosInf);
    if (lhs.kind == Kind::kPosConst && rhs.kind == Kind::kPosInf)
      return ValueType(Kind::kPosInf);
    // -inf * -inf = inf
    if (lhs.kind == Kind::kNegInf && rhs.kind == Kind::kNegInf)
      return ValueType(Kind::kPosInf);
    // inf * inf = inf
    if (lhs.kind == Kind::kPosInf && rhs.kind == Kind::kPosInf)
      return ValueType(Kind::kPosInf);

    // c * -c = -c
    if (lhs.kind == Kind::kNegConst && rhs.kind == Kind::kPosConst)
      return ValueType(Kind::kNegConst);
    if (lhs.kind == Kind::kPosConst && rhs.kind == Kind::kNegConst)
      return ValueType(Kind::kNegConst);
    // c * c = c
    if (lhs.kind == Kind::kPosConst && rhs.kind == Kind::kPosConst)
      return ValueType(Kind::kPosConst);
    // -c * -c = c
    if (lhs.kind == Kind::kNegConst && rhs.kind == Kind::kNegConst)
      return ValueType(Kind::kPosConst);

    // 0 * ? = 0
    if (lhs.kind == Kind::kZero || rhs.kind == Kind::kZero)
      return ValueType(Kind::kZero);
    return ValueType(Kind::kVar);
  }

  static ValueType join_exp(ValueType v) {
    // e ^ nan = nan
    if (v.kind == Kind::kNan)
      return ValueType(Kind::kNan);
    // e ^ inf = inf
    if (v.kind == Kind::kPosInf)
      return ValueType(Kind::kPosInf);
    // e ^ -inf = 0
    if (v.kind == Kind::kNegInf)
      return ValueType(Kind::kZero);
    // e ^ -c = +c
    if (v.kind == Kind::kNegConst)
      return ValueType(Kind::kPosConst);
    // e ^ +c = +c
    if (v.kind == Kind::kPosConst)
      return ValueType(Kind::kPosConst);
    // e ^ 0 = +c
    if (v.kind == Kind::kZero)
      return ValueType(Kind::kPosConst);
    return ValueType(Kind::kVar);
  }

  static ValueType join_pow(ValueType base, ValueType exp) {
    // nan ^ ? = nan
    if (base.kind == Kind::kNan || exp.kind == Kind::kNan)
      return ValueType(Kind::kNan);
    // 0 ^ inf = nan
    if (base.kind == Kind::kZero && exp.kind == Kind::kPosInf)
      return ValueType(Kind::kNan);
    // 0 ^ -inf = nan
    if (base.kind == Kind::kZero && exp.kind == Kind::kNegInf)
      return ValueType(Kind::kNan);
    // inf ^ 0 = nan
    if (base.kind == Kind::kPosInf && exp.kind == Kind::kZero)
      return ValueType(Kind::kNan);
    // -inf ^ 0 = nan
    if (base.kind == Kind::kNegInf && exp.kind == Kind::kZero)
      return ValueType(Kind::kNan);
    // 0 ^ 0 = nan
    if (base.kind == Kind::kZero && exp.kind == Kind::kZero)
      return ValueType(Kind::kNan);
    // -inf ^ -inf = nan
    if (base.kind == Kind::kNegInf && exp.kind == Kind::kNegInf)
      return ValueType(Kind::kNan);
    // -inf ^ inf = nan
    if (base.kind == Kind::kNegInf && exp.kind == Kind::kPosInf)
      return ValueType(Kind::kNan);

    // inf ^ inf = inf
    if (base.kind == Kind::kPosInf && exp.kind == Kind::kPosInf)
      return ValueType(Kind::kPosInf);
    // inf ^ +c = inf
    if (base.kind == Kind::kPosInf && exp.kind == Kind::kPosConst)
      return ValueType(Kind::kPosInf);
    // inf ^ -inf = 0
    if (base.kind == Kind::kPosInf && exp.kind == Kind::kNegInf)
      return ValueType(Kind::kZero);

    // 0 ^ -c = 0
    if (base.kind == Kind::kZero && exp.kind == Kind::kNegConst)
      return ValueType(Kind::kZero);
    // 0 ^ c = 0
    if (base.kind == Kind::kZero && exp.kind == Kind::kPosConst)
      return ValueType(Kind::kZero);

    // FIXME: more cases
    return ValueType(Kind::kVar);
  }

  static ValueType join_log(ValueType v) {
    // log(nan) = nan
    if (v.kind == Kind::kNan)
      return ValueType(Kind::kNan);
    // log(0) = -inf
    if (v.kind == Kind::kZero)
      return ValueType(Kind::kNegInf);
    // log(-inf) = nan
    if (v.kind == Kind::kNegInf)
      return ValueType(Kind::kNan);
    return ValueType(Kind::kVar);
  }

  static ValueType join_neg(ValueType v) {
    // -nan = nan
    if (v.kind == Kind::kNan)
      return ValueType(Kind::kNan);
    // -inf = inf
    if (v.kind == Kind::kNegInf)
      return ValueType(Kind::kPosInf);
    // -0 = 0
    if (v.kind == Kind::kZero)
      return ValueType(Kind::kZero);
    // -c = c
    if (v.kind == Kind::kPosConst)
      return ValueType(Kind::kNegConst);
    // c = -c
    if (v.kind == Kind::kNegConst)
      return ValueType(Kind::kPosConst);
    return ValueType(Kind::kVar);
  }

  static ValueType join_cmp_ge(ValueType lhs, ValueType rhs) {
    // 0 >= +c = 0
    if (lhs.kind == Kind::kZero && rhs.kind == Kind::kPosConst)
      return ValueType(Kind::kZero);
    return ValueType(Kind::kVar);
  }
};

struct EarlyStopByMaskPattern : public OpRewritePattern<MaskOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  // FIXME: this is a temporary implementation
  bool isValidOp(MaskOp mask_op) const {
    auto region = &mask_op.getRegion();
    SmallVector<Operation *> ops;
    region->front().walk([&](Operation *op) { ops.push_back(op); });
    if (ops.size() != 6)
      return false;
    if (!isa<arith::AddIOp>(ops[0]))
      return false;
    auto add_op = cast<arith::AddIOp>(ops[0]);
    if (add_op.getLhs() != mask_op.getIterArgInEntry(0))
      return false;
    if (!isa<arith::CmpIOp>(ops[1]))
      return false;
    auto cmpi_op = cast<arith::CmpIOp>(ops[1]);
    if (cmpi_op.getRhs() != mask_op.getIterArgInEntry(1))
      return false;
    auto cmpi_pred = cmpi_op.getPredicate();
    if (cmpi_pred != arith::CmpIPredicate::ule)
      return false;
    if (!isa<scf::YieldOp>(ops[2]))
      return false;
    if (!isa<scf::YieldOp>(ops[3]))
      return false;
    if (!isa<scf::IfOp>(ops[4]))
      return false;
    if (!isa<MaskYieldOp>(ops[5]))
      return false;
    return true;
  }

  mlir::LogicalResult matchAndRewrite(MaskOp mask_op, mlir::PatternRewriter &rewriter) const override {
    auto for_op = mask_op->getParentOfType<DynamicBlockForOp>();
    if (for_op == nullptr)
      return failure();
    auto for_iter = for_op.getIterArgInEntry();
    int block_size = for_op.getStepAttr().getInt();
    int yield_res_num = cast<BlockYieldOp>(for_op.getRegion().front().getTerminator()).getNumBlocks();
    // we cannot early stop if there are BLOCKING results
    if (yield_res_num > 0)
      return failure();
    auto lb_val = for_op.getLowerBound();
    auto ub_val = for_op.getUpperBound();
    auto lb_const_op = lb_val.getDefiningOp<arith::ConstantOp>();
    auto ub_const_op = ub_val.getDefiningOp<arith::ConstantOp>();
    if (lb_const_op == nullptr || ub_const_op == nullptr)
      return failure();

    SmallVector<Value> starts(mask_op.getStarts());
    if (starts.size() != 2)
      return failure();
    if (starts[0].getDefiningOp() != nullptr || starts[1].getDefiningOp() != nullptr)
      return failure();
    if (starts[0] != for_iter && starts[1] != for_iter)
      return failure();

    if (!isValidOp(mask_op))
      return failure();

    // dbg("early stop");
    // for_op->dump();
    // dbg(for_op.getNumBlockArgs());
    // dbg(for_op.getNumInitArgs());
    // analyze ops in for_op
    DenseMap<Value, ValueType> type_map;
    type_map[for_iter] = ValueType::kVar;
    for (int i = 0; i < for_op.getNumBlockArgs(); i++) {
      type_map[for_op.getBlockArgInEntry(i)] = ValueType::kVar;
    }
    for (int i = 0; i < for_op.getNumInitArgs(); i++) {
      auto init = for_op.getInitArg(i);
      auto def_op = init.getDefiningOp();
      // if (def_op == nullptr) {
      //   llvm::errs() << "i: " << i << ", def_op: nullptr"
      //                << "\n";
      // } else {
      //   llvm::errs() << "i: " << i << ", def_op: " << *def_op << "\n";
      // }
      auto zero_op = init.getDefiningOp<ZeroOp>();
      if (zero_op == nullptr) {
        type_map[for_op.getInitArgInEntry(i)] = ValueType::kVar;
      } else {
        type_map[for_op.getInitArgInEntry(i)] = ValueType::kZero;
      }
    }

    auto find_type = [&](Value val) -> ValueType {
      if (type_map.find(val) != type_map.end()) {
        return type_map[val];
      }
      auto def_op = val.getDefiningOp();
      // llvm::errs() << "def: " << *def_op << "\n";
      auto const_op = val.getDefiningOp<arith::ConstantOp>();
      if (const_op == nullptr) {
        return ValueType(ValueType::kVar);
      }
      auto const_v = const_op.getValue();
      bool zero = true;
      bool pos_const = true;
      bool neg_const = true;
      bool pos_inf = true;
      bool neg_inf = true;
      auto update_apint_type = [&](APInt v) {
        pos_inf = false;
        neg_inf = false;
        if (v.isZero()) {
          pos_const = false;
          neg_const = false;
        } else if (v.isNegative()) {
          zero = false;
          pos_const = false;
        } else {
          zero = false;
          neg_const = false;
        }
      };
      auto update_apfloat_type = [&](APFloat v) {
        if (v.isZero()) {
          pos_const = false;
          neg_const = false;
          pos_inf = false;
          neg_inf = false;
        } else if (v.isPosInfinity()) {
          zero = false;
          pos_const = false;
          neg_const = false;
          neg_inf = false;
        } else if (v.isNegInfinity()) {
          zero = false;
          pos_const = false;
          neg_const = false;
          pos_inf = false;
        } else {
          assert(!v.isInfinity());
          zero = false;
          pos_inf = false;
          neg_inf = false;
          if (v.isNegative()) {
            pos_const = false;
          } else {
            neg_const = false;
          }
        }
      };
      if (isa<DenseElementsAttr>(const_v)) {
        auto dense_attr = cast<DenseElementsAttr>(const_v);
        auto elem_type = dense_attr.getElementType();
        if (isa<IntegerType>(elem_type)) {
          pos_inf = false;
          neg_inf = false;
          for (auto v : dense_attr.getValues<APInt>()) {
            update_apint_type(v);
          }
        } else if (isa<FloatType>(elem_type)) {
          for (auto v : dense_attr.getValues<APFloat>()) {
            update_apfloat_type(v);
          }
        } else {
          llvm::errs() << "const val: " << const_v << "\n";
          llvm_unreachable("not supported");
        }
      } else if (isa<IntegerAttr>(const_v)) {
        auto v = cast<IntegerAttr>(const_v).getValue();
        update_apint_type(v);
      } else if (isa<FloatAttr>(const_v)) {
        auto v = cast<FloatAttr>(const_v).getValue();
        update_apfloat_type(v);
      } else {
        llvm::errs() << "const val: " << const_v << "\n";
        llvm_unreachable("not supported");
      }

      ValueType res;
      if (zero)
        res.kind = ValueType::kZero;
      if (pos_const)
        res.kind = ValueType::kPosConst;
      if (neg_const)
        res.kind = ValueType::kNegConst;
      if (pos_inf)
        res.kind = ValueType::kPosInf;
      if (neg_inf)
        res.kind = ValueType::kNegInf;

      // llvm::errs() << "res type: " << res.str() << "\n";
      return res;
    };

    for_op.getRegion().front().walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto _mask_op = dyn_cast<MaskOp>(op)) {
        if (_mask_op == mask_op) {
          type_map[_mask_op.getResult()] = ValueType::kNegInf;
        } else {
          type_map[_mask_op.getResult()] = ValueType::kVar;
        }
        return WalkResult::skip();
      } else if (auto _add_op = dyn_cast<AddOp>(op)) {
        auto lhs = find_type(_add_op.getLhs());
        auto rhs = find_type(_add_op.getRhs());
        auto res = ValueType::join_add(lhs, rhs);
        type_map[_add_op.getResult()] = res;
      } else if (isa<MulOp, DivOp, DotOp>(op)) {
        auto lhs = find_type(op->getOperand(0));
        auto rhs = find_type(op->getOperand(1));
        auto res = ValueType::join_mul(lhs, rhs);
        type_map[op->getResult(0)] = res;
      } else if (isa<PowOp>(op)) {
        auto base = find_type(op->getOperand(0));
        auto exp = find_type(op->getOperand(1));
        auto res = ValueType::join_pow(base, exp);
        type_map[op->getResult(0)] = res;
      } else if (isa<ExpOp, Exp2Op>(op)) {
        auto v = find_type(op->getOperand(0));
        auto res = ValueType::join_exp(v);
        type_map[op->getResult(0)] = res;
      } else if (isa<LogOp, Log2Op>(op)) {
        auto v = find_type(op->getOperand(0));
        auto res = ValueType::join_log(v);
        type_map[op->getResult(0)] = res;
      } else if (isa<NegOp>(op)) {
        auto v = find_type(op->getOperand(0));
        auto res = ValueType::join_neg(v);
        type_map[op->getResult(0)] = res;
      } else if (auto _reduce_op = dyn_cast<ReduceOp>(op)) {
        // auto init = _reduce_op.getInit();
        auto reduce_type = _reduce_op.getReduceType();
        // llvm::errs() << "op: " << *op << "\n";
        // llvm::errs() << "\t arg:" << type_map[_reduce_op.getOperand()].str() << "\n";
        // llvm::errs() << "\t init:" << type_map[_reduce_op.getInit()].str() << "\n";
        auto v = find_type(_reduce_op.getOperand());
        if (reduce_type == ReduceType::ADD) {
          auto res = ValueType::join_add(v, v);
          if (_reduce_op.getInit() == nullptr) {
            type_map[_reduce_op.getResult()] = res;
          } else {
            auto init_v = find_type(_reduce_op.getInit());
            res = ValueType::join_add(res, init_v);
            type_map[_reduce_op.getResult()] = res;
          }
        } else if (reduce_type == ReduceType::ANY) {
          auto res = ValueType::join_mul(v, v);
          if (_reduce_op.getInit() == nullptr) {
            type_map[_reduce_op.getResult()] = res;
          } else {
            auto init_v = find_type(_reduce_op.getInit());
            res = ValueType::join_mul(res, init_v);
            type_map[_reduce_op.getResult()] = res;
          }
        } else {
          op->dump();
          llvm_unreachable("unsupported reduce type");
        }
      } else if (auto _cmp_op = dyn_cast<CmpOp>(op)) {
        auto cmp_type = _cmp_op.getCmpType();
        if (cmp_type == CmpType::GE) {
          auto lhs = find_type(_cmp_op.getLhs());
          auto rhs = find_type(_cmp_op.getRhs());
          auto res = ValueType::join_cmp_ge(lhs, rhs);
          type_map[_cmp_op.getResult()] = res;
        } else {
          op->dump();
          llvm_unreachable("unsupported cmp type");
        }
      } else {
        for (auto res : op->getResults()) {
          type_map[res] = ValueType::kVar;
        }
      }
      return WalkResult::advance();
    });

    // for_op.getRegion().front().walk<WalkOrder::PreOrder>([&](Operation *op) {
    //   llvm::errs() << *op << "\n";
    //   for (auto res : op->getResults()) {
    //     llvm::errs() << "\t" << type_map[res].str() << ", ";
    //   }
    //   llvm::errs() << "\n";
    //   if (isa<MaskOp>(op)) {
    //     return WalkResult::skip();
    //   } else {
    //     return WalkResult::advance();
    //   }
    // });
    auto yield_op = cast<BlockYieldOp>(for_op.getRegion().front().getTerminator());
    // yield_op->dump();
    // for (int i = 0; i < yield_op.getNumIters(); ++i) {
    //   auto iter = yield_op.getIter(i);
    //   llvm::errs() << "iter: " << ", " << type_map[iter].str() << "\n";
    // }
    assert(yield_op.getNumBlocks() == 0);
    bool has_effect = false;
    for (int i = 0; i < yield_op.getNumIters(); ++i) {
      auto iter = yield_op.getIter(i);
      auto type = type_map[iter];
      auto init_arg = for_op.getInitArg(i);
      auto zero_op = init_arg.getDefiningOp<ZeroOp>();
      if (zero_op == nullptr) {
        has_effect = true;
        break;
      }
      if (type.kind != ValueType::kZero) {
        has_effect = true;
        break;
      }
    }
    if (has_effect) {
      return mlir::failure();
    }

    Value diagonal = nullptr;
    mask_op.getRegion().front().walk([&](Operation *op) {
      if (auto add_op = dyn_cast<arith::AddIOp>(op)) {
        diagonal = add_op.getRhs();
      }
    });
    assert(diagonal != nullptr);
    auto diagonal_const_op = diagonal.getDefiningOp<arith::ConstantOp>();
    assert(diagonal_const_op != nullptr);
    auto diagonal_val = cast<IntegerAttr>(diagonal_const_op.getValue()).getInt();
    // rewrite
    if (starts[0] == for_iter) {
      // add new lb
      int off = (int)std::ceil(diagonal_val / (double)block_size) * block_size;
      // dbg(off);
      // no sub
      // FIXME: why?
      rewriter.setInsertionPoint(for_op);
      // auto sub_op = rewriter.create<arith::SubIOp>(
      //     for_op.getLoc(), starts[1], rewriter.create<arith::ConstantOp>(for_op.getLoc(),
      //     rewriter.getIndexAttr(off)));
      IRMapping val_map;
      // val_map.map(for_op.getLowerBound(), sub_op.getResult());
      val_map.map(for_op.getLowerBound(), starts[1]);
      auto new_for_op = rewriter.clone(*for_op, val_map);

      rewriter.replaceOp(for_op, new_for_op);
      return success();
    } else {
      // add new ub
      assert(starts[1] == for_iter);
      int off = (int)std::ceil(diagonal_val / (double)block_size) * block_size;
      // dbg(off);
      rewriter.setInsertionPoint(for_op);
      auto add_op = rewriter.create<arith::AddIOp>(
          for_op.getLoc(), starts[0], rewriter.create<arith::ConstantOp>(for_op.getLoc(), rewriter.getIndexAttr(off)));
      IRMapping val_map;
      val_map.map(for_op.getUpperBound(), add_op.getResult());
      auto new_for_op = rewriter.clone(*for_op, val_map);

      rewriter.replaceOp(for_op, new_for_op);
      return success();
    }
  }
};

class DynamicForPass : public ::mlir::asuka::impl::AsukaDynamicForBase<DynamicForPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    KernelOp m = getOperation();

    patterns.add<ToDynamicBlockForPattern>(context);
    patterns.add<EarlyStopByMaskPattern>(context);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createDynamicForPass() { return std::make_unique<DynamicForPass>(); }

} // namespace mlir::asuka