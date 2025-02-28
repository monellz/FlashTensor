#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"
#include "asuka/Dialect/Asuka/Transforms/Passes.h"

#include "dbg.h"

namespace mlir::asuka {

#define GEN_PASS_DEF_ASUKARECOVERTYPEINKERNEL
#include "asuka/Dialect/Asuka/Transforms/Passes.h.inc"

} // namespace mlir::asuka

namespace mlir::asuka {

namespace {

bool needConvert(SmallVector<Value> vals) {
  bool has_same_type = true;
  for (auto val : vals) {
    if (val.getType() != vals[0].getType()) {
      has_same_type = false;
      break;
    }
  }
  if (has_same_type)
    return false;

  SmallVector<FloatType> types;
  for (auto val : vals) {
    assert(isa<RankedTensorType>(val.getType()));
    auto t = cast<RankedTensorType>(val.getType());
    assert(isa<FloatType>(t.getElementType()));
    types.push_back(cast<FloatType>(t.getElementType()));
  }
  bool has_same_elem_type = true;
  for (size_t i = 1; i < types.size(); ++i) {
    if (types[i] != types[0]) {
      has_same_elem_type = false;
      break;
    }
  }
  if (has_same_elem_type)
    return false;
  return true;
}

// convert all vals to the least element type
SmallVector<Value> convertLeastElementType(SmallVector<Value> vals, OpBuilder &builder) {
  bool has_same_type = true;
  for (auto val : vals) {
    if (val.getType() != vals[0].getType()) {
      has_same_type = false;
      break;
    }
  }
  if (has_same_type) {
    return vals;
  }

  // all is ranked tensor type
  SmallVector<Type> elem_types;
  for (auto val : vals) {
    assert(isa<RankedTensorType>(val.getType()));
    auto t = cast<RankedTensorType>(val.getType());
    elem_types.push_back(t.getElementType());
  }
  bool has_same_elem_type = true;
  for (size_t i = 1; i < elem_types.size(); ++i) {
    if (elem_types[i] != elem_types[0]) {
      has_same_elem_type = false;
      break;
    }
  }
  if (has_same_elem_type) {
    return vals;
  }

  // all type must be float type
  SmallVector<FloatType> types;
  for (auto t : elem_types) {
    assert(isa<FloatType>(t));
    types.push_back(cast<FloatType>(t));
  }

  FloatType least_type = types[0];
  for (auto type : types) {
    if (type.getWidth() < least_type.getWidth()) {
      least_type = type;
    }
  }

  // try convert
  SmallVector<Value> rets;
  for (size_t i = 0; i < vals.size(); ++i) {
    if (types[i] != least_type) {
      auto val = vals[i];
      auto new_val = builder.create<ConvertOp>(val.getLoc(), val, least_type);
      rets.push_back(new_val);
    } else {
      rets.push_back(vals[i]);
    }
  }
  return rets;
}

void infer_type(Operation *op) {
  assert(isa<InferTypeOpInterface>(op));
  auto type_infer = cast<InferTypeOpInterface>(op);
  SmallVector<Type> new_types;
  auto success = type_infer.inferReturnTypes(op->getContext(), op->getLoc(), op->getOperands(), op->getAttrDictionary(),
                                             op->getPropertiesStorage(), op->getRegions(), new_types);

  assert(succeeded(success));
  for (size_t i = 0; i < new_types.size(); ++i) {
    op->getResult(i).setType(new_types[i]);
  }
}

constexpr char ATTR_NAME[] = "asuka.erased_type";

class RecoverTypeInKernelPass : public ::mlir::asuka::impl::AsukaRecoverTypeInKernelBase<RecoverTypeInKernelPass> {
public:
  LogicalResult match(KernelOp kernel_op) {
    auto arg_attrs = kernel_op.getCallableArgAttrs();
    auto res_attrs = kernel_op.getCallableResAttrs();
    if (arg_attrs == nullptr || res_attrs == nullptr) {
      return failure();
    }
    if (arg_attrs != nullptr) {
      for (auto attr : arg_attrs) {
        assert(isa<DictionaryAttr>(attr));
        auto dict = cast<DictionaryAttr>(attr);
        if (!dict.contains(ATTR_NAME)) {
          return failure();
        }
      }
    }
    if (res_attrs != nullptr) {
      for (auto attr : arg_attrs) {
        assert(isa<DictionaryAttr>(attr));
        auto dict = cast<DictionaryAttr>(attr);
        if (!dict.contains(ATTR_NAME)) {
          return failure();
        }
      }
    }
    return success();
  }

  void rewrite(arith::ConstantOp const_op, OpBuilder &builder) {
    auto default_type = builder.getF32Type();
    auto v = const_op.getValue();
    if (auto dense_attr = dyn_cast<DenseElementsAttr>(v)) {
      auto elem_type = dense_attr.getElementType();
      if (isa<FloatType>(elem_type)) {
        if (elem_type != default_type) {
          // reset
          SmallVector<APFloat> vals;
          for (auto f : dense_attr.getValues<APFloat>()) {
            // dbg(f.convertToDouble());
            // FIXME: APFloat has its own semantic, which should be matched with default_type
            vals.push_back(APFloat(f.convertToFloat()));
          }
          auto shape = dense_attr.getType().getShape();
          auto new_dense_attr = DenseElementsAttr::get(RankedTensorType::get(shape, default_type), vals);

          const_op.setValueAttr(new_dense_attr);
        }
      }
    } else if (auto float_attr = dyn_cast<FloatAttr>(v)) {
      auto elem_type = float_attr.getType();
      if (elem_type != default_type) {
        // reset
        const_op.setValueAttr(builder.getFloatAttr(default_type, float_attr.getValueAsDouble()));
      }
    }

    infer_type(const_op);
  }

  void rewrite(ZeroOp zero_op, OpBuilder &builder) {
    auto elem_type = zero_op.getElementType();
    if (!isa<FloatType>(elem_type))
      return;

    auto default_type = builder.getF32Type();
    if (elem_type != default_type) {
      zero_op.setElementType(default_type);
    }

    infer_type(zero_op);
  }

  void rewrite(Block *block, OpBuilder &builder) {
    auto update_elem_type = [](Value new_arg, Value arg) {
      auto new_elem_type = cast<RankedTensorType>(new_arg.getType()).getElementType();
      auto elem_type = cast<RankedTensorType>(arg.getType()).getElementType();
      if (elem_type != new_elem_type) {
        auto shape = cast<RankedTensorType>(arg.getType()).getShape();
        arg.setType(RankedTensorType::get(shape, new_elem_type));
      }
    };

    block->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto para_for_op = dyn_cast<ParallelForOp>(op)) {
        auto para_block = &para_for_op.getRegion().front();
        auto kernel_op = para_for_op->getParentOfType<KernelOp>();
        assert(kernel_op != nullptr);
        auto new_arg_types = kernel_op.getArgumentTypes();
        auto new_res_types = kernel_op.getResultTypes();
        assert(kernel_op.hasParallelMap());
        auto parallel_maps = kernel_op.getParallelMaps();
        auto para_dim_num = parallel_maps.size();
        assert(para_block->getNumArguments() == para_dim_num + new_arg_types.size());
        for (size_t i = 0; i < para_block->getNumArguments(); ++i) {
          auto arg = para_block->getArgument(i);
          if (i >= para_dim_num) {
            auto elem_type = cast<RankedTensorType>(arg.getType()).getElementType();
            auto new_elem_type = cast<RankedTensorType>(new_arg_types[i - para_dim_num]).getElementType();
            if (elem_type != new_elem_type) {
              auto shape = cast<RankedTensorType>(arg.getType()).getShape();
              arg.setType(RankedTensorType::get(shape, new_elem_type));
            }
          }
        }
        for (size_t i = 0; i < new_res_types.size(); ++i) {
          auto new_elem_type = cast<RankedTensorType>(new_res_types[i]).getElementType();
          auto res = para_for_op->getResult(i);
          auto shape = cast<RankedTensorType>(res.getType()).getShape();
          res.setType(RankedTensorType::get(shape, new_elem_type));
        }

        rewrite(para_block, builder);
      } else if (auto const_op = dyn_cast<arith::ConstantOp>(op)) {
        rewrite(const_op, builder);
      } else if (auto zero_op = dyn_cast<ZeroOp>(op)) {
        rewrite(zero_op, builder);
      } else if (auto dot_op = dyn_cast<DotOp>(op)) {
        builder.setInsertionPoint(dot_op);
        auto operands = convertLeastElementType({dot_op.getLhs(), dot_op.getRhs()}, builder);
        auto lhs = operands[0];
        auto rhs = operands[1];
        // FIXME: f32 is enough?
        Type acc_type = builder.getF32Type();
        auto new_dot_op = builder.create<PreciseDotOp>(dot_op.getLoc(), lhs, rhs, acc_type);
        dot_op.getResult().replaceAllUsesWith(new_dot_op.getResult());
        dot_op->erase();
      } else if (auto mask_op = dyn_cast<MaskOp>(op)) {
        auto elem_type = mask_op.getElementType();
        assert(isa<FloatType>(elem_type));
        auto default_type = builder.getF32Type();
        if (elem_type != default_type) {
          mask_op.setElementType(default_type);
        }
        infer_type(&*mask_op);

        auto mask_block = &mask_op.getRegion().front();
        rewrite(mask_block, builder);
      } else if (auto block_for_op = dyn_cast<BlockForOp>(op)) {
        // FIXME: redundant code, need an block_for interface
        for (int i = 0; i < block_for_op.getNumBlockArgs(); ++i) {
          auto arg = block_for_op.getBlockArg(i);
          auto entry_arg = block_for_op.getBlockArgInEntry(i);
          update_elem_type(arg, entry_arg);
        }
        for (int i = 0; i < block_for_op.getNumInitArgs(); ++i) {
          auto arg = block_for_op.getInitArg(i);
          auto entry_arg = block_for_op.getInitArgInEntry(i);
          update_elem_type(arg, entry_arg);
        }
        auto for_block = &block_for_op.getRegion().front();
        rewrite(for_block, builder);

        auto yield_op = cast<BlockYieldOp>(for_block->getTerminator());
        assert(yield_op.getNumOperands() == block_for_op.getNumResults());
        for (size_t i = 0; i < yield_op->getNumOperands(); ++i) {
          auto arg = yield_op->getOperand(i);
          auto ret = block_for_op->getResult(i);
          update_elem_type(arg, ret);
        }
      } else if (auto dyn_block_for_op = dyn_cast<DynamicBlockForOp>(op)) {
        for (int i = 0; i < dyn_block_for_op.getNumBlockArgs(); ++i) {
          auto arg = dyn_block_for_op.getBlockArg(i);
          auto entry_arg = dyn_block_for_op.getBlockArgInEntry(i);
          update_elem_type(arg, entry_arg);
        }
        for (int i = 0; i < dyn_block_for_op.getNumInitArgs(); ++i) {
          auto arg = dyn_block_for_op.getInitArg(i);
          auto entry_arg = dyn_block_for_op.getInitArgInEntry(i);
          update_elem_type(arg, entry_arg);
        }
        auto for_block = &dyn_block_for_op.getRegion().front();
        rewrite(for_block, builder);

        auto yield_op = cast<BlockYieldOp>(for_block->getTerminator());
        assert(yield_op.getNumOperands() == dyn_block_for_op.getNumResults());
        for (size_t i = 0; i < yield_op->getNumOperands(); ++i) {
          auto arg = yield_op->getOperand(i);
          auto ret = dyn_block_for_op->getResult(i);
          update_elem_type(arg, ret);
        }
      } else if (isa<InferTypeOpInterface>(op)) {
        SmallVector<Value> operands(op->getOperands());
        builder.setInsertionPoint(op);
        auto new_operands = convertLeastElementType(operands, builder);
        for (size_t i = 0; i < new_operands.size(); ++i) {
          op->setOperand(i, new_operands[i]);
        }
        infer_type(op);
      } else if (auto yield_op = dyn_cast<ParallelYieldOp>(op)) {
        builder.setInsertionPoint(yield_op);
        auto parallel_for_op = yield_op->getParentOfType<ParallelForOp>();
        assert(parallel_for_op != nullptr);
        auto kernel_op = parallel_for_op->getParentOfType<KernelOp>();
        assert(kernel_op != nullptr);
        auto new_res_types = kernel_op.getResultTypes();
        assert(yield_op.getNumOperands() == new_res_types.size());
        for (size_t i = 0; i < yield_op->getNumOperands(); ++i) {
          auto arg = yield_op->getOperand(i);
          auto new_elem_type = cast<RankedTensorType>(new_res_types[i]).getElementType();
          auto elem_type = cast<RankedTensorType>(arg.getType()).getElementType();
          if (elem_type != new_elem_type) {
            // need convert
            auto converted_arg = builder.create<ConvertOp>(arg.getLoc(), arg, new_elem_type);
            yield_op->setOperand(i, converted_arg.getResult());
          }
        }
      } else if (op->getNumResults() == 0) {
        // pass
      } else {
        block->dump();
        llvm::errs() << "op: " << *op << "\n";
        llvm_unreachable("not a type infer");
      }
      return WalkResult::advance();
    });
  }

  void rewrite(KernelOp kernel_op, OpBuilder &builder) {
    auto arg_types = kernel_op.getArgumentTypes();
    auto arg_attrs = kernel_op.getCallableArgAttrs();
    auto res_types = kernel_op.getResultTypes();
    auto res_attrs = kernel_op.getCallableResAttrs();
    SmallVector<Type> new_arg_types;
    SmallVector<Type> new_res_types;

    for (auto [arg_type, arg_attr] : llvm::zip(arg_types, arg_attrs)) {
      assert(isa<DictionaryAttr>(arg_attr));
      auto dict = cast<DictionaryAttr>(arg_attr);
      assert(dict.size() == 1);
      assert(dict.contains(ATTR_NAME));
      auto erased_type = cast<TypeAttr>(dict.get(ATTR_NAME)).getValue();

      auto arg_shape = cast<RankedTensorType>(arg_type).getShape();
      auto new_arg_type = RankedTensorType::get(arg_shape, erased_type);
      new_arg_types.push_back(new_arg_type);
    }
    kernel_op.setArgAttrsAttr({});
    for (auto [res_type, res_attr] : llvm::zip(res_types, res_attrs)) {
      assert(isa<DictionaryAttr>(res_attr));
      auto dict = cast<DictionaryAttr>(res_attr);
      assert(dict.size() == 1);
      assert(dict.contains(ATTR_NAME));
      auto erased_type = cast<TypeAttr>(dict.get(ATTR_NAME)).getValue();

      auto res_shape = cast<RankedTensorType>(res_type).getShape();
      auto new_res_type = RankedTensorType::get(res_shape, erased_type);
      new_res_types.push_back(new_res_type);
    }
    kernel_op.setResAttrsAttr({});

    // reset type
    auto new_function_type = builder.getFunctionType(new_arg_types, new_res_types);
    kernel_op.setFunctionType(new_function_type);

    // change type of all ops in kernel
    auto region = kernel_op.getCallableRegion();
    auto block = &region->front();

    for (auto [arg_type, arg] : llvm::zip(new_arg_types, block->getArguments())) {
      if (arg.getType() != arg_type) {
        arg.setType(arg_type);
      }
    }

    rewrite(block, builder);
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    KernelOp k = getOperation();

    if (match(k).succeeded()) {
      OpBuilder builder(context);
      rewrite(k, builder);
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createRecoverTypeInKernelPass() { return std::make_unique<RecoverTypeInKernelPass>(); }

} // namespace mlir::asuka