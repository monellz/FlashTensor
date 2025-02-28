#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"
#include "asuka/Dialect/Asuka/Transforms/Passes.h"

#include "dbg.h"

namespace mlir::asuka {

#define GEN_PASS_DEF_ASUKAERASETYPEINKERNEL
#include "asuka/Dialect/Asuka/Transforms/Passes.h.inc"

} // namespace mlir::asuka

namespace mlir::asuka {

namespace {

constexpr char ATTR_NAME[] = "asuka.erased_type";

class EraseTypeInKernelPass : public ::mlir::asuka::impl::AsukaEraseTypeInKernelBase<EraseTypeInKernelPass> {
public:
  LogicalResult match(KernelOp kernel_op) const {
    auto arg_attrs = kernel_op.getCallableArgAttrs();
    auto res_attrs = kernel_op.getCallableResAttrs();
    if (arg_attrs == nullptr || res_attrs == nullptr)
      return success();
    if (arg_attrs != nullptr) {
      for (auto attr : arg_attrs) {
        assert(isa<DictionaryAttr>(attr));
        auto dict = cast<DictionaryAttr>(attr);
        if (!dict.contains(ATTR_NAME)) {
          return success();
        }
      }
    }
    if (res_attrs != nullptr) {
      for (auto attr : arg_attrs) {
        assert(isa<DictionaryAttr>(attr));
        auto dict = cast<DictionaryAttr>(attr);
        if (!dict.contains(ATTR_NAME)) {
          return success();
        }
      }
    }
    return failure();
  }

  void rewrite(KernelOp kernel_op, OpBuilder &builder) const {
    auto arg_types = kernel_op.getArgumentTypes();
    auto res_types = kernel_op.getResultTypes();

    SmallVector<Type> new_arg_types;
    SmallVector<Type> new_res_types;

    SmallVector<Attribute> new_arg_attrs;
    SmallVector<Attribute> new_res_attrs;
    for (auto arg_type : arg_types) {
      assert(isa<RankedTensorType>(arg_type));
      auto tensor_type = cast<RankedTensorType>(arg_type);
      if (isa<FloatType>(tensor_type.getElementType())) {
        // TODO: f64 is enough?
        auto new_type = builder.getF64Type();
        auto new_arg_type = RankedTensorType::get(tensor_type.getShape(), new_type);
        new_arg_types.push_back(new_arg_type);
      } else {
        new_arg_types.push_back(arg_type);
      }

      auto type_attr = TypeAttr::get(tensor_type.getElementType());
      NamedAttrList attrs;
      attrs.append(ATTR_NAME, type_attr);
      auto dict_attr = builder.getDictionaryAttr(attrs);
      new_arg_attrs.push_back(dict_attr);
    }
    for (auto res_type : res_types) {
      assert(isa<RankedTensorType>(res_type));
      auto tensor_type = cast<RankedTensorType>(res_type);
      if (isa<FloatType>(tensor_type.getElementType())) {
        // TODO: f64 is enough?
        auto new_type = builder.getF64Type();
        auto new_res_type = RankedTensorType::get(tensor_type.getShape(), new_type);
        new_res_types.push_back(new_res_type);
      } else {
        new_res_types.push_back(res_type);
      }
      auto type_attr = TypeAttr::get(tensor_type.getElementType());
      NamedAttrList attrs;
      attrs.append(ATTR_NAME, type_attr);
      auto dict_attr = builder.getDictionaryAttr(attrs);
      new_res_attrs.push_back(dict_attr);
    }

    // reset type
    auto new_function_type = builder.getFunctionType(new_arg_types, new_res_types);
    kernel_op.setFunctionType(new_function_type);

    // set attrs
    ArrayAttr arg_attrs = builder.getArrayAttr(new_arg_attrs);
    ArrayAttr res_attrs = builder.getArrayAttr(new_res_attrs);
    kernel_op.setArgAttrsAttr(arg_attrs);
    kernel_op.setResAttrsAttr(res_attrs);

    // kernel_op->dump();
    // rebuild all ops in kernel

    auto region = kernel_op.getCallableRegion();
    auto original_block = &region->front();
    assert(region != nullptr);
    auto build_block = builder.createBlock(region);
    builder.setInsertionPointToStart(build_block);

    IRMapping val_map;
    for (auto [arg_type, original_arg] : llvm::zip(new_arg_types, original_block->getArguments())) {
      auto new_arg = build_block->addArgument(arg_type, original_arg.getLoc());
      val_map.map(original_arg, new_arg);
    }

    original_block->walk([&](Operation *op) {
      if (auto constant_op = dyn_cast<arith::ConstantOp>(op)) {
        auto original_val = constant_op.getValue();
        if (isa<DenseElementsAttr>(original_val)) {
          auto dense_attr = cast<DenseElementsAttr>(original_val);
          auto elem_type = dense_attr.getElementType();
          auto shape = dense_attr.getType().getShape();
          if (isa<FloatType>(elem_type)) {
            SmallVector<Attribute> new_vals;
            for (auto v : dense_attr.getValues<Attribute>()) {
              assert(isa<FloatAttr>(v));
              double _v = cast<FloatAttr>(v).getValueAsDouble();
              new_vals.push_back(builder.getFloatAttr(builder.getF64Type(), _v));
            }
            auto new_elem_type = builder.getF64Type();
            auto new_dense_attr = DenseElementsAttr::get(RankedTensorType::get(shape, new_elem_type), new_vals);
            auto new_op = builder.create<arith::ConstantOp>(op->getLoc(), new_dense_attr);

            val_map.map(constant_op.getResult(), new_op.getResult());
          } else {
            op->dump();
            llvm_unreachable("not supported");
          }
        } else if (isa<FloatAttr>(original_val)) {
          auto float_attr = cast<FloatAttr>(original_val);
          auto new_attr = builder.getFloatAttr(builder.getF64Type(), float_attr.getValueAsDouble());
          auto new_op = builder.create<arith::ConstantOp>(op->getLoc(), new_attr);

          val_map.map(constant_op.getResult(), new_op.getResult());
        } else if (isa<IntegerAttr>(original_val)) {
          // pass
        } else {
          llvm::errs() << "const val: " << original_val << "\n";
          llvm_unreachable("not supported");
        }
      } else if (auto trilu_op = dyn_cast<TriluOp>(op)) {
        if (auto float_attr = dyn_cast<FloatAttr>(trilu_op.getValAttr())) {
          double _v = float_attr.getValueAsDouble();
          auto new_attr = builder.getFloatAttr(builder.getF64Type(), _v);
          auto diagonal = trilu_op.getDiagonal();
          auto is_upper = trilu_op.getIsUpper();
          auto shape = trilu_op.getShape();
          auto new_op = builder.create<TriluOp>(op->getLoc(), diagonal, is_upper, shape, new_attr);

          val_map.map(trilu_op.getResult(), new_op.getResult());
        } else {
          op->dump();
          llvm_unreachable("not supported");
        }
      } else if (auto convert_op = dyn_cast<ConvertOp>(op)) {
        auto dst_type = convert_op.getDstType();
        auto operand = convert_op.getOperand();
        auto operand_type = cast<RankedTensorType>(operand.getType());
        if (isa<FloatType>(dst_type)) {
          if (isa<FloatType>(operand_type.getElementType())) {
            val_map.map(convert_op.getResult(), val_map.lookup(operand));
          } else {
            auto new_dst_type = builder.getF64Type();
            auto new_op = builder.create<ConvertOp>(op->getLoc(), val_map.lookup(operand), new_dst_type);

            val_map.map(convert_op.getResult(), new_op.getResult());
          }
        } else {
          auto new_op = builder.create<ConvertOp>(op->getLoc(), val_map.lookup(operand), dst_type);
          val_map.map(convert_op.getResult(), new_op.getResult());
        }
      } else {
        auto new_op = builder.clone(*op, val_map);
        auto type_infer = dyn_cast<InferTypeOpInterface>(new_op);
        if (type_infer) {
          SmallVector<Type> new_types;
          auto success = type_infer.inferReturnTypes(new_op->getContext(), new_op->getLoc(), new_op->getOperands(),
                                                     new_op->getAttrDictionary(), new_op->getPropertiesStorage(),
                                                     new_op->getRegions(), new_types);
          if (succeeded(success)) {
            for (size_t i = 0; i < new_types.size(); ++i) {
              new_op->getResult(i).setType(new_types[i]);
            }
          }
        } else if (!isa<ReturnOp>(new_op)) {
          new_op->dump();
          llvm_unreachable("not supported");
        }
        for (auto [original_res, new_res] : llvm::zip(op->getResults(), new_op->getResults())) {
          val_map.map(original_res, new_res);
        }
      }
    });

    // FIXME: any api we can use to erase block?
    // erase block
    assert(original_block->use_empty());
    for (auto &op : llvm::make_early_inc_range(llvm::reverse(*original_block))) {
      assert(op.use_empty());
      op.erase();
    }
    original_block->erase();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    KernelOp kernel_op = getOperation();

    if (failed(match(kernel_op))) {
      signalPassFailure();
      return;
    }

    OpBuilder builder(context);
    rewrite(kernel_op, builder);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createEraseTypeInKernelPass() { return std::make_unique<EraseTypeInKernelPass>(); }

} // namespace mlir::asuka