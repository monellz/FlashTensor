#include <vector>
#include "dbg.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"
#include "asuka/Dialect/Asuka/IR/AsukaEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "asuka/Dialect/Asuka/IR/AsukaAttrs.cpp.inc"
#define GET_OP_CLASSES
#include "asuka/Dialect/Asuka/IR/AsukaOps.cpp.inc"

namespace mlir::asuka {
#include "asuka/Dialect/Asuka/IR/AsukaInterfaces.cpp.inc"
}

// move dialect def in this file to make compiler happy
#include "asuka/Dialect/Asuka/IR/AsukaDialect.cpp.inc"
namespace mlir {
namespace asuka {
void AsukaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "asuka/Dialect/Asuka/IR/AsukaOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "asuka/Dialect/Asuka/IR/AsukaAttrs.cpp.inc"
      >();
}
} // namespace asuka
} // namespace mlir

namespace mlir {
namespace asuka {

// -- ConvertOp --
LogicalResult ConvertOp::inferReturnTypes(::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
                                          Adaptor adaptor, ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto dst_type = adaptor.getDstType();
  auto operand_type = cast<RankedTensorType>(adaptor.getOperand().getType());
  auto ret_type = RankedTensorType::get(operand_type.getShape(), dst_type);
  inferredReturnTypes.push_back(ret_type);
  return success();
}

// -- EraseTypeOp --
LogicalResult EraseTypeOp::inferReturnTypes(::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
                                            Adaptor adaptor,
                                            ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  for (auto operand : adaptor.getOperands()) {
    auto type = cast<RankedTensorType>(operand.getType());
    auto new_type = RankedTensorType::get(type.getShape(), IndexType::get(context));
    inferredReturnTypes.push_back(new_type);
  }
  return success();
}

// -- SpecifyTypeOp --
LogicalResult SpecifyTypeOp::inferReturnTypes(::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
                                              Adaptor adaptor,
                                              ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto types_attr = adaptor.getElementTypesAttr();
  llvm::SmallVector<::mlir::Type, 4> types(types_attr.size());
  llvm::transform(types_attr, types.begin(), [&](Attribute attr) { return cast<TypeAttr>(attr).getValue(); });

  for (auto [operand, element_type_attr] : llvm::zip(adaptor.getOperands(), adaptor.getElementTypesAttr())) {
    auto operand_type = cast<RankedTensorType>(operand.getType());
    auto new_type = RankedTensorType::get(operand_type.getShape(), cast<TypeAttr>(element_type_attr).getValue());
    inferredReturnTypes.push_back(new_type);
  }
  return success();
}

// -- BinaryElementWiseOp --
#define BINARY_ELEMENTWISE_OP_VERIFY(Op)                                                                               \
  LogicalResult Op::verify() {                                                                                         \
    auto _op = cast<BroadcastableBinaryOpInterface>(getOperation());                                                   \
    auto max_rank = _op.getBroadcastedRank();                                                                          \
    auto lhs_shape = _op.getLhsBroadcastedShape();                                                                     \
    auto rhs_shape = _op.getRhsBroadcastedShape();                                                                     \
    auto expected_result_shape = _op.getExpectedResultShape();                                                         \
    for (int64_t dim = 0; dim < max_rank; ++dim) {                                                                     \
      if (lhs_shape[dim] != 1 && rhs_shape[dim] != 1) {                                                                \
        if (lhs_shape[dim] != rhs_shape[dim]) {                                                                        \
          return emitOpError() << "non-broadcast dim " << dim << " doesn't match ([" << lhs_shape << "], ["            \
                               << rhs_shape << "])";                                                                   \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
    auto result_type = dyn_cast<RankedTensorType>(getResult().getType());                                              \
    if (result_type.getRank() != max_rank) {                                                                           \
      return emitOpError() << "result rank should be " << max_rank << " instead of " << result_type.getRank();         \
    }                                                                                                                  \
    auto result_shape = result_type.getShape();                                                                        \
    for (int64_t dim = 0; dim < max_rank; ++dim) {                                                                     \
      if (result_shape[dim] != expected_result_shape[dim]) {                                                           \
        return emitOpError() << "result shape should be [" << expected_result_shape << "] instead of ["                \
                             << result_shape << "]";                                                                   \
      }                                                                                                                \
    }                                                                                                                  \
    return success();                                                                                                  \
  }

#define BINARY_ELEMENTWISE_OP_INFER_RETURN_TYPES(Op)                                                                   \
  LogicalResult Op::inferReturnTypes(MLIRContext *context, std::optional<Location> location, ValueRange operands,      \
                                     DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,      \
                                     SmallVectorImpl<Type> &inferredReturnTypes) {                                     \
    auto expected_result_shape =                                                                                       \
        BroadcastableBinaryOpInterface::Trait<Op>::getExpectedResultShapeBy(operands[0], operands[1]);                 \
    auto element_type = cast<RankedTensorType>(operands[0].getType()).getElementType();                                \
    auto return_type = RankedTensorType::get(expected_result_shape, element_type);                                     \
    inferredReturnTypes.push_back(return_type);                                                                        \
    return success();                                                                                                  \
  }

BINARY_ELEMENTWISE_OP_VERIFY(AddOp)
BINARY_ELEMENTWISE_OP_VERIFY(SubOp)
BINARY_ELEMENTWISE_OP_VERIFY(MulOp)
BINARY_ELEMENTWISE_OP_VERIFY(DivOp)
// TODO: verify elemnt_type combination
BINARY_ELEMENTWISE_OP_VERIFY(PowOp)
// TODO: verify element type?
BINARY_ELEMENTWISE_OP_VERIFY(CmpOp)

BINARY_ELEMENTWISE_OP_INFER_RETURN_TYPES(AddOp)
BINARY_ELEMENTWISE_OP_INFER_RETURN_TYPES(SubOp)
BINARY_ELEMENTWISE_OP_INFER_RETURN_TYPES(MulOp)
BINARY_ELEMENTWISE_OP_INFER_RETURN_TYPES(DivOp)
BINARY_ELEMENTWISE_OP_INFER_RETURN_TYPES(PowOp)

LogicalResult CmpOp::inferReturnTypes(::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
                                      Adaptor adaptor, ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto expected_result_shape =
      BroadcastableBinaryOpInterface::Trait<CmpOp>::getExpectedResultShapeBy(adaptor.getLhs(), adaptor.getRhs());
  Type ret_elem_type = IntegerType::get(context, 8);
  auto return_type = RankedTensorType::get(expected_result_shape, ret_elem_type);
  inferredReturnTypes.push_back(return_type);
  return success();
}

// -- ReduceOp --
LogicalResult ReduceOp::inferReturnTypes(::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
                                         Adaptor adaptor, ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto keep_dim = adaptor.getKeepDimAttr().getValue();
  int64_t reduce_dim = adaptor.getReduceDimensionAttr().getValue().getSExtValue();
  auto operand_type = cast<RankedTensorType>(adaptor.getOperand().getType());

  auto shape = operand_type.getShape();
  auto element_type = operand_type.getElementType();

  if (reduce_dim >= (int64_t)shape.size() ||
      (reduce_dim < 0 &&
       (reduce_dim + (int64_t)shape.size() < 0 || reduce_dim + (int64_t)shape.size() >= (int64_t)shape.size()))) {
    return emitOptionalError(location, "reduce_dim error: ", reduce_dim, ", shape.size=", shape.size());
  }

  if (reduce_dim < 0) {
    reduce_dim += (int64_t)shape.size();
  }
  if (keep_dim) {
    llvm::SmallVector<int64_t, 4> return_shape(shape);
    return_shape[reduce_dim] = 1;
    auto return_type = RankedTensorType::get(return_shape, element_type);
    inferredReturnTypes.push_back(return_type);
  } else {
    assert(shape.size() >= 1);
    llvm::SmallVector<int64_t, 4> return_shape;
    for (int64_t i = 0; i < (int64_t)shape.size(); ++i) {
      if (i != reduce_dim) {
        return_shape.push_back(shape[i]);
      }
    }
    if (return_shape.size() == 0) {
      return_shape.push_back(1);
    }
    auto return_type = RankedTensorType::get(return_shape, element_type);
    inferredReturnTypes.push_back(return_type);
  }
  return success();
}

// -- SplitOp --
LogicalResult SplitOp::verify() {
  int64_t split_dim = getSplitDimensionAttr().getValue().getSExtValue();
  if (split_dim < 0) {
    return emitOpError() << "split dim should be positive but got " << split_dim;
  }
  assert(split_dim >= 0);
  int64_t split_size_sum = 0;
  for (const auto &type : getResultTypes()) {
    auto tensor_type = dyn_cast<RankedTensorType>(type);
    auto shape = tensor_type.getShape();
    if (split_dim >= tensor_type.getRank()) {
      return emitOpError() << "split dim " << split_dim << " more than result rank " << tensor_type.getRank();
    }
    split_size_sum += shape[split_dim];
  }

  auto operand_type = dyn_cast<RankedTensorType>(getOperand().getType());
  if (split_dim >= operand_type.getRank()) {
    return emitOpError() << "split dim " << split_dim << " more than operand rank " << operand_type.getRank();
  }

  if (split_size_sum != operand_type.getShape()[split_dim]) {
    return emitOpError() << "split size sum " << split_size_sum << " doesn't match operand size "
                         << operand_type.getShape()[split_dim];
  }
  return success();
}

// -- DotOp --
LogicalResult DotOp::inferReturnTypes(::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
                                      Adaptor adaptor, ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto lhs_type = cast<RankedTensorType>(adaptor.getLhs().getType());
  auto rhs_type = cast<RankedTensorType>(adaptor.getRhs().getType());

  auto element_type = lhs_type.getElementType();

  auto lhs_rank = lhs_type.getRank();
  auto rhs_rank = rhs_type.getRank();
  if (lhs_rank < 2 || rhs_rank < 2) {
    return emitOptionalError(location, "lhs and rhs rank both need >= 2");
  }

  auto lhs_shape = lhs_type.getShape();
  auto rhs_shape = rhs_type.getShape();
  if (lhs_shape[lhs_rank - 1] != rhs_shape[rhs_rank - 2]) {
    return emitOptionalError(location, "lhs_shape[-2] and rhs_shape[-1] doesn't match");
  }

  if (lhs_rank != rhs_rank) {
    // batch dim broadcast
    if (lhs_rank > 2 && rhs_rank > 2) {
      return emitOptionalError(location, "operands broadcast batch dim differently, unaccepted");
    }

    if (lhs_rank > rhs_rank) {
      assert(rhs_rank == 2);
      llvm::SmallVector<int64_t, 4> return_shape(lhs_shape);
      return_shape[lhs_rank - 1] = rhs_shape[rhs_rank - 1];
      auto return_type = RankedTensorType::get(return_shape, element_type);
      inferredReturnTypes.push_back(return_type);
    } else {
      assert(lhs_rank == 2);
      llvm::SmallVector<int64_t, 4> return_shape(rhs_shape);
      return_shape[rhs_rank - 2] = lhs_shape[lhs_rank - 1];
      auto return_type = RankedTensorType::get(return_shape, element_type);
      inferredReturnTypes.push_back(return_type);
    }
  } else {
    llvm::SmallVector<int64_t, 4> return_shape(lhs_shape);
    return_shape[lhs_rank - 1] = rhs_shape[rhs_rank - 1];
    auto return_type = RankedTensorType::get(return_shape, element_type);
    inferredReturnTypes.push_back(return_type);
  }
  return success();
}

// -- PreciseDotOp --
LogicalResult PreciseDotOp::inferReturnTypes(::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
                                             Adaptor adaptor,
                                             ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto lhs_type = cast<RankedTensorType>(adaptor.getLhs().getType());
  auto rhs_type = cast<RankedTensorType>(adaptor.getRhs().getType());
  auto acc_type = adaptor.getAccType();

  auto lhs_elem_type = lhs_type.getElementType();
  auto rhs_elem_type = rhs_type.getElementType();
  if (lhs_elem_type != rhs_elem_type) {
    return emitOptionalError(location, "lhs and rhs elem_type mismatch");
  }

  auto lhs_rank = lhs_type.getRank();
  auto rhs_rank = rhs_type.getRank();
  if (lhs_rank < 2 || rhs_rank < 2) {
    return emitOptionalError(location, "lhs and rhs rank both need >= 2");
  }

  auto lhs_shape = lhs_type.getShape();
  auto rhs_shape = rhs_type.getShape();
  if (lhs_shape[lhs_rank - 1] != rhs_shape[rhs_rank - 2]) {
    return emitOptionalError(location, "lhs_shape[-2] and rhs_shape[-1] doesn't match");
  }

  if (lhs_rank != rhs_rank) {
    // batch dim broadcast
    if (lhs_rank > 2 && rhs_rank > 2) {
      return emitOptionalError(location, "operands broadcast batch dim differently, unaccepted");
    }

    if (lhs_rank > rhs_rank) {
      assert(rhs_rank == 2);
      llvm::SmallVector<int64_t, 4> return_shape(lhs_shape);
      return_shape[lhs_rank - 1] = rhs_shape[rhs_rank - 1];
      auto return_type = RankedTensorType::get(return_shape, acc_type);
      inferredReturnTypes.push_back(return_type);
    } else {
      assert(lhs_rank == 2);
      llvm::SmallVector<int64_t, 4> return_shape(rhs_shape);
      return_shape[rhs_rank - 2] = lhs_shape[lhs_rank - 1];
      auto return_type = RankedTensorType::get(return_shape, acc_type);
      inferredReturnTypes.push_back(return_type);
    }
  } else {
    llvm::SmallVector<int64_t, 4> return_shape(lhs_shape);
    return_shape[lhs_rank - 1] = rhs_shape[rhs_rank - 1];
    auto return_type = RankedTensorType::get(return_shape, acc_type);
    inferredReturnTypes.push_back(return_type);
  }
  return success();
}

// -- AvgPoolOp --
LogicalResult AvgPoolOp::inferReturnTypes(::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
                                          Adaptor adaptor, ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto src = adaptor.getOperand();
  auto kernel_size = adaptor.getKernelSize();
  auto stride = adaptor.getStride();
  auto padding = adaptor.getPadding();
  auto ceil_mode = adaptor.getCeilMode();

  auto src_type = cast<RankedTensorType>(src.getType());
  auto src_shape = src_type.getShape();
  auto rank = src_type.getRank();

  auto spatial_rank = kernel_size.size();
  assert(spatial_rank <= rank);
  assert(stride.size() == spatial_rank);
  assert(padding.size() == spatial_rank);

  SmallVector<int64_t> ret_shape(src_shape);
  size_t rank_off = rank - spatial_rank;
  if (ceil_mode) {
    for (size_t i = 0; i < spatial_rank; ++i) {
      ret_shape[rank_off + i] = static_cast<int>(
          std::ceil((src_shape[rank_off + i] + 2 * padding[i] - kernel_size[i]) / static_cast<double>(stride[i]) + 1));
    }
  } else {
    for (size_t i = 0; i < spatial_rank; ++i) {
      ret_shape[rank_off + i] = static_cast<int>(
          std::floor((src_shape[rank_off + i] + 2 * padding[i] - kernel_size[i]) / static_cast<double>(stride[i]) + 1));
    }
  }
  inferredReturnTypes.push_back(RankedTensorType::get(ret_shape, src_type.getElementType()));
  return success();
}

// -- PermuteOp --
LogicalResult PermuteOp::inferReturnTypes(::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
                                          Adaptor adaptor, ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto src = adaptor.getOperand();
  auto src_type = cast<RankedTensorType>(src.getType());
  llvm::SmallVector<int64_t, 4> src_shape(src_type.getShape());
  llvm::SmallVector<int64_t, 4> return_shape(src_type.getShape());
  auto dims = adaptor.getDims();
  if (dims.size() != src_shape.size()) {
    return emitOptionalError(location, "size(dims) != rank(src)");
  }
  for (size_t i = 0; i < dims.size(); ++i) {
    return_shape[i] = src_shape[dims[i]];
  }
  inferredReturnTypes.push_back(RankedTensorType::get(return_shape, src_type.getElementType()));
  return success();
}

// -- TriluOp ---
LogicalResult TriluOp::inferReturnTypes(::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
                                        Adaptor adaptor, ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto shape = adaptor.getShape();
  ::mlir::TypedAttr val = adaptor.getVal();
  auto type = val.getType();
  auto ret_type = RankedTensorType::get(shape, type);
  inferredReturnTypes.push_back(ret_type);
  return success();
}

// -- MaskOp ---
LogicalResult MaskOp::inferReturnTypes(::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
                                       Adaptor adaptor, ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto shape = adaptor.getSizes();
  auto type = adaptor.getElementType();
  auto ret_type = RankedTensorType::get(shape, type);
  inferredReturnTypes.push_back(ret_type);
  return success();
}

// -- ZeroOp --
LogicalResult ZeroOp::inferReturnTypes(::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
                                       Adaptor adaptor, ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto shape = adaptor.getShape();
  auto type = adaptor.getElementType();
  auto ret_type = RankedTensorType::get(shape, type);
  inferredReturnTypes.push_back(ret_type);
  return success();
}

// -- TransposeOp --
LogicalResult TransposeOp::inferReturnTypes(::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
                                            Adaptor adaptor,
                                            ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto src = adaptor.getOperand();
  auto src_type = cast<RankedTensorType>(src.getType());
  llvm::SmallVector<int64_t, 4> src_shape(src_type.getShape());
  llvm::SmallVector<int64_t, 4> return_shape(src_type.getShape());
  auto dims = adaptor.getDims();
  if (dims.size() != 2) {
    return emitOptionalError(location, "size(dims) != 2");
  }
  auto src_dim = dims[0];
  auto dst_dim = dims[1];
  return_shape[src_dim] = src_shape[dst_dim];
  return_shape[dst_dim] = src_shape[src_dim];
  inferredReturnTypes.push_back(RankedTensorType::get(return_shape, src_type.getElementType()));
  return success();
}

// -- UnsqueezeOp --
LogicalResult UnsqueezeOp::inferReturnTypes(::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
                                            Adaptor adaptor,
                                            ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto src = adaptor.getOperand();
  auto src_type = cast<RankedTensorType>(src.getType());
  auto elem_type = src_type.getElementType();
  int64_t dim = adaptor.getDimAttr().getInt();
  SmallVector<int64_t> shape(src_type.getShape());
  shape.insert(shape.begin() + dim, 1);
  inferredReturnTypes.push_back(RankedTensorType::get(shape, elem_type));
  return success();
}

void KernelOp::build(OpBuilder &builder, OperationState &state, StringRef name, FunctionType type,
                     ArrayRef<NamedAttribute> attrs, ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_interface_impl::addArgAndResultAttrs(builder, state, argAttrs, /*resultAttrs=*/std::nullopt,
                                                getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

ParseResult KernelOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
                          function_interface_impl::VariadicFlag,
                          std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(parser, result, /*allowVariadic=*/false,
                                                  getFunctionTypeAttrName(result.name), buildFuncType,
                                                  getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void KernelOp::print(OpAsmPrinter &printer) {
  function_interface_impl::printFunctionOp(printer, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
                                           getArgAttrsAttrName(), getResAttrsAttrName());
}

// -- ReturnOp --
LogicalResult ReturnOp::verify() {
  auto function = cast<KernelOp>((*this)->getParentOp());
  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has ") << getNumOperands() << " operands, but enclosing function (@" << function.getName()
                               << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of return operand " << i << " (" << getOperand(i).getType()
                         << ") doesn't match function result type (" << results[i] << ")" << " in function @"
                         << function.getName();

  return success();
}

// -- CallOp --
LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this).getProperties().callee;
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  KernelOp fn = symbolTable.lookupNearestSymbolFrom<KernelOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue() << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided " << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  return success();
}

// -- BlockForOp --
LogicalResult BlockForOp::verify() {
  auto yield_op = cast<BlockYieldOp>(getRegion().front().getTerminator());

  auto res_num = getOperation()->getNumResults();
  auto yield_res_num = yield_op.getNumBlocks();
  auto yield_iter_num = yield_op.getNumIters();
  if (res_num != yield_res_num + yield_iter_num) {
    return emitOpError("result number mismatch: ") << res_num << " != " << yield_res_num + yield_iter_num;
  }

  auto init_num = getNumInitArgs();
  if (yield_iter_num != init_num) {
    return emitOpError("iter number mismatch: ") << init_num << " != " << yield_iter_num;
  }
  return success();
}

} // namespace asuka
} // namespace mlir