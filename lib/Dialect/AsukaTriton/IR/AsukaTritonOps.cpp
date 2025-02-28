#include "asuka/Dialect/AsukaTriton/IR/AsukaTritonDialect.h"
#include "asuka/Dialect/AsukaTriton/IR/AsukaTritonTypes.h"

#include "dbg.h"

#define GET_OP_CLASSES
#include "asuka/Dialect/AsukaTriton/IR/AsukaTritonOps.cpp.inc"

#include "asuka/Dialect/AsukaTriton/IR/AsukaTritonDialect.cpp.inc"
namespace mlir {
namespace asuka::triton {
void AsukaTritonDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "asuka/Dialect/AsukaTriton/IR/AsukaTritonOps.cpp.inc"
      >();
}

} // namespace asuka::triton
} // namespace mlir

namespace mlir {
namespace asuka {
namespace triton {

// -- PointerOfOp --
LogicalResult PointerOfOp::inferReturnTypes(::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
                                            Adaptor adaptor,
                                            ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto type = cast<RankedTensorType>(adaptor.getOperand().getType());
  auto ret_type = PointerType::get(type);
  inferredReturnTypes.push_back(ret_type);
  return success();
}

// -- EmptyPointerOp --
LogicalResult EmptyPointerOp::inferReturnTypes(::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
                                               Adaptor adaptor,
                                               ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto tensor_type = cast<RankedTensorType>(adaptor.getTensorType());
  auto ret_type = PointerType::get(tensor_type);
  inferredReturnTypes.push_back(ret_type);
  return success();
}

// -- TensorFromOp --
LogicalResult TensorFromOp::inferReturnTypes(::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
                                             Adaptor adaptor,
                                             ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto operand = adaptor.getOperand();
  auto tensor_type = cast<PointerType>(operand.getType()).getPointeeType();
  inferredReturnTypes.push_back(tensor_type);
  return success();
}

// -- BlockPointerOfOp --
LogicalResult BlockPointerOfOp::inferReturnTypes(::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
                                                 Adaptor adaptor,
                                                 ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto base_ptr = adaptor.getBasePointer();
  auto tensor_type = cast<PointerType>(base_ptr.getType()).getPointeeType();
  auto elem_type = tensor_type.getElementType();
  auto block_shape = adaptor.getBlockShape();

  auto block_tensor_type = RankedTensorType::get(block_shape, elem_type);
  auto ret_type = BlockPointerType::get(block_tensor_type);
  inferredReturnTypes.push_back(ret_type);
  return success();
}

// -- BlockLoadOp --
LogicalResult BlockLoadOp::inferReturnTypes(::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
                                            Adaptor adaptor,
                                            ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto type = cast<BlockPointerType>(adaptor.getSrcPointer().getType());
  auto tensor_type = type.getPointeeType();
  inferredReturnTypes.push_back(tensor_type);
  return success();
}

} // namespace triton
} // namespace asuka
} // namespace mlir