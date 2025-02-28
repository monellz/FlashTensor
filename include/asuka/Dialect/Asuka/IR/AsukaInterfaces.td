#ifndef ASUKA_INTERFACES
#define ASUKA_INTERFACES

include "mlir/IR/OpBase.td"

def FlopsInterface: OpInterface<"FlopsInterface"> {
  let description = [{}];
  let methods = [
    InterfaceMethod<"get flops", "int64_t", "getFlops", (ins), [{}], [{
      return 0;
    }]>
  ];
}

def TotalAccessedElementsInterface: OpInterface<"TotalAccessedElementsInterface"> {
  let description = [{
    compute accessed ranked tensor elements
  }];

  let methods = [
    InterfaceMethod<"Compute read elements number", "int64_t", "totalReadElements", (ins), [{
      int64_t num = 0;
      for (const auto& val: $_op.getOperation()->getOperands()) {
        if (auto type = dyn_cast<RankedTensorType>(val.getType())) {
          auto shape = type.getShape();
          int64_t size = 1;
          for (auto c: shape) {
            size *= c;
          }
          num += size;
        }
      }
      return num;
    }]>,

    InterfaceMethod<"Compute write elements number", "int64_t", "totalWriteElements", (ins), [{
      int64_t num = 0;
      for (const auto& val: $_op.getOperation()->getResults()) {
        if (auto type = dyn_cast<RankedTensorType>(val.getType())) {
          auto shape = type.getShape();
          int64_t size = 1;
          for (auto c: shape) {
            size *= c;
          }
          num += size;
        }
      }
      return num;
    }]>,

    InterfaceMethod<"Compute accessed elements number", "int64_t", "totalAccessedElements", (ins), [{
      int64_t num = 0;
      for (const auto& val: $_op.getOperation()->getOperands()) {
        if (auto type = dyn_cast<RankedTensorType>(val.getType())) {
          auto shape = type.getShape();
          int64_t size = 1;
          for (auto c: shape) {
            size *= c;
          }
          num += size;
        }
      }
      for (const auto& val: $_op.getOperation()->getResults()) {
        if (auto type = dyn_cast<RankedTensorType>(val.getType())) {
          auto shape = type.getShape();
          int64_t size = 1;
          for (auto c: shape) {
            size *= c;
          }
          num += size;
        }
      }
      return num;
    }]>
  ];
}

def BroadcastableBinaryOpInterface: OpInterface<"BroadcastableBinaryOpInterface"> {
  let description = [{
    binary op which support broadcast(right align)
  }];

  let methods = [
    InterfaceMethod<"", "::mlir::Value", "getLhs", (ins)>,
    InterfaceMethod<"", "::mlir::Value", "getRhs", (ins)>,
    InterfaceMethod<"", "::mlir::Value", "getResult", (ins)>,

    InterfaceMethod<"", "int64_t", "getBroadcastedRank", (ins), [{
      return getBroadcastedRankBy($_op.getLhs(), $_op.getRhs());
    }]>,

    StaticInterfaceMethod<"", "int64_t", "getBroadcastedRankBy", (ins "::mlir::Value":$lhs, "::mlir::Value":$rhs), [{}], [{
      int64_t max_rank = 0;
      auto lhs_type = dyn_cast<RankedTensorType>(lhs.getType());
      auto rhs_type = dyn_cast<RankedTensorType>(rhs.getType());
      max_rank = std::max(max_rank, lhs_type.getRank());
      max_rank = std::max(max_rank, rhs_type.getRank());
      return max_rank;
    }]>,

    InterfaceMethod<"", "llvm::SmallVector<int64_t>", "getBroadcastedShape", (ins), [{
      return getBroadcastedShapeBy($_op.getLhs(), $_op.getRhs());
    }]>,

    StaticInterfaceMethod<"", "llvm::SmallVector<int64_t>", "getBroadcastedShapeBy", (ins "::mlir::Value":$lhs, "::mlir::Value":$rhs), [{}], [{
      int max_rank = getBroadcastedRankBy(lhs, rhs);
      llvm::SmallVector<int64_t> ret_shape(max_rank, 1);

      auto type = dyn_cast<RankedTensorType>(lhs.getType());
      auto rank = type.getRank();
      auto shape = type.getShape();
      for (int64_t dim = max_rank - rank; dim < max_rank; ++dim) {
        ret_shape[dim] = shape[dim - (max_rank - rank)];
      }
      return ret_shape;
    }]>,

    InterfaceMethod<"", "llvm::SmallVector<int64_t>", "getLhsBroadcastedShape", (ins), [{
      return getLhsBroadcastedShapeBy($_op.getLhs(), $_op.getRhs());
    }]>,

    StaticInterfaceMethod<"", "llvm::SmallVector<int64_t>", "getLhsBroadcastedShapeBy", (ins "::mlir::Value":$lhs, "::mlir::Value":$rhs), [{}], [{
      int max_rank = getBroadcastedRankBy(lhs, rhs);
      llvm::SmallVector<int64_t> ret_shape(max_rank, 1);

      auto type = dyn_cast<RankedTensorType>(lhs.getType());
      auto rank = type.getRank();
      auto shape = type.getShape();
      for (int64_t dim = max_rank - rank; dim < max_rank; ++dim) {
        ret_shape[dim] = shape[dim - (max_rank - rank)];
      }
      return ret_shape;      
    }]>,

    InterfaceMethod<"", "llvm::SmallVector<int64_t>", "getRhsBroadcastedShape", (ins), [{
      return getRhsBroadcastedShapeBy($_op.getLhs(), $_op.getRhs());
    }]>,

    StaticInterfaceMethod<"", "llvm::SmallVector<int64_t>", "getRhsBroadcastedShapeBy", (ins "::mlir::Value":$lhs, "::mlir::Value":$rhs), [{}], [{
      int max_rank = getBroadcastedRankBy(lhs, rhs);
      llvm::SmallVector<int64_t> ret_shape(max_rank, 1);

      auto type = dyn_cast<RankedTensorType>(rhs.getType());
      auto rank = type.getRank();
      auto shape = type.getShape();
      for (int64_t dim = max_rank - rank; dim < max_rank; ++dim) {
        ret_shape[dim] = shape[dim - (max_rank - rank)];
      }
      return ret_shape;
    }]>,


    InterfaceMethod<"", "llvm::SmallVector<int64_t>", "getExpectedResultShape", (ins), [{
      return getExpectedResultShapeBy($_op.getLhs(), $_op.getRhs());
    }]>,

    StaticInterfaceMethod<"", "llvm::SmallVector<int64_t>", "getExpectedResultShapeBy", (ins "::mlir::Value":$lhs, "::mlir::Value":$rhs), [{}], [{
      int max_rank = getBroadcastedRankBy(lhs, rhs);
      auto lhs_shape = getLhsBroadcastedShapeBy(lhs, rhs);
      auto rhs_shape = getRhsBroadcastedShapeBy(lhs, rhs);
      llvm::SmallVector<int64_t> shape(max_rank, 1);
      for (int64_t dim = 0; dim < max_rank; ++dim) {
        shape[dim] = std::max(lhs_shape[dim], rhs_shape[dim]);
      }
      return shape;
    }]>,
  ];
}

#endif // ASUKA_INTERFACES