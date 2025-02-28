#ifndef ASUKA_DIELACT_H_
#define ASUKA_DIELACT_H_

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// // Triton
// #include "triton/Dialect/Triton/IR/Dialect.h"
// #include "triton/Dialect/TritonGPU/IR/Dialect.h"
// #include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
// #include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"

#include "asuka/Dialect/Asuka/IR/AsukaDialect.h.inc"

#include "asuka/Dialect/Asuka/IR/AsukaTypes.h"

namespace mlir::asuka {
#include "asuka/Dialect/Asuka/IR/AsukaInterfaces.h.inc"
}

#include "asuka/Dialect/Asuka/IR/AsukaEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "asuka/Dialect/Asuka/IR/AsukaAttrs.h.inc"

#define GET_OP_CLASSES
#include "asuka/Dialect/Asuka/IR/AsukaOps.h.inc"

#endif