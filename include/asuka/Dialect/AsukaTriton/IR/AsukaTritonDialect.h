#ifndef ASUKATRITON_DIELACT_H_
#define ASUKATRITON_DIELACT_H_

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// // Triton
// #include "triton/Dialect/Triton/IR/Dialect.h"
// #include "triton/Dialect/TritonGPU/IR/Dialect.h"
// #include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
// #include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"

// Asuka
#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"

#include "asuka/Dialect/AsukaTriton/IR/AsukaTritonTypes.h"

#include "asuka/Dialect/AsukaTriton/IR/AsukaTritonDialect.h.inc"
#define GET_OP_CLASSES
#include "asuka/Dialect/AsukaTriton/IR/AsukaTritonOps.h.inc"

#endif