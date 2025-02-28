#ifndef ASUKATRITON_TYPES_H_
#define ASUKATRITON_TYPES_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "asuka/Dialect/AsukaTriton/IR/AsukaTritonTypes.h.inc"

#endif // ASUKATRITON_TYPES_H_