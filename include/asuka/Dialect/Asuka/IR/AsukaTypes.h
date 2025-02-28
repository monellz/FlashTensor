#ifndef ASUKA_TYPES_H_
#define ASUKA_TYPES_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "asuka/Dialect/Asuka/IR/AsukaTypes.h.inc"

#endif // ASUKA_TYPES_H_