#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"
#include "asuka/Dialect/Asuka/IR/AsukaTypes.h"

using namespace mlir;
using namespace mlir::asuka;

#define GET_TYPEDEF_CLASSES
#include "asuka/Dialect/Asuka/IR/AsukaTypes.cpp.inc"
