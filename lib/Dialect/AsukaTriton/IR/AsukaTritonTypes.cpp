#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "asuka/Dialect/AsukaTriton/IR/AsukaTritonDialect.h"
#include "asuka/Dialect/AsukaTriton/IR/AsukaTritonTypes.h"

using namespace mlir;
using namespace mlir::asuka;
using namespace mlir::asuka::triton;

#define GET_TYPEDEF_CLASSES
#include "asuka/Dialect/AsukaTriton/IR/AsukaTritonTypes.cpp.inc"

void AsukaTritonDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "asuka/Dialect/AsukaTriton/IR/AsukaTritonTypes.cpp.inc"
      >();
}
