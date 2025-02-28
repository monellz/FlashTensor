#ifndef ASUKA_TRANSLATE_H
#define ASUKA_TRANSLATE_H

#include <string>

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/TypeSwitch.h"
#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"
#include "asuka/Dialect/AsukaTriton/IR/AsukaTritonDialect.h"

namespace mlir::asuka {

::mlir::LogicalResult module_to_py_impl(::mlir::ModuleOp, ::mlir::raw_ostream &, bool benchmark = true);
::mlir::LogicalResult kernel_to_py_impl(::mlir::asuka::KernelOp, ::mlir::raw_ostream &, bool import = true,
                                        bool benchmark = true);

} // namespace mlir::asuka

#endif // ASUKA_TRANSLATE_H