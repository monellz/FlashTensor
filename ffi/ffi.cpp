#include "ffi.h"

PYBIND11_MODULE(asuka_ffi, m) {
  m.doc() = "TODO";
  mlir::asuka::init_ffi_ir(m.def_submodule("ir"));
  mlir::asuka::init_ffi_passes(m.def_submodule("passes"));
}