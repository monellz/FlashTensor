#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

// Dialect
#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"
#include "asuka/Dialect/AsukaTriton/IR/AsukaTritonDialect.h"

namespace py = pybind11;

namespace mlir::asuka {

namespace {

class OpBuilderWithLoc {
public:
  OpBuilderWithLoc(MLIRContext *context) {
    builder = std::make_unique<OpBuilder>(context);
    lastLoc = std::make_unique<Location>(builder->getUnknownLoc());
  }

  OpBuilder &getBuilder() { return *builder; }

  void setLastLoc(Location loc) { lastLoc = std::make_unique<Location>(loc); }

  void setLastLoc(const std::string &fileName, int line, int column) {
    auto context = builder->getContext();
    setLastLoc(FileLineColLoc::get(context, fileName, line, column));
  }

  Location getLastLoc() {
    assert(lastLoc);
    return *lastLoc;
  }

  void setInsertionPointToStart(Block &block) {
    if (!block.empty())
      setLastLoc(block.begin()->getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->setInsertionPointToStart(&block);
  }

  void setInsertionPointToEnd(Block &block) {
    if (!block.empty())
      setLastLoc(block.back().getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->setInsertionPointToEnd(&block);
  }

  void setInsertionPointAfter(Operation &op) {
    setLastLoc(op.getLoc());
    builder->setInsertionPointAfter(&op);
  }

  void restoreInsertionPoint(OpBuilder::InsertPoint pt) {
    if (pt.isSet() && pt.getPoint() != pt.getBlock()->end())
      setLastLoc(pt.getPoint()->getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->restoreInsertionPoint(pt);
  }

  Operation *clone(Operation &op) { return builder->clone(op); }

  template <typename OpTy, typename... Args>
  OpTy create(Args &&...args) {
    auto loc = getLastLoc();
    return builder->create<OpTy>(loc, std::forward<Args>(args)...);
  }

  // Overload to create or fold a single result operation.
  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<OpTrait::OneResult>(), Value> createOrFold(Args &&...args) {
    auto loc = getLastLoc();
    return builder->createOrFold<OpTy>(loc, std::forward<Args>(args)...);
  }

  // Overload to create or fold a zero result operation.
  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<OpTrait::ZeroResults>(), OpTy> createOrFold(Args &&...args) {
    auto loc = getLastLoc();
    return builder->createOrFold<OpTy>(loc, std::forward<Args>(args)...);
  }

private:
  std::unique_ptr<OpBuilder> builder;
  std::unique_ptr<Location> lastLoc;
};

} // namespace

using ret = py::return_value_policy;

void init_ffi_ir_asuka(py::module_ &&m) {
  m.def("load_dialects", [](MLIRContext &context) {
    mlir::DialectRegistry registry;
    // clang-format off
    registry.insert<mlir::asuka::AsukaDialect,
                    mlir::asuka::triton::AsukaTritonDialect,

                    // mlir::triton::TritonDialect,
                    // mlir::triton::gpu::TritonGPUDialect,
                    // mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
                    // mlir::triton::nvgpu::NVGPUDialect,

                    mlir::func::FuncDialect,
                    mlir::arith::ArithDialect,
                    mlir::math::MathDialect,
                    mlir::scf::SCFDialect,
                    mlir::cf::ControlFlowDialect>();
    // clang-format on
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  py::enum_<ReduceType>(m, "REDUCE_TYPE", py::module_local())
      .value("REDUCE_ADD", ReduceType::ADD)
      .value("REDUCE_MUL", ReduceType::MUL)
      .value("REDUCE_ANY", ReduceType::ANY)
      .export_values();
  py::enum_<CmpType>(m, "CMP_TYPE", py::module_local())
      .value("CMP_GE", CmpType::GE)
      .value("CMP_GT", CmpType::GT)
      .export_values();

  py::class_<KernelOp, OpState>(m, "kernel", py::module_local())
      .def("arg",
           [](KernelOp &self, unsigned idx) -> BlockArgument {
             if (idx >= self.getNumArguments())
               throw pybind11::index_error("Kernel argument index out of range");
             return self.getArgument(idx);
           })
      .def("dump", &KernelOp::dump)
      // FIXME: memory leak?
      .def("erase", [](KernelOp &self) { self->erase(); })
      .def("get_parallelism", &KernelOp::getParallelism)
      .def("get_num_arguments", &KernelOp::getNumArguments)
      .def(
          "get_callable_region", [](KernelOp &self) -> Region * { return self.getCallableRegion(); }, ret::reference)
      .def(
          "add_entry_block", [](KernelOp &self) -> Block * { return self.addEntryBlock(); }, ret::reference)
      .def_property_readonly("type", &KernelOp::getFunctionType);

  py::class_<ReshapeOp, OpState>(m, "reshape", py::module_local())
      .def("from_operation", [](Operation &operation) -> py::object {
        if (auto op = dyn_cast<ReshapeOp>(operation))
          return py::cast(op);
        else
          return py::none();
      });

  py::class_<PermuteOp, OpState>(m, "permute", py::module_local())
      .def("from_operation",
           [](Operation &operation) -> py::object {
             if (auto op = dyn_cast<PermuteOp>(operation))
               return py::cast(op);
             else
               return py::none();
           })
      .def("get_dims", [](PermuteOp &self) -> std::vector<int64_t> { return self.getDims(); });

  py::class_<DivOp, OpState>(m, "div", py::module_local())
      .def("from_operation", [](Operation &operation) -> py::object {
        if (auto op = dyn_cast<DivOp>(operation))
          return py::cast(op);
        else
          return py::none();
      });

  py::class_<DotOp, OpState>(m, "dot", py::module_local())
      .def("from_operation", [](Operation &operation) -> py::object {
        if (auto op = dyn_cast<DotOp>(operation))
          return py::cast(op);
        else
          return py::none();
      });

  py::class_<ConvertOp, OpState>(m, "convert", py::module_local())
      .def("from_operation",
           [](Operation &operation) -> py::object {
             if (auto op = dyn_cast<ConvertOp>(operation))
               return py::cast(op);
             else
               return py::none();
           })
      .def("get_dst_type", &ConvertOp::getDstType);

  py::class_<ReduceOp, OpState>(m, "reduce", py::module_local())
      .def("from_operation",
           [](Operation &operation) -> py::object {
             if (auto op = dyn_cast<ReduceOp>(operation))
               return py::cast(op);
             else
               return py::none();
           })
      .def("get_reduce_dim",
           [](ReduceOp &self) -> int64_t {
             int64_t dim = self.getReduceDimensionAttr().getValue().getSExtValue();
             return dim;
           })
      .def("get_keep_dim", &ReduceOp::getKeepDim)
      .def("is_reduce_add", [](ReduceOp &op) { return op.getReduceTypeAttr().getValue() == ReduceType::ADD; })
      .def("is_reduce_mul", [](ReduceOp &op) { return op.getReduceTypeAttr().getValue() == ReduceType::MUL; });
  py::class_<SoftmaxOp, OpState>(m, "softmax", py::module_local())
      .def("from_operation",
           [](Operation &operation) -> py::object {
             if (auto op = dyn_cast<SoftmaxOp>(operation))
               return py::cast(op);
             else
               return py::none();
           })
      .def("get_reduce_dim", [](SoftmaxOp &self) -> int64_t {
        int64_t dim = self.getReduceDimensionAttr().getValue().getSExtValue();
        return dim;
      });
  py::class_<TriluOp, OpState>(m, "trilu", py::module_local())
      .def("from_operation",
           [](Operation &operation) -> py::object {
             if (auto op = dyn_cast<TriluOp>(operation))
               return py::cast(op);
             else
               return py::none();
           })
      .def("get_diagonal", &TriluOp::getDiagonal)
      .def("get_is_upper", &TriluOp::getIsUpper)
      .def("get_shape", [](TriluOp &self) -> std::vector<int64_t> { return self.getShape(); })
      .def("is_inf_val", &TriluOp::isInfVal)
      .def("is_neg_inf_val", &TriluOp::isNegInfVal);
}

void init_ffi_ir_builder(py::module_ &m) {
  py::class_<OpBuilder::InsertPoint>(m, "InsertPoint", py::module_local());

  py::class_<OpBuilderWithLoc>(m, "builder")
      .def(py::init<MLIRContext *>())
      .def("create_module", [](OpBuilderWithLoc &self) -> ModuleOp { return self.create<ModuleOp>(); })
      // insertion block/point
      .def("set_insertion_point_to_start",
           [](OpBuilderWithLoc &self, Block &block) -> void { self.setInsertionPointToStart(block); })
      .def("set_insertion_point_to_end",
           [](OpBuilderWithLoc &self, Block &block) { self.setInsertionPointToEnd(block); })
      .def("set_insertion_point_after", [](OpBuilderWithLoc &self, Operation &op) { self.setInsertionPointAfter(op); })
      .def(
          "get_insertion_block",
          [](OpBuilderWithLoc &self) -> Block * { return self.getBuilder().getInsertionBlock(); }, ret::reference)
      .def("get_insertion_point", [](OpBuilderWithLoc &self) { return self.getBuilder().saveInsertionPoint(); })
      .def("restore_insertion_point",
           [](OpBuilderWithLoc &self, OpBuilder::InsertPoint pt) { self.restoreInsertionPoint(pt); })
      // Attr
      .def("get_bool_attr", [](OpBuilderWithLoc &self, bool value) { return self.getBuilder().getBoolAttr(value); })
      .def("get_int32_attr",
           [](OpBuilderWithLoc &self, int32_t value) { return self.getBuilder().getI32IntegerAttr(value); })
      // constant
      .def("get_int64",
           [](OpBuilderWithLoc &self, int64_t v) -> Value {
             return Value(self.create<arith::ConstantIntOp>(v, self.getBuilder().getI64Type()));
           })
      .def("get_f16",
           [](OpBuilderWithLoc &self, float v) -> Value {
             return self.create<arith::ConstantOp>(self.getBuilder().getF16FloatAttr(v));
           })
      .def("get_f32",
           [](OpBuilderWithLoc &self, float v) -> Value {
             return self.create<arith::ConstantOp>(self.getBuilder().getF32FloatAttr(v));
           })
      .def("get_f64",
           [](OpBuilderWithLoc &self, double v) -> Value {
             return self.create<arith::ConstantOp>(self.getBuilder().getF64FloatAttr(v));
           })
      .def("get_f16_tensor",
           [](OpBuilderWithLoc &self, std::vector<float> &v, std::vector<int64_t> &shape) -> Value {
             auto elem_type = self.getBuilder().getF16Type();
             auto tensor_type = RankedTensorType::get(shape, elem_type);
             llvm::SmallVector<Attribute> vals;
             for (auto e : v) {
               vals.push_back(self.getBuilder().getF16FloatAttr(e));
             }
             auto data = DenseElementsAttr::get(tensor_type, vals);
             return self.create<arith::ConstantOp>(tensor_type, data);
           })
      .def("get_f32_tensor",
           [](OpBuilderWithLoc &self, std::vector<float> &v, std::vector<int64_t> &shape) -> Value {
             auto elem_type = self.getBuilder().getF32Type();
             auto tensor_type = RankedTensorType::get(shape, elem_type);
             llvm::SmallVector<Attribute> vals;
             for (auto e : v) {
               vals.push_back(self.getBuilder().getF32FloatAttr(e));
             }
             auto data = DenseElementsAttr::get(tensor_type, vals);
             return self.create<arith::ConstantOp>(tensor_type, data);
           })
      .def("get_f64_tensor",
           [](OpBuilderWithLoc &self, std::vector<double> &v, std::vector<int64_t> &shape) -> Value {
             auto elem_type = self.getBuilder().getF64Type();
             auto tensor_type = RankedTensorType::get(shape, elem_type);
             llvm::SmallVector<Attribute> vals;
             for (auto e : v) {
               vals.push_back(self.getBuilder().getF64FloatAttr(e));
             }
             auto data = DenseElementsAttr::get(tensor_type, vals);
             return self.create<arith::ConstantOp>(tensor_type, data);
           })
      .def("get_int64_tensor",
           [](OpBuilderWithLoc &self, std::vector<int64_t> &v, std::vector<int64_t> &shape) -> Value {
             auto elem_type = self.getBuilder().getI64Type();
             auto tensor_type = RankedTensorType::get(shape, elem_type);
             llvm::SmallVector<Attribute> vals;
             for (auto e : v) {
               vals.push_back(self.getBuilder().getI64IntegerAttr(e));
             }
             auto data = DenseElementsAttr::get(tensor_type, vals);
             return self.create<arith::ConstantOp>(tensor_type, data);
           })

      // type
      .def("get_bool_ty", [](OpBuilderWithLoc &self) -> Type { return self.getBuilder().getI8Type(); })
      .def("get_int64_ty", [](OpBuilderWithLoc &self) -> Type { return self.getBuilder().getI64Type(); })
      .def("get_f64_ty", [](OpBuilderWithLoc &self) -> Type { return self.getBuilder().getF64Type(); })
      .def("get_f16_ty", [](OpBuilderWithLoc &self) -> Type { return self.getBuilder().getF16Type(); })
      .def("get_bf16_ty", [](OpBuilderWithLoc &self) -> Type { return self.getBuilder().getBF16Type(); })
      .def("get_f32_ty", [](OpBuilderWithLoc &self) -> Type { return self.getBuilder().getF32Type(); })
      .def("get_ranked_tensor_ty",
           [](OpBuilderWithLoc &self, Type &elementType, std::vector<int64_t> &shape) -> Type {
             return RankedTensorType::get(shape, elementType);
           })
      .def("get_function_ty",
           [](OpBuilderWithLoc &self, std::vector<Type> inTypes, std::vector<Type> outTypes) -> Type {
             return self.getBuilder().getFunctionType(inTypes, outTypes);
           })

      // locs
      .def("set_loc", [](OpBuilderWithLoc &self, Location loc) { self.setLastLoc(loc); })
      .def("set_loc", [](OpBuilderWithLoc &self, const std::string &fileName, int line,
                         int column) { self.setLastLoc(fileName, line, column); })
      .def("get_loc", [](OpBuilderWithLoc &self) -> Location { return self.getLastLoc(); })

      // op
      .def(
          "clone", [](OpBuilderWithLoc &self, Operation &op) -> Operation * { return self.clone(op); }, ret::reference)
      .def("get_or_insert_function",
           [](OpBuilderWithLoc &self, ModuleOp &module, std::string &funcName, Type &funcType) -> func::FuncOp {
             if (Operation *funcOperation = module.lookupSymbol(funcName))
               return llvm::dyn_cast<func::FuncOp>(funcOperation);
             if (auto funcTy = dyn_cast<FunctionType>(funcType)) {
               return self.create<func::FuncOp>(funcName, funcTy);
             }
             throw std::invalid_argument("invalid function type");
           })
      .def("get_or_insert_kernel",
           [](OpBuilderWithLoc &self, ModuleOp &module, std::string &kernelName, Type &funcType) -> KernelOp {
             if (Operation *kernelOperation = module.lookupSymbol(kernelName))
               return llvm::dyn_cast<KernelOp>(kernelOperation);
             if (auto funcTy = dyn_cast<FunctionType>(funcType)) {
               return self.create<KernelOp>(kernelName, funcTy);
             }
             throw std::invalid_argument("invalid kernel type");
           })
      .def(
          "create_block",
          [](OpBuilderWithLoc &self) -> Block * {
            Region *parent = self.getBuilder().getBlock()->getParent();
            return self.getBuilder().createBlock(parent);
          },
          ret::reference)
      .def(
          "create_block_with_parent",
          [](OpBuilderWithLoc &self, Region &parent, std::vector<Type> &argTypes) -> Block * {
            // TODO: update arg loc
            auto loc = self.getBuilder().getUnknownLoc();
            llvm::SmallVector<Location, 8> argLocs(argTypes.size(), loc);
            return self.getBuilder().createBlock(&parent, {}, argTypes, argLocs);
          },
          ret::reference)
      .def(
          "new_block", [](OpBuilderWithLoc &self) -> Block * { return new Block(); }, ret::reference)
      .def(
          "ret",
          [](OpBuilderWithLoc &self, std::vector<Value> &vals) -> OpState { return self.create<func::ReturnOp>(vals); })

      // asuka
      .def("create_add",
           [](OpBuilderWithLoc &self, Value &lhs, Value &rhs) -> Value { return self.create<AddOp>(lhs, rhs); })
      .def("create_sub",
           [](OpBuilderWithLoc &self, Value &lhs, Value &rhs) -> Value { return self.create<SubOp>(lhs, rhs); })
      .def("create_div",
           [](OpBuilderWithLoc &self, Value &lhs, Value &rhs) -> Value { return self.create<DivOp>(lhs, rhs); })
      .def("create_pow",
           [](OpBuilderWithLoc &self, Value &lhs, Value &rhs) -> Value { return self.create<PowOp>(lhs, rhs); })
      .def("create_mul",
           [](OpBuilderWithLoc &self, Value &lhs, Value &rhs) -> Value { return self.create<MulOp>(lhs, rhs); })
      .def("create_dot",
           [](OpBuilderWithLoc &self, Value &lhs, Value &rhs) -> Value { return self.create<DotOp>(lhs, rhs); })
      .def("create_tanh", [](OpBuilderWithLoc &self, Value &operand) -> Value { return self.create<TanhOp>(operand); })
      .def("create_exp", [](OpBuilderWithLoc &self, Value &operand) -> Value { return self.create<ExpOp>(operand); })
      .def("create_log", [](OpBuilderWithLoc &self, Value &operand) -> Value { return self.create<LogOp>(operand); })
      .def("create_neg", [](OpBuilderWithLoc &self, Value &operand) -> Value { return self.create<NegOp>(operand); })
      .def("create_abs", [](OpBuilderWithLoc &self, Value &operand) -> Value { return self.create<AbsOp>(operand); })
      .def("create_cmp",
           [](OpBuilderWithLoc &self, Value &lhs, Value &rhs, CmpType cmp_type) -> Value {
             auto ty = CmpTypeAttr::get(self.getBuilder().getContext(), cmp_type);
             return self.create<CmpOp>(lhs, rhs, ty);
           })
      .def("create_avgpool",
           [](OpBuilderWithLoc &self, Value &operand, std::vector<int64_t> &kernel_size, std::vector<int64_t> &stride,
              std::vector<int64_t> &padding, bool ceil_mode, bool count_include_pad) -> Value {
             return self.create<AvgPoolOp>(operand, kernel_size, stride, padding, ceil_mode, count_include_pad);
           })
      .def("create_softmax",
           [](OpBuilderWithLoc &self, Value &operand, int64_t dim) -> Value {
             return self.create<SoftmaxOp>(operand, self.getBuilder().getI64IntegerAttr(dim));
           })
      .def("create_reduce",
           [](OpBuilderWithLoc &self, Value &operand, int64_t dim, ReduceType reduce_type, bool keep_dim) -> Value {
             auto ty = ReduceTypeAttr::get(self.getBuilder().getContext(), reduce_type);
             return self.create<ReduceOp>(operand, self.getBuilder().getI64IntegerAttr(dim), ty,
                                          self.getBuilder().getBoolAttr(keep_dim));
           })
      .def("create_permute",
           [](OpBuilderWithLoc &self, Value &operand, std::vector<int64_t> &dims) -> Value {
             return self.create<PermuteOp>(operand, dims);
           })
      .def("create_reshape",
           [](OpBuilderWithLoc &self, Value &operand, std::vector<int64_t> &shape) -> Value {
             auto elem_type = cast<RankedTensorType>(operand.getType()).getElementType();
             return self.create<ReshapeOp>(RankedTensorType::get(shape, elem_type), operand);
           })
      .def("create_convert",
           [](OpBuilderWithLoc &self, Value &operand, Type &type) -> Value {
             return self.create<ConvertOp>(operand, type);
           })
      .def("create_inf_trilu",
           [](OpBuilderWithLoc &self, int64_t diagonal, bool is_upper, std::vector<int64_t> &shape, Type &type,
              bool negative) -> Value {
             if (auto float_type = dyn_cast<FloatType>(type)) {
               return self.create<TriluOp>(diagonal, is_upper, shape, float_type, negative);
             }
             throw std::invalid_argument("type is not float-like");
           })
      .def("create_return",
           [](OpBuilderWithLoc &self, std::vector<Value> &operands) { self.create<ReturnOp>(operands); });
}

void init_ffi_ir_common_op(py::module_ &m) {
  // dynamic_attr is used to transfer ownership of the MLIR context to the module
  py::class_<ModuleOp, OpState>(m, "module", py::module_local(), py::dynamic_attr())
      .def("dump", &ModuleOp::dump)
      .def("__str__",
           [](ModuleOp &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             auto printingFlags = OpPrintingFlags();
             printingFlags.enableDebugInfo();
             self.print(os, printingFlags);
             return str;
           })
      .def("push_back", [](ModuleOp &self, func::FuncOp &funcOp) -> void { self.push_back(funcOp); })
      .def("push_back", [](ModuleOp &self, KernelOp &kernelOp) -> void { self.push_back(kernelOp); })
      .def("has_function",
           [](ModuleOp &self, std::string &funcName) -> bool {
             if (self.lookupSymbol(funcName))
               return true;
             return false;
           })
      .def("get_function",
           [](ModuleOp &self, std::string &funcName) -> func::FuncOp {
             return self.lookupSymbol<func::FuncOp>(funcName);
           })
      .def("get_kernel",
           [](ModuleOp &self, std::string &kernelName) -> KernelOp { return self.lookupSymbol<KernelOp>(kernelName); })
      .def("erase_kernel",
           [](ModuleOp &self, std::string &kernelName) -> void {
            auto kernel = self.lookupSymbol<KernelOp>(kernelName);
            assert(kernel->use_empty() && "KernelOp still has uses!");
            kernel->erase();
          })
      .def("get_int_attr",
           [](ModuleOp &self, std::string name) -> py::object {
             auto ret = self->getAttrOfType<IntegerAttr>(name);
             if (!ret)
               return py::none();
             return py::int_(ret.getInt());
           })
      .def("walk", [](ModuleOp &self, const std::function<void(Operation *)> &fn) { self.walk(fn); });

  py::class_<arith::ConstantOp, OpState>(m, "constant", py::module_local())
      .def("get_splat_float_value", [](arith::ConstantOp &self) -> py::object {
        if (auto dense_attr = dyn_cast<DenseElementsAttr>(self.getValueAttr())) {
          if (dense_attr.isSplat()) {
            double val = dense_attr.getSplatValue<FloatAttr>().getValueAsDouble();
            return py::float_(val);
          }
        }
        return py::none();
      });

  py::class_<func::FuncOp, OpState>(m, "function", py::module_local())
      .def("arg",
           [](func::FuncOp &self, unsigned idx) -> BlockArgument {
             if (idx >= self.getNumArguments())
               throw pybind11::index_error("Function argument index out of range");
             return self.getArgument(idx);
           })
      .def("get_num_arguments", &func::FuncOp::getNumArguments)
      .def(
          "get_callable_region", [](func::FuncOp &self) -> Region * { return self.getCallableRegion(); },
          ret::reference)
      .def(
          "get_terminator",
          [](func::FuncOp &self) -> Operation * { return self.getCallableRegion()->front().getTerminator(); },
          ret::reference)
      .def(
          "add_entry_block", [](func::FuncOp &self) -> Block * { return self.addEntryBlock(); }, ret::reference)
      .def(
          "set_arg_attr",
          [](func::FuncOp &self, int arg_no, const std::string &name, int val) {
            if (arg_no >= self.getNumArguments())
              throw pybind11::index_error("Function argument index out of range");
            // set arg attributes "name" to value "val"
            auto attrTy = IntegerType::get(self.getContext(), 32);
            self.setArgAttr(arg_no, name, IntegerAttr::get(attrTy, val));
          },
          ret::reference)
      .def_property_readonly("type", &func::FuncOp::getFunctionType)
      .def("reset_type", &func::FuncOp::setType);
}

void init_ffi_ir_operation(py::module_ &m) {
  py::class_<OpState>(m, "OpState", py::module_local())
      .def("set_attr", [](OpState &self, std::string &name, Attribute &attr) -> void { self->setAttr(name, attr); })
      .def("get_num_results", [](OpState &self) -> unsigned { return self->getNumResults(); })
      .def("get_result",
           [](OpState &self, unsigned idx) -> Value {
             if (idx >= self->getNumResults())
               throw pybind11::index_error("Op result index out of range");
             return self->getResult(idx);
           })
      .def(
          "get_region",
          [](OpState &self, unsigned idx) -> Region & {
            if (idx >= self->getNumRegions())
              throw pybind11::index_error("Op region index out of range");
            return self->getRegion(idx);
          },
          ret::reference)
      .def("dump", [](OpState &self) { self->dump(); })
      .def("__str__",
           [](OpState &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             auto printingFlags = OpPrintingFlags();
             printingFlags.enableDebugInfo();
             self->print(os, printingFlags);
             return str;
           })
      .def("append_operand", [](OpState &self, Value &val) { self->insertOperands(self->getNumOperands(), val); })
      .def("verify", [](OpState &self) -> bool { return succeeded(verify(self.getOperation())); });

  py::class_<Operation, std::unique_ptr<Operation, py::nodelete>>(m, "operation", py::module_local())
      .def("__str__",
           [](Operation &self) {
             llvm::StringRef opName = self.getName().getStringRef();
             return opName.str();
           })
      .def("__repr__",
           [](Operation &self) {
             llvm::StringRef opName = self.getName().getStringRef();
             return opName.str();
           })
      .def("get_name",
           [](Operation &self) {
             llvm::StringRef opName = self.getName().getStringRef();
             return opName.str();
           })
      .def("get_num_operands", &Operation::getNumOperands)
      .def("get_operand", &Operation::getOperand)
      .def("set_operand", &Operation::setOperand)
      .def("get_num_results", &Operation::getNumResults)
      .def("get_result", &Operation::getResult)
      .def("get_num_regions", &Operation::getNumRegions)
      .def("get_region", &Operation::getRegion, ret::reference)
      .def("get_block", &Operation::getBlock, ret::reference)
      .def("get_str_attr",
           [](Operation &self, const std::string &name) -> py::object {
             auto ret = self.getAttrOfType<StringAttr>(name);
             if (!ret)
               return py::none();
             return py::str(ret.getValue().str());
           })
      .def("get_flat_symbol_ref_attr",
           [](Operation &self, const std::string &name) -> py::object {
             auto ret = self.getAttrOfType<FlatSymbolRefAttr>(name);
             if (!ret)
               return py::none();
             return py::str(ret.getValue().str());
           })
      .def("is_constant_op",
           [](Operation &self) -> bool { return isa<arith::ConstantOp>(self) || isa<arith::ConstantIntOp>(self); })
      .def("to_constant_op",
           [](Operation &self) -> py::object {
             if (auto op = dyn_cast<arith::ConstantOp>(self)) {
               return py::cast(op);
             }
             return py::none();
           })
      .def("get_flops", [](Operation &self) -> int64_t {
        if (auto flops_interface = dyn_cast<FlopsInterface>(self)) {
          return flops_interface.getFlops();
        } else {
          return 0;
        }
      });
}

void init_ffi_ir_common(py::module_ &m) {
  py::class_<MLIRContext>(m, "context", py::module_local())
      .def(py::init<>())
      .def("disable_multithreading", [](MLIRContext &self) { self.disableMultithreading(); });

  py::class_<Type>(m, "type", py::module_local())
      .def("__str__",
           [](Type &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return os.str();
           })
      .def("to_ranked_tensor_ty",
           [](Type &self) -> py::object {
             if (auto ty = dyn_cast<RankedTensorType>(self)) {
               return py::cast(ty);
             }
             return py::none();
           })
      .def("is_f16", &Type::isF16)
      .def("is_f32", &Type::isF32)
      .def("is_f64", &Type::isF64)
      .def("is_float", [](Type &self) -> bool { return isa<FloatType>(self); })
      .def("is_index", &Type::isIndex)
      .def("is_int", [](Type &self) -> bool { return isa<IntegerType>(self); })
      .def("is_signed_int", py::overload_cast<>(&Type::isSignedInteger, py::const_))
      .def("is_unsigned_int", py::overload_cast<>(&Type::isUnsignedInteger, py::const_))
      .def("is_signless_int", py::overload_cast<>(&Type::isSignlessInteger, py::const_));

  py::class_<RankedTensorType, Type>(m, "ranked_tensor", py::module_local())
      .def("get_rank", &RankedTensorType::getRank)
      .def("get_shape", [](RankedTensorType &self) -> std::vector<int64_t> { return self.getShape(); })
      .def("get_element_ty", &RankedTensorType::getElementType);

  py::class_<FunctionType>(m, "function_type", py::module_local()).def("param_types", [](FunctionType &self) {
    return std::vector<Type>(self.getInputs().begin(), self.getInputs().end());
  });

  py::class_<Location>(m, "location", py::module_local()).def("__str__", [](Location &self) {
    std::string str;
    llvm::raw_string_ostream os(str);
    self.print(os);
    return os.str();
  });

  py::class_<Value>(m, "value", py::module_local())
      .def("get_context", &Value::getContext)
      .def(
          "get_defining_op", [](Value &self) -> Operation * { return self.getDefiningOp(); }, ret::reference)
      .def("get_users",
           [](Value &self) -> py::list {
             py::list ret;
             for (auto &use : self.getUses()) {
               ret.append(use.getOwner());
             }
             return ret;
           })
      .def("replace_all_uses_with", [](Value &self, Value &newValue) { self.replaceAllUsesWith(newValue); })
      .def("get_type", &Value::getType)
      .def("id", [](Value &self) { return (uint64_t)self.getImpl(); })
      .def("__eq__", &Value::operator==)
      .def("__hash__", [](Value &self) -> unsigned { return hash_value(self); });

  py::class_<OpResult, Value>(m, "op_result", py::module_local());

  py::class_<BlockArgument, Value>(m, "block_argument", py::module_local());

  py::class_<Region>(m, "region", py::module_local())
      .def("get_parent_region", &Region::getParentRegion, ret::reference)
      .def("size", [](Region &self) { return self.getBlocks().size(); })
      .def(
          "front", [](Region &self) -> Block * { return &self.front(); }, ret::reference)
      .def("empty", &Region::empty)
      .def("walk", [](Block &self, const std::function<void(Operation *)> &fn) { self.walk(fn); })
      .def("id", [](Region &self) { return (uint64_t)&self; });

  py::class_<Block>(m, "block", py::module_local())
      .def("arg",
           [](Block &self, int index) -> BlockArgument {
             if (index >= self.getNumArguments())
               throw pybind11::index_error("Block argument index out of range");
             return self.getArgument(index);
           })
      .def("add_argument",
           [](Block &self, Type ty) {
             auto loc = UnknownLoc::get(ty.getContext());
             self.addArgument(ty, loc);
           })
      .def("get_num_arguments", &Block::getNumArguments)
      .def("get_argument", &Block::getArgument)
      .def("dump", &Block::dump)
      .def("move_before", [](Block &self, Block &dst) { self.moveBefore(&dst); })
      .def("insert_before", &Block::insertBefore)
      .def("get_parent", &Block::getParent, ret::reference)
      .def("merge_block_before",
           [](Block &self, Block &dst) {
             // ref: RewriterBase::mergeBlocks()
             if (self.getNumArguments() != 0)
               throw std::runtime_error("This block has arguments, don't merge");
             dst.getOperations().splice(dst.begin(), self.getOperations());
             self.dropAllUses();
             self.erase();
           })
      .def("replace_use_in_block_with",
           [](Block &self, Value &v, Value &newVal) {
             v.replaceUsesWithIf(newVal, [&](OpOperand &operand) {
               Operation *user = operand.getOwner();
               Block *currentBlock = user->getBlock();
               while (currentBlock) {
                 if (currentBlock == &self)
                   return true;
                 // Move up one level
                 currentBlock = currentBlock->getParent()->getParentOp()->getBlock();
               }
               return false;
             });
           })
      .def("__str__",
           [](Block &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return str;
           })
      .def("has_terminator", [](Block &self) { return !self.empty() && self.back().hasTrait<OpTrait::IsTerminator>(); })
      .def("has_return", [](Block &self) { return !self.empty() && self.back().hasTrait<OpTrait::ReturnLike>(); })
      .def("erase", [](Block &self) { self.erase(); })
      .def("walk", [](Block &self, const std::function<void(Operation *)> &fn) { self.walk(fn); })
      .def("id", [](Block &self) { return (uint64_t)&self; });

  py::class_<Attribute>(m, "attribute", py::module_local());
  py::class_<IntegerAttr, Attribute>(m, "integer_attr", py::module_local());
  py::class_<BoolAttr, Attribute>(m, "bool_attr", py::module_local());
}

void init_ffi_ir(py::module_ &&m) {
  init_ffi_ir_common(m);
  init_ffi_ir_operation(m);
  init_ffi_ir_common_op(m);
  init_ffi_ir_builder(m);
  init_ffi_ir_asuka(m.def_submodule("asuka"));
}

} // namespace mlir::asuka