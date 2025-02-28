#include "asuka/Translate/translate.h"

#include "dbg.h"

// FIXME: use twine instead of std::string
using llvm::Twine;

namespace mlir::asuka {

class PyKernelTranslator {
public:
  struct CounterGuard {
    explicit CounterGuard(int &counter) : counter(counter), preserved(counter) { counter = 0; }
    ~CounterGuard() { counter = preserved; }
    int &counter;
    int preserved;
  };
  struct Indent {
    explicit Indent(int &level, ::mlir::raw_ostream &os) : level(level), os(os) { ++level; }
    ::mlir::raw_ostream &operator()(bool indented = true) {
      if (indented) {
        for (int i = 0; i < level; ++i) {
          os << "  ";
        }
      }
      return os;
    }

    ~Indent() { --level; }
    int &level;
    ::mlir::raw_ostream &os;
  };

  int cur_level;
  llvm::raw_ostream &os;
  llvm::ScopedHashTable<mlir::Value, std::string> symbol_table;
  int name_counter = 0;

  explicit PyKernelTranslator(::mlir::raw_ostream &os) : cur_level(0), os(os) {}
  int get_counter() { return name_counter++; }
  void reset_counter() { name_counter = 0; }

  std::string get_name(Twine tag = "reg") {
    auto counter = Twine(get_counter());
    auto name = tag + "_" + counter;
    return name.str();
  }

  std::string get_torch_dtype(Type type) {
    if (type.isF16()) {
      return "torch.float16";
    } else if (type.isF32()) {
      return "torch.float32";
    } else if (type.isF64()) {
      return "torch.float64";
    } else if (type.isInteger(8)) {
      return "torch.bool";
    } else {
      llvm::errs() << "type: " << type << "\n";
      llvm_unreachable("not support");
    }
  }

  std::string get_tl_dtype(Type type) {
    if (type.isF16()) {
      return "tl.float16";
    } else if (type.isF32()) {
      return "tl.float32";
    } else if (type.isF64()) {
      return "tl.float64";
    } else if (type.isInteger(8)) {
      return "tl.int8";
    } else if (type.isInteger(1)) {
      return "tl.int1";
    } else {
      llvm::errs() << "type: " << type << "\n";
      llvm_unreachable("not support");
    }
  }

  void trans(Block *block) {
    llvm::ScopedHashTableScope<Value, std::string> kernel_scope(symbol_table);
    Indent indent(cur_level, os);

    block->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto const_op = dyn_cast<arith::ConstantOp>(op)) {
        auto v = const_op.getValue();
        auto name = get_name("const");
        if (auto dense_attr = dyn_cast<DenseElementsAttr>(v)) {
          auto shape = dense_attr.getType().getShape();
          auto elem_type = dense_attr.getElementType();
          assert(shape.size() == 1 && shape[0] == 1);
          if (isa<FloatType>(elem_type)) {
            auto val = dense_attr.getValues<APFloat>()[0];
            if (val.isPosInfinity()) {
              indent() << name << " = " << "float('inf')\n";
            } else if (val.isNegInfinity()) {
              indent() << name << " = " << "float('-inf')\n";
            } else {
              indent() << name << " = " << val.convertToDouble() << "\n";
            }
          } else if (isa<IntegerType>(elem_type)) {
            auto val = dense_attr.getValues<IntegerAttr>()[0];
            indent() << name << " = " << val.getInt() << "\n";
          } else {
            llvm_unreachable("not support");
          }
        } else if (auto float_attr = dyn_cast<FloatAttr>(v)) {
          auto val = float_attr.getValue();
          if (val.isPosInfinity()) {
            indent() << name << " = " << "float('inf')\n";
          } else if (val.isNegInfinity()) {
            indent() << name << " = " << "float('-inf')\n";
          } else {
            indent() << name << " = " << val.convertToDouble() << "\n";
          }
        } else if (auto int_attr = dyn_cast<IntegerAttr>(v)) {
          indent() << name << " = " << int_attr.getInt() << "\n";
        } else {
          llvm_unreachable("not support");
        }
        symbol_table.insert(const_op.getResult(), name);
      } else if (auto zero_op = dyn_cast<ZeroOp>(op)) {
        assert(zero_op.getResult().hasOneUse());
        auto shape = zero_op.getShape();
        auto elem_type = zero_op.getElementType();
        auto name = get_name("zero");
        indent() << name << " = tl.zeros([";
        llvm::interleaveComma(shape, indent(false));
        indent(false) << "], dtype=" << get_tl_dtype(elem_type) << ")\n";

        symbol_table.insert(zero_op.getResult(), name);
      } else if (auto convert_op = dyn_cast<ConvertOp>(op)) {
        auto arg = convert_op.getOperand();
        auto arg_type = cast<RankedTensorType>(convert_op.getType());
        auto arg_shape = arg_type.getShape();
        auto dst_elem_type = convert_op.getDstType();
        auto name = get_name("converted");
        if (arg_shape.size() == 1 && arg_shape[0] == 1) {
          // do nothing for scalar
          indent() << name << " = " << symbol_table.lookup(arg) << "\n";
        } else {
          indent() << name << " = " << symbol_table.lookup(arg) << ".to(" << get_tl_dtype(dst_elem_type) << ")\n";
        }
        symbol_table.insert(convert_op.getResult(), name);
      } else if (auto mask_op = dyn_cast<MaskOp>(op)) {
        // try convert mask to zero and where
        // TODO: there need a dataflow analysis to get all possible mask_yield operands
        SmallVector<Operation *> ops;
        mask_op.getRegion().front().walk([&](Operation *inner_op) { ops.push_back(inner_op); });
        // FIXME: hard code !
        auto is_where_op = [&]() -> bool {
          if (ops.size() != 6)
            return false;
          if (!isa<arith::AddIOp>(ops[0]))
            return false;
          auto add_op = cast<arith::AddIOp>(ops[0]);
          if (add_op.getLhs() != mask_op.getIterArgInEntry(0))
            return false;
          // FIXME: many cases!
          return true;
        };
        assert(is_where_op());

        auto starts = mask_op.getStarts();
        auto sizes = mask_op.getSizes();
        auto elem_type = mask_op.getElementType();
        assert(starts.size() == 2);

        auto for_op = mask_op->getParentOfType<scf::ForOp>();
        assert(for_op);

        auto name = get_name("where");
        // create zero
        indent() << name << " = " << "tl.zeros([";
        llvm::interleaveComma(sizes, indent(false));
        indent(false) << "], dtype=" << get_tl_dtype(elem_type) << ")\n";

        indent() << name << " = tl.where(";
        indent(false) << symbol_table.lookup(starts[0]) << " + tl.arange(0, " << sizes[0] << ")[:, None]";
        indent(false) << " >= " << symbol_table.lookup(starts[1]) << " + tl.arange(0, " << sizes[1] << ")[None, :], ";
        indent(false) << name << ", " << "float('-inf'))\n";
        symbol_table.insert(mask_op.getResult(), name);
        return WalkResult::skip();
      } else if (auto reduce_op = dyn_cast<ReduceOp>(op)) {
        auto operand = reduce_op.getOperand();
        auto elem_type = cast<RankedTensorType>(operand.getType()).getElementType();
        auto init = reduce_op.getInit();
        auto dim = reduce_op.getReduceDimensionAttr().getInt();
        auto reduce_type = reduce_op.getReduceType();
        auto keep_dim = reduce_op.getKeepDim();

        std::string name_tag;
        std::string callee;
        std::string inplace_op_str;
        assert(reduce_type == ReduceType::ADD || reduce_type == ReduceType::ANY);
        switch (reduce_type) {
        case ReduceType::ADD:
          name_tag = "reduce_sum";
          callee = "tl.sum";
          inplace_op_str = "+=";
          break;
        case ReduceType::ANY:
          // workaround for any because triton not support any
          assert(cast<RankedTensorType>(operand.getType()).getElementType().isInteger(8));
          name_tag = "reduce_any";
          callee = "tl.max";
          inplace_op_str = "|=";
          break;
        default:
          op->dump();
          llvm_unreachable("not support reduce type");
        }
        auto name = get_name(name_tag);
        indent() << name << " = " << callee << "(" << symbol_table.lookup(operand) << ", axis=" << dim
                 << ", keep_dims=" << (keep_dim ? "True" : "False") << ").to(" << get_tl_dtype(elem_type) << ")\n";
        if (init != nullptr) {
          indent() << name << " " << inplace_op_str << " " << symbol_table.lookup(init) << "\n";
        }
        symbol_table.insert(reduce_op.getResult(), name);
      } else if (auto dot_op = dyn_cast<PreciseDotOp>(op)) {
        auto name = get_name("dot");
        auto lhs = dot_op.getLhs();
        auto rhs = dot_op.getRhs();
        auto lhs_type = cast<RankedTensorType>(lhs.getType());
        auto rhs_type = cast<RankedTensorType>(rhs.getType());
        auto res_type = cast<RankedTensorType>(dot_op.getResult().getType());
        assert(lhs_type.getRank() == 2 && rhs_type.getRank() == 2);
        bool is_f16 = lhs_type.getElementType().isF16() && rhs_type.getElementType().isF16();
        bool is_bf16 = lhs_type.getElementType().isBF16() && rhs_type.getElementType().isBF16();
        assert(is_f16 || is_bf16);
        // FIXME: f32? verify
        assert(res_type.getRank() == 2 && res_type.getElementType().isF32());
        indent() << name << " = tl.dot(" << symbol_table.lookup(lhs) << ", " << symbol_table.lookup(rhs) << ")\n";
        symbol_table.insert(dot_op.getResult(), name);
      } else if (auto pow_op = dyn_cast<PowOp>(op)) {
        auto name = get_name("pow");
        auto base = pow_op.getLhs();
        auto exp = pow_op.getRhs();
        if (auto const_op = exp.getDefiningOp<arith::ConstantOp>()) {
          auto val = const_op.getValue();
          assert(isa<DenseElementsAttr>(val));
          auto dense_attr = cast<DenseElementsAttr>(val);
          auto shape = dense_attr.getType().getShape();
          assert(shape.size() == 1 && shape[0] == 1);

          auto v = dense_attr.getValues<APFloat>()[0].convertToDouble();
          assert(v == 2.0);
          auto name = get_name("square");
          indent() << name << " = " << symbol_table.lookup(base) << " * " << symbol_table.lookup(base) << "\n";
          symbol_table.insert(pow_op.getResult(), name);
        } else {
          llvm_unreachable("not support");
        }
      } else if (isa<BroadcastableBinaryOpInterface, arith::AddIOp, arith::SubIOp, arith::MulIOp>(op)) {
        std::string op_str;
        std::string name_tag;
        llvm::TypeSwitch<Operation *>(op)
            .Case<AddOp, arith::AddIOp>([&](auto) {
              op_str = "+";
              name_tag = "add";
            })
            .Case<SubOp, arith::SubIOp>([&](auto) {
              op_str = "-";
              name_tag = "sub";
            })
            .Case<MulOp, arith::MulIOp>([&](auto) {
              op_str = "*";
              name_tag = "mul";
            })
            .Case<DivOp>([&](auto) {
              op_str = "/";
              name_tag = "div";
            })
            .Case<CmpOp>([&](CmpOp cmp_op) {
              auto cmp_type = cmp_op.getCmpType();
              switch (cmp_type) {
              case CmpType::GE:
                op_str = ">=";
                name_tag = "ge";
                break;
              case CmpType::GT:
                op_str = ">";
                name_tag = "gt";
                break;
              default:
                cmp_op->dump();
                llvm_unreachable("not support cmp type");
              }
            })
            .Default([&](Operation *_op) {
              _op->dump();
              llvm_unreachable("unknown binary op");
            });
        auto name = get_name(name_tag);
        auto lhs = op->getOperand(0);
        auto rhs = op->getOperand(1);
        indent() << name << " = " << symbol_table.lookup(lhs) << " " << op_str << " " << symbol_table.lookup(rhs)
                 << "\n";

        // FIXME: const scalar operand will not be converted to correct type, so we need mannually convert it after
        // binary op
        if (auto convert_op = lhs.getDefiningOp<ConvertOp>()) {
          auto tensor_shape = cast<RankedTensorType>(convert_op.getOperand().getType()).getShape();
          auto dst_type = convert_op.getDstType();
          if (tensor_shape.size() == 1 && tensor_shape[0] == 1) {
            // is scalar
            indent() << name << " = " << name << ".to(" << get_tl_dtype(dst_type) << ")\n";
          }
        } else if (auto convert_op = rhs.getDefiningOp<ConvertOp>()) {
          auto tensor_shape = cast<RankedTensorType>(convert_op.getOperand().getType()).getShape();
          auto dst_type = convert_op.getDstType();
          if (tensor_shape.size() == 1 && tensor_shape[0] == 1) {
            // is scalar
            indent() << name << " = " << name << ".to(" << get_tl_dtype(dst_type) << ")\n";
          }
        }

        symbol_table.insert(op->getResult(0), name);
      } else if (isa<ExpOp, Exp2Op, LogOp, Log2Op, NegOp, AbsOp>(op)) {
        std::string func_str;
        std::string name_tag;
        llvm::TypeSwitch<Operation *>(op)
            .Case<ExpOp>([&](auto) {
              func_str = "tl.math.exp";
              name_tag = "exp";
            })
            .Case<Exp2Op>([&](auto) {
              func_str = "tl.math.exp2";
              name_tag = "exp2";
            })
            .Case<LogOp>([&](auto) {
              func_str = "tl.math.log";
              name_tag = "log";
            })
            .Case<Log2Op>([&](auto) {
              func_str = "tl.math.log2";
              name_tag = "log2";
            })
            .Case<NegOp>([&](auto) {
              func_str = "-";
              name_tag = "neg";
            })
            .Default([&](Operation *_op) {
              _op->dump();
              llvm_unreachable("unknown unary op");
            });
        auto name = get_name(name_tag);
        indent() << name << " = " << func_str << "(" << symbol_table.lookup(op->getOperand(0)) << ")\n";
        symbol_table.insert(op->getResult(0), name);
      } else if (isa<TanhOp>(op)) {
        // these ops are not supported by triton directly
        auto operand = op->getOperand(0);
        llvm::TypeSwitch<Operation *>(op)
            .Case<TanhOp>([&](auto) {
              auto name = get_name("tanh");
              auto elem_type = cast<RankedTensorType>(operand.getType()).getElementType();
              assert(elem_type.isF32());
              indent() << name << " = tl.inline_asm_elementwise(" << "\n";
              {
                Indent tab(cur_level, os);
                tab() << "asm=" << "'tanh.approx.f32 $0, $1;'," << "\n";
                tab() << "constraints=('=r,r')," << "\n";
                tab() << "args=[" << symbol_table.lookup(operand) << "]," << "\n";
                tab() << "dtype=(" << get_tl_dtype(elem_type) << ",)," << "\n";
                tab() << "is_pure=True," << "\n";
                tab() << "pack=1," << "\n";
              }
              indent() << ")\n";
              symbol_table.insert(op->getResult(0), name);
            })
            .Default([&](Operation *_op) {
              _op->dump();
              llvm_unreachable("unknown unary op that triton not supported yet");
            });
      } else if (auto unsqueeze_op = dyn_cast<UnsqueezeOp>(op)) {
        auto operand = unsqueeze_op.getOperand();
        auto dim = unsqueeze_op.getDimAttr().getInt();
        auto name = get_name("unsqueeze");
        auto res_rank = cast<RankedTensorType>(unsqueeze_op.getResult().getType()).getRank();
        indent() << name << " = " << symbol_table.lookup(operand) << "[";
        llvm::interleaveComma(llvm::seq<size_t>(0, res_rank), indent(false), [&](size_t i) {
          if ((int)i == dim) {
            indent(false) << "None";
          } else {
            indent(false) << ":";
          }
        });
        indent(false) << "]\n";
        symbol_table.insert(unsqueeze_op.getResult(), name);
      } else if (auto block_ptr_of_op = dyn_cast<triton::BlockPointerOfOp>(op)) {
        auto base_ptr = block_ptr_of_op.getBasePointer();
        auto base_offset = block_ptr_of_op.getBaseOffset();
        auto shape = block_ptr_of_op.getShape();
        auto stride = block_ptr_of_op.getStride();
        auto offset = block_ptr_of_op.getOffset();
        auto block_shape = block_ptr_of_op.getBlockShape();
        auto order = block_ptr_of_op.getOrder();

        auto name = get_name("block_ptr");
        indent() << name << " = tl.make_block_ptr(\n";
        {
          Indent tab(cur_level, os);
          auto make_block_ptr_attr = [&](Twine name, ArrayRef<int64_t> attr) {
            tab() << name << "=(";
            llvm::interleaveComma(attr, tab(false));
            tab(false) << ",),\n";
          };
          tab() << "base=" << symbol_table.lookup(base_ptr) << " + " << symbol_table.lookup(base_offset) << ",\n";
          make_block_ptr_attr("shape", shape);
          make_block_ptr_attr("strides", stride);
          make_block_ptr_attr("offsets", offset);
          make_block_ptr_attr("block_shape", block_shape);
          make_block_ptr_attr("order", order);
        }
        indent() << ")\n";
        symbol_table.insert(block_ptr_of_op.getResult(), name);
      } else if (auto for_op = dyn_cast<scf::ForOp>(op)) {
        auto iv = for_op.getInductionVar();
        auto lb = for_op.getLowerBound();
        auto ub = for_op.getUpperBound();
        auto step = for_op.getStep();

        auto iv_name = get_name("i");
        indent() << "for " << iv_name << " in range(" << symbol_table.lookup(lb) << ", " << symbol_table.lookup(ub)
                 << ", " << symbol_table.lookup(step) << "):\n";
        symbol_table.insert(iv, iv_name);

        // iter args are inplacely updated
        {
          Indent tab(cur_level, os);
          for (auto [res, init_arg, iter_arg] :
               llvm::zip(for_op->getResults(), for_op.getInitArgs(), for_op.getRegionIterArgs())) {
            auto name = symbol_table.lookup(init_arg);
            symbol_table.insert(iter_arg, name);
            symbol_table.insert(res, name);
          }
        }
        trans(for_op.getBody());
        return WalkResult::skip();
      } else if (auto yield_op = dyn_cast<scf::YieldOp>(op)) {
        if (auto for_op = dyn_cast<scf::ForOp>(op->getParentOp())) {
          for (auto [res, init_arg] : llvm::zip(yield_op->getOperands(), for_op.getInitArgs())) {
            auto type = init_arg.getType();
            auto name = symbol_table.lookup(res);
            indent() << symbol_table.lookup(init_arg) << " = " << name << "\n";
          }
        } else {
          op->getParentOp()->dump();
          llvm_unreachable("not support");
        }
      } else if (auto block_load_op = dyn_cast<triton::BlockLoadOp>(op)) {
        auto src_ptr = block_load_op.getSrcPointer();
        auto name = get_name("block_load");
        indent() << name << " = tl.load(" << symbol_table.lookup(src_ptr) << ")\n";
        symbol_table.insert(block_load_op.getResult(), name);
      } else if (auto block_advance_op = dyn_cast<triton::BlockAdvanceOp>(op)) {
        auto src_ptr = block_advance_op.getSrcPointer();
        auto offsets = block_advance_op.getOffsets();
        auto name = get_name("block_advance");
        indent() << name << " = tl.advance(" << symbol_table.lookup(src_ptr) << ", (";
        llvm::interleaveComma(offsets, indent(false));
        indent(false) << ",))\n";
        symbol_table.insert(block_advance_op.getResult(), name);
      } else if (auto block_store_op = dyn_cast<triton::BlockStoreOp>(op)) {
        auto dst_ptr = block_store_op.getDstPointer();
        auto val = block_store_op.getValue();
        auto name = get_name("block_store");
        indent() << name << " = tl.store(" << symbol_table.lookup(dst_ptr) << ", " << symbol_table.lookup(val) << ")\n";
      } else if (isa<triton::DeviceYieldOp>(op)) {
        // pass
      } else {
        llvm::errs() << "op: " << *op << "\n";
        llvm_unreachable("not support");
      }
      return WalkResult::advance();
    });
  }

  void trans(triton::DeviceKernelOp dev_kernel_op, Twine triton_kernel_name, Twine autotune_key_name) {
    CounterGuard guard(name_counter);
    llvm::ScopedHashTableScope<Value, std::string> kernel_scope(symbol_table);

    auto grid = dev_kernel_op.getGrid();
    auto block = &dev_kernel_op.getRegion().front();

    // arg_names
    SmallVector<std::string> arg_names;
    for (size_t i = 0; i < block->getNumArguments() - grid.size(); ++i) {
      auto arg = block->getArgument(i + grid.size());
      auto name = get_name("arg");
      arg_names.push_back(name);
      symbol_table.insert(arg, name);
    }
    // last arg is autotune key
    arg_names.push_back(autotune_key_name.str());

    os << "\n";
    // autotune
    os << "@triton.autotune(configs=[" << "\n";
    SmallVector<int> num_warps;
    num_warps.push_back(4);
    num_warps.push_back(8);
    for (auto num_warp : num_warps) {
      Indent tab(cur_level, os);
      tab() << "triton.Config({}, num_warps=" << num_warp << "),\n";
    }
    os << "], key=['" << autotune_key_name << "'])\n";

    os << "@triton.jit\n";
    os << "def " << triton_kernel_name << "(\n";
    for (auto arg_name : arg_names) {
      Indent tab(cur_level, os);
      tab() << arg_name << ",\n";
    }
    os << "):\n";

    // get program id
    for (size_t i = 0; i < grid.size(); ++i) {
      Indent tab(cur_level, os);
      auto name = get_name("pid");
      tab() << name << " = " << "tl.program_id(" << i << ")\n";
      symbol_table.insert(dev_kernel_op.getIterArgInEntry(i), name);
    }

    trans(block);
  }

  void trans(KernelOp kernel_op, bool import, bool benchmark) {
    if (import) {
      add_imports();
    }
    if (benchmark) {
      add_benchmark(kernel_op);
    }

    CounterGuard guard(name_counter);
    llvm::ScopedHashTableScope<Value, std::string> kernel_scope(symbol_table);

    os << "\ndef " << kernel_op.getSymName() << "(";
    llvm::interleaveComma(llvm::enumerate(kernel_op.getArguments()), os, [&](auto it) {
      auto arg_name = "arg" + std::to_string(it.index());
      os << arg_name << ": torch.Tensor";
      symbol_table.insert(it.value(), arg_name);
    });
    os << ") -> ";
    if (kernel_op.getResultTypes().size() == 1) {
      os << "torch.Tensor:\n";
    } else {
      os << "Tuple[";
      assert(kernel_op.getResultTypes().size() > 1);
      llvm::interleaveComma(kernel_op.getResultTypes(), os, [&](Type type) {
        assert(isa<TensorType>(type));
        os << "torch.Tensor";
      });
      os << "]:\n";
    }

    triton::DeviceKernelOp dev_kernel_op = nullptr;
    // cuda dev
    // FIXME: we use device major cuda capability as autotune key
    Twine dev = "dev";
    Twine autotune_key_name = "autotune_key";
    {
      Indent indent(cur_level, os);
      indent() << dev << " = " << symbol_table.lookup(*kernel_op.getArguments().begin()) << ".device\n";
      indent() << autotune_key_name << " = torch.cuda.get_device_capability(dev)[0]\n";
    }
    Twine triton_kernel_name = kernel_op.getSymName() + "_kernel";
    kernel_op.getCallableRegion()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      Indent indent(cur_level, os);
      if (isa<triton::PointerOfOp, triton::TensorFromOp>(op)) {
        assert(op->getNumOperands() == 1);
        assert(op->getNumResults() == 1);
        // non op
        auto name = get_name("tensor");
        indent() << name << " = " << symbol_table.lookup(op->getOperand(0)) << "\n";
        symbol_table.insert(op->getResult(0), name);
      } else if (auto empty_ptr_op = dyn_cast<triton::EmptyPointerOp>(op)) {
        auto name = get_name("empty_ptr");
        auto tensor_type = cast<RankedTensorType>(empty_ptr_op.getTensorType());
        // alloc
        auto shape = tensor_type.getShape();
        auto elem_type = tensor_type.getElementType();
        indent() << name << " = " << "torch.empty(";
        for (auto dim : shape) {
          indent(false) << dim << ", ";
        }
        indent(false) << "dtype=" << get_torch_dtype(elem_type) << ", device=" << dev << ")\n";
        symbol_table.insert(empty_ptr_op.getResult(), name);
      } else if (isa<triton::DeviceKernelOp>(op)) {
        assert(dev_kernel_op == nullptr);
        dev_kernel_op = cast<triton::DeviceKernelOp>(op);
        auto grid = dev_kernel_op.getGrid();
        Twine grid_name = "grid";
        indent() << grid_name << " = (";
        llvm::interleaveComma(grid, indent(false));
        indent(false) << ")\n";

        indent() << triton_kernel_name << "[" << grid_name << "](";
        llvm::interleaveComma(dev_kernel_op->getOperands(), indent(false),
                              [&](Value v) { indent(false) << symbol_table.lookup(v); });
        indent(false) << ", " << autotune_key_name << ")\n";
        return WalkResult::skip();
      } else if (auto ret_op = dyn_cast<ReturnOp>(op)) {
        indent() << "return ";
        llvm::interleaveComma(ret_op.getOperands(), indent(false),
                              [&](Value v) { indent(false) << symbol_table.lookup(v); });
        indent(false) << "\n";
      } else if (auto avg_pool_op = dyn_cast<AvgPoolOp>(op)) {
        auto name = get_name("avg_pool");
        auto operand = avg_pool_op.getOperand();
        auto kernel_size = avg_pool_op.getKernelSize();
        auto stride = avg_pool_op.getStride();
        auto padding = avg_pool_op.getPadding();
        bool ceil_mode = avg_pool_op.getCeilMode();
        bool count_include_pad = avg_pool_op.getCountIncludePad();

        if (kernel_size.size() == 1) {
          indent() << name << " = F.avg_pool1d(" << symbol_table.lookup(operand) << ", kernel_size=" << kernel_size[0]
                   << ", stride=" << stride[0] << ", padding=" << padding[0]
                   << ", ceil_mode=" << (ceil_mode ? "True" : "False")
                   << ", count_include_pad=" << (count_include_pad ? "True" : "False") << ")\n";
        } else {
          llvm_unreachable("not support");
        }
        symbol_table.insert(avg_pool_op.getResult(), name);
      } else {
        kernel_op->dump();
        llvm::errs() << "op: " << *op << "\n";
        llvm_unreachable("not support");
      }
      return WalkResult::advance();
    });
    if (dev_kernel_op != nullptr) {
      trans(dev_kernel_op, triton_kernel_name, autotune_key_name);
    }
  }

  void trans(ModuleOp module_op, bool benchmark) {
    add_imports();
    SmallVector<std::string> kernel_names;
    for (auto &op : module_op.getOps()) {
      if (auto kernel_op = dyn_cast<KernelOp>(op)) {
        trans(kernel_op, false, benchmark);
        kernel_names.push_back(kernel_op.getSymName().str());
      } else {
        llvm::errs() << "Skip op: " << op.getName() << "\n";
      }
    }

    if (benchmark) {
      // add main
      os << "\n";
      os << "if __name__ == '__main__':\n";
      for (auto kernel_name : kernel_names) {
        Indent indent(cur_level, os);
        indent() << "bench_" + kernel_name << "()\n";
      }
    }
  }

  void add_imports() {
    os << "import math\n";
    os << "import torch\n";
    os << "import torch.nn as nn\n";
    os << "import torch.nn.functional as F\n";
    os << "import triton\n";
    os << "import triton.language as tl\n";
    os << "from typing import Callable, Any, Optional, Tuple\n";
  }

  void add_benchmark(KernelOp kernel_op) {
    CounterGuard guard(name_counter);
    llvm::ScopedHashTableScope<Value, std::string> kernel_scope(symbol_table);

    auto bench_name = "bench_" + kernel_op.getSymName();
    os << "\ndef " << bench_name << "():\n";

    Indent indent(cur_level, os);
    std::string dev = "dev";
    indent() << dev << " = torch.cuda.current_device()\n";
    for (auto arg : kernel_op.getArguments()) {
      auto type = cast<TensorType>(arg.getType());
      auto name = get_name("rand_arg");
      indent() << name << " = torch.randn(";
      llvm::interleaveComma(type.getShape(), indent(false));
      indent(false) << ", dtype=" << get_torch_dtype(type.getElementType()) << ", device=dev)\n";
      symbol_table.insert(arg, name);
    }

    bool use_triton_bench = true;
    if (use_triton_bench) {
      indent() << "avg_ms = triton.testing.do_bench(lambda: " << kernel_op.getSymName() << "(";
      llvm::interleaveComma(kernel_op.getArguments(), indent(false),
                            [&](Value v) { indent(false) << symbol_table.lookup(v); });
      indent(false) << "))\n";
      indent() << "print('[" << kernel_op.getSymName() << "] avg_ms:', avg_ms)\n";
    } else {
      llvm_unreachable("not support");
    }
  }
};

LogicalResult module_to_py_impl(ModuleOp module_op, raw_ostream &os, bool benchmark) {
  PyKernelTranslator(os).trans(module_op, benchmark);
  return success();
}

LogicalResult kernel_to_py_impl(KernelOp kernel_op, raw_ostream &os, bool import, bool benchmark) {
  PyKernelTranslator(os).trans(kernel_op, import, benchmark);
  return success();
}

} // namespace mlir::asuka