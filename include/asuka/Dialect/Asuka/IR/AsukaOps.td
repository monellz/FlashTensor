#ifndef ASUKA_OPS
#define ASUKA_OPS

include "asuka/Dialect/Asuka/IR/AsukaDialect.td"
include "asuka/Dialect/Asuka/IR/AsukaTypes.td"
include "asuka/Dialect/Asuka/IR/AsukaEnums.td"
include "asuka/Dialect/Asuka/IR/AsukaInterfaces.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td" // SymbolUserOpInterface
include "mlir/Interfaces/CallInterfaces.td" // CallOpInterface
include "mlir/IR/OpAsmInterface.td" // OpAsmOpInterface
include "mlir/Interfaces/FunctionInterfaces.td" // FunctionOpInterface
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/ControlFlowInterfaces.td" // ReturnLike

include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultRank


// Op Base
class Asuka_Op<string mnemonic, list<Trait> traits = []> :
    Op<Asuka_Dialect, mnemonic, !listconcat(traits, [DeclareOpInterfaceMethods<FlopsInterface>])> {}


def Asuka_ConvertOp: Asuka_Op<"convert", [Pure, InferTypeOpAdaptor]> {
  let summary = "convert";
  let description = [{}];
  let arguments = (ins
    AS_TypedTensor:$operand,
    TypeAttr:$dst_type
  );

  let results = (outs AS_TypedTensor:$result);
  let assemblyFormat = "$operand `,` `type` `=` $dst_type attr-dict `:` functional-type(operands, results)";
}

def Asuka_EraseTypeOp: Asuka_Op<"erase_type", [Pure, InferTypeOpAdaptor]> {
  let summary = "erase type";

  let description = [{
    erase type
  }];

  let arguments = (ins Variadic<AS_AnyTensor>:$inputs);

  let results = (outs Variadic<AS_ShapedOnlyTensor>:$outputs);

  let assemblyFormat = "$inputs attr-dict `:` functional-type(operands, results)";
}

def Asuka_SpecifyTypeOp: Asuka_Op<"specify_type", [Pure, InferTypeOpAdaptor]> {
  let summary = "specify type";
  let description = [{
    specify type
  }];
  let arguments = (ins
    Variadic<AS_ShapedOnlyTensor>:$inputs,
    TypeArrayAttr:$element_types
  );
  let results = (outs Variadic<AS_TypedTensor>:$outputs);
  let assemblyFormat = "$inputs `with` `type` `=` $element_types attr-dict `:` functional-type(operands, results)";
}

class Asuka_BinaryElementWiseOp<string mnemonic, list<Trait> traits = []> :
  Asuka_Op<mnemonic,
  !listconcat(traits, [Pure, SameOperandsAndResultElementType, BroadcastableBinaryOpInterface, DeclareOpInterfaceMethods<InferTypeOpInterface>, TotalAccessedElementsInterface])> {

  let arguments = (ins
    AS_AnyTensor:$lhs,
    AS_AnyTensor:$rhs
  );

  let results = (outs AS_AnyTensor:$result);

  let assemblyFormat =  "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";

  let hasVerifier = 1;
}

class Asuka_UnaryElementWiseOp<string mnemonic, list<Trait> traits = []> :
  Asuka_Op<mnemonic, !listconcat(traits, [Pure, SameOperandsAndResultType, TotalAccessedElementsInterface])> {

  let arguments = (ins AS_AnyTensor:$operand);

  let results = (outs AS_AnyTensor:$result);

  let assemblyFormat = "$operand attr-dict `:` functional-type(operands, results)";
}

def AddOp: Asuka_BinaryElementWiseOp<"add", [Commutative]> {
  let summary = "add";

  let description = [{
    add
  }];
}

def SubOp: Asuka_BinaryElementWiseOp<"sub", []> {
  let summary = "sub";

  let description = [{
    sub
  }];
}


def MulOp: Asuka_BinaryElementWiseOp<"mul", [Commutative]> {
  let summary = "mul";

  let description = [{
    mul
  }];
}


def DivOp: Asuka_BinaryElementWiseOp<"div", []> {
  let summary = "div";

  let description = [{
    div
  }];
}

def CmpOp: Asuka_Op<"cmp", [Pure, BroadcastableBinaryOpInterface, InferTypeOpAdaptor, TotalAccessedElementsInterface]> {
  let summary = "cmp";
  let description = [{cmp}];
  let arguments = (ins
    AS_AnyTensor:$lhs,
    AS_AnyTensor:$rhs,
    Asuka_CmpTypeAttr:$cmp_type
  );
  let results = (outs AS_BoolTensor:$result);
  let assemblyFormat =  "$cmp_type $lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

// TODO: support different element_type for base/power
def PowOp: Asuka_BinaryElementWiseOp<"pow", []> {
  let summary = "pow";
  let description = [{pow}];
}

def RngOp: Asuka_Op<"rng", [Pure]> {
  let summary = "rng";

  let description = [{
    rng
  }];

  let arguments = (ins
    Asuka_RngDistributionAttr:$rng_distribution
  );

  let results = (outs AS_AnyTensor:$result);

  let assemblyFormat =  "`distribution` `=` $rng_distribution attr-dict `:` type($result)";

}

def SplitOp: Asuka_Op<"split", [Pure]> {
  let summary = "rng";

  let description = [{
    split
  }];

  let arguments = (ins
    AS_AnyTensor:$operand,
    I64Attr:$split_dimension
  );

  let results = (outs Variadic<AS_AnyTensor>:$outputs);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$operand, "ArrayRef<int64_t>":$split_size, "int64_t":$split_dim), [{
      auto type = dyn_cast<RankedTensorType>(operand.getType());
      int64_t rank = type.getRank();
      if (split_dim < 0) {
        split_dim = rank + split_dim;
      }
      assert(split_dim >= 0);
      assert(split_dim < rank);

      auto split_dim_attr = $_builder.getI64IntegerAttr(split_dim);
      auto shape = type.getShape();
      size_t result_num = split_size.size();
      llvm::SmallVector<::mlir::Type, 4> result_types;
      for (size_t i = 0; i < result_num; ++i) {
        llvm::SmallVector<int64_t, 8> new_shape(shape);
        new_shape[split_dim] = split_size[i];
        auto new_type = RankedTensorType::get(new_shape, type.getElementType());
        result_types.push_back(new_type);
      }

      build($_builder, $_state, result_types, operand, split_dim_attr);
    }]>
  ];

  let extraClassDeclaration = [{
    std::vector<int64_t> getSplitSize() {
      std::vector<int64_t> split_size;
      int64_t split_dim = getSplitDimensionAttr().getValue().getSExtValue();
      auto result_types = getResultTypes();
      for (const auto& type: result_types) {
        auto tensor_type = dyn_cast<RankedTensorType>(type);
        auto shape = tensor_type.getShape();
        auto rank = tensor_type.getRank();
        assert(split_dim < rank);
        split_size.push_back(shape[split_dim]);
      }
      return split_size;
    }
  }];


  let assemblyFormat =  "$operand `,` `dim` `=` $split_dimension attr-dict `:` functional-type(operands, results)";

  let hasVerifier = 1;

}

def DotOp: Asuka_Op<"dot", [Pure, SameOperandsAndResultElementType, InferTypeOpAdaptor, TotalAccessedElementsInterface]> {
  // TODO: need interface for input/output element_type
  let summary = "asuka dot";
  let description = [{
    dot op
    for lhs: [x.., m, k]
    for rhs: [y.., k, n]
    k is reduce dim, x.. and y.. are batch dim
  }];
  let arguments = (ins
    AS_AnyTensor:$lhs,
    AS_AnyTensor:$rhs
  );

  let results = (outs AS_AnyTensor:$result);

  let extraClassDeclaration = [{
    int64_t getFlops() {
      auto lhs = getLhs();
      auto rhs = getRhs();
      auto lhs_shape = cast<RankedTensorType>(lhs.getType()).getShape();
      auto rhs_shape = cast<RankedTensorType>(rhs.getType()).getShape();
      int64_t m = lhs_shape[(int)lhs_shape.size() - 2];
      int64_t k = lhs_shape[(int)lhs_shape.size() - 1];
      int64_t n = rhs_shape[(int)rhs_shape.size() - 1];
      int64_t batch = 1;
      for (int i = 0; i < (int)lhs_shape.size() - 2; ++i) {
        batch *= lhs_shape[i];
      }

      return batch * (2 * m * k * n);
    }
  }];

  let assemblyFormat =  "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";

  // let hasVerifier = 1;
}

// FIXME: this op is only used in the final pass that recovers type
def PreciseDotOp: Asuka_Op<"precise_dot_op", [Pure, InferTypeOpAdaptor]> {
  let summary = "Dot operation with output precision at least as high as input precision";
  let description = [{
    This operation performs a dot product on two input tensors with the same element type,
    and ensures the output tensor has a precision (element type) that is at least as high as the inputs.
  }];
  let arguments = (ins
    AS_AnyTensor:$lhs,
    AS_AnyTensor:$rhs,
    TypeAttr:$acc_type
  );
  let results = (outs AS_AnyTensor:$result);
  let assemblyFormat =  "$lhs `,` $rhs `,` `acc` `=` $acc_type attr-dict `:` functional-type(operands, results)";
}

def AvgPoolOp: Asuka_Op<"avg_pool", [Pure, SameOperandsAndResultElementType, InferTypeOpAdaptor]> {
  let summary = "asuka avgpool";  
  let description = [{}];
  let arguments = (ins
    AS_AnyTensor:$operand,
    DenseI64ArrayAttr:$kernel_size,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$padding,
    BoolAttr:$ceil_mode,
    BoolAttr:$count_include_pad
  );

  let results = (outs AS_AnyTensor:$result);

  let assemblyFormat =  "$operand `,` `kernel_size` `=` $kernel_size `,` `stride` `=` $stride `,` `padding` `=` $padding `,` `ceil_mode` `=` $ceil_mode `,` `count_include_pad` `=` $count_include_pad attr-dict `:` functional-type(operands, results)";
}

def SoftmaxOp: Asuka_Op<"softmax", [Pure, SameOperandsAndResultType]> {
  let summary = "asuka softmax";
  let description = [{
    softmax
  }];

  let arguments = (ins
    AS_AnyTensor:$operand,
    I64Attr:$reduce_dimension
  );

  let results = (outs AS_AnyTensor:$result);

  let assemblyFormat =  "$operand `,` `dim` `=` $reduce_dimension attr-dict `:` functional-type(operands, results)";

  // let hasVerifier = 1;
}

def NormalizeOp: Asuka_Op<"normalize", [Pure, SameOperandsAndResultType]> {
  let summary = "asuka normalize";
  let description = [{
    normalize
  }];

  let arguments = (ins
    AS_AnyTensor:$operand,
    I64Attr:$reduce_dimension,
    I64Attr:$lp
  );

  let results = (outs AS_AnyTensor:$result);

  let assemblyFormat =  "$operand `,` `dim` `=` $reduce_dimension `,` `Lp` `=` $lp attr-dict `:` functional-type(operands, results)";

  // let hasVerifier = 1;
}

def ReduceOp: Asuka_Op<"reduce", [Pure, SameOperandsAndResultElementType, InferTypeOpAdaptor]> {
  let summary = "asuka reduce";
  let description = [{
    reduce
  }];

  let arguments = (ins
    AS_AnyTensor:$operand, 
    Optional<AS_AnyTensor>:$init,
    I64Attr:$reduce_dimension,
    Asuka_ReduceTypeAttr:$reduce_type,
    BoolAttr:$keep_dim
  );

  let results = (outs AS_AnyTensor:$result);

  let builders = [
    OpBuilder<(ins "Value":$operand, "IntegerAttr":$reduce_dimension, "::mlir::asuka::ReduceTypeAttr":$reduce_type, "BoolAttr":$keep_dim), [{
      build($_builder, $_state, operand, {}, reduce_dimension, reduce_type, keep_dim);
    }]>,
    OpBuilder<(ins "Value":$operand, "int64_t":$reduce_dimension, "::mlir::asuka::ReduceType":$reduce_type, "bool":$keep_dim), [{
      auto dim = $_builder.getI64IntegerAttr(reduce_dimension);
      auto reduce_ty = ReduceTypeAttr::get($_builder.getContext(), reduce_type);
      auto keep_dim_attr = $_builder.getBoolAttr(keep_dim);
      build($_builder, $_state, operand, {}, dim, reduce_ty, keep_dim_attr);
    }]>
  ];

  let assemblyFormat =  "`(` $operand ( `,` `init` `=` $init^ )? `)` `,` `dim` `=` $reduce_dimension `,` `op` `=` $reduce_type attr-dict `,` `keep_dim` `=` $keep_dim `:` functional-type(operands, results)";

  // let hasVerifier = 1;
}

def TanhOp: Asuka_UnaryElementWiseOp<"tanh", []> {
  let summary = "tanh";
  let description = [{}];
}

def ExpOp: Asuka_UnaryElementWiseOp<"exp", []> {
  let summary = "exp";

  let description = [{
    exp
  }];
}

def Exp2Op: Asuka_UnaryElementWiseOp<"exp2", []> {
  let summary = "exp2";

  let description = [{
    exp2
  }];
}

def LogOp: Asuka_UnaryElementWiseOp<"log", []> {
  let summary = "log";

  let description = [{
    log
  }];
}

def Log2Op: Asuka_UnaryElementWiseOp<"log2", []> {
  let summary = "log2";

  let description = [{
    log2
  }];
}

def NegOp: Asuka_UnaryElementWiseOp<"neg", []> {
  let summary = "neg";

  let description = [{
    neg
  }];
}


def AbsOp: Asuka_UnaryElementWiseOp<"abs", []> {
  let summary = "abs";

  let description = [{
    abs
  }];
}

def ViewOp: Asuka_Op<"view", [Pure]> {
  let summary = "";
  let description = [{
  }];

  let arguments = (ins AS_AnyTensor:$operand);

  let results = (outs AS_AnyTensor:$result);

  let assemblyFormat = "$operand attr-dict `:` functional-type(operands, results)";

  // let hasVerifier = 1;
}


def ReshapeOp: Asuka_Op<"reshape", [Pure]> {
  let summary = "";
  let description = [{
  }];

  let arguments = (ins AS_AnyTensor:$operand);

  let results = (outs AS_AnyTensor:$result);

  let assemblyFormat = "$operand attr-dict `:` functional-type(operands, results)";

  // let hasVerifier = 1;
}

def PermuteOp: Asuka_Op<"permute", [Pure, SameOperandsAndResultElementType, SameOperandsAndResultRank, InferTypeOpAdaptor]> {
  let summary = "";
  let description = [{
  }];

  let arguments = (ins
    AS_AnyTensor:$operand,
    DenseI64ArrayAttr:$dims
  );

  let results = (outs AS_AnyTensor:$result);

  let assemblyFormat = "$operand `,` `dims` `=` $dims attr-dict `:` functional-type(operands, results)";

  // let hasVerifier = 1;
}

def TransposeOp: Asuka_Op<"transpose", [Pure, SameOperandsAndResultElementType, SameOperandsAndResultRank, InferTypeOpAdaptor]> {
  let summary = "";
  let description = [{
  }];

  let arguments = (ins
    AS_AnyTensor:$operand,
    DenseI64ArrayAttr:$dims
  );

  let results = (outs AS_AnyTensor:$result);

  let assemblyFormat = "$operand `,` `dims` `=` $dims attr-dict `:` functional-type(operands, results)";

  // let hasVerifier = 1;
}

def UnsqueezeOp: Asuka_Op<"unsqueeze", [Pure, SameOperandsAndResultElementType, InferTypeOpAdaptor]> {
  let summary = "";
  let description = [{}];
  let arguments = (ins
    AS_AnyTensor:$operand,
    I64Attr:$dim
  );
  let results = (outs AS_AnyTensor:$result);
  let assemblyFormat = "$operand `,` `dim` `=` $dim attr-dict `:` functional-type(operands, results)";
}

def TriluOp: Asuka_Op<"trilu", [Pure, ConstantLike, InferTypeOpAdaptor]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins
    I64Attr:$diagonal,
    BoolAttr:$is_upper,
    DenseI64ArrayAttr:$shape,
    TypedAttrInterface:$val
  );

  let results = (outs AS_AnyTensor:$result);

  let builders = [
    // inf/-inf builder
    OpBuilder<(ins "int64_t":$diagonal, "bool":$is_upper, "ArrayRef<int64_t>":$shape, "::mlir::FloatType":$type, "bool":$negative), [{
      auto ap_inf = ::llvm::APFloat::getInf(type.getFloatSemantics(), negative);
      auto attr = $_builder.getFloatAttr(type, ap_inf);
      build($_builder, $_state, diagonal, is_upper, shape, attr);
    }]>
  ];

  let extraClassDeclaration = [{
    bool isInfVal() {
      if (auto float_attr = dyn_cast<FloatAttr>(getValAttr())) {
        auto val = float_attr.getValue();
        if (val.isInfinity() && !val.isNegative()) return true;
      }
      return false;
    }
    bool isNegInfVal() {
      if (auto float_attr = dyn_cast<FloatAttr>(getValAttr())) {
        auto val = float_attr.getValue();
        if (val.isInfinity() && val.isNegative()) return true;
      }
      return false;
    }
  }];

  let assemblyFormat = "`diagonal` `=` $diagonal `,` `is_upper` `=` $is_upper `,` `shape` `=` $shape `,` `val` `=` $val attr-dict";

  // let hasVerifier = 1;
}

// FIXME: it should be Pure and ConstantLike (maybe we need a pass to explicitly replicate its use or make another zero_op in asuka::triton that is not pure)
def ZeroOp: Asuka_Op<"zero", [InferTypeOpAdaptor]> {
  let summary = "";  
  let description = [{}];
  let arguments = (ins
    DenseI64ArrayAttr:$shape,
    TypeAttr:$element_type
  );
  let results = (outs AS_AnyTensor:$result);
  let assemblyFormat = "`shape` `=` $shape `,` `type` `=` $element_type attr-dict `:` functional-type(operands, results)";
}

def MaskOp: Asuka_Op<"mask", [Pure, SingleBlockImplicitTerminator<"asuka::MaskYieldOp">, InferTypeOpAdaptor]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins
    Variadic<Index>:$starts,
    DenseI64ArrayAttr:$sizes,
    TypeAttr:$element_type
  );

  let results = (outs AS_AnyTensor:$result);

  let regions = (region SizedRegion<1>:$region);

  // let builders = [
  // ];

  let extraClassDeclaration = [{
    Value getStart(int i) {
      return getOperation()->getOperand(i);      
    }
    Value getIterArgInEntry(int i) {
      auto entry = &getRegion().front();
      return entry->getArgument(i);
    }
  }];

  let assemblyFormat = "`starts` `=` `[` $starts `]` `,` `sizes` `=` $sizes `,` `type` `=` $element_type $region attr-dict `:` functional-type(operands, results)";

  // let hasVerifier = 1;
}

def MaskYieldOp: Asuka_Op<"mask_yield", [Pure, ReturnLike, Terminator, HasParent<"MaskOp">]> {
  let summary = "";
  let description = [{}];
  let arguments = (ins Variadic<AS_FloatIntLike>:$srcs);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];

  let assemblyFormat = "attr-dict $srcs `:` type($srcs)";
}


def KernelOp: Asuka_Op<"kernel", [
  AffineScope, AutomaticAllocationScope,
  CallableOpInterface, FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface
]> {
  let summary = "asuka kernel";
  let description = [{
    kernel
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type, 
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$arg_attrs
  )>];

  let extraClassDeclaration = [{
    bool hasParallelMap() {
      return (*this)->hasAttr("parallel_map");
    }

    // FIXME: add squeezed dim/shape explicitly? or dim is squeezed only if size_per_unit == 1?
    struct ParallelMap {
      int unit_num = -1;
      int size_per_unit = -1;
      llvm::SmallVector<int, 4> arg_dims;
      llvm::SmallVector<int, 4> res_dims;

      void show() {
        llvm::errs() << "ParallelMap(";
        llvm::errs() << "unit_num=" << unit_num;
        llvm::errs() << ", size_per_unit=" << size_per_unit;
        llvm::errs() << ", arg_dims=[";
        for (auto arg_dim: arg_dims) {
          llvm::errs() << arg_dim << ",";
        }
        llvm::errs() << "]";
        llvm::errs() << ", res_dims=[";
        for (auto res_dim: res_dims) {
          llvm::errs() << res_dim << ",";
        }
        llvm::errs() << "]";
        llvm::errs() << ")\n";
      }
    };
    
    llvm::SmallVector<ParallelMap> getParallelMaps() {
      llvm::SmallVector<ParallelMap> maps;
      if (!hasParallelMap()) return maps;

      auto array = (*this)->getAttrOfType<ArrayAttr>("parallel_map");
      for (auto attr: array) {
        auto dict = dyn_cast<DictionaryAttr>(attr);
        ParallelMap cur_map;

        cur_map.unit_num = (int) cast<IntegerAttr>(dict.get("unit_num")).getValue().getSExtValue();
        cur_map.size_per_unit = (int) cast<IntegerAttr>(dict.get("size_per_unit")).getValue().getSExtValue();

        auto arg_dims_array = cast<ArrayAttr>(dict.get("arg_dims"));
        for (auto arg_dim_attr: arg_dims_array) {
          cur_map.arg_dims.push_back((int)cast<IntegerAttr>(arg_dim_attr).getValue().getSExtValue());
        }
        auto res_dims_array = cast<ArrayAttr>(dict.get("res_dims"));
        for (auto res_dim_attr: res_dims_array) {
          cur_map.res_dims.push_back((int)cast<IntegerAttr>(res_dim_attr).getValue().getSExtValue());
        }
        maps.push_back(cur_map);
      }
      return maps;
    }

    void setParallelMaps(ArrayRef<ParallelMap> maps) {
      OpBuilder builder(getContext());
      SmallVector<Attribute> array;
      for (const auto& map: maps) {
        SmallVector<NamedAttribute> dict;
        dict.push_back(NamedAttribute(
          builder.getStringAttr("unit_num"),
          builder.getI64IntegerAttr(map.unit_num)
        ));
        dict.push_back(NamedAttribute(
          builder.getStringAttr("size_per_unit"),
          builder.getI64IntegerAttr(map.size_per_unit)
        ));
        SmallVector<Attribute> arg_dims, res_dims;
        for (auto arg_dim: map.arg_dims) {
          arg_dims.push_back(builder.getI64IntegerAttr(arg_dim));
        }
        for (auto res_dim: map.res_dims) {
          res_dims.push_back(builder.getI64IntegerAttr(res_dim));
        }
        dict.push_back(NamedAttribute(
          builder.getStringAttr("arg_dims"),
          builder.getArrayAttr(arg_dims)
        ));
        dict.push_back(NamedAttribute(
          builder.getStringAttr("res_dims"),
          builder.getArrayAttr(res_dims)
        ));

        array.push_back(builder.getDictionaryAttr(dict));
      }
      (*this)->setAttr("parallel_map", builder.getArrayAttr(array));
    }

    int getParallelism() {
      assert(hasParallelMap());
      auto maps = getParallelMaps();
      int ret = 1;
      for (auto& map: maps) {
        ret *= map.unit_num;
      }
      return ret;
    }

    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    ::mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    ::mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }

    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    bool isDeclaration() { return isExternal(); }
  }];
  let hasCustomAssemblyFormat = 1;
}


def ReturnOp : Asuka_Op<"return", [Pure, HasParent<"KernelOp">, /*MemRefsNormalizable, */ReturnLike, Terminator]> {
  let summary = "return";
  let description = [{
    return
  }];

  let arguments = (ins Variadic<AnyType>:$srcs);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];

  let assemblyFormat = "attr-dict ($srcs^ `:` type($srcs))?";
  let hasVerifier = 1;
}

def CallOp : Asuka_Op<"call", [CallOpInterface, /*MemRefsNormalizable, */DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call";
  let description = [{
    call
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "KernelOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
            results, operands);
    }]>];

  let extraClassDeclaration = [{
    FunctionType getCalleeType() {
      return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
    }

    // Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    // Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    // Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
    }

    // Required by CallOpInterface.
    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def ParallelForOp: Asuka_Op<"parallel_for", [Pure, IsolatedFromAbove, HasParent<"KernelOp">, SingleBlockImplicitTerminator<"asuka::ParallelYieldOp">]> {
  let summary = "";
  let description = [{}];
  let arguments = (ins Variadic<AS_AnyTensor>:$inputs);
  let results = (outs Variadic<AS_AnyTensor>:$outputs);
  let regions = (region SizedRegion<1>:$region);
  let extraClassDeclaration = [{
    Value getIterArgInEntry(int i) {
      auto entry = &getRegion().front();
      return entry->getArgument(i);
    }
  }];
  let assemblyFormat = "$inputs $region attr-dict `:` functional-type(operands, results)";
}

def ParallelYieldOp: Asuka_Op<"parallel_yield", [Pure, ReturnLike, Terminator, HasParent<"ParallelForOp">]> {
  let summary = "";
  let description = [{}];
  let arguments = (ins Variadic<AS_AnyTensor>:$srcs);
  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];
  let assemblyFormat = "attr-dict $srcs `:` type($srcs)";
}

// TODO: use an interface to implement common func
def DynamicBlockForOp : Asuka_Op<"dynamic_block_for", [AttrSizedOperandSegments, Pure, AutomaticAllocationScope, SingleBlockImplicitTerminator<"asuka::BlockYieldOp">]> {
  let summary = "block for";
  let description = [{}];

  let arguments = (ins
    Index:$lower_bound,
    Index:$upper_bound,
    IndexAttr:$step, // block_size
    Variadic<AS_AnyTensor>:$block_args,
    DenseI64ArrayAttr:$block_dims,
    Variadic<AS_AnyTensor>:$init_args
  );

  let results = (outs Variadic<AS_AnyTensor>:$results);

  let regions = (region SizedRegion<1>:$region);

  let extraClassDeclaration = [{
    int64_t getNumBlockArgs() {
      auto sizes = getProperties().getOperandSegmentSizes();
      return sizes[2];
    }
    int64_t getNumInitArgs() {
      auto sizes = getProperties().getOperandSegmentSizes();
      return sizes[3];
    }
    Value getBlockArg(int i) {
      assert(i >= 0);
      assert(i < getNumBlockArgs());
      return getOperation()->getOperand(2 + i);
    }
    Value getInitArg(int i) {
      assert(i >= 0);
      assert(i < getNumInitArgs()); 
      int block_arg_num = getNumBlockArgs();
      return getOperation()->getOperand(2 + block_arg_num + i);
    }

    int64_t getBlockDim(int i) {
      assert(i >= 0);
      assert(i < getNumBlockArgs());
      auto block_dims = getBlockDims();
      return block_dims[i];
    }

    Value getIterArgInEntry() {
      auto entry = &getRegion().front();
      return entry->getArgument(0);
    }

    Value getBlockArgInEntry(int i) {
      assert(i >= 0);
      assert(i < getNumBlockArgs());
      auto entry = &getRegion().front();
      return entry->getArgument(1 + i);
    }

    Value getInitArgInEntry(int i) {
      assert(i >= 0);
      assert(i < getNumInitArgs());
      auto entry = &getRegion().front();
      auto block_arg_num = getNumBlockArgs();
      return entry->getArgument(1 + block_arg_num + i);
    }

    Block* addEntryBlock(Location loc) {
      SmallVector<Location> locs(1 + getNumBlockArgs() + getNumInitArgs(), loc);
      return addEntryBlock(locs);
    }
    Block* addEntryBlock(ArrayRef<Location> locs) {
      auto context = getOperation()->getContext();
      auto region = &getRegion();
      assert(region->empty());
      Block *entry = new Block();
      region->push_back(entry);
      // 0 is iter arg
      int block_arg_num = getNumBlockArgs();
      int init_arg_num = getNumInitArgs();
      assert((int)locs.size() == 1 + block_arg_num + init_arg_num);
      entry->addArgument(IndexType::get(context), locs[0]);
      int block_size = getStepAttr().getInt();
      for (int i = 0; i < block_arg_num; ++i) {
        auto arg = getBlockArg(i); 
        auto dim = getBlockDim(i);
        auto arg_type = cast<RankedTensorType>(arg.getType());
        auto shape = arg_type.getShape();
        assert(shape[dim] > block_size && shape[dim] % block_size == 0);
        SmallVector<int64_t> tiled_shape(shape);
        assert(dim < tiled_shape.size());
        tiled_shape[dim] = block_size;

        entry->addArgument(RankedTensorType::get(tiled_shape, arg_type.getElementType()), locs[1 + i]);
      }
      for (int i = 0; i < init_arg_num; ++i) {
        auto arg = getInitArg(i); 
        entry->addArgument(arg.getType(), locs[1 + block_arg_num + i]);
      }

      return entry;
    }
    
  }];

  let builders = [
    OpBuilder<(ins "SmallVector<Type>":$res_types, "Value":$lower_bound, "Value":$upper_bound, "int64_t":$step, "SmallVector<Value>":$block_args, "SmallVector<int64_t>":$block_dims, "SmallVector<Value>":$init_args), [{
      build($_builder, $_state,
        res_types,
        lower_bound,
        upper_bound,
        $_builder.getIndexAttr(step),
        block_args,
        $_builder.getDenseI64ArrayAttr(block_dims),
        init_args
      );
    }]>
  ];

  let assemblyFormat = "`lb` `=` $lower_bound `,` `ub` `=` $upper_bound `,` `step` `=` $step `,` `args` `=` `[` $block_args `]` `,` `dims` `=` $block_dims `,` `init` `=` `[` $init_args `]` $region attr-dict `:` functional-type(operands, results)";
}



def BlockForOp : Asuka_Op<"block_for", [AttrSizedOperandSegments, Pure, AutomaticAllocationScope, SingleBlockImplicitTerminator<"asuka::BlockYieldOp">]> {
  let summary = "block for";
  let description = [{}];

  let arguments = (ins
    IndexAttr:$lower_bound,
    IndexAttr:$upper_bound,
    IndexAttr:$step, // block_size
    Variadic<AS_AnyTensor>:$block_args,
    DenseI64ArrayAttr:$block_dims,
    Variadic<AS_AnyTensor>:$init_args
  );

  let results = (outs Variadic<AS_AnyTensor>:$results);

  let regions = (region SizedRegion<1>:$region);

  let extraClassDeclaration = [{
    int64_t getNumBlockArgs() {
      auto sizes = getProperties().getOperandSegmentSizes();
      return sizes[0];
    }
    int64_t getNumInitArgs() {
      auto sizes = getProperties().getOperandSegmentSizes();
      return sizes[1];
    }
    Value getBlockArg(int i) {
      assert(i >= 0);
      assert(i < getNumBlockArgs());
      return getOperation()->getOperand(i);
    }
    Value getInitArg(int i) {
      assert(i >= 0);
      assert(i < getNumInitArgs()); 
      int block_arg_num = getNumBlockArgs();
      return getOperation()->getOperand(block_arg_num + i);
    }

    int64_t getBlockDim(int i) {
      assert(i >= 0);
      assert(i < getNumBlockArgs());
      auto block_dims = getBlockDims();
      return block_dims[i];
    }

    Value getIterArgInEntry() {
      auto entry = &getRegion().front();
      return entry->getArgument(0);
    }

    Value getBlockArgInEntry(int i) {
      assert(i >= 0);
      assert(i < getNumBlockArgs());
      auto entry = &getRegion().front();
      return entry->getArgument(1 + i);
    }

    Value getInitArgInEntry(int i) {
      assert(i >= 0);
      assert(i < getNumInitArgs());
      auto entry = &getRegion().front();
      auto block_arg_num = getNumBlockArgs();
      return entry->getArgument(1 + block_arg_num + i);
    }

    Block* addEntryBlock(Location loc) {
      SmallVector<Location> locs(1 + getNumBlockArgs() + getNumInitArgs(), loc);
      return addEntryBlock(locs);
    }
    Block* addEntryBlock(ArrayRef<Location> locs) {
      auto context = getOperation()->getContext();
      auto region = &getRegion();
      assert(region->empty());
      Block *entry = new Block();
      region->push_back(entry);
      // 0 is iter arg
      int block_arg_num = getNumBlockArgs();
      int init_arg_num = getNumInitArgs();
      assert((int)locs.size() == 1 + block_arg_num + init_arg_num);
      entry->addArgument(IndexType::get(context), locs[0]);
      int block_size = getStepAttr().getInt();
      for (int i = 0; i < block_arg_num; ++i) {
        auto arg = getBlockArg(i); 
        auto dim = getBlockDim(i);
        auto arg_type = cast<RankedTensorType>(arg.getType());
        auto shape = arg_type.getShape();
        assert(shape[dim] > block_size && shape[dim] % block_size == 0);
        SmallVector<int64_t> tiled_shape(shape);
        assert(dim < tiled_shape.size());
        tiled_shape[dim] = block_size;

        entry->addArgument(RankedTensorType::get(tiled_shape, arg_type.getElementType()), locs[1 + i]);
      }
      for (int i = 0; i < init_arg_num; ++i) {
        auto arg = getInitArg(i); 
        entry->addArgument(arg.getType(), locs[1 + block_arg_num + i]);
      }

      return entry;
    }
    
  }];

  let builders = [
    OpBuilder<(ins "SmallVector<Type>":$res_types, "int64_t":$lower_bound, "int64_t":$upper_bound, "int64_t":$step, "SmallVector<Value>":$block_args, "SmallVector<int64_t>":$block_dims, "SmallVector<Value>":$init_args), [{
      build($_builder, $_state,
        res_types,
        $_builder.getIndexAttr(lower_bound),
        $_builder.getIndexAttr(upper_bound),
        $_builder.getIndexAttr(step),
        block_args,
        $_builder.getDenseI64ArrayAttr(block_dims),
        init_args
      );
    }]>
  ];

  let assemblyFormat = "`lb` `=` $lower_bound `,` `ub` `=` $upper_bound `,` `step` `=` $step `,` `args` `=` `[` $block_args `]` `,` `dims` `=` $block_dims `,` `init` `=` `[` $init_args `]` $region attr-dict `:` functional-type(operands, results)";

  let hasVerifier = 1;
}

def BlockYieldOp : Asuka_Op<"block_yield", [AttrSizedOperandSegments, Pure, ReturnLike, Terminator, ParentOneOf<["BlockForOp", "DynamicBlockForOp"]>]> {
  let summary = "block yield in block for op";
  let description = [{}];
  let arguments = (ins
    Variadic<AS_AnyTensor>:$block_outs,
    Variadic<AS_AnyTensor>:$iter_outs
  );

  let builders = [OpBuilder<(ins), [{
    // workaround for Terminator trait
    ::mlir::ValueRange blocks;
    ::mlir::ValueRange iters;
    build($_builder, $_state, blocks, iters);
  }]>];

  let extraClassDeclaration = [{
    int64_t getNumBlocks() {
      auto sizes = getProperties().getOperandSegmentSizes();
      return sizes[0];
    }
    int64_t getNumIters() {
      auto sizes = getProperties().getOperandSegmentSizes();
      return sizes[1];
    }    

    Value getBlock(int i) {
      assert(i >= 0);
      assert(i < getNumBlocks());
      return getOperation()->getOperand(i);
    }
    Value getIter(int i) {
      assert(i >= 0);
      assert(i < getNumIters());
      auto block_num = getNumBlocks();
      return getOperation()->getOperand(block_num + i);
    }
  }];

  let assemblyFormat = "`block_outs` `=` `[` $block_outs `]` `,` `iter_outs` `=` `[` $iter_outs `]` attr-dict `:` type(operands)";
}


#endif // ASUKA_OPS