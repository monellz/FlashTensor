#ifndef ASUKATRITON_OPS
#define ASUKATRITON_OPS

include "asuka/Dialect/AsukaTriton/IR/AsukaTritonDialect.td"
include "asuka/Dialect/AsukaTriton/IR/AsukaTritonTypes.td"

include "asuka/Dialect/Asuka/IR/AsukaTypes.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/IR/OpAsmInterface.td" // OpAsmOpInterface
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td" // ReturnLike

class AsukaTriton_Op<string mnemonic, list<Trait> traits = []> :
  Op<AsukaTriton_Dialect, mnemonic, traits> {}


def PointerOfOp: AsukaTriton_Op<"ptr_of", [Pure, InferTypeOpAdaptor]> {
  let summary = "point from";
  let description = [{}];

  let arguments = (ins AS_AnyTensor:$operand);
  let results = (outs ASTT_PtrType:$result);

  let assemblyFormat = "$operand attr-dict `:` functional-type(operands, results)";
}

def EmptyPointerOp: AsukaTriton_Op<"empty_ptr", [InferTypeOpAdaptor]> {
  let summary = "point from shape/type";
  let description = [{}];

  let arguments = (ins TypeAttr:$tensor_type);
  let results = (outs ASTT_PtrType:$result);
  let assemblyFormat = "`type` `=` $tensor_type attr-dict `:` type($result)";
}


def TensorFromOp: AsukaTriton_Op<"tensor_from", [Pure, InferTypeOpAdaptor]> {
  let summary = "tensor from pointer";
  let description = [{}];

  let arguments = (ins ASTT_PtrType:$operand);
  let results = (outs AS_AnyTensor:$result);
  let assemblyFormat = "$operand attr-dict `:` functional-type(operands, results)";
}

def DeviceKernelOp: AsukaTriton_Op<"device_kernel",
  [OpAsmOpInterface, IsolatedFromAbove, RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"DeviceYieldOp">,
    HasParent<"::mlir::asuka::KernelOp">]> {
  let summary = "device_kernel";
  let description = [{}];

  // TODO: support non-tensor pointer type
  // let arguments = (ins Variadic<AnyTypeOf<[ASTT_PtrType, AS_Float, AS_Int]>>:$args);
  let arguments = (ins
    Variadic<ASTT_PtrType>:$args,
    DenseI64ArrayAttr:$grid
  );
  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<(ins "ValueRange":$operands, "ArrayRef<int64_t>":$grid), [{
      $_state.addOperands(operands);
      $_state.getOrAddProperties<Properties>().grid = $_builder.getDenseI64ArrayAttr(grid);
      auto region = $_state.addRegion();
      DeviceKernelOp::ensureTerminator(*region, $_builder, $_state.location);
      for (size_t i = 0; i < grid.size(); ++i) {
        region->front().addArgument($_builder.getIndexType(), $_state.location); 
      }
      for (auto v: operands) {
        region->front().addArgument(v.getType(), v.getLoc());
      }
    }]>
  ];

  let extraClassDeclaration = [{
    Value getIterArgInEntry(int i) {
      auto grid = getGrid();
      assert(i >= 0);
      assert(i < (int)grid.size());
      return getRegion().front().getArgument(i);
    }
    Value getNonIterArgInEntry(int i) {
      auto grid = getGrid();
      assert(i >= 0);
      auto block = &getRegion().front();
      auto total_arg_num = block->getNumArguments();
      assert(i < (int)(total_arg_num - grid.size()));
      return block->getArgument(grid.size() + i);
    }
  }];

  let skipDefaultBuilders = 1;
  let assemblyFormat = "`args` `=` `[` $args `]` `,` `grid` `=` $grid $region attr-dict `:` functional-type(operands, results)";
}

def DeviceYieldOp: AsukaTriton_Op<"device_yield", [Pure, Terminator, ReturnLike, HasParent<"DeviceKernelOp">]> {
  let summary = "device_yield";
  let description = [{}];

  let assemblyFormat = "attr-dict";
}

def BlockPointerOfOp: AsukaTriton_Op<"block_ptr_of", [Pure, InferTypeOpAdaptor]> {
  let summary = "block pointer from";
  let description = [{}];

  let arguments = (ins
    ASTT_PtrType:$base_pointer,
    Index:$base_offset,
    DenseI64ArrayAttr:$shape,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$offset,
    DenseI64ArrayAttr:$block_shape,
    DenseI64ArrayAttr:$order
  );

  let builders = [
    // default order (n-1, n-2, ..., 0)
    OpBuilder<(ins "Value":$base_pointer, "Value":$base_offset, "ArrayRef<int64_t>":$shape, "ArrayRef<int64_t>":$stride, "ArrayRef<int64_t>":$offset, "ArrayRef<int64_t>":$block_shape), [{
      SmallVector<int64_t> order;
      for (int i = (int)block_shape.size() - 1; i >= 0; --i) {
        order.push_back(i); 
      }
      build($_builder, $_state, base_pointer, base_offset, shape, stride, offset, block_shape, order);
    }]>
  ];

  let results = (outs ASTT_BlockPtrType:$result);
  let assemblyFormat = "`base` `=` $base_pointer `,` `base_offset` `=` $base_offset `,` `shape` `=` $shape `,` `stride` `=` $stride `,` `offset` `=` $offset `,` `block_shape` `=` $block_shape `,` `order` `=` $order attr-dict `:` functional-type(operands, results)";
}

def BlockLoadOp: AsukaTriton_Op<"block_load", [Pure, InferTypeOpAdaptor]> {
  let summary = "block load";
  let description = [{}];

  let arguments = (ins
    ASTT_BlockPtrType:$src_pointer
  );

  let results = (outs
    AS_AnyTensor:$result
  );

  let assemblyFormat = "$src_pointer attr-dict `:` functional-type(operands, results)";
}

def BlockAdvanceOp: AsukaTriton_Op<"block_advance", [SameOperandsAndResultType]> {
  let summary = "block advance";
  let description = [{}];
  let arguments = (ins
    ASTT_BlockPtrType:$src_pointer,
    DenseI64ArrayAttr:$offsets
  );
  let results = (outs ASTT_BlockPtrType:$next_pointer);
  let assemblyFormat = "$src_pointer `,` `offsets` `=` $offsets attr-dict `:` functional-type(operands, results)";
}

def BlockStoreOp: AsukaTriton_Op<"block_store", []> {
  let summary = "block store";
  let description = [{}];

  let arguments = (ins
    ASTT_BlockPtrType:$dst_pointer,
    AS_AnyTensor:$value
  );
  let assemblyFormat = "$dst_pointer `,` $value attr-dict `:` functional-type(operands, results)";
}

#endif // ASUKATRITON_OPS
