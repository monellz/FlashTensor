#ifndef ASUKATRITON_TYPES
#define ASUKATRITON_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

include "asuka/Dialect/AsukaTriton/IR/AsukaTritonDialect.td"

class AsukaTritonTypeDef<string name, string _mnemonic, list<Trait> traits = []>
  :  TypeDef<AsukaTriton_Dialect, name, traits> {
  let  mnemonic = _mnemonic;
}

def ASTT_PtrType : AsukaTritonTypeDef<"Pointer", "ptr"> {
  let summary = "Pointer type";
  let description = [{}];
  let parameters = (ins "RankedTensorType":$pointee_type);

  let builders = [
    TypeBuilderWithInferredContext<(ins "RankedTensorType":$pointee_type), [{
      return $_get(pointee_type.getContext(), pointee_type);
    }]>
  ];

  let assemblyFormat = "`<` $pointee_type `>`";
}

def ASTT_BlockPtrType : AsukaTritonTypeDef<"BlockPointer", "block_ptr"> {
  let summary = "Block Pointer type";
  let description = [{}];
  let parameters = (ins "RankedTensorType":$pointee_type);

  let builders = [
    TypeBuilderWithInferredContext<(ins "RankedTensorType":$pointee_type), [{
      return $_get(pointee_type.getContext(), pointee_type);
    }]>
  ];

  let assemblyFormat = "`{` $pointee_type `}`";
}



#endif // ASUKATRITON_TYPES